from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import openai
import requests
from dotenv import load_dotenv

from rlm.clients import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.utils.prompts import RLMStructuredResponse

load_dotenv('./.env')


def _build_messages(prompt: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    messages = prompt
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return messages


def _extract_chat_content(response_payload: dict[str, Any]) -> str:
    return response_payload["choices"][0]["message"]["content"]


def _extract_responses_output_text(response_payload: Any) -> str:
    if hasattr(response_payload, "model_dump"):
        response_payload = response_payload.model_dump()

    if isinstance(response_payload, dict):
        error = response_payload.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
            return f"Error: {message}"

        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Some Aruba responses come back in a chat-completion-like shape.
        choices = response_payload.get("choices")
        if choices:
            try:
                return choices[0]["message"]["content"]
            except Exception:
                pass

        output = response_payload.get("output")
    else:
        output = getattr(response_payload, "output", None)

    if not output:
        raise ValueError("Unable to extract Aruba responses output text.")

    for item in output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "message":
            continue
        content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for chunk in content or []:
            chunk_type = chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
            if chunk_type == "output_text":
                return chunk["text"] if isinstance(chunk, dict) else getattr(chunk, "text", "")
    raise ValueError("Unable to extract Aruba responses output text.")


def _serialize_parsed(parsed: Any, fallback_text: Any = None) -> str:
    if parsed is None:
        if isinstance(fallback_text, str):
            return fallback_text
        return str(fallback_text or "")

    if hasattr(parsed, "model_dump_json"):
        return parsed.model_dump_json()
    if hasattr(parsed, "model_dump"):
        return json.dumps(parsed.model_dump(), ensure_ascii=False)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed)


def _extract_chat_parsed_text(response: Any) -> str:
    try:
        parsed = response.choices[0].message.parsed
    except Exception:
        parsed = None
    content = None
    try:
        content = response.choices[0].message.content
    except Exception:
        content = None
    return _serialize_parsed(parsed, content)


def _extract_responses_parsed_text(response: Any) -> str:
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        parsed = getattr(response, "parsed", None)
    output_text = getattr(response, "output_text", None)
    return _serialize_parsed(parsed, output_text)

class ArubaClientChatCompletion(BaseLM):
    def __init__(self, model_name:str, api_key: str | None = None, keycloak_client_id: str | None = None, keycloak_client_secret: str | None = None, method: str = "legacy"):
        super().__init__(model_name=model_name)
        #self.full_url = "https://api.ai.devops.aruba.it/api/llm/v1/chat/completions"
        self.full_url = "https://aiconsole.ai.devops.aruba.it/v1/chat/completions"
        self.base_url = "https://aiconsole.ai.devops.aruba.it/v1"

        self.api_key = api_key or os.environ.get("ARUBA_API_KEY")
        self.keycloak_client_id = keycloak_client_id or os.environ.get("KEYCLOAK_CLIENT_ID")
        self.keycloak_client_secret = keycloak_client_secret or os.environ.get("KEYCLOAK_CLIENT_SECRET")
        if self.api_key is None or self.keycloak_client_id is None or self.keycloak_client_secret is None:
            raise ValueError("ArubaClient requires api_key, keycloak_client_id, keycloak_client_secret")

        self.model_call_counts = defaultdict(int)
        self.model_input_tokens = defaultdict(int)
        self.model_output_tokens = defaultdict(int)
        self.model_total_tokens = defaultdict(int)
        self.method = method
        self.response_model = RLMStructuredResponse
        self._get_jwt_token()
        self._build_clients()


    def _get_jwt_token(self):
        url = "https://keycloak.ai.devops.aruba.it/realms/ai-platform-aruba/protocol/openid-connect/token"
        payload = f'client_id={self.keycloak_client_id}&client_secret={self.keycloak_client_secret}&grant_type=client_credentials&scope=openid'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        self.jwt_token = response.json()['access_token']

    def _build_headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if 'devops.aruba.it' in self.full_url:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
            headers["ai-platform-key"] = f"Bearer {self.api_key}"
        return headers

    def _build_clients(self) -> None:
        headers = self._build_headers()
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)

    def _refresh_auth(self) -> None:
        self._get_jwt_token()
        self._build_clients()

    async def acompletion( self, prompt: list[dict[str, Any]]) -> str:
        return self.completion(prompt)

    def completion( self, prompt: list[dict[str, Any]],) -> str:
        messages = _build_messages(prompt)
        try:
            if self.method == "parsed_completions":
                response = self.client.beta.chat.completions.parse(  # type: ignore[arg-type]
                    model=self.model_name,
                    messages=messages,
                    response_format=self.response_model,
                )
                self._track_cost(response, self.model_name)
                return _extract_chat_parsed_text(response)

            response = self.client.chat.completions.create(  # type: ignore[arg-type]
                model=self.model_name,
                messages=messages,
            )
            self._track_cost(response, self.model_name)
            return response.choices[0].message.content
        except openai.APIStatusError as exc:
            if getattr(exc, "status_code", None) == 401:
                self._refresh_auth()
                return self.completion(prompt)
            raise
        except openai.AuthenticationError:
            self._refresh_auth()
            return self.completion(prompt)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if 'aiconsole' in self.full_url:
            headers['Authorization'] = f"Bearer {self.api_key}"
        elif 'devops.aruba.it' in self.full_url:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
            headers["ai-platform-key"] = f"Bearer {self.api_key}"
        else:
            raise ValueError("Aruba Endpoint is not correct")
        return headers

    def _track_cost(self, response: dict, model: str):
        self.model_call_counts[model] += 1

        usage = response["usage"] if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            prompt_tokens = completion_tokens = total_tokens = 0
        else:
            prompt_tokens = usage['prompt_tokens'] if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
            completion_tokens = usage['completion_tokens'] if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)
            total_tokens = usage['total_tokens'] if isinstance(usage, dict) else getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens
        self.model_total_tokens[model] += total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=getattr(self, "last_prompt_tokens", 0),
            total_output_tokens=getattr(self, "last_completion_tokens", 0),
        )

class ArubaClientResponses(BaseLM):
    def __init__(self, model_name:str, api_key: str | None = None, keycloak_client_id: str | None = None, keycloak_client_secret: str | None = None, method: str = "parsed_responses"):
        super().__init__(model_name=model_name)
        self.full_url = "https://aiconsole.ai.devops.aruba.it/v1/responses"
        self.base_url = "https://aiconsole.ai.devops.aruba.it/v1"

        self.api_key = api_key or os.environ.get("ARUBA_API_KEY")
        self.keycloak_client_id = keycloak_client_id or os.environ.get("KEYCLOAK_CLIENT_ID")
        self.keycloak_client_secret = keycloak_client_secret or os.environ.get("KEYCLOAK_CLIENT_SECRET")
        if self.api_key is None or self.keycloak_client_id is None or self.keycloak_client_secret is None:
            raise ValueError("ArubaClient requires api_key, keycloak_client_id, keycloak_client_secret")

        self.model_call_counts = defaultdict(int)
        self.model_input_tokens = defaultdict(int)
        self.model_output_tokens = defaultdict(int)
        self.model_total_tokens = defaultdict(int)
        self.method = method
        self.response_model = RLMStructuredResponse
        self._get_jwt_token()
        self._build_clients()


    def _get_jwt_token(self):
        url = "https://keycloak.ai.devops.aruba.it/realms/ai-platform-aruba/protocol/openid-connect/token"
        payload = f'client_id={self.keycloak_client_id}&client_secret={self.keycloak_client_secret}&grant_type=client_credentials&scope=openid'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        self.jwt_token = response.json()['access_token']

    def _build_headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if 'devops.aruba.it' in self.full_url:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
            headers["ai-platform-key"] = f"Bearer {self.api_key}"
        return headers

    def _build_clients(self) -> None:
        headers = self._build_headers()
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)

    def _refresh_auth(self) -> None:
        self._get_jwt_token()
        self._build_clients()

    async def acompletion( self, prompt: list[dict[str, Any]]) -> str:
        return self.completion(prompt)

    def completion( self, prompt: list[dict[str, Any]],) -> str:
        messages = _build_messages(prompt)
        try:
            if self.method == "legacy":
                response = self.client.responses.create(  # type: ignore[arg-type]
                    model=self.model_name,
                    input=messages,
                )
                self._track_cost(response, self.model_name)
                return _extract_responses_output_text(response)

            response = self.client.responses.parse(  # type: ignore[arg-type]
                model=self.model_name,
                input=messages,
                text_format=self.response_model,
            )
            self._track_cost(response, self.model_name)
            return _extract_responses_parsed_text(response)
        except openai.APIStatusError as exc:
            if getattr(exc, "status_code", None) == 401:
                self._refresh_auth()
                return self.completion(prompt)
            raise
        except openai.AuthenticationError:
            self._refresh_auth()
            return self.completion(prompt)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if 'aiconsole' in self.full_url:
            headers['Authorization'] = f"Bearer {self.api_key}"
        elif 'devops.aruba.it' in self.full_url:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
            headers["ai-platform-key"] = f"Bearer {self.api_key}"
        else:
            raise ValueError("Aruba Endpoint is not correct")
        return headers

    def _track_cost(self, response: dict, model: str):
        self.model_call_counts[model] += 1

        usage = response["usage"] if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            input_tokens = output_tokens = total_tokens = 0
        else:
            input_tokens = usage['input_tokens'] if isinstance(usage, dict) else getattr(usage, "input_tokens", 0)
            output_tokens = usage['output_tokens'] if isinstance(usage, dict) else getattr(usage, "output_tokens", 0)
            total_tokens = usage['total_tokens'] if isinstance(usage, dict) else getattr(usage, "total_tokens", input_tokens + output_tokens)

        self.model_input_tokens[model] += input_tokens
        self.model_output_tokens[model] += output_tokens
        self.model_total_tokens[model] += total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = input_tokens
        self.last_completion_tokens = output_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=getattr(self, "last_prompt_tokens", 0),
            total_output_tokens=getattr(self, "last_completion_tokens", 0),
        )
