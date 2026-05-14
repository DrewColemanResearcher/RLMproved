import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, cast

import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.utils.prompts import RLMStructuredResponse

load_dotenv()

# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        method: str = "legacy",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY

        # For vLLM, set base_url to local vLLM server address.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.method = method
        self.response_model = RLMStructuredResponse

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def _build_messages(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _serialize_parsed(self, parsed: Any, fallback_text: Any = None) -> str:
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

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = getattr(response, "output", None)
        if output:
            collected: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    content = item.get("content", [])
                else:
                    content = getattr(item, "content", [])
                for chunk in content or []:
                    chunk_type = getattr(chunk, "type", None) or (chunk.get("type") if isinstance(chunk, dict) else None)
                    if chunk_type in {"output_text", "text"}:
                        text = getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else None)
                        if text:
                            collected.append(str(text))
            if collected:
                return "".join(collected)

        choices = getattr(response, "choices", None)
        if choices:
            message = choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                collected = []
                for chunk in content:
                    text = getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else None)
                    if text:
                        collected.append(str(text))
                if collected:
                    return "".join(collected)

        raise ValueError("Unable to extract text from response payload.")

    def _extract_chat_parsed_text(self, response: Any) -> str:
        try:
            parsed = response.choices[0].message.parsed
        except Exception:
            parsed = None
        try:
            content = response.choices[0].message.content
        except Exception:
            content = None
        return self._serialize_parsed(parsed, content)

    def _extract_responses_parsed_text(self, response: Any) -> str:
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            parsed = getattr(response, "parsed", None)
        output_text = getattr(response, "output_text", None)
        return self._serialize_parsed(parsed, output_text)

    def _extract_usage(self, response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = getattr(usage, "input_tokens", 0)

        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None:
            completion_tokens = getattr(usage, "output_tokens", 0)

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        return int(prompt_tokens), int(completion_tokens), int(total_tokens)

    def _track_cost(self, response: Any, model: str):
        self.model_call_counts[model] += 1

        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)

        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens
        self.model_total_tokens[model] += total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def _chat_completion_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        if str(self.client.base_url).rstrip("/") == DEFAULT_PRIME_INTELLECT_BASE_URL.rstrip("/"):
            kwargs["extra_body"] = {"usage": {"include": True}}
        return kwargs

    def _responses_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if str(self.client.base_url).rstrip("/") == DEFAULT_PRIME_INTELLECT_BASE_URL.rstrip("/"):
            kwargs["extra_body"] = {"usage": {"include": True}}
        return kwargs

    def _chat_parse_kwargs(self) -> Dict[str, Any]:
        kwargs = self._chat_completion_kwargs()
        kwargs["response_format"] = self.response_model
        return kwargs

    def _responses_parse_kwargs(self) -> Dict[str, Any]:
        kwargs = self._responses_kwargs()
        kwargs["text_format"] = self.response_model
        return kwargs

    def completion(self, prompt: Union[str, List[Dict[str, Any]]], model: Optional[str] = None) -> str:
        messages = self._build_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        messages_for_sdk = cast(Any, messages)
        if self.method == "parsed_completions":
            response = self.client.beta.chat.completions.parse(  # type: ignore[arg-type]
                model=model, messages=messages_for_sdk, **self._chat_parse_kwargs()
            )
            self._track_cost(response, model)
            return self._extract_chat_parsed_text(response)

        if self.method == "parsed_responses":
            response = self.client.responses.parse(  # type: ignore[arg-type]
                model=model, input=messages_for_sdk, **self._responses_parse_kwargs()
            )
            self._track_cost(response, model)
            return self._extract_responses_parsed_text(response)

        response = self.client.chat.completions.create(
            model=model, messages=messages_for_sdk, **self._chat_completion_kwargs()
        )
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self, prompt: Union[str, List[Dict[str, Any]]], model: Optional[str] = None
    ) -> str:
        messages = self._build_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        messages_for_sdk = cast(Any, messages)
        if self.method == "parsed_completions":
            response = await self.async_client.beta.chat.completions.parse(  # type: ignore[arg-type]
                model=model, messages=messages_for_sdk, **self._chat_parse_kwargs()
            )
            self._track_cost(response, model)
            return self._extract_chat_parsed_text(response)

        if self.method == "parsed_responses":
            response = await self.async_client.responses.parse(  # type: ignore[arg-type]
                model=model, input=messages_for_sdk, **self._responses_parse_kwargs()
            )
            self._track_cost(response, model)
            return self._extract_responses_parsed_text(response)

        response = await self.async_client.chat.completions.create(
            model=model, messages=messages_for_sdk, **self._chat_completion_kwargs()
        )
        self._track_cost(response, model)
        return response.choices[0].message.content


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
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
