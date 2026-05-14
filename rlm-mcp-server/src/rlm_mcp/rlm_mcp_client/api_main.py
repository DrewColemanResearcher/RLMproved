import os
import time
import uuid

import httpx
from fastapi import FastAPI
from fastapi import HTTPException

from rlm_mcp.rlm_mcp_client.domain.models import ChatCompletionRequest
from rlm_mcp.rlm_mcp_client.agents.rlm_agent import build_agent, build_mcp_http_client

app = FastAPI()
MCP_SERVER_BASE_URL = os.getenv("MCP_SERVER_BASE_URL", "http://localhost:8000")


async def _preload_document_into_repl(
    document: str,
    mcp_headers: dict[str, str],
    mcp_server_base_url: str = MCP_SERVER_BASE_URL,
) -> None:
    print(mcp_headers)
    timeout = httpx.Timeout(None)
    async with build_mcp_http_client(timeout = timeout, mcp_headers=mcp_headers) as mcp_http_client:
        response = await mcp_http_client.post(
            f"{mcp_server_base_url}/internal/repl/context",
            json={"context": document},
        )
    print(response)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to preload document in the MCP REPL environment: "
                f"{exc.response.status_code} {exc.response.text}"
            ),
        ) from exc


@app.post("/v1/chat/completions")
async def create_chat_completion(chat_completion_request: ChatCompletionRequest):
    username: str = chat_completion_request.username
    chat_id: str = chat_completion_request.chat_id

    root_model: str = chat_completion_request.root_model
    sub_model: str = chat_completion_request.sub_model
    openrouter_api_key: str = chat_completion_request.openrouter_api_key

    messages = chat_completion_request.messages
    # Extract the last user message as the prompt
    user_prompt = next(
        (msg["content"] for msg in reversed(messages) if msg.get("role") == "user"),
        "No user message found"
    )
    print(f"Got user message: {user_prompt}")
    message_history = messages[:-1]  # All but the last message
    print(f"Extracted history: {message_history}")

    mcp_headers = {
        "X-User-Id": username,
        "X-Chat-Id" : chat_id,
        "X-Sub-LLM-Model": sub_model,
        "X-Openrouter-API-Key": openrouter_api_key,
    }

    # Add the document to the MCP environment before the agent starts calling tools.
    # TODO handle many documents in different moments in the chat
    print("Sending document to mcp server...")
    document = chat_completion_request.document
    await _preload_document_into_repl(document=document, mcp_headers=mcp_headers)

    agent = build_agent(mcp_headers=mcp_headers, openrouter_api_key=openrouter_api_key, root_model=root_model)
    async with agent:
        result = await agent.run(
            user_prompt=user_prompt,
            message_history=message_history
        )
    ret = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.response.model_name,
        "sub_model": sub_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.output,
                    "refusal": None,
                    "annotations": [],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.usage().input_tokens,
            "completion_tokens": result.usage().output_tokens,
            "reasoning_tokens": result.usage().details['reasoning_tokens'],
            "total_tokens": result.usage().input_tokens + result.usage().output_tokens,
            "total_tool_calls": result.usage().tool_calls,
            "finish_reason": result.response.finish_reason,

            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "extra": {
            "parts": [message.parts for message in result.all_messages()],
        },
        "service_tier": "default",
    }

    return ret
