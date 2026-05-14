import time
import uuid
from fastapi import FastAPI

from domain.entities.request.rlm_chat_completions_request import RLMChatCompletionRequest
from rlm import RLM
from rlm.logger import RLMLogger
import logging
from typing import Any


logging.basicConfig(level=logging.INFO, encoding="utf-8")
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/rlm/v1/chat/completions")
async def complete_chat(rlm_chat_completion_request: RLMChatCompletionRequest):
    root_prompt, resource = rlm_chat_completion_request.root_prompt, rlm_chat_completion_request.resource
    model = rlm_chat_completion_request.model
    method: Any = rlm_chat_completion_request.method

    rlm = RLM(
        logger=RLMLogger(log_dir='./logs'),
        backend=rlm_chat_completion_request.backend,  # type: ignore[arg-type]
        # Might want to add a custom system prompt
        backend_kwargs={
            "model_name": model,
            "method": method,
        },
       verbose=True,
    )
    print(f'Handling request {resource[:100]}')
    completion = rlm.completion(prompt=resource, root_prompt=root_prompt)
    ret = {
        "id": f'chatcmpl-{uuid.uuid4().hex}',
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion.root_model,
        "sub_model": completion.root_model, # If we want to specify different sub_llms we need to update the signature and return it here. For now it is the same as the root_model
        "choices":[
            {
                "index":0,
                "message":{
                    "role": 'assistant',
                    "content": completion.response,
                    "refusal": "null",
                    "annotations": [],
                },
                "logprobs": 'null',
                "finish_reason": 'stop'
            }
        ],
        "usage": {
            "prompt_tokens": completion.usage_summary.model_usage_summaries[model].total_input_tokens,
            "completion_tokens": completion.usage_summary.model_usage_summaries[model].total_output_tokens,
            "total_tokens": completion.usage_summary.model_usage_summaries[model].total_input_tokens + completion.usage_summary.model_usage_summaries[model].total_output_tokens,
            "total_calls": completion.usage_summary.model_usage_summaries[model].total_calls,
            "total_time": completion.execution_time,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "service_tier": 'default'
    }
    print(f'returning {ret}')
    return ret
