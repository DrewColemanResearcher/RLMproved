"""
RLM Generic chat completion request entity.
This entity is used to represent the request body for the RLM chat completion endpoint.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rlm.core.types import ClientBackend

class RLMChatCompletionRequest(BaseModel):
    """ RLMChatCompletionRequest entity. """
    root_prompt: str = Field(
        ...,
        title="root_prompt",
        description="Either the prompt from the user directly as a string",
        examples=["Summarize this insanely long document."],
    )
    resource: str = Field(
        ...,
        title="resource",
        description="A string with a long document/resource that should be parsed",
    )
    model: str = Field(
        ...,
        title="model",
        description="A string with the name of the backend large language model to be queried",
    )
    backend: ClientBackend = Field(
        default='openrouter',
        title="backend model service",
        description="A string with the name of the backend to be used for inference",
    )

    method: Literal["legacy", "parsed_completions", "parsed_responses"] = Field(
        default="parsed_completions",
        title="backend method",
        description="Select legacy chat completions, parsed chat completions, or parsed responses.",
    )

    @model_validator(mode="after")
    def validate_backend_method(self):
        if self.method != "legacy" and self.backend not in {"aruba", "openrouter"}:
            raise ValueError(
                "Parsed backend methods are only supported for the 'aruba' and 'openrouter' backends."
            )
        return self

