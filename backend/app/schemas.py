from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("system_prompt")
    @classmethod
    def blank_prompt_means_default(cls, value: Optional[str]) -> Optional[str]:
        return value or None


class ChatMetadata(BaseModel):
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata
