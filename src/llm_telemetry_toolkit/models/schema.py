# ./src/llm_telemetry_toolkit/models/schema.py
"""
Define the canonical telemetry interaction schema used across toolkit components.
Used by logger, formatters, tests, and integrations producing structured records.
Run: Imported as package model definitions; not a standalone executable module.
Inputs: Raw interaction/tool/message payloads from LLM and agent workflows.
Outputs: Validated Pydantic models with typed chat/tool/embedding metadata.
Side effects: None.
Operational notes: `extra="allow"` preserves provider-specific fields for forward compatibility.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


class ToolFunctionCall(BaseModel):
    """Normalized tool function call payload."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ToolFunctionCall.name must be non-empty.")
        return normalized


class ToolCall(BaseModel):
    """Structured tool-call metadata supporting Ollama/OpenAI style payloads."""

    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[ToolFunctionCall] = None
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _normalize_shape(self) -> "ToolCall":
        if self.function is None and self.name:
            self.function = ToolFunctionCall(
                name=self.name,
                arguments=self.arguments or {},
            )
        if self.function is None:
            raise ValueError(
                "ToolCall must include either `function` or top-level `name`."
            )
        return self


class ChatMessage(BaseModel):
    """Structured chat message payload with optional multimodal/tool metadata."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    images: Optional[List[str]] = None
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ChatMessage.role must be non-empty.")
        return normalized


class LLMInteraction(BaseModel):
    """
    Represents a single interaction with an LLM.
    Used for structured logging and telemetry persistence.
    """

    # Core Identity
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp_utc: str = Field(default_factory=lambda: now_utc().isoformat())

    # Model & Provider
    model_name: str
    provider: Optional[str] = None

    # Payload
    prompt: str
    response: str
    thought_process: Optional[str] = None

    # Performance
    response_time_seconds: float
    token_count_prompt: Optional[int] = None
    token_count_response: Optional[int] = None
    cost_usd: Optional[float] = None

    # Chat/Tool metadata
    request_messages: Optional[List[ChatMessage]] = None
    response_message_role: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    # Embedding metadata
    embedding_input_count: Optional[int] = None
    embedding_vector_count: Optional[int] = None
    embedding_dimensions: Optional[int] = None

    # Context
    tool_name: Optional[str] = None
    agent_name: Optional[str] = None
    task_context: Optional[str] = None
    interaction_type: Optional[str] = None

    # Entity context
    entity_id: Optional[str] = None
    entity_label: Optional[str] = None

    # Validation
    confidence_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    error_message: Optional[str] = None

    # Flexible metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="allow")
