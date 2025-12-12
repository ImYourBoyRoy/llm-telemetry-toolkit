# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/models/schema.py
"""
Data models for LLM Interactions.
Defines the structure of the telemetry data using Pydantic.
Inputs: Raw interaction data.
Outputs: Validated LLMInteraction objects.
"""

import uuid
from typing import Dict, Any, Optional

from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


def now_utc() -> datetime:
    """Returns the current UTC datetime."""
    return datetime.now(timezone.utc)


class LLMInteraction(BaseModel):
    """
    Represents a single interaction with an LLM (Large Language Model).
    Used for structured logging and telemetry.
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
    thought_process: Optional[str] = None  # For Chain-of-Thought

    # Performance
    response_time_seconds: float
    token_count_prompt: Optional[int] = None
    token_count_response: Optional[int] = None
    cost_usd: Optional[float] = None

    # Context
    tool_name: Optional[str] = None
    agent_name: Optional[str] = None
    task_context: Optional[str] = None
    interaction_type: Optional[str] = (
        None  # e.g. "tool_call", "conversation", "embedding"
    )

    # Entity Context (for localized logging)
    entity_id: Optional[str] = None  # e.g. "Company_123"
    entity_label: Optional[str] = None  # e.g. "iComply"

    # Validation
    confidence_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    error_message: Optional[str] = None

    # Flexible Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="allow")
