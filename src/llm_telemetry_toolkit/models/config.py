# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/models/config.py
"""
Configuration model for the LLM Telemetry Toolkit.
Defines Pydantic models for strictly typed configuration injection.
Inputs: Dictionary or kwargs for config.
Outputs: Validated TelemetryConfig object.
"""

from pathlib import Path
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class TelemetryConfig(BaseModel):
    """
    Configuration for the LLM Telemetry Toolkit.
    """

    session_id: str = Field(
        default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Unique identifier for the current session. Defaults to 'session_YYYYMMDD_HHMMSS'.",
    )
    base_log_dir: Path = Field(
        default=Path("./logs"),
        description="Root directory for storing logs. Defaults to './logs'.",
    )

    # Feature Flags
    enable_session_logging: bool = Field(
        default=True, description="Enable main session logging."
    )
    enable_entity_logging: bool = Field(
        default=False, description="Enable separate logs per entity (e.g. company)."
    )

    # Paths
    session_log_subdir: str = Field(
        default="llm_interactions", description="Subdirectory for session logs."
    )
    entity_log_subdir: str = Field(
        default="entity_llm_interactions", description="Subdirectory for entity logs."
    )

    # Formatting
    output_formats: List[str] = Field(
        default=["json"], description="List of formats to output: 'json', 'csv', 'md'."
    )
    filename_template: str = Field(
        default="{timestamp}_{interaction_id}_{type}.{ext}",
        description="Template for log filenames.",
    )
    json_indent: int = Field(default=2, description="Indentation level for JSON logs.")
    ensure_ascii: bool = Field(
        default=False, description="Escape non-ASCII characters in JSON."
    )

    # Data Hygiene
    max_content_length: Optional[int] = Field(
        default=None,
        description="Maximum characters for prompt/response in logs. None means no truncation.",
    )

    # Privacy / Security
    mask_pii: bool = Field(
        default=False,
        description="Enable smart PII redaction (Email, IP, Phone, Credit Card).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
