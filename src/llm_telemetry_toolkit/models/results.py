# ./src/llm_telemetry_toolkit/models/results.py
"""
Define structured responses returned by logging operations.
Used by callers to determine whether writes were queued, confirmed, or failed.
Run: Imported by logger and tests; no direct execution entry point.
Inputs: Logger execution status, interaction identifiers, file paths, and timing data.
Outputs: `LogResult` models suitable for API responses and diagnostics.
Side effects: None.
Operational notes: Distinguishes queued async writes from fully confirmed sync writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class LogResult(BaseModel):
    """Result payload returned by `LLMLogger.log`."""

    success: bool = Field(..., description="Whether logging operation succeeded.")
    interaction_id: str = Field(
        ..., description="Interaction identifier associated with this result."
    )

    # Completion semantics
    queued: bool = Field(
        default=False,
        description="True when the interaction is queued for background write.",
    )
    write_confirmed: bool = Field(
        default=False,
        description="True when file write has completed (sync mode or post-processing confirmation).",
    )

    # Paths for different formats
    primary_log_path: Optional[Path] = Field(
        default=None, description="Path to the primary created file (typically JSON)."
    )
    created_files: List[Path] = Field(
        default_factory=list, description="All files created for this interaction."
    )

    # Feedback
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings generated during processing.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Fatal or actionable errors encountered.",
    )

    # Stats
    latency_ms: float = Field(
        default=0.0, description="Total log call latency in milliseconds."
    )
