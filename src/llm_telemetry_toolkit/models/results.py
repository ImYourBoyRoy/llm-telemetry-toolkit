# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/models/results.py
"""
Response models for logging operations.
Provides distinct feedback objects (LogResult) to callers.
Inputs: Operation status and paths.
Outputs: LogResult object.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class LogResult(BaseModel):
    """
    Feedback object returned after logging an interaction.
    Allows the caller (Agent/User) to know the outcome and location of the logs.
    """

    success: bool = Field(..., description="Whether the log was successfully written.")

    # Paths for different formats
    primary_log_path: Optional[Path] = Field(
        default=None, description="Path to the primary log file (usually JSON)."
    )
    created_files: List[Path] = Field(
        default_factory=list, description="List of all files created."
    )

    # Feedback
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during logging (e.g. 'Truncated output').",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered (e.g. 'CSV write failed').",
    )

    # Stats
    interaction_id: str
    latency_ms: float = Field(default=0.0, description="Time taken to write logs.")
