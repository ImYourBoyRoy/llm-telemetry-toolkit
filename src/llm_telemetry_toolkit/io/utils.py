# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/io/utils.py
"""
Utility functions for the LLM Telemetry Toolkit.
Provides helpers for timestamp generation and safe filename construction.
Inputs: Strings, raw data.
Outputs: Sanitized strings, datetimes.
"""

import re
import uuid
from datetime import datetime, timezone


def now_utc() -> datetime:
    """Returns the current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_safe_filename(
    name: str, suffix: str = "", timestamp: bool = True, unique_id: bool = False
) -> str:
    """Generates a safe filename from a string, optionally adding a timestamp and/or a short UUID."""
    if not name:
        name = "Unknown_Entity"

    # Allow alphanumerics, spaces, hyphens, and underscores; replace others with underscore
    safe_name = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name
    ).strip()
    safe_name = safe_name.replace(" ", "_")
    safe_name = re.sub(r"[_]+", "_", safe_name)
    safe_name = re.sub(r"[-]+", "-", safe_name)
    safe_name = safe_name[:128]  # Truncate to avoid filesystem limits

    parts = [safe_name]
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    if unique_id:
        # Use only first 8 chars of a UUID for brevity
        parts.append(str(uuid.uuid4())[:8])

    base_name = "_".join(parts)
    return f"{base_name}{suffix}"
