# ./src/llm_telemetry_toolkit/io/utils.py
"""
Provide reusable normalization helpers for log paths and filenames.
Used across logger internals to enforce deterministic and filesystem-safe naming.
Run: Imported as library utilities; no direct command-line execution.
Inputs: Raw names, optional suffix flags, and text components from telemetry records.
Outputs: Sanitized path-safe strings and UTC datetimes.
Side effects: None.
Operational notes: Sanitization removes traversal characters and caps length for portability.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_\- ]+")
_MULTI_UNDERSCORE = re.compile(r"_+")
_MULTI_HYPHEN = re.compile(r"-+")


def now_utc() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def sanitize_path_component(name: str, fallback: str = "unknown") -> str:
    """Return a single safe path segment (no traversal or separators)."""
    if not name:
        return fallback

    normalized = name.replace("\\", "_").replace("/", "_")
    normalized = normalized.replace("..", "_")
    normalized = _UNSAFE_CHARS.sub("_", normalized).strip()
    normalized = normalized.replace(" ", "_")
    normalized = _MULTI_UNDERSCORE.sub("_", normalized)
    normalized = _MULTI_HYPHEN.sub("-", normalized)
    normalized = normalized.strip("._-")
    return normalized[:128] or fallback


def generate_safe_filename(
    name: str, suffix: str = "", timestamp: bool = True, unique_id: bool = False
) -> str:
    """Generate a safe filename with optional timestamp and short UUID segments."""
    safe_name = sanitize_path_component(name, fallback="unknown_entity")
    parts = [safe_name]
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    if unique_id:
        parts.append(str(uuid.uuid4())[:8])

    base_name = "_".join(parts)
    return f"{base_name}{suffix}"
