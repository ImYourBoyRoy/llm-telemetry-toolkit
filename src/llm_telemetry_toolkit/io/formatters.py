# ./src/llm_telemetry_toolkit/io/formatters.py
"""
Render telemetry interactions into on-disk output formats.
Used by `LLMLogger` write paths to produce JSON, Markdown, and CSV records.
Run: Imported by logger internals; not intended for standalone CLI execution.
Inputs: `LLMInteraction` plus `TelemetryConfig` formatting options.
Outputs: Serialized text payloads and deterministic file extensions.
Side effects: None in-memory; file writes are handled by logger components.
Operational notes: Unknown formats fail fast to prevent silent format drift in production.
"""

from __future__ import annotations

import csv
import io
import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models.config import SUPPORTED_OUTPUT_FORMATS, TelemetryConfig
from ..models.schema import LLMInteraction


class LogFormatter(ABC):
    """Abstract contract for telemetry formatters."""

    @abstractmethod
    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        """Return a serialized string representation for one interaction."""

    @abstractmethod
    def file_extension(self) -> str:
        """Return the format extension (without leading dot)."""


class JsonFormatter(LogFormatter):
    """Serialize interactions as pretty JSON while honoring config settings."""

    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        data = interaction.model_dump(mode="json", exclude_none=True)
        return json.dumps(
            data,
            indent=config.json_indent,
            ensure_ascii=config.ensure_ascii,
        )

    def file_extension(self) -> str:
        return "json"


class MarkdownFormatter(LogFormatter):
    """Serialize interactions into a readable markdown report."""

    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        lines = []
        lines.append(f"# Interaction: {interaction.interaction_id}")
        lines.append(
            f"**Session:** {interaction.session_id} | **Time:** {interaction.timestamp_utc}"
        )
        lines.append(
            f"**Model:** {interaction.model_name} | **Type:** {interaction.interaction_type or 'N/A'}"
        )
        lines.append(
            f"**Cost:** ${interaction.cost_usd or 0.0:.6f} | **Latency:** {interaction.response_time_seconds:.2f}s"
        )
        lines.append("")

        lines.append("## Prompt")
        lines.append("```")
        lines.append(interaction.prompt)
        lines.append("```")
        lines.append("")

        if interaction.thought_process:
            lines.append("## Thought Process")
            lines.append("> " + interaction.thought_process.replace("\n", "\n> "))
            lines.append("")

        lines.append("## Response")
        lines.append(interaction.response)
        lines.append("")

        if interaction.metadata:
            lines.append("## Metadata")
            lines.append("```json")
            lines.append(json.dumps(interaction.metadata, indent=2, ensure_ascii=False))
            lines.append("```")

        return "\n".join(lines)

    def file_extension(self) -> str:
        return "md"


class CsvFormatter(LogFormatter):
    """Serialize interactions as a one-row CSV payload with flattened metadata."""

    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        del config  # Reserved for future CSV-specific options.
        flat = interaction.model_dump(mode="json", exclude_none=True)
        metadata = flat.pop("metadata", {})
        for key, value in metadata.items():
            flat[f"meta_{key}"] = _coerce_csv_value(value)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
        return output.getvalue()

    def file_extension(self) -> str:
        return "csv"


def _coerce_csv_value(value: Any) -> str:
    """Convert nested values to safe CSV cells without losing semantic structure."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


class FormatterFactory:
    """Resolve formatter implementations by configured output format name."""

    _formatters: Dict[str, LogFormatter] = {
        "json": JsonFormatter(),
        "md": MarkdownFormatter(),
        "csv": CsvFormatter(),
    }

    @classmethod
    def get_formatter(cls, fmt: str) -> LogFormatter:
        normalized = fmt.strip().lower()
        if normalized not in cls._formatters:
            supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
            raise ValueError(
                f"Unsupported output format '{fmt}'. Supported formats: {supported}."
            )
        return cls._formatters[normalized]
