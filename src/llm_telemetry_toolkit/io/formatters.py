# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/io/formatters.py
"""
Output formatters for telemetry logs.
Implements the Strategy pattern to support JSON, CSV, and Markdown outputs.
Inputs: LLMInteraction objects.
Outputs: Formatted strings (JSON/CSV/MD).
"""

import json
import csv
import io
from abc import ABC, abstractmethod

from ..models.schema import LLMInteraction
from ..models.config import TelemetryConfig


class LogFormatter(ABC):
    """Abstract base class for log formatters."""

    @abstractmethod
    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        """Formats the interaction into a string representation."""
        pass

    @abstractmethod
    def file_extension(self) -> str:
        """Returns the file extension for this format (e.g. 'json')."""
        pass


class JsonFormatter(LogFormatter):
    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        # Use simple json.dumps to control ensure_ascii
        data = interaction.model_dump(exclude_none=True)
        return json.dumps(data, indent=config.json_indent, ensure_ascii=False)

    def file_extension(self) -> str:
        return "json"


class MarkdownFormatter(LogFormatter):
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
            lines.append(json.dumps(interaction.metadata, indent=2))
            lines.append("```")

        return "\n".join(lines)

    def file_extension(self) -> str:
        return "md"


class CsvFormatter(LogFormatter):
    def format(self, interaction: LLMInteraction, config: TelemetryConfig) -> str:
        # CSV log files are typically one file per interaction,
        # but flattened structure is key here.
        flat = interaction.model_dump(exclude_none=True)
        # We need to flatten metadata
        meta = flat.pop("metadata", {})
        for k, v in meta.items():
            flat[f"meta_{k}"] = v

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
        return output.getvalue()

    def file_extension(self) -> str:
        return "csv"


class FormatterFactory:
    _formatters = {
        "json": JsonFormatter(),
        "md": MarkdownFormatter(),
        "csv": CsvFormatter(),
    }

    @classmethod
    def get_formatter(cls, fmt: str) -> LogFormatter:
        return cls._formatters.get(fmt.lower(), JsonFormatter())
