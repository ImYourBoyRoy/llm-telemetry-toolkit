# ./tests/test_models.py
"""
Validate core Pydantic model behavior for schema, config, and result objects.
Used by CI to prevent regressions in config validation and result semantics.
Run: `python -m pytest tests/test_models.py` or full suite execution.
Inputs: In-memory model constructor arguments.
Outputs: Assertions covering defaults, validation failures, and field contracts.
Side effects: None.
Operational notes: Works with cache-scrub setup to avoid stale import artifacts.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit.models.config import TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.models.results import LogResult  # noqa: E402
from llm_telemetry_toolkit.models.schema import (  # noqa: E402
    ChatMessage,
    LLMInteraction,
    ToolCall,
)


class TestModels(unittest.TestCase):
    def test_schema_defaults(self) -> None:
        """Interaction model should auto-populate key defaults."""
        interaction = LLMInteraction(
            session_id="test_sess",
            model_name="gpt-4",
            prompt="hello",
            response="world",
            response_time_seconds=1.0,
        )
        self.assertTrue(len(interaction.interaction_id) > 0)
        self.assertIsNotNone(interaction.timestamp_utc)
        self.assertIsNone(interaction.cost_usd)

    def test_schema_structured_chat_embedding_fields(self) -> None:
        """Structured chat/tool/embed fields should validate and remain accessible."""
        interaction = LLMInteraction(
            session_id="structured_sess",
            model_name="qwen3:8b",
            prompt="Describe telemetry.",
            response="Telemetry is runtime signal capture.",
            response_time_seconds=0.12,
            request_messages=[{"role": "user", "content": "Describe telemetry."}],
            response_message_role="assistant",
            finish_reason="stop",
            tool_calls=[{"name": "lookup", "arguments": {"topic": "telemetry"}}],
            embedding_input_count=1,
            embedding_vector_count=1,
            embedding_dimensions=1024,
        )
        self.assertEqual(interaction.response_message_role, "assistant")
        self.assertEqual(interaction.finish_reason, "stop")
        self.assertEqual(interaction.embedding_dimensions, 1024)
        self.assertEqual(len(interaction.request_messages or []), 1)
        self.assertEqual(len(interaction.tool_calls or []), 1)
        self.assertIsInstance((interaction.request_messages or [])[0], ChatMessage)
        self.assertIsInstance((interaction.tool_calls or [])[0], ToolCall)

    def test_tool_call_requires_name_or_function(self) -> None:
        with self.assertRaises(ValueError):
            LLMInteraction(
                session_id="bad_tool",
                model_name="qwen3:8b",
                prompt="x",
                response="y",
                response_time_seconds=0.1,
                tool_calls=[{}],
            )

    def test_config_paths(self) -> None:
        """Config should normalize paths and default output formats."""
        config = TelemetryConfig(session_id="config_sess", base_log_dir=Path("./logs"))
        self.assertIsInstance(config.base_log_dir, Path)
        self.assertEqual(config.json_indent, 2)
        self.assertEqual(config.output_formats, ["json"])

    def test_invalid_output_format_raises(self) -> None:
        """Unknown output formats must fail fast for deterministic behavior."""
        with self.assertRaises(ValueError):
            TelemetryConfig(
                session_id="bad_format",
                base_log_dir=Path("./logs"),
                output_formats=["json", "xml"],
            )

    def test_invalid_filename_template_field_raises(self) -> None:
        """Filename templates should only allow whitelisted placeholders."""
        with self.assertRaises(ValueError):
            TelemetryConfig(
                session_id="bad_template",
                base_log_dir=Path("./logs"),
                filename_template="{timestamp}_{unknown}.{ext}",
            )

    def test_invalid_otel_fields_raise(self) -> None:
        """OpenTelemetry name fields should reject empty values."""
        with self.assertRaises(ValueError):
            TelemetryConfig(
                session_id="otel_bad",
                base_log_dir=Path("./logs"),
                otel_span_name="   ",
            )

    def test_invalid_otel_sampler_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryConfig(
                session_id="otel_ratio_bad",
                base_log_dir=Path("./logs"),
                otel_sampler_ratio=1.5,
            )

    def test_otel_auto_configure_requires_exporter_or_console(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryConfig(
                session_id="otel_auto_bad",
                base_log_dir=Path("./logs"),
                otel_auto_configure=True,
                otel_exporter="none",
                otel_enable_console_export=False,
            )

    def test_log_result_init(self) -> None:
        """LogResult should initialize with safe defaults."""
        result = LogResult(success=True, interaction_id="123")
        self.assertTrue(result.success)
        self.assertEqual(result.interaction_id, "123")
        self.assertEqual(result.warnings, [])
        self.assertFalse(result.queued)
        self.assertFalse(result.write_confirmed)


if __name__ == "__main__":
    unittest.main()
