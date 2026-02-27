# ./tests/test_io.py
"""
Test parsing and formatting behavior for telemetry I/O components.
Used by CI to ensure redaction, think-tag parsing, and formatter output integrity.
Run: `python -m pytest tests/test_io.py` or full-suite execution.
Inputs: In-memory interaction payloads and config instances.
Outputs: Assertions for JSON/Markdown/CSV rendering and parser transforms.
Side effects: None.
Operational notes: Includes format validation coverage for fail-fast unknown formats.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit.io.formatters import FormatterFactory  # noqa: E402
from llm_telemetry_toolkit.io.parser import ContentParser  # noqa: E402
from llm_telemetry_toolkit.models.config import TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.models.schema import ChatMessage, LLMInteraction  # noqa: E402


class TestIO(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TelemetryConfig(session_id="io_sess", base_log_dir=Path("."))
        self.interaction = LLMInteraction(
            session_id="io_sess",
            interaction_id="test_id_123",
            model_name="model",
            prompt="Hello <think>skipme</think> world",
            response="Answer",
            response_time_seconds=0.5,
        )

    def test_parser_truncation(self) -> None:
        cfg = TelemetryConfig(
            session_id="trunc", base_log_dir=Path("."), max_content_length=5
        )
        text = "Hello World"
        clean = ContentParser.clean_and_truncate(text, cfg)
        self.assertEqual(clean, "Hello...[TRUNCATED]")

    def test_parser_think_tag(self) -> None:
        text = "Start <think>Thinking...</think> End"
        thought, final = ContentParser.extract_thought_process(text)
        self.assertEqual(thought, "Thinking...")
        self.assertEqual(final, "Start  End")

        text2 = "<THINK>Upper</THINK> Done"
        thought2, final2 = ContentParser.extract_thought_process(text2)
        self.assertEqual(thought2, "Upper")
        self.assertEqual(final2, "Done")

    def test_recursive_redaction_handles_nested_payloads(self) -> None:
        payload = {
            "email": "roy@example.com",
            "nested": [{"ip": "203.0.113.10"}, {"phone": "+1 (555) 123-4567"}],
        }
        redacted = ContentParser.redact_pii_recursive(payload)
        self.assertEqual(redacted["email"], "r***@example.com")
        self.assertEqual(redacted["nested"][0]["ip"], "203.xxx.xxx.xxx")
        self.assertIn("***", redacted["nested"][1]["phone"])

    def test_recursive_redaction_handles_pydantic_models(self) -> None:
        message = ChatMessage(role="user", content="email roy@example.com")
        redacted = ContentParser.redact_pii_recursive(message)
        self.assertIsInstance(redacted, ChatMessage)
        self.assertEqual(redacted.content, "email r***@example.com")

    def test_json_formatter_honors_ensure_ascii(self) -> None:
        fmt = FormatterFactory.get_formatter("json")
        interaction = self.interaction.model_copy(
            update={"prompt": "café", "response": "naïve"}
        )

        escaped = TelemetryConfig(
            session_id="ascii_true",
            base_log_dir=Path("."),
            ensure_ascii=True,
        )
        escaped_output = fmt.format(interaction, escaped)
        self.assertIn("\\u00e9", escaped_output)

        unescaped = TelemetryConfig(
            session_id="ascii_false",
            base_log_dir=Path("."),
            ensure_ascii=False,
        )
        unescaped_output = fmt.format(interaction, unescaped)
        self.assertIn("café", unescaped_output)

    def test_json_formatter_output(self) -> None:
        fmt = FormatterFactory.get_formatter("json")
        output = fmt.format(self.interaction, self.config)
        data = json.loads(output)
        self.assertEqual(data["interaction_id"], "test_id_123")

    def test_markdown_formatter_output_structure(self) -> None:
        fmt = FormatterFactory.get_formatter("md")
        self.interaction.thought_process = "Thinking..."
        output = fmt.format(self.interaction, self.config)
        self.assertIn("# Interaction: test_id_123", output)
        self.assertIn("> Thinking...", output)
        self.assertIn("```\nHello <think>skipme</think> world", output)

    def test_unknown_formatter_raises(self) -> None:
        with self.assertRaises(ValueError):
            FormatterFactory.get_formatter("xml")


if __name__ == "__main__":
    unittest.main()
