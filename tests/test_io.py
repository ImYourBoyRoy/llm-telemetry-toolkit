# ./llm-telemetry-toolkit/tests/test_io.py
"""
Unit tests for IO components (Formatters, Parser).
"""

import unittest
import json
from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit.models.schema import LLMInteraction  # noqa: E402
from llm_telemetry_toolkit.models.config import TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.io.formatters import FormatterFactory  # noqa: E402
from llm_telemetry_toolkit.io.parser import ContentParser  # noqa: E402


class TestIO(unittest.TestCase):
    def setUp(self):
        self.config = TelemetryConfig(session_id="io_sess", base_log_dir=".")
        self.interaction = LLMInteraction(
            session_id="io_sess",
            interaction_id="test_id_123",
            model_name="model",
            prompt="Hello <think>skipme</think> world",
            response="Answer",
            response_time_seconds=0.5,
        )

    def test_parser_truncation(self):
        """Test content truncation logic."""
        cfg = TelemetryConfig(
            session_id="trunc", base_log_dir=".", max_content_length=5
        )
        text = "Hello World"
        clean = ContentParser.clean_and_truncate(text, cfg)
        self.assertEqual(clean, "Hello...[TRUNCATED]")

    def test_parser_think_tag(self):
        """Test <think> tag extraction."""
        text = "Start <think>Thinking...</think> End"
        thought, final = ContentParser.extract_thought_process(text)
        self.assertEqual(thought, "Thinking...")
        self.assertEqual(final, "Start  End")

        # Test Case Insensitivity
        text2 = "<THINK>Upper</THINK> Done"
        thought2, final2 = ContentParser.extract_thought_process(text2)
        self.assertEqual(thought2, "Upper")
        self.assertEqual(final2, "Done")

    def test_json_formatter(self):
        """Test JSON formatter output."""
        fmt = FormatterFactory.get_formatter("json")
        output = fmt.format(self.interaction, self.config)
        data = json.loads(output)
        self.assertEqual(data["interaction_id"], "test_id_123")

    def test_markdown_formatter(self):
        """Test MD formatter output structure."""
        fmt = FormatterFactory.get_formatter("md")
        # Add thought process manually to test rendering
        self.interaction.thought_process = "Thinking..."
        output = fmt.format(self.interaction, self.config)
        self.assertIn("# Interaction: test_id_123", output)
        self.assertIn("> Thinking...", output)
        self.assertIn("```\nHello <think>skipme</think> world", output)


if __name__ == "__main__":
    unittest.main()
