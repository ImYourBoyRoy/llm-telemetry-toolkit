# ./llm-telemetry-toolkit/tests/test_models.py
"""
Unit tests for Data Models (Schema, Config, Results).
"""

import unittest
from pathlib import Path
from tests.test_helper import setup_test_environment

# clean pycache before imports
setup_test_environment()

from llm_telemetry_toolkit.models.schema import LLMInteraction  # noqa: E402
from llm_telemetry_toolkit.models.config import TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.models.results import LogResult  # noqa: E402


class TestModels(unittest.TestCase):
    def test_schema_defaults(self):
        """Test default values and UUID generation in LLMInteraction."""
        interaction = LLMInteraction(
            session_id="test_sess",
            model_name="gpt-4",
            prompt="hello",
            response="world",
            response_time_seconds=1.0,
        )
        self.assertTrue(len(interaction.interaction_id) > 0)
        self.assertTrue(interaction.timestamp_utc is not None)
        self.assertEqual(interaction.cost_usd, None)

    def test_config_paths(self):
        """Test TelemetryConfig path handling."""
        config = TelemetryConfig(session_id="config_sess", base_log_dir=Path("./logs"))
        self.assertTrue(isinstance(config.base_log_dir, Path))
        self.assertEqual(config.json_indent, 2)

    def test_log_result_init(self):
        """Test LogResult instantiation."""
        res = LogResult(success=True, interaction_id="123")
        self.assertTrue(res.success)
        self.assertEqual(res.interaction_id, "123")
        self.assertEqual(res.warnings, [])


if __name__ == "__main__":
    unittest.main()
