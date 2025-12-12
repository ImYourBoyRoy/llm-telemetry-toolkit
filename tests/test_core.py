# ./llm-telemetry-toolkit/tests/test_core.py
"""
Unit tests for Core components (Logger, Context, Decorators).
Uses partial mocking to avoid real file I/O where possible.
"""

import unittest
import time
import shutil
from pathlib import Path
from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit.models.config import TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.core.logger import LLMLogger  # noqa: E402
from llm_telemetry_toolkit.core.context import SessionContext, get_current_session_id  # noqa: E402
from llm_telemetry_toolkit.core.decorators import monitor_interaction  # noqa: E402


class TestCore(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_core_logs")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        self.config = TelemetryConfig(
            session_id="core_test_sess",
            base_log_dir=self.test_dir,
            enable_entity_logging=False,
        )
        # Use a fresh logger instance key per test if possible, or reset
        # Since logger is multiton keyed by session_id, we use unique IDs or rely on cleanup
        self.logger = LLMLogger(self.config)

    def tearDown(self):
        self.logger.shutdown()
        LLMLogger._instances.clear()  # Force reset for next test
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_context_management(self):
        """Test implicit session ID switching."""
        # base_id = self.config.session_id # Unused

        self.assertEqual(get_current_session_id(), None)

        with SessionContext("new_session_123"):
            self.assertEqual(get_current_session_id(), "new_session_123")

        self.assertEqual(get_current_session_id(), None)

    def test_decorator_logging(self):
        """Test @monitor_interaction writes to queue."""

        @monitor_interaction(self.logger, interaction_type="unit_test")
        def dummy_func(x):
            return x * 2

        # Run function
        res = dummy_func(10)
        self.assertEqual(res, 20)

        # Check logger queue indirectly?
        # Or wait for file? Let's wait for file as integration check is stronger here.
        time.sleep(1.2)  # Wait for worker

        subdir = self.test_dir / "llm_interactions" / "core_test_sess"
        files = list(subdir.glob("*.json"))
        files = [f for f in subdir.glob("*.json") if f.name != "session_config.json"]
        self.assertEqual(len(files), 1)

    def test_logger_multiton(self):
        """Ensure multiton returns same instance for same session_id."""
        l1 = LLMLogger(TelemetryConfig(session_id="same", base_log_dir=Path(".")))
        l2 = LLMLogger(TelemetryConfig(session_id="same", base_log_dir=Path(".")))
        self.assertIs(l1, l2)

        l3 = LLMLogger(TelemetryConfig(session_id="diff", base_log_dir=Path(".")))
        self.assertIsNot(l1, l3)

        # Cleanup
        l1.shutdown()
        l3.shutdown()


if __name__ == "__main__":
    unittest.main()
