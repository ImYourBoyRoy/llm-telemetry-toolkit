# ./tests/test_core.py
"""
Exercise core logger, context, and decorator runtime behavior.
Used by CI to validate queue lifecycle, sync semantics, and safe session routing.
Run: `python -m pytest tests/test_core.py` or full-suite invocation.
Inputs: In-memory interactions and temporary filesystem log directories.
Outputs: Assertions around files created, errors raised, and telemetry correctness.
Side effects: Creates and removes temporary directories under the project root.
Operational notes: Ensures shutdown flushes queue so no interactions are dropped.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit.core.context import (  # noqa: E402
    SessionContext,
    get_current_session_id,
)
from llm_telemetry_toolkit.core.decorators import monitor_interaction  # noqa: E402
from llm_telemetry_toolkit.core.logger import LLMLogger  # noqa: E402
from llm_telemetry_toolkit.models.config import TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.models.schema import LLMInteraction  # noqa: E402


class TestCore(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("test_core_logs")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        LLMLogger.clear_instances()
        self.config = TelemetryConfig(
            session_id="core_test_sess",
            base_log_dir=self.test_dir,
            enable_entity_logging=False,
        )
        self.logger = LLMLogger(self.config)

    def tearDown(self) -> None:
        self.logger.shutdown()
        LLMLogger.clear_instances()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_context_management(self) -> None:
        self.assertIsNone(get_current_session_id())
        with SessionContext("new_session_123"):
            self.assertEqual(get_current_session_id(), "new_session_123")
        self.assertIsNone(get_current_session_id())

    def test_sync_logging_waits_and_returns_files(self) -> None:
        original_writer = self.logger._write_to_disk

        def delayed_write(interaction: LLMInteraction) -> list[Path]:
            time.sleep(0.2)
            return original_writer(interaction)

        self.logger._write_to_disk = delayed_write  # type: ignore[method-assign]
        start = time.perf_counter()
        result = self.logger.log(
            LLMInteraction(
                session_id="core_test_sess",
                model_name="sync-model",
                prompt="sync prompt",
                response="sync response",
                response_time_seconds=0.1,
            ),
            sync=True,
        )
        elapsed = time.perf_counter() - start

        self.assertGreaterEqual(elapsed, 0.19)
        self.assertTrue(result.success)
        self.assertTrue(result.write_confirmed)
        self.assertFalse(result.queued)
        self.assertTrue(result.created_files)
        for file_path in result.created_files:
            self.assertTrue(file_path.exists())

    def test_shutdown_flushes_all_queued_logs(self) -> None:
        interaction_count = 12
        for index in range(interaction_count):
            self.logger.log(
                LLMInteraction(
                    session_id="core_test_sess",
                    model_name="queue-model",
                    prompt=f"prompt-{index}",
                    response=f"response-{index}",
                    response_time_seconds=0.01,
                )
            )

        self.logger.shutdown()
        session_dir = self.test_dir / "llm_interactions" / "core_test_sess"
        files = [
            f for f in session_dir.glob("*.json") if f.name != "session_config.json"
        ]
        self.assertEqual(len(files), interaction_count)

    def test_async_write_failures_are_recorded(self) -> None:
        def failing_write(_: LLMInteraction) -> list[Path]:
            raise RuntimeError("simulated disk failure")

        self.logger._write_to_disk = failing_write  # type: ignore[assignment,method-assign]
        result = self.logger.log(
            LLMInteraction(
                session_id="core_test_sess",
                model_name="failing-model",
                prompt="prompt",
                response="response",
                response_time_seconds=0.01,
            )
        )
        self.assertTrue(result.success)
        self.assertFalse(result.write_confirmed)

        self.logger.shutdown()
        errors = self.logger.get_write_errors(result.interaction_id)
        self.assertIn(result.interaction_id, errors)
        self.assertIn("simulated disk failure", errors[result.interaction_id][0])

    def test_async_decorator_logs_actual_result(self) -> None:
        @monitor_interaction(self.logger, interaction_type="async_test")
        async def async_double(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        result = asyncio.run(async_double(6))
        self.assertEqual(result, 12)

        self.logger.shutdown()
        session_dir = self.test_dir / "llm_interactions" / "core_test_sess"
        files = sorted(
            [f for f in session_dir.glob("*.json") if f.name != "session_config.json"],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        self.assertTrue(files)
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["response"], "12")
        self.assertEqual(payload["interaction_type"], "async_test")

    def test_log_errors_false_skips_error_logging(self) -> None:
        @monitor_interaction(self.logger, interaction_type="failing", log_errors=False)
        def explode() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            explode()

        self.logger.shutdown()
        session_dir = self.test_dir / "llm_interactions" / "core_test_sess"
        files = [
            f for f in session_dir.glob("*.json") if f.name != "session_config.json"
        ]
        self.assertEqual(files, [])

    def test_logger_multiton_isolation_uses_full_config_key(self) -> None:
        base_a = Path("test_core_logs_a")
        base_b = Path("test_core_logs_b")
        for folder in (base_a, base_b):
            if folder.exists():
                shutil.rmtree(folder)

        logger_a = None
        logger_b = None
        try:
            cfg_a = TelemetryConfig(session_id="same", base_log_dir=base_a)
            cfg_a_clone = TelemetryConfig(session_id="same", base_log_dir=base_a)
            cfg_b = TelemetryConfig(session_id="same", base_log_dir=base_b)

            logger_a = LLMLogger(cfg_a)
            logger_a_clone = LLMLogger(cfg_a_clone)
            logger_b = LLMLogger(cfg_b)

            self.assertIs(logger_a, logger_a_clone)
            self.assertIsNot(logger_a, logger_b)
            self.assertEqual(logger_b.config.base_log_dir, base_b)
        finally:
            if logger_a is not None:
                logger_a.shutdown()
            if logger_b is not None:
                logger_b.shutdown()
            LLMLogger.clear_instances()
            for folder in (base_a, base_b):
                if folder.exists():
                    shutil.rmtree(folder)

    def test_filename_template_is_applied(self) -> None:
        custom_base = Path("test_core_logs_template")
        if custom_base.exists():
            shutil.rmtree(custom_base)

        custom_logger = None
        try:
            custom_config = TelemetryConfig(
                session_id="templated_session",
                base_log_dir=custom_base,
                filename_template="{session_id}_{type}_{interaction_id}.{ext}",
            )
            custom_logger = LLMLogger(custom_config)
            result = custom_logger.log(
                LLMInteraction(
                    session_id="templated_session",
                    model_name="template-model",
                    prompt="prompt",
                    response="response",
                    response_time_seconds=0.01,
                    interaction_type="tool_call",
                ),
                sync=True,
            )
            self.assertTrue(result.success)
            self.assertTrue(result.created_files)
            name = result.created_files[0].name
            self.assertTrue(name.startswith("templated_session_tool_call_"))
            self.assertTrue(name.endswith(".json"))
        finally:
            if custom_logger is not None:
                custom_logger.shutdown()
            LLMLogger.clear_instances()
            if custom_base.exists():
                shutil.rmtree(custom_base)

    def test_session_directory_is_sanitized(self) -> None:
        safe_base = self.test_dir / "sanitized_root"
        unsafe_config = TelemetryConfig(
            session_id="../../dangerous\\segment",
            base_log_dir=safe_base,
        )
        unsafe_logger = LLMLogger(unsafe_config)
        try:
            safe_root = (safe_base / unsafe_config.session_log_subdir).resolve()
            actual_session_dir = unsafe_logger.session_dir.resolve()
            self.assertTrue(str(actual_session_dir).startswith(str(safe_root)))
            self.assertNotIn("..", str(actual_session_dir))
        finally:
            unsafe_logger.shutdown()
            LLMLogger.clear_instances()

    def test_custom_exporter_is_invoked_for_sync_logs(self) -> None:
        exported_ids: list[str] = []

        class StubExporter:
            def export(self, interaction: LLMInteraction) -> None:
                exported_ids.append(interaction.interaction_id)

        self.logger.register_exporter(StubExporter())
        interaction = LLMInteraction(
            session_id="core_test_sess",
            model_name="export-model",
            prompt="p",
            response="r",
            response_time_seconds=0.01,
        )
        result = self.logger.log(interaction, sync=True)

        self.assertTrue(result.success)
        self.assertEqual(exported_ids, [result.interaction_id])

    def test_otel_exporter_config_path_is_wired(self) -> None:
        exported_ids: list[str] = []

        class StubOtelExporter:
            def __init__(self, **_: str) -> None:
                pass

            def export(self, interaction: LLMInteraction) -> None:
                exported_ids.append(interaction.interaction_id)

        self.logger.shutdown()
        LLMLogger.clear_instances()

        with patch(
            "llm_telemetry_toolkit.observability.OpenTelemetryInteractionExporter",
            StubOtelExporter,
        ):
            otel_logger = LLMLogger(
                TelemetryConfig(
                    session_id="otel_session",
                    base_log_dir=self.test_dir,
                    enable_otel_export=True,
                )
            )
            try:
                interaction = LLMInteraction(
                    session_id="otel_session",
                    model_name="otel-model",
                    prompt="otel prompt",
                    response="otel response",
                    response_time_seconds=0.02,
                )
                result = otel_logger.log(interaction, sync=True)
                self.assertTrue(result.success)
                self.assertEqual(exported_ids, [result.interaction_id])
            finally:
                otel_logger.shutdown()
                LLMLogger.clear_instances()

    def test_mask_pii_redacts_structured_fields(self) -> None:
        self.logger.shutdown()
        LLMLogger.clear_instances()

        secure_logger = LLMLogger(
            TelemetryConfig(
                session_id="pii_structured",
                base_log_dir=self.test_dir,
                mask_pii=True,
                output_formats=["json"],
            )
        )
        try:
            interaction = LLMInteraction(
                session_id="pii_structured",
                model_name="mask-model",
                prompt="email roy@example.com",
                response="ip 203.0.113.10",
                response_time_seconds=0.01,
                request_messages=[
                    {"role": "user", "content": "call +1 (555) 123-4567"}
                ],
                tool_calls=[
                    {
                        "name": "contact_lookup",
                        "arguments": {"email": "roy@example.com"},
                    }
                ],
                metadata={"ip": "198.51.100.42"},
            )
            result = secure_logger.log(interaction, sync=True)
            self.assertTrue(result.success)
            self.assertTrue(result.created_files)

            payload = json.loads(result.created_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["prompt"], "email r***@example.com")
            self.assertIn("xxx.xxx.xxx", payload["response"])
            self.assertIn("***", payload["request_messages"][0]["content"])
            self.assertEqual(
                payload["tool_calls"][0]["arguments"]["email"], "r***@example.com"
            )
            self.assertEqual(payload["metadata"]["ip"], "198.xxx.xxx.xxx")
        finally:
            secure_logger.shutdown()
            LLMLogger.clear_instances()


if __name__ == "__main__":
    unittest.main()
