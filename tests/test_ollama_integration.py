# ./tests/test_ollama_integration.py
"""
Run optional live integration checks against a real Ollama server.
Used to validate client compatibility and telemetry writes with currently installed models.
Run: `OLLAMA_INTEGRATION=1 OLLAMA_BASE_URL=<host> python -m pytest tests/test_ollama_integration.py -q`.
Inputs: Environment variables for host/model selection and local temporary output directory.
Outputs: Assertions on live API responses and generated telemetry files.
Side effects: Makes network calls to Ollama and writes temporary log artifacts under project root.
Operational notes: Skips automatically unless explicitly enabled via `OLLAMA_INTEGRATION=1`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import unittest
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict

from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit import LLMInteraction, LLMLogger, TelemetryConfig  # noqa: E402
from llm_telemetry_toolkit.providers import AsyncOllamaClient, OllamaClient  # noqa: E402


class TestOllamaIntegration(unittest.TestCase):
    """Integration tests for Ollama API compatibility and telemetry output flow."""

    @classmethod
    def setUpClass(cls) -> None:
        settings = _load_local_ollama_settings()

        enabled = os.getenv(
            "OLLAMA_INTEGRATION", settings.get("enable_integration_tests", "0")
        ).strip()
        if enabled.lower() not in {"1", "true", "yes", "on"}:
            raise unittest.SkipTest(
                "Set OLLAMA_INTEGRATION=1 to enable live Ollama integration tests."
            )

        cls.base_url = os.getenv(
            "OLLAMA_BASE_URL", settings.get("base_url", "")
        ).strip()
        if not cls.base_url:
            raise unittest.SkipTest(
                "Set OLLAMA_BASE_URL (for example: http://localhost:11434)."
            )

        cls.text_model = os.getenv(
            "OLLAMA_TEST_TEXT_MODEL",
            settings.get("test_text_model", "tinyllama:latest"),
        ).strip()
        cls.embed_model = os.getenv(
            "OLLAMA_TEST_EMBED_MODEL",
            settings.get("test_embed_model", "mxbai-embed-large:latest"),
        ).strip()

        cls.api_key = (
            os.getenv("OLLAMA_API_KEY", settings.get("api_key", "")).strip() or None
        )
        cls.client = OllamaClient(cls.base_url, api_key=cls.api_key)
        if not cls.client.check_connection():
            raise unittest.SkipTest(f"Unable to connect to Ollama host: {cls.base_url}")

        tags_payload = cls.client.list_models()
        if "error" in tags_payload:
            raise unittest.SkipTest(
                f"Failed to load model list: {tags_payload['error']}"
            )

        cls.installed_models = {
            str(model.get("name"))
            for model in tags_payload.get("models", [])
            if isinstance(model, dict) and model.get("name")
        }
        if not cls.installed_models:
            raise unittest.SkipTest("No models found from Ollama /api/tags response.")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_server_exposes_model_list(self) -> None:
        self.assertGreater(len(self.installed_models), 0)

    def test_chat_or_generate_returns_text(self) -> None:
        if self.text_model not in self.installed_models:
            self.skipTest(f"Model not installed: {self.text_model}")

        prompt = "Reply in 5 words: why sky blue?"
        result = self.client.chat(
            self.text_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.0, "num_predict": 24},
            think=False,
        )
        if "error" in result:
            result = self.client.generate(
                self.text_model,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.0, "num_predict": 24},
                think=False,
            )

        self.assertNotIn("error", result, str(result))
        response_text = _extract_response_text(result)
        self.assertIsInstance(response_text, str)
        self.assertTrue(response_text.strip())

    def test_embed_returns_vector(self) -> None:
        if self.embed_model not in self.installed_models:
            self.skipTest(f"Embedding model not installed: {self.embed_model}")

        result = self.client.embed(self.embed_model, "Telemetry integration check.")
        self.assertNotIn("error", result, str(result))
        embeddings = result.get("embeddings", [])
        self.assertIsInstance(embeddings, list)
        self.assertGreater(len(embeddings), 0)
        self.assertIsInstance(embeddings[0], list)
        self.assertGreater(len(embeddings[0]), 0)

    def test_embed_batch_input_returns_multiple_vectors(self) -> None:
        if self.embed_model not in self.installed_models:
            self.skipTest(f"Embedding model not installed: {self.embed_model}")

        inputs = [
            "Telemetry lets us observe model behavior.",
            "Consistent schema improves downstream analysis.",
        ]
        result = self.client.embed(self.embed_model, inputs)
        self.assertNotIn("error", result, str(result))
        embeddings = result.get("embeddings", [])
        self.assertIsInstance(embeddings, list)
        self.assertGreaterEqual(len(embeddings), len(inputs))

    def test_async_chat_or_generate_returns_text(self) -> None:
        if self.text_model not in self.installed_models:
            self.skipTest(f"Model not installed: {self.text_model}")

        async def run_case() -> Dict[str, Any]:
            async with AsyncOllamaClient(
                self.base_url,
                api_key=self.api_key,
            ) as async_client:
                result = await async_client.chat(
                    self.text_model,
                    messages=[
                        {"role": "user", "content": "Reply in 4 words: telemetry?"}
                    ],
                    stream=False,
                    options={"temperature": 0.0, "num_predict": 24},
                    think=False,
                )
                if "error" in result:
                    result = await async_client.generate(
                        self.text_model,
                        prompt="Reply in 4 words: telemetry?",
                        stream=False,
                        options={"temperature": 0.0, "num_predict": 24},
                        think=False,
                    )
                return result

        result = asyncio.run(run_case())
        self.assertNotIn("error", result, str(result))
        self.assertTrue(_extract_response_text(result).strip())

    def test_chat_tools_payload_roundtrip(self) -> None:
        if self.text_model not in self.installed_models:
            self.skipTest(f"Model not installed: {self.text_model}")

        tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_fact",
                    "description": "Return short fact.",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                    },
                },
            }
        ]
        result = self.client.chat(
            self.text_model,
            messages=[
                {"role": "user", "content": "What is telemetry in one sentence?"}
            ],
            stream=False,
            options={"temperature": 0.0, "num_predict": 48},
            think=False,
            tools=tools_payload,
        )
        if "error" in result and _tool_call_not_supported_error(result):
            self.skipTest(
                "Installed model or runtime does not support tool-call schema."
            )

        self.assertNotIn("error", result, str(result))
        response_text = _extract_response_text(result)
        self.assertTrue(response_text.strip() or _contains_tool_calls(result))

    def test_stream_generate_emits_events(self) -> None:
        if self.text_model not in self.installed_models:
            self.skipTest(f"Model not installed: {self.text_model}")

        events = []
        for event in self.client.stream_generate(
            self.text_model,
            "Reply with one short sentence about telemetry.",
            options={"temperature": 0.0, "num_predict": 40},
            think=False,
        ):
            events.append(event)
            if len(events) >= 3:
                break

        self.assertGreater(len(events), 0)
        self.assertIsInstance(events[0], dict)

    def test_async_stream_chat_emits_events(self) -> None:
        if self.text_model not in self.installed_models:
            self.skipTest(f"Model not installed: {self.text_model}")

        async def run_case() -> int:
            async with AsyncOllamaClient(self.base_url, api_key=self.api_key) as client:
                event_count = 0
                async for _ in client.stream_chat(
                    model=self.text_model,
                    messages=[{"role": "user", "content": "Say telemetry in 5 words."}],
                    options={"temperature": 0.0, "num_predict": 24},
                    think=False,
                ):
                    event_count += 1
                    if event_count >= 3:
                        break
                return event_count

        event_count = asyncio.run(run_case())
        self.assertGreater(event_count, 0)

    def test_live_response_can_be_logged_to_telemetry(self) -> None:
        if self.text_model not in self.installed_models:
            self.skipTest(f"Model not installed: {self.text_model}")

        output_dir = Path("test_ollama_logs")
        if output_dir.exists():
            shutil.rmtree(output_dir)

        logger = LLMLogger(
            TelemetryConfig(
                session_id="ollama_integration",
                base_log_dir=output_dir,
                output_formats=["json"],
                max_content_length=4000,
            )
        )
        try:
            prompt = "Give a one sentence definition of telemetry."
            response_payload = self.client.generate(
                self.text_model,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.1, "num_predict": 60},
                think=False,
            )
            self.assertNotIn("error", response_payload, str(response_payload))

            interaction = LLMInteraction(
                session_id="ollama_integration",
                model_name=self.text_model,
                provider="ollama",
                prompt=prompt,
                response=_extract_response_text(response_payload),
                response_time_seconds=0.0,
                interaction_type="integration_test",
                response_message_role="assistant",
                finish_reason=response_payload.get("done_reason"),
                metadata={
                    "ollama_done_reason": response_payload.get("done_reason"),
                    "ollama_eval_count": response_payload.get("eval_count"),
                },
            )
            result = logger.log(interaction, sync=True)
            self.assertTrue(result.success)
            self.assertTrue(result.write_confirmed)
            self.assertTrue(result.created_files)
            file_path = result.created_files[0]
            self.assertTrue(file_path.exists())

            data: Dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
            self.assertEqual(data["model_name"], self.text_model)
            self.assertEqual(data["provider"], "ollama")
            self.assertEqual(data.get("response_message_role"), "assistant")
        finally:
            logger.shutdown()
            LLMLogger.clear_instances()
            if output_dir.exists():
                shutil.rmtree(output_dir)


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Extract assistant text from either /api/chat or /api/generate response shapes."""
    message = payload.get("message")
    if isinstance(message, dict) and "content" in message:
        content = message.get("content")
        if isinstance(content, str):
            return content

    response = payload.get("response")
    if isinstance(response, str):
        return response

    return ""


def _contains_tool_calls(payload: Dict[str, Any]) -> bool:
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
        return len(message["tool_calls"]) > 0
    return False


def _tool_call_not_supported_error(payload: Dict[str, Any]) -> bool:
    detail = str(payload.get("detail", "")).lower()
    error = str(payload.get("error", "")).lower()
    combined = f"{error} {detail}"
    signals = [
        "tool",
        "unsupported",
        "not implemented",
        "invalid field",
        "does not support",
    ]
    return any(token in combined for token in signals)


def _load_local_ollama_settings() -> Dict[str, str]:
    """Load optional local test settings from ./settings.cfg."""
    settings_path = Path("settings.cfg")
    if not settings_path.exists():
        return {}

    parser = ConfigParser()
    parser.read(settings_path, encoding="utf-8")
    if not parser.has_section("ollama"):
        return {}

    return {
        key.strip(): value.strip()
        for key, value in parser.items("ollama")
        if key and value
    }
