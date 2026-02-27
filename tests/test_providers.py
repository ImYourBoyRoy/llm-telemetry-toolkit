# ./tests/test_providers.py
"""
Validate Ollama provider adapters for retries, circuit-breaker behavior, and async transport.
Used by CI to keep resilient HTTP behavior stable for agentic integration layers.
Run: `python -m pytest tests/test_providers.py` or full suite execution.
Inputs: In-memory httpx mock transports and synthetic API payloads.
Outputs: Assertions for response handling and reliability controls.
Side effects: None beyond local in-memory HTTP simulation.
Operational notes: Avoids external network calls so tests remain deterministic.
"""

from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from tests.test_helper import setup_test_environment

setup_test_environment()

from llm_telemetry_toolkit.providers import (  # noqa: E402
    AsyncOllamaClient,
    OllamaClient,
    OllamaTransportConfig,
)
from llm_telemetry_toolkit.models.schema import ChatMessage  # noqa: E402


class TestOllamaProviders(unittest.TestCase):
    def test_sync_client_retries_then_succeeds(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    status_code=500,
                    request=request,
                    text="temporary server error",
                )
            return httpx.Response(
                status_code=200,
                request=request,
                json={"models": [{"name": "tinyllama:latest"}]},
            )

        raw_client = httpx.Client(
            base_url="http://test.local",
            transport=httpx.MockTransport(handler),
        )
        client = OllamaClient(
            base_url="http://test.local",
            client=raw_client,
            transport_config=OllamaTransportConfig(
                max_retries=1,
                backoff_initial_seconds=0.0,
                backoff_multiplier=1.0,
            ),
        )
        try:
            payload = client.list_models()
            self.assertNotIn("error", payload)
            self.assertEqual(call_count, 2)
        finally:
            client.close()
            raw_client.close()

    def test_sync_circuit_breaker_opens_after_threshold(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("service unavailable", request=request)

        raw_client = httpx.Client(
            base_url="http://test.local",
            transport=httpx.MockTransport(handler),
        )
        client = OllamaClient(
            base_url="http://test.local",
            client=raw_client,
            transport_config=OllamaTransportConfig(
                max_retries=0,
                circuit_breaker_failure_threshold=1,
                circuit_breaker_recovery_seconds=60.0,
            ),
        )
        try:
            first = client.list_models()
            self.assertIn("error", first)

            second = client.list_models()
            self.assertIn("error", second)
            self.assertIn("Circuit breaker is open", second["error"])
            self.assertEqual(call_count, 1)
        finally:
            client.close()
            raw_client.close()

    def test_async_client_chat_returns_payload(self) -> None:
        async def run_case() -> dict[str, object]:
            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.url.path, "/api/chat")
                return httpx.Response(
                    status_code=200,
                    request=request,
                    json={
                        "message": {"role": "assistant", "content": "Telemetry ready."},
                        "done_reason": "stop",
                    },
                )

            raw_client = httpx.AsyncClient(
                base_url="http://test.local",
                transport=httpx.MockTransport(handler),
            )
            client = AsyncOllamaClient(
                base_url="http://test.local",
                client=raw_client,
                transport_config=OllamaTransportConfig(max_retries=0),
            )
            try:
                return await client.chat(
                    model="tinyllama:latest",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=False,
                )
            finally:
                await client.aclose()
                await raw_client.aclose()

        payload = asyncio.run(run_case())
        self.assertNotIn("error", payload)
        message = payload.get("message")
        self.assertIsInstance(message, dict)
        if isinstance(message, dict):
            self.assertEqual(message.get("role"), "assistant")

    def test_sync_stream_generate_events_and_aggregate(self) -> None:
        stream_bytes = (
            b'{"model":"tinyllama:latest","response":"Hel","done":false}\n'
            b'{"model":"tinyllama:latest","response":"lo","done":false}\n'
            b'{"model":"tinyllama:latest","response":"","done":true,"done_reason":"stop"}\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/generate")
            return httpx.Response(
                status_code=200,
                request=request,
                content=stream_bytes,
                headers={"content-type": "application/x-ndjson"},
            )

        with httpx.Client(
            base_url="http://test.local",
            transport=httpx.MockTransport(handler),
        ) as raw_client:
            client = OllamaClient(
                base_url="http://test.local",
                client=raw_client,
                transport_config=OllamaTransportConfig(max_retries=0),
            )

            events = list(
                client.stream_generate(
                    model="tinyllama:latest",
                    prompt="say hello",
                )
            )
            self.assertEqual(len(events), 3)
            self.assertEqual(events[0].get("response"), "Hel")

            aggregated = client.generate(
                model="tinyllama:latest",
                prompt="say hello",
                stream=True,
            )
            self.assertEqual(aggregated.get("response"), "Hello")
            self.assertEqual(aggregated.get("done_reason"), "stop")

    def test_sync_stream_chat_events_and_aggregate(self) -> None:
        stream_bytes = (
            b'{"message":{"role":"assistant","content":"Hello "},"done":false}\n'
            b'{"message":{"role":"assistant","content":"world"},"done":false}\n'
            b'{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"ping","arguments":{}}}]},"done":true,"done_reason":"stop"}\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/chat")
            return httpx.Response(
                status_code=200,
                request=request,
                content=stream_bytes,
                headers={"content-type": "application/x-ndjson"},
            )

        with httpx.Client(
            base_url="http://test.local",
            transport=httpx.MockTransport(handler),
        ) as raw_client:
            client = OllamaClient(
                base_url="http://test.local",
                client=raw_client,
                transport_config=OllamaTransportConfig(max_retries=0),
            )

            events = list(
                client.stream_chat(
                    model="tinyllama:latest",
                    messages=[{"role": "user", "content": "hello"}],
                )
            )
            self.assertEqual(len(events), 3)
            aggregated = client.chat(
                model="tinyllama:latest",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )

            self.assertEqual(aggregated.get("done_reason"), "stop")
            message = aggregated.get("message")
            self.assertIsInstance(message, dict)
            if isinstance(message, dict):
                self.assertEqual(message.get("content"), "Hello world")
                self.assertIsInstance(message.get("tool_calls"), list)

    def test_async_stream_generate_events_and_aggregate(self) -> None:
        async def run_case() -> tuple[list[dict[str, object]], dict[str, object]]:
            stream_bytes = (
                b'{"response":"A","done":false}\n'
                b'{"response":"B","done":false}\n'
                b'{"response":"C","done":true,"done_reason":"stop"}\n'
            )

            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.url.path, "/api/generate")
                return httpx.Response(
                    status_code=200,
                    request=request,
                    content=stream_bytes,
                    headers={"content-type": "application/x-ndjson"},
                )

            raw_client = httpx.AsyncClient(
                base_url="http://test.local",
                transport=httpx.MockTransport(handler),
            )
            client = AsyncOllamaClient(
                base_url="http://test.local",
                client=raw_client,
                transport_config=OllamaTransportConfig(max_retries=0),
            )
            try:
                events: list[dict[str, object]] = []
                async for event in client.stream_generate(
                    model="tinyllama:latest",
                    prompt="abc",
                ):
                    events.append(event)

                aggregated = await client.generate(
                    model="tinyllama:latest",
                    prompt="abc",
                    stream=True,
                )
                return events, aggregated
            finally:
                await client.aclose()
                await raw_client.aclose()

        events, aggregated = asyncio.run(run_case())
        self.assertEqual(len(events), 3)
        self.assertEqual(aggregated.get("response"), "ABC")
        self.assertEqual(aggregated.get("done_reason"), "stop")

    def test_chat_accepts_typed_messages(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/chat")
            payload = json.loads(request.content.decode("utf-8"))
            self.assertEqual(payload["messages"][0]["role"], "user")
            return httpx.Response(
                status_code=200,
                request=request,
                json={"message": {"role": "assistant", "content": "ok"}},
            )

        with httpx.Client(
            base_url="http://test.local",
            transport=httpx.MockTransport(handler),
        ) as raw_client:
            client = OllamaClient(
                base_url="http://test.local",
                client=raw_client,
                transport_config=OllamaTransportConfig(max_retries=0),
            )
            payload = client.chat(
                model="tinyllama:latest",
                messages=[ChatMessage(role="user", content="hello")],
            )
            self.assertNotIn("error", payload)


if __name__ == "__main__":
    unittest.main()
