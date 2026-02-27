# ./src/llm_telemetry_toolkit/providers/ollama.py
"""
Provide resilient sync/async Ollama API adapters with stream/event support.
Used by integrations that need reliable generate/chat/embed requests with retries.
Run: Imported by application code; no standalone command-line entry point.
Inputs: Model IDs, prompts/messages, embedding payloads, transport options.
Outputs: Parsed JSON payloads and streaming chunk/event iterators from Ollama APIs.
Side effects: Performs HTTP requests to local/remote Ollama services.
Operational notes: Circuit breaker avoids repeated hammering during outage windows.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

import httpx

_RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class OllamaTransportConfig:
    """Runtime HTTP reliability settings for Ollama API calls."""

    timeout_seconds: float = 60.0
    max_retries: int = 2
    backoff_initial_seconds: float = 0.4
    backoff_multiplier: float = 2.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 30.0


class _CircuitBreaker:
    """Thread-safe circuit breaker for sync and async request flows."""

    def __init__(self, config: OllamaTransportConfig):
        self._config = config
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: Optional[float] = None

    def before_request(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return

            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.circuit_breaker_recovery_seconds:
                self._opened_at = None
                self._failures = 0
                return

            remaining = self._config.circuit_breaker_recovery_seconds - elapsed
            raise RuntimeError(
                f"Circuit breaker is open (cooldown {remaining:.1f}s remaining)."
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._config.circuit_breaker_failure_threshold:
                self._opened_at = time.monotonic()


class OllamaClient:
    """Synchronous Ollama client with retries/backoff and stream-chunk APIs."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        transport_config: Optional[OllamaTransportConfig] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "").strip() or None
        self.transport_config = transport_config or OllamaTransportConfig()
        self._breaker = _CircuitBreaker(self.transport_config)
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=self.transport_config.timeout_seconds,
            headers=self._headers(),
        )
        self._owns_client = client is None

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        backoff = self.transport_config.backoff_initial_seconds
        last_error = "unknown_error"

        for attempt in range(self.transport_config.max_retries + 1):
            try:
                self._breaker.before_request()
                response = self._client.request(method, endpoint, json=payload)
                if response.status_code in _RETRY_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                if response.status_code >= 400:
                    self._breaker.record_success()
                    return {
                        "error": f"HTTP {response.status_code}",
                        "detail": _extract_error_detail(response),
                    }

                self._breaker.record_success()
                return response.json()
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ) as exc:
                self._breaker.record_failure()
                last_error = str(exc)
                if attempt >= self.transport_config.max_retries:
                    return {"error": last_error}
                time.sleep(backoff)
                backoff *= self.transport_config.backoff_multiplier
            except RuntimeError as exc:
                return {"error": str(exc)}
            except Exception as exc:  # pragma: no cover - defensive catch-all
                return {"error": str(exc)}

        return {"error": last_error}

    def _stream_json(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        backoff = self.transport_config.backoff_initial_seconds
        last_error = "unknown_error"

        for attempt in range(self.transport_config.max_retries + 1):
            try:
                self._breaker.before_request()
                with self._client.stream(method, endpoint, json=payload) as response:
                    if response.status_code in _RETRY_STATUS_CODES:
                        raise httpx.HTTPStatusError(
                            f"Server error {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    if response.status_code >= 400:
                        detail = _extract_error_detail(response)
                        self._breaker.record_success()
                        raise RuntimeError(
                            f"HTTP {response.status_code} while streaming: {detail}"
                        )

                    self._breaker.record_success()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        yield _parse_ndjson_line(line)
                    return
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ) as exc:
                self._breaker.record_failure()
                last_error = str(exc)
                if attempt >= self.transport_config.max_retries:
                    raise RuntimeError(last_error) from exc
                time.sleep(backoff)
                backoff *= self.transport_config.backoff_multiplier
            except RuntimeError:
                raise
            except Exception as exc:  # pragma: no cover - defensive catch-all
                raise RuntimeError(str(exc)) from exc

        raise RuntimeError(last_error)

    def list_models(self) -> Dict[str, Any]:
        return self._request_json("GET", "/api/tags")

    def show_model_info(self, model: str) -> Dict[str, Any]:
        return self._request_json("POST", "/api/show", {"name": model})

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        *,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = _build_generate_payload(
            model=model,
            prompt=prompt,
            system=system,
            stream=stream,
            options=options,
            think=think,
        )
        if stream:
            try:
                return _collect_generate_events(
                    self._stream_json("POST", "/api/generate", payload)
                )
            except Exception as exc:
                return {"error": str(exc)}

        return self._request_json("POST", "/api/generate", payload)

    def stream_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        *,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload = _build_generate_payload(
            model=model,
            prompt=prompt,
            system=system,
            stream=True,
            options=options,
            think=think,
        )
        yield from self._stream_json("POST", "/api/generate", payload)

    def chat(
        self,
        model: str,
        messages: Sequence[Any],
        *,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = _build_chat_payload(
            model=model,
            messages=messages,
            stream=stream,
            options=options,
            think=think,
            tools=tools,
            response_format=response_format,
            keep_alive=keep_alive,
        )
        if stream:
            try:
                return _collect_chat_events(
                    self._stream_json("POST", "/api/chat", payload)
                )
            except Exception as exc:
                return {"error": str(exc)}

        return self._request_json("POST", "/api/chat", payload)

    def stream_chat(
        self,
        model: str,
        messages: Sequence[Any],
        *,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload = _build_chat_payload(
            model=model,
            messages=messages,
            stream=True,
            options=options,
            think=think,
            tools=tools,
            response_format=response_format,
            keep_alive=keep_alive,
        )
        yield from self._stream_json("POST", "/api/chat", payload)

    def embed(self, model: str, input_text: str | List[str]) -> Dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/embed",
            {"model": model, "input": input_text},
        )

    def check_connection(self) -> bool:
        result = self.list_models()
        return "error" not in result


class AsyncOllamaClient:
    """Asynchronous Ollama client with retries/backoff and stream-chunk APIs."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        transport_config: Optional[OllamaTransportConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "").strip() or None
        self.transport_config = transport_config or OllamaTransportConfig()
        self._breaker = _CircuitBreaker(self.transport_config)
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.transport_config.timeout_seconds,
            headers=self._headers(),
        )
        self._owns_client = client is None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def __aenter__(self) -> "AsyncOllamaClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        backoff = self.transport_config.backoff_initial_seconds
        last_error = "unknown_error"

        for attempt in range(self.transport_config.max_retries + 1):
            try:
                self._breaker.before_request()
                response = await self._client.request(method, endpoint, json=payload)
                if response.status_code in _RETRY_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    self._breaker.record_success()
                    return {
                        "error": f"HTTP {response.status_code}",
                        "detail": _extract_error_detail(response),
                    }

                self._breaker.record_success()
                return response.json()
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ) as exc:
                self._breaker.record_failure()
                last_error = str(exc)
                if attempt >= self.transport_config.max_retries:
                    return {"error": last_error}
                await asyncio.sleep(backoff)
                backoff *= self.transport_config.backoff_multiplier
            except RuntimeError as exc:
                return {"error": str(exc)}
            except Exception as exc:  # pragma: no cover - defensive catch-all
                return {"error": str(exc)}

        return {"error": last_error}

    async def _stream_json(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        backoff = self.transport_config.backoff_initial_seconds
        last_error = "unknown_error"

        for attempt in range(self.transport_config.max_retries + 1):
            try:
                self._breaker.before_request()
                async with self._client.stream(
                    method, endpoint, json=payload
                ) as response:
                    if response.status_code in _RETRY_STATUS_CODES:
                        raise httpx.HTTPStatusError(
                            f"Server error {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    if response.status_code >= 400:
                        detail = _extract_error_detail(response)
                        self._breaker.record_success()
                        raise RuntimeError(
                            f"HTTP {response.status_code} while streaming: {detail}"
                        )

                    self._breaker.record_success()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        yield _parse_ndjson_line(line)
                    return
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ) as exc:
                self._breaker.record_failure()
                last_error = str(exc)
                if attempt >= self.transport_config.max_retries:
                    raise RuntimeError(last_error) from exc
                await asyncio.sleep(backoff)
                backoff *= self.transport_config.backoff_multiplier
            except RuntimeError:
                raise
            except Exception as exc:  # pragma: no cover - defensive catch-all
                raise RuntimeError(str(exc)) from exc

        raise RuntimeError(last_error)

    async def list_models(self) -> Dict[str, Any]:
        return await self._request_json("GET", "/api/tags")

    async def show_model_info(self, model: str) -> Dict[str, Any]:
        return await self._request_json("POST", "/api/show", {"name": model})

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        *,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = _build_generate_payload(
            model=model,
            prompt=prompt,
            system=system,
            stream=stream,
            options=options,
            think=think,
        )
        if stream:
            try:
                return await _collect_async_generate_events(
                    self._stream_json("POST", "/api/generate", payload)
                )
            except Exception as exc:
                return {"error": str(exc)}

        return await self._request_json("POST", "/api/generate", payload)

    async def stream_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        *,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        payload = _build_generate_payload(
            model=model,
            prompt=prompt,
            system=system,
            stream=True,
            options=options,
            think=think,
        )
        async for event in self._stream_json("POST", "/api/generate", payload):
            yield event

    async def chat(
        self,
        model: str,
        messages: Sequence[Any],
        *,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = _build_chat_payload(
            model=model,
            messages=messages,
            stream=stream,
            options=options,
            think=think,
            tools=tools,
            response_format=response_format,
            keep_alive=keep_alive,
        )
        if stream:
            try:
                return await _collect_async_chat_events(
                    self._stream_json("POST", "/api/chat", payload)
                )
            except Exception as exc:
                return {"error": str(exc)}

        return await self._request_json("POST", "/api/chat", payload)

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Any],
        *,
        options: Optional[Dict[str, Any]] = None,
        think: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        payload = _build_chat_payload(
            model=model,
            messages=messages,
            stream=True,
            options=options,
            think=think,
            tools=tools,
            response_format=response_format,
            keep_alive=keep_alive,
        )
        async for event in self._stream_json("POST", "/api/chat", payload):
            yield event

    async def embed(self, model: str, input_text: str | List[str]) -> Dict[str, Any]:
        return await self._request_json(
            "POST",
            "/api/embed",
            {"model": model, "input": input_text},
        )

    async def check_connection(self) -> bool:
        result = await self.list_models()
        return "error" not in result


def _build_generate_payload(
    *,
    model: str,
    prompt: str,
    system: Optional[str],
    stream: bool,
    options: Optional[Dict[str, Any]],
    think: Optional[bool],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options or {"temperature": 0.7},
    }
    if system:
        payload["system"] = system
    if think is not None:
        payload["think"] = think
    return payload


def _build_chat_payload(
    *,
    model: str,
    messages: Sequence[Any],
    stream: bool,
    options: Optional[Dict[str, Any]],
    think: Optional[bool],
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Dict[str, Any]],
    keep_alive: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [_normalize_message_payload(item) for item in messages],
        "stream": stream,
    }
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["format"] = response_format
    if keep_alive:
        payload["keep_alive"] = keep_alive
    return payload


def _normalize_message_payload(message: Any) -> Dict[str, Any]:
    """Normalize message items to plain dict payloads for Ollama API requests."""
    if isinstance(message, dict):
        return dict(message)

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=True)
        if isinstance(dumped, dict):
            return dumped

    raise TypeError(
        "messages entries must be dictionaries or Pydantic models supporting model_dump()."
    )


def _parse_ndjson_line(line: str) -> Dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid NDJSON chunk from Ollama: {line!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid stream payload: expected a JSON object chunk.")
    return payload


def _collect_generate_events(events: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    response_parts: List[str] = []
    for event in events:
        merged.update(event)
        chunk = event.get("response")
        if isinstance(chunk, str):
            response_parts.append(chunk)
    if response_parts:
        merged["response"] = "".join(response_parts)
    return merged


def _collect_chat_events(events: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    response_parts: List[str] = []
    role: Optional[str] = None
    tool_calls: List[Any] = []

    for event in events:
        merged.update(event)
        message = event.get("message")
        if not isinstance(message, dict):
            continue

        maybe_role = message.get("role")
        if isinstance(maybe_role, str) and maybe_role:
            role = maybe_role

        content = message.get("content")
        if isinstance(content, str):
            response_parts.append(content)

        chunk_tool_calls = message.get("tool_calls")
        if isinstance(chunk_tool_calls, list):
            tool_calls.extend(chunk_tool_calls)

    if response_parts or role or tool_calls:
        merged["message"] = {
            "role": role or "assistant",
            "content": "".join(response_parts),
        }
        if tool_calls:
            merged["message"]["tool_calls"] = tool_calls

    return merged


async def _collect_async_generate_events(
    events: AsyncIterator[Dict[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    response_parts: List[str] = []
    async for event in events:
        merged.update(event)
        chunk = event.get("response")
        if isinstance(chunk, str):
            response_parts.append(chunk)
    if response_parts:
        merged["response"] = "".join(response_parts)
    return merged


async def _collect_async_chat_events(
    events: AsyncIterator[Dict[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    response_parts: List[str] = []
    role: Optional[str] = None
    tool_calls: List[Any] = []

    async for event in events:
        merged.update(event)
        message = event.get("message")
        if not isinstance(message, dict):
            continue

        maybe_role = message.get("role")
        if isinstance(maybe_role, str) and maybe_role:
            role = maybe_role

        content = message.get("content")
        if isinstance(content, str):
            response_parts.append(content)

        chunk_tool_calls = message.get("tool_calls")
        if isinstance(chunk_tool_calls, list):
            tool_calls.extend(chunk_tool_calls)

    if response_parts or role or tool_calls:
        merged["message"] = {
            "role": role or "assistant",
            "content": "".join(response_parts),
        }
        if tool_calls:
            merged["message"]["tool_calls"] = tool_calls

    return merged


def _extract_error_detail(response: httpx.Response) -> str:
    """Extract provider error detail text from JSON payload or raw response body."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                return error
    except Exception:
        pass
    return response.text
