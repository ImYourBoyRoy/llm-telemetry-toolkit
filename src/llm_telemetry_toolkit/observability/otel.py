# ./src/llm_telemetry_toolkit/observability/otel.py
"""
Implement optional OpenTelemetry span export for logged LLM interactions.
Used by `LLMLogger` when OpenTelemetry export is enabled in `TelemetryConfig`.
Run: Imported lazily by logger; package still works without OTel extras installed.
Inputs: Prepared `LLMInteraction` records and optional bootstrap/export settings.
Outputs: Tracing spans written to configured OpenTelemetry provider/exporters.
Side effects: May initialize global tracer provider when `auto_configure=True`.
Operational notes: Auto-bootstrap is guarded and only occurs once per process.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from ..models.schema import LLMInteraction


class OpenTelemetryInteractionExporter:
    """Export each interaction as a trace span with optional SDK/bootstrap wiring."""

    _bootstrap_lock = threading.Lock()
    _bootstrapped = False

    def __init__(
        self,
        *,
        tracer_name: str = "llm_telemetry_toolkit",
        span_name: str = "llm.interaction",
        service_name: str = "llm-telemetry-toolkit",
        auto_configure: bool = False,
        exporter: str = "otlp_http",
        otlp_endpoint: str = "http://localhost:4318/v1/traces",
        otlp_headers: Optional[Dict[str, str]] = None,
        otlp_timeout_seconds: float = 10.0,
        enable_console_export: bool = False,
        sampler_ratio: float = 1.0,
        resource_attributes: Optional[Dict[str, str]] = None,
    ):
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError(
                "OpenTelemetry export requested but `opentelemetry-api` is not installed."
            ) from exc

        self._trace = trace
        self._span_name = span_name
        self._service_name = service_name
        self._owns_provider = False

        if auto_configure:
            self._owns_provider = self._bootstrap_provider(
                service_name=service_name,
                exporter=exporter,
                otlp_endpoint=otlp_endpoint,
                otlp_headers=otlp_headers or {},
                otlp_timeout_seconds=otlp_timeout_seconds,
                enable_console_export=enable_console_export,
                sampler_ratio=sampler_ratio,
                resource_attributes=resource_attributes or {},
            )

        self._tracer = trace.get_tracer(tracer_name)

    @classmethod
    def _bootstrap_provider(
        cls,
        *,
        service_name: str,
        exporter: str,
        otlp_endpoint: str,
        otlp_headers: Dict[str, str],
        otlp_timeout_seconds: float,
        enable_console_export: bool,
        sampler_ratio: float,
        resource_attributes: Dict[str, str],
    ) -> bool:
        """Create and register a process-wide tracer provider when currently unset."""
        with cls._bootstrap_lock:
            if cls._bootstrapped:
                return False

            from opentelemetry import trace

            current_provider = trace.get_tracer_provider()
            if current_provider.__class__.__name__ != "ProxyTracerProvider":
                cls._bootstrapped = True
                return False

            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import (
                    BatchSpanProcessor,
                    ConsoleSpanExporter,
                    SimpleSpanProcessor,
                )
                from opentelemetry.sdk.trace.sampling import (
                    ParentBased,
                    TraceIdRatioBased,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency branch
                raise RuntimeError(
                    "OpenTelemetry auto-config requested, but SDK/exporter packages are missing. "
                    'Install with `pip install "llm-telemetry-toolkit[otel]"`.'
                ) from exc

            merged_resource_attrs: Dict[str, str] = {
                "service.name": service_name,
                **resource_attributes,
            }
            sampler = ParentBased(TraceIdRatioBased(sampler_ratio))
            provider = TracerProvider(
                resource=Resource.create(merged_resource_attrs),
                sampler=sampler,
            )

            if exporter == "otlp_http":
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    headers=otlp_headers,
                    timeout=otlp_timeout_seconds,
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

            if enable_console_export:
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

            trace.set_tracer_provider(provider)
            cls._bootstrapped = True
            return True

    def export(self, interaction: LLMInteraction) -> None:
        """Create one span containing core interaction attributes."""
        with self._tracer.start_as_current_span(self._span_name) as span:
            self._set_if_not_none(span, "service.name", self._service_name)
            self._set_if_not_none(span, "gen_ai.request.model", interaction.model_name)
            self._set_if_not_none(
                span, "gen_ai.system", interaction.provider or "unknown"
            )
            self._set_if_not_none(
                span, "gen_ai.operation.name", interaction.interaction_type or "unknown"
            )
            self._set_if_not_none(span, "gen_ai.session.id", interaction.session_id)
            self._set_if_not_none(span, "gen_ai.tool.name", interaction.tool_name)
            self._set_if_not_none(span, "gen_ai.agent.name", interaction.agent_name)
            self._set_if_not_none(
                span,
                "gen_ai.request.message_count",
                self._count(interaction.request_messages),
            )
            self._set_if_not_none(
                span, "gen_ai.usage.input_tokens", interaction.token_count_prompt
            )
            self._set_if_not_none(
                span, "gen_ai.usage.output_tokens", interaction.token_count_response
            )
            self._set_if_not_none(
                span, "gen_ai.response.finish_reason", interaction.finish_reason
            )
            self._set_if_not_none(
                span, "gen_ai.response.role", interaction.response_message_role
            )
            self._set_if_not_none(
                span,
                "gen_ai.response.tool_call_count",
                self._count(interaction.tool_calls),
            )
            self._set_if_not_none(
                span,
                "gen_ai.embedding.input_count",
                interaction.embedding_input_count,
            )
            self._set_if_not_none(
                span,
                "gen_ai.embedding.vector_count",
                interaction.embedding_vector_count,
            )
            self._set_if_not_none(
                span, "gen_ai.embedding.dimensions", interaction.embedding_dimensions
            )
            self._set_if_not_none(
                span, "llm.telemetry.latency_seconds", interaction.response_time_seconds
            )
            self._set_if_not_none(span, "llm.telemetry.cost_usd", interaction.cost_usd)
            self._set_if_not_none(
                span, "llm.telemetry.validation_passed", interaction.validation_passed
            )
            self._set_if_not_none(
                span, "llm.telemetry.error_message", interaction.error_message
            )

    def shutdown(self) -> None:
        """Shutdown auto-configured tracer provider to flush pending spans."""
        if not self._owns_provider:
            return

        provider = self._trace.get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()

    @staticmethod
    def _count(value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        return 0

    @staticmethod
    def _set_if_not_none(span: Any, key: str, value: Any) -> None:
        if value is not None:
            span.set_attribute(key, value)
