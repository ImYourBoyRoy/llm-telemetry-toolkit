# ./src/llm_telemetry_toolkit/models/config.py
"""
Define validated runtime configuration for telemetry logging behavior.
Used by `LLMLogger` and CLI integrations to control outputs and safety defaults.
Run: Imported by package code; no direct CLI entry point.
Inputs: Constructor kwargs, env-loaded values, or config-file mappings passed by callers.
Outputs: Strongly typed `TelemetryConfig` with normalized format list and template validation.
Side effects: None; validation errors are raised early for invalid settings.
Operational notes: Enforces supported formats and safe template placeholders for predictable files.
"""

from __future__ import annotations

import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_OUTPUT_FORMATS = frozenset({"json", "md", "csv"})
SUPPORTED_TEMPLATE_FIELDS = frozenset(
    {
        "timestamp",
        "interaction_id",
        "type",
        "ext",
        "session_id",
        "model_name",
        "tool_name",
    }
)


class TelemetryConfig(BaseModel):
    """Runtime configuration for the telemetry logger and output formatters."""

    session_id: str = Field(
        default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Unique identifier for the current session. Defaults to 'session_YYYYMMDD_HHMMSS'.",
    )
    base_log_dir: Path = Field(
        default=Path("./logs"),
        description="Root directory for storing logs. Defaults to './logs'.",
    )

    # Feature Flags
    enable_session_logging: bool = Field(
        default=True, description="Enable main session logging."
    )
    enable_entity_logging: bool = Field(
        default=False, description="Enable separate logs per entity (e.g. company)."
    )

    # Paths
    session_log_subdir: str = Field(
        default="llm_interactions", description="Subdirectory for session logs."
    )
    entity_log_subdir: str = Field(
        default="entity_llm_interactions", description="Subdirectory for entity logs."
    )

    # Formatting
    output_formats: List[str] = Field(
        default_factory=lambda: ["json"],
        description="Ordered formats to write per interaction. Supported: 'json', 'csv', 'md'.",
    )
    filename_template: str = Field(
        default="{timestamp}_{interaction_id}_{type}.{ext}",
        description=(
            "Template for output filenames. Allowed placeholders: "
            "{timestamp}, {interaction_id}, {type}, {ext}, {session_id}, {model_name}, {tool_name}."
        ),
    )
    json_indent: int = Field(default=2, description="Indentation level for JSON logs.")
    ensure_ascii: bool = Field(
        default=False, description="Escape non-ASCII characters in JSON output."
    )

    # Data Hygiene
    max_content_length: Optional[int] = Field(
        default=None,
        description="Maximum characters for prompt/response in logs. None means no truncation.",
    )

    # Privacy / Security
    mask_pii: bool = Field(
        default=False,
        description="Enable smart PII redaction (Email, IP, Phone, Credit Card).",
    )

    # Observability exporters
    enable_otel_export: bool = Field(
        default=False,
        description="Enable optional OpenTelemetry span export for each interaction.",
    )
    otel_tracer_name: str = Field(
        default="llm_telemetry_toolkit",
        description="Tracer name used when OpenTelemetry export is enabled.",
    )
    otel_span_name: str = Field(
        default="llm.interaction",
        description="Span name emitted for each logged interaction.",
    )
    otel_service_name: str = Field(
        default="llm-telemetry-toolkit",
        description="Service name attribute recorded on emitted OpenTelemetry spans.",
    )
    otel_auto_configure: bool = Field(
        default=False,
        description="Automatically bootstrap OpenTelemetry SDK/exporter when no provider exists.",
    )
    otel_exporter: str = Field(
        default="otlp_http",
        description="Exporter type when auto-configuring OpenTelemetry. Supported: otlp_http, none.",
    )
    otel_otlp_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="OTLP/HTTP trace endpoint used when auto-configure is enabled.",
    )
    otel_otlp_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional OTLP/HTTP headers for auth/tenant routing.",
    )
    otel_otlp_timeout_seconds: float = Field(
        default=10.0,
        description="OTLP/HTTP exporter timeout in seconds when auto-configure is enabled.",
    )
    otel_enable_console_export: bool = Field(
        default=False,
        description="Also write spans to console when auto-configure is enabled.",
    )
    otel_sampler_ratio: float = Field(
        default=1.0,
        description="Trace sampling ratio between 0.0 and 1.0 for auto-configured SDK.",
    )
    otel_resource_attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional OpenTelemetry resource attributes to attach to emitted spans.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("session_id must be a non-empty string.")
        return normalized

    @field_validator("output_formats")
    @classmethod
    def _validate_output_formats(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("output_formats must include at least one format.")

        normalized: List[str] = []
        for raw in value:
            format_name = raw.strip().lower()
            if format_name not in SUPPORTED_OUTPUT_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
                raise ValueError(
                    f"Unsupported output format '{raw}'. Supported formats: {supported}."
                )
            if format_name not in normalized:
                normalized.append(format_name)
        return normalized

    @field_validator("filename_template")
    @classmethod
    def _validate_filename_template(cls, value: str) -> str:
        template = value.strip()
        if not template:
            raise ValueError("filename_template must be a non-empty string.")

        formatter = string.Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name and field_name not in SUPPORTED_TEMPLATE_FIELDS:
                allowed = ", ".join(sorted(SUPPORTED_TEMPLATE_FIELDS))
                raise ValueError(
                    f"Unsupported template field '{field_name}'. Allowed fields: {allowed}."
                )

        return template

    @field_validator("otel_tracer_name", "otel_span_name", "otel_service_name")
    @classmethod
    def _validate_non_empty_otel_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("OpenTelemetry name fields must be non-empty strings.")
        return normalized

    @field_validator("otel_exporter")
    @classmethod
    def _validate_otel_exporter(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"otlp_http", "none"}:
            raise ValueError("otel_exporter must be one of: 'otlp_http', 'none'.")
        return normalized

    @field_validator("otel_otlp_endpoint")
    @classmethod
    def _validate_otel_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("otel_otlp_endpoint must be a non-empty string.")
        return normalized

    @field_validator("otel_otlp_timeout_seconds")
    @classmethod
    def _validate_otel_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("otel_otlp_timeout_seconds must be greater than 0.")
        return value

    @field_validator("otel_sampler_ratio")
    @classmethod
    def _validate_sampler_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("otel_sampler_ratio must be between 0.0 and 1.0.")
        return value

    @model_validator(mode="after")
    def _validate_auto_config_requirements(self) -> "TelemetryConfig":
        if self.otel_auto_configure and self.otel_exporter == "none":
            if not self.otel_enable_console_export:
                raise ValueError(
                    "When otel_auto_configure is enabled and otel_exporter='none', "
                    "otel_enable_console_export must be True."
                )
        return self
