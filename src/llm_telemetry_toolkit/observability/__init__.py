# ./src/llm_telemetry_toolkit/observability/__init__.py
"""
Expose optional observability exporters that complement file-based telemetry logs.
Used by logger initialization when advanced tracing integration is enabled by config.
Run: Imported by core logger internals and advanced integration code.
Inputs: Interaction objects emitted by logging pipelines.
Outputs: Exporter classes implementing `.export(interaction)` behavior.
Side effects: Depends on exporter implementation (for example OpenTelemetry span creation).
Operational notes: Exporters are optional; package remains usable without external tracing deps.
"""

from .otel import OpenTelemetryInteractionExporter

__all__ = ["OpenTelemetryInteractionExporter"]
