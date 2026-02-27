# ./src/llm_telemetry_toolkit/__init__.py
"""
Expose the public API surface for the LLM Telemetry Toolkit package.
Used by library consumers for logger, schema, and decorator imports from one namespace.
Run: Imported by applications or CLI entry points; not executed directly.
Inputs: None.
Outputs: Public symbols and package version metadata.
Side effects: None.
Operational notes: Version resolves from installed package metadata with a dev fallback.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .core.context import SessionContext
from .core.decorators import monitor_interaction
from .core.logger import InteractionExporter, LLMLogger
from .models.config import TelemetryConfig
from .models.results import LogResult
from .models.schema import ChatMessage, LLMInteraction, ToolCall, ToolFunctionCall
from .observability import OpenTelemetryInteractionExporter
from .providers import AsyncOllamaClient, OllamaClient, OllamaTransportConfig

try:
    __version__ = version("llm-telemetry-toolkit")
except PackageNotFoundError:  # pragma: no cover - local editable fallback
    __version__ = "0.0.0+dev"


def cli_main() -> None:
    """Run the package CLI entry point lazily to avoid import-time side effects."""
    from .interface.cli import main

    main()


__all__ = [
    "LLMLogger",
    "InteractionExporter",
    "LLMInteraction",
    "ChatMessage",
    "ToolCall",
    "ToolFunctionCall",
    "TelemetryConfig",
    "LogResult",
    "SessionContext",
    "monitor_interaction",
    "OllamaClient",
    "AsyncOllamaClient",
    "OllamaTransportConfig",
    "OpenTelemetryInteractionExporter",
    "cli_main",
    "__version__",
]
