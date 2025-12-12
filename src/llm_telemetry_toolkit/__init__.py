# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/__init__.py
"""
LLM Telemetry Toolkit.
A production-grade observability system for AI Agents.
Exposes core components for easy access.
"""

from .core.logger import LLMLogger
from .models.schema import LLMInteraction
from .models.config import TelemetryConfig
from .models.results import LogResult
from .core.context import SessionContext
from .core.decorators import monitor_interaction
from .interface import main as cli_main

__all__ = [
    "LLMLogger",
    "LLMInteraction",
    "TelemetryConfig",
    "LogResult",
    "SessionContext",
    "monitor_interaction",
    "cli_main",
]
