# ./src/llm_telemetry_toolkit/providers/__init__.py
"""
Expose provider adapters used by integration scripts and external toolchains.
Used when callers want standardized transport wrappers for model backends.
Run: Imported by package consumers and compatibility wrappers.
Inputs: Provider-specific request payloads and runtime transport settings.
Outputs: Structured API responses and helper client classes.
Side effects: Network I/O to provider endpoints.
Operational notes: Keep adapters modular so downstream tools can install only what they need.
"""

from .ollama import AsyncOllamaClient, OllamaClient, OllamaTransportConfig

__all__ = ["OllamaClient", "AsyncOllamaClient", "OllamaTransportConfig"]
