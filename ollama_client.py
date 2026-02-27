# ./ollama_client.py
"""
Maintain backward-compatible imports for Ollama client adapters.
Used by legacy scripts that previously imported `OllamaClient` from project root.
Run: Imported by local scripts; package-native adapter lives under `src/.../providers/`.
Inputs: N/A (module-level re-exports only).
Outputs: Re-exported sync/async Ollama client classes and transport config.
Side effects: None.
Operational notes: Prefer `from llm_telemetry_toolkit.providers import OllamaClient`.
"""

from llm_telemetry_toolkit.providers.ollama import (
    AsyncOllamaClient,
    OllamaClient,
    OllamaTransportConfig,
)

__all__ = ["OllamaClient", "AsyncOllamaClient", "OllamaTransportConfig"]
