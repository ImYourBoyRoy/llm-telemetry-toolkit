# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/io/__init__.py
"""
IO sub-package.
Contains Input/Output strategies: Formatters (Strategy Pattern), Parsing, and Utils.
"""

from .formatters import FormatterFactory
from .parser import ContentParser
# utils is mostly internal, but we can expose it if needed.
# For now, keeping the public API clean.

__all__ = ["FormatterFactory", "ContentParser"]
