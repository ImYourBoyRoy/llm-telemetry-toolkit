# ./src/llm_telemetry_toolkit/io/__init__.py
"""
Expose I/O helpers used by the telemetry logger pipeline.
Used internally and by advanced integrations needing direct parser/formatter access.
Run: Imported by logger components and optional extension code.
Inputs: None.
Outputs: Public formatter factory and content parser exports.
Side effects: None.
Operational notes: Utility helpers stay internal unless explicitly exported here.
"""

from .formatters import FormatterFactory
from .parser import ContentParser
# utils is mostly internal, but we can expose it if needed.
# For now, keeping the public API clean.

__all__ = ["FormatterFactory", "ContentParser"]
