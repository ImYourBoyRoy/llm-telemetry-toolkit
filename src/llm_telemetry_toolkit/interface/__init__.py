# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/interface/__init__.py
"""
Interface sub-package.
Contains User Interfaces such as the CLI.
"""

# CLI is typically run via 'python -m ...' or entry point, so exports are less critical here,
# but good practice to allow importing 'main' for programmatic invocation.
from .cli import main

__all__ = ["main"]
