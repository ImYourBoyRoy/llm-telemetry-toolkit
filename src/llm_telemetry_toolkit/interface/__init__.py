# ./src/llm_telemetry_toolkit/interface/__init__.py
"""
Expose interface-layer entry points for the telemetry toolkit package.
Used by package consumers that invoke CLI functionality programmatically.
Run: Imported by `llm_telemetry_toolkit.__init__` and packaging entry points.
Inputs: None.
Outputs: Re-exported `main` callable from the CLI module.
Side effects: None.
Operational notes: Keeps interface API stable even if internal CLI modules change.
"""


def main() -> None:
    """Run CLI entry point lazily to avoid eager module import side effects."""
    from .cli import main as cli_main

    cli_main()


__all__ = ["main"]
