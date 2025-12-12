# ./llm-telemetry-toolkit/src/llm_telemetry_toolkit/interface/cli.py
"""
CLI Interface for the LLM Telemetry Toolkit.
Provides commands to view logs and calculate statistics using Rich for a "Sexy" UI.
Inputs: Command line arguments (session_id, dir).
Outputs: Rich console dashboard.
"""

import argparse
import json
from pathlib import Path

# Formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.tree import Tree
except ImportError:
    print("Error: 'rich' library is required for this CLI. Run 'pip install rich'.")
    exit(1)


console = Console(force_terminal=True, color_system="auto", legacy_windows=False)


def main():
    parser = argparse.ArgumentParser(description="LLM Telemetry Toolkit CLI (v0.1.0)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # VIEW Command
    view_parser = subparsers.add_parser("view", help="View recent logs for a session")
    view_parser.add_argument("--session", required=True, help="Session ID to view")
    view_parser.add_argument("--dir", default="./logs", help="Base log directory")
    view_parser.add_argument(
        "--limit", type=int, default=5, help="Number of logs to show"
    )

    # STATS Command
    stats_parser = subparsers.add_parser("stats", help="Calculate token usage stats")
    stats_parser.add_argument("--session", required=True, help="Session ID")
    stats_parser.add_argument("--dir", default="./logs", help="Base log directory")

    args = parser.parse_args()

    if args.command == "view":
        _handle_view(args)
    elif args.command == "stats":
        _handle_stats(args)
    else:
        # Show a nice welcome banner if no args
        try:
            _show_banner()
            parser.print_help()
        except Exception:
            parser.print_help()


def _show_banner():
    """Displays a cool ASCII art banner."""
    console.print(
        Panel.fit(
            "[bold cyan]LLM Telemetry Toolkit[/bold cyan]\n[dim]Standardized Observability for Agents[/dim]",
            border_style="cyan",
            title="v0.1.0",
        )
    )


def _handle_view(args):
    """Rich Viewer for JSON logs."""
    base_dir = Path(args.dir)
    session_dir = base_dir / "llm_interactions" / args.session

    if not session_dir.exists():
        console.print(
            f"[bold red]X[/bold red] No logs found for session '[bold yellow]{args.session}[/bold yellow]' in {base_dir}"
        )
        return

    # Find all .json files (excluding config files)
    files = sorted(
        [
            p
            for p in session_dir.glob("*.json")
            if not p.name.endswith("session_config.json")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    console.print(
        f"[bold green]>[/bold green] Found {len(files)} interactions for session [cyan]{args.session}[/cyan]"
    )

    for f in files[: args.limit]:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)

                # Status Color
                status_color = "green"
                if not data.get("validation_passed", True):
                    status_color = "red"
                elif data.get("error_message"):
                    status_color = "red"

                # Header Panel
                i_id = data.get("interaction_id", "Unknown")
                model = data.get("model_name", "Unknown")
                latency = data.get("response_time_seconds", 0)
                timestamp = data.get("timestamp_utc", "")

                header_text = (
                    f"[bold {status_color}]ID:[/bold {status_color}] {i_id} | "
                    f"[bold]Model:[/bold] {model} | "
                    f"[bold]Latency:[/bold] {latency:.2f}s | "
                    f"[dim]{timestamp}[/dim]"
                )

                # Content Grid (Table)
                grid = Table(show_header=True, header_style="bold magenta", expand=True)
                grid.add_column("Component", width=15)
                grid.add_column("Content")

                # Prompt
                grid.add_row(
                    "Prompt",
                    str(data.get("prompt", ""))[:500] + "..."
                    if len(str(data.get("prompt", ""))) > 500
                    else str(data.get("prompt", "")),
                )

                # Thought Process (if any)
                if data.get("thought_process"):
                    grid.add_row(
                        "[italic]Thought[/italic]",
                        Syntax(
                            str(data.get("thought_process")),
                            "text",
                            theme="monokai",
                            word_wrap=True,
                        ),
                    )

                # Response (Syntax Highlighted)
                resp_text = str(data.get("response", ""))
                grid.add_row(
                    "Response",
                    # Using generic syntax if we don't know content type, but Markdown is safe guess for LLMs
                    Syntax(resp_text, "markdown", theme="monokai", word_wrap=True),
                )

                # Metadata Tree
                if data.get("metadata"):
                    meta_tree = Tree("[dim]Metadata[/dim]")
                    for k, v in data.get("metadata").items():
                        meta_tree.add(f"[cyan]{k}[/cyan]: {v}")
                    grid.add_row("Metadata", meta_tree)

                console.print(Panel(grid, title=header_text, border_style=status_color))
                console.print("")  # Spacer

        except Exception as e:
            console.print(f"[red]Error reading {f.name}: {e}[/red]")


def _handle_stats(args):
    """Rich Stats Aggregator."""
    base_dir = Path(args.dir)
    session_dir = base_dir / "llm_interactions" / args.session

    if not session_dir.exists():
        console.print(
            f"[bold red]X[/bold red] Session '[bold yellow]{args.session}[/bold yellow]' not found."
        )
        return

    files = [
        p
        for p in session_dir.glob("*.json")
        if not p.name.endswith("session_config.json")
    ]
    total_latency = 0.0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_resp_tokens = 0
    count = 0

    # Model breakdown
    models = {}

    with console.status("[bold green]Calculating statistics...[/bold green]"):
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    lat = data.get("response_time_seconds", 0)
                    cost = data.get("cost_usd", 0) or 0
                    p_tok = data.get("token_count_prompt", 0) or 0
                    r_tok = data.get("token_count_response", 0) or 0
                    model = data.get("model_name", "unknown")

                    total_latency += lat
                    total_cost += cost
                    total_prompt_tokens += p_tok
                    total_resp_tokens += r_tok
                    count += 1

                    if model not in models:
                        models[model] = 0
                    models[model] += 1

            except Exception:
                pass

    if count == 0:
        console.print("[yellow]No valid logs found.[/yellow]")
        return

    # Create Summary Table
    table = Table(title=f"Session Statistics: [bold]{args.session}[/bold]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Total Interactions", str(count))
    table.add_row("Total Latency", f"{total_latency:.2f}s")
    table.add_row("Avg Latency", f"{total_latency / count:.2f}s")
    table.add_row("Total Cost", f"[green]${total_cost:.5f}[/green]")
    table.add_row("Total Tokens", f"{total_prompt_tokens + total_resp_tokens}")
    table.add_row(" - Prompt", f"{total_prompt_tokens}")
    table.add_row(" - Response", f"{total_resp_tokens}")

    console.print(table)

    # Simple Model Breakdown
    console.print("[dim]Models used:[/dim]")
    for m, c in models.items():
        console.print(f" - [blue]{m}[/blue]: {c}")


if __name__ == "__main__":
    main()
