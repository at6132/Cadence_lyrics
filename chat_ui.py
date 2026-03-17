"""
Chat-style CLI for the anti-AI lyric pipeline.
Rich-based UI: banner, panels, streaming output, clear turn-taking.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.theme import Theme

# Subtle, readable theme (works on light/dark backgrounds)
CONSOLE_THEME = Theme({
    "user": "bold cyan",
    "assistant": "bold green",
    "meta": "dim italic",
    "error": "bold red",
    "banner": "bold magenta",
    "lyrics": "white",
})

console = Console(theme=CONSOLE_THEME)


def banner() -> None:
    console.print()
    console.print(
        Panel(
            "[banner]♪ Anti-AI Lyric Studio ♪[/banner]\n\n"
            "[meta]Describe a song (mood, topic, style). Commands: /quit  /help  /debug[/meta]",
            border_style="magenta",
            padding=(0, 2),
        )
    )
    console.print()


def show_help() -> None:
    help_text = """
**[user]/help[/user]** — Show this message  
**[user]/quit[/user]** or **[user]/exit[/user]** — Exit the chat  
**[user]/debug[/user]** — Toggle debug mode (show score, passes, banned phrases)  
**[user]/clear[/user]** — Clear the screen  

Just type a request like *"sad song about a rainy bus stop"* or *"upbeat pop chorus"* and press Enter.
"""
    console.print(Panel(Markdown(help_text), title="[assistant] Commands[/assistant]", border_style="dim"))
    console.print()


def _phase_title(phase: str) -> str:
    """Human-readable title for pipeline phase."""
    if phase == "draft":
        return "[assistant]Writing draft…[/assistant]"
    if phase == "evaluating":
        return "[meta]Evaluating…[/meta]"
    if phase.startswith("rewrite_"):
        n = phase.replace("rewrite_", "")
        return f"[assistant]Rewriting ({n})…[/assistant]"
    return phase or "…"


class _StreamPanel:
    """Mutable renderable for Live: updates phase/text and renders as a Panel."""

    def __init__(self) -> None:
        self.phase = "draft"
        self.text = ""

    def __rich_console__(self, console: Console, options) -> None:
        title = _phase_title(self.phase)
        content = (self.text or "").strip() or "[dim]…[/dim]"
        yield Panel(
            content,
            title=title,
            border_style="green",
            padding=(1, 2),
            expand=False,
        )


def show_lyrics(lyrics: str, *, debug: bool = False, debug_data: Optional[dict] = None) -> None:
    """Render lyrics in a panel; if debug, show score and meta below."""
    content = lyrics.strip() or "[dim](no lyrics generated)[/dim]"
    panel = Panel(
        content,
        title="[assistant] Lyrics[/assistant]",
        border_style="green",
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)
    if debug and debug_data:
        meta = []
        if isinstance(debug_data.get("score"), (int, float)):
            meta.append(f"Score: {debug_data['score']:.0f}")
        if isinstance(debug_data.get("passes_used"), int):
            meta.append(f"Passes: {debug_data['passes_used']}")
        if debug_data.get("banned_phrases"):
            meta.append(f"Banned: {debug_data['banned_phrases']}")
        if debug_data.get("evaluator_issues"):
            meta.append("Issues: " + "; ".join(debug_data["evaluator_issues"][:3]))
        if meta:
            console.print("  [meta]" + "  |  ".join(meta) + "[/meta]")
    console.print()


def run_chat() -> None:
    """Load model once, then run chat loop."""
    from generate import load_model_and_tokenizer
    from pipeline import run_pipeline

    banner()
    show_help()

    console.print("[meta]Loading model…[/meta]")
    load_model_and_tokenizer()
    console.print("[meta]Ready. Type your request below.[/meta]\n")

    debug_mode = False
    config = {"max_rewrite_passes": 3, "pass_threshold": 80}

    while True:
        try:
            user_input = Prompt.ask("[user]You[/user]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[meta]Goodbye.[/meta]")
            break

        if not user_input:
            continue

        # Commands
        lower = user_input.lower()
        if lower in ("/quit", "/exit", "/q"):
            console.print("[meta]Goodbye.[/meta]")
            break
        if lower == "/help":
            show_help()
            continue
        if lower == "/debug":
            debug_mode = not debug_mode
            console.print("[meta]Debug mode: " + ("on" if debug_mode else "off") + "[/meta]\n")
            continue
        if lower == "/clear":
            console.clear()
            banner()
            continue

        # Generate with streaming live panel (mutable renderable for Rich compatibility)
        stream_panel = _StreamPanel()

        def stream_cb(phase: str, text: str) -> None:
            stream_panel.phase = phase
            stream_panel.text = text or ""

        with Live(
            stream_panel,
            console=console,
            refresh_per_second=12,
        ):
            result = run_pipeline(
                user_input,
                debug=debug_mode,
                config=config,
                stream_callback=stream_cb,
            )
            # Show final result in the same panel briefly
            final_text = result.get("final_lyrics", result) if isinstance(result, dict) else result
            stream_panel.phase = "done"
            stream_panel.text = final_text

        if debug_mode and isinstance(result, dict):
            show_lyrics(
                result.get("final_lyrics", ""),
                debug=True,
                debug_data=result,
            )
        else:
            show_lyrics(result if isinstance(result, str) else result.get("final_lyrics", ""))


def main() -> None:
    try:
        run_chat()
    except Exception as e:
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
