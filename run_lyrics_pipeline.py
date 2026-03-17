"""
Entry point for the anti-AI lyric pipeline.
Single-shot: prints lyrics only (or --debug for details).
Chat UI: use --chat for an interactive CLI chat.
"""
import sys
from pathlib import Path

# Ensure Lyric_model is on path when run from project root
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Anti-AI lyric pipeline: draft → detect → rewrite → score → retry."
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Lyric request (omit when using --chat)",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="Start the interactive chat UI",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print structured object: final_lyrics, score, passes_used, banned_phrases, evaluator_issues, draft, rewrite_passes",
    )
    p.add_argument("--max-rewrites", type=int, default=3, help="Max total rewrite passes")
    p.add_argument("--threshold", type=int, default=80, help="Score threshold to pass")
    args = p.parse_args()

    if args.chat:
        from chat_ui import main as chat_main
        chat_main()
        return

    # Single-shot mode
    from generate import load_model_and_tokenizer
    load_model_and_tokenizer()
    from pipeline import run_pipeline

    prompt = args.prompt or "Write a verse and chorus about heartbreak."
    config = {"max_rewrite_passes": args.max_rewrites, "pass_threshold": args.threshold}
    result = run_pipeline(prompt, debug=args.debug, config=config)

    if args.debug and isinstance(result, dict):
        import json
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
