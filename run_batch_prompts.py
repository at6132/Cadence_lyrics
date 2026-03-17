"""
Run a fixed set of 6 prompts through the lyric pipeline and save a run_log file for each.
Usage: python run_batch_prompts.py
       python run_batch_prompts.py --log-dir ./my_logs
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

BATCH_PROMPTS = [
    "Write a sad pop song about a breakup that uses specific real-life details and scenes, avoids all clichés and generic phrases, keeps the lyrics conversational and human, and includes a clear, catchy, singable chorus (not prose-like) while implying emotion instead of stating it.",
    "Write a sad pop song about a breakup.",
    "Write a pop song with a strong, repeatable chorus about missing someone after a fight; make the chorus something people could actually sing and remember, use real-life details, no clichés, and keep it natural.",
    "Write a song about sitting in your room after an argument; keep lines short and singable, use no long sentences, no metaphors, only real actions and thoughts, and make sure it still feels like a song, not a paragraph.",
    "Write a sad indie-pop song about seeing your ex at a party; do not use any emotional words like sad, hurt, miss, love, or pain, only show the story through actions, dialogue, and details, and include a chorus that is simple and repeatable.",
    "Write a pop song about a breakup that feels musical and singable, not like a story being narrated; keep the verses concise, make the chorus simpler and catchier than the verses, and avoid long sentence-like lines.",
]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run 6 fixed prompts through the pipeline and save run_log for each.")
    p.add_argument("--log-dir", type=str, default=None, help="Directory for run logs (default: Lyric_model/run_logs)")
    p.add_argument("--quiet", action="store_true", help="Only print log paths and summary, not lyrics")
    args = p.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else _root / "run_logs"
    log_dir = log_dir.resolve()
    print(f"Run logs will be saved to: {log_dir}\n")

    from generate import load_model_and_tokenizer
    from pipeline import run_pipeline

    print("Loading model...")
    load_model_and_tokenizer()
    print("Model loaded.\n")

    config = {"max_rewrite_passes": 1, "pass_threshold": 85}
    saved_paths = []

    for i, prompt in enumerate(BATCH_PROMPTS, 1):
        print(f"[{i}/6] Running: {prompt[:70]}...")
        run_log_paths = []
        result = run_pipeline(
            prompt,
            debug=False,
            config=config,
            run_log_dir=log_dir,
            run_log_path_out=run_log_paths,
        )
        if run_log_paths:
            saved_paths.append(run_log_paths[0])
            print(f"      Saved: {run_log_paths[0]}")
        if not args.quiet and result:
            lyrics_preview = (result[:200] + "...") if len(result) > 200 else result
            print(f"      Lyrics: {lyrics_preview}\n")
        else:
            print("")

    print("=" * 60)
    print("Batch complete. Run logs saved:")
    for p in saved_paths:
        print(f"  {p}")
    print(f"\nTotal: {len(saved_paths)} runs.")


if __name__ == "__main__":
    main()
