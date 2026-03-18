"""
Run a fixed set of prompts through the lyric pipeline and save a run_log file for each.
Usage: python run_batch_prompts.py
       python run_batch_prompts.py --log-dir ./my_logs
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

BATCH_PROMPTS = [
    "Write a pop song about a breakup with a chorus that feels instantly memorable after one listen, but do not use any stock breakup phrases, city/streetlight imagery, or \"we're broken / undone / apart\" type wording.",
    "Write a pop song about missing someone after a fight, with a short repeatable chorus that sounds fresh and human, not like a template people have heard a hundred times.",
    "Write a song about driving home after a breakup, but avoid every obvious lyric angle; no \"city at night,\" no \"wrong turn,\" no \"radio playing our song,\" and no generic emotional lines.",
    "Write a song with an unlabeled repeated refrain instead of a labeled chorus; make the refrain 2 lines long and naturally recur three times without sounding repetitive or generic.",
    "Write a pop song where the verses are specific and scene-based, but the chorus is simpler, shorter, and catchier than the verses without becoming cliché.",
    "Write a breakup song that feels musical and singable, but not like a narrated story; keep lines compressed and lyrical, and make the chorus clearly distinct from the verses.",
    "Write an indie-pop song about seeing your ex at a grocery store, using awkward real-life details and a simple repeated hook that does not rely on generic longing language.",
    "Write a song about sleeping in the same room after a breakup, but do not use \"your side of the bed,\" \"empty room,\" \"cold pillow,\" or similar phrases.",
    "Write a pop chorus only, 6 lines max, about wanting someone back after a fight; it should be catchy and repeatable, but every line must feel original and non-template.",
    "Write a full song where the chorus uses the pattern \"same ___, different ___\" if you want, but make it feel genuinely fresh and not like a stock pop lyric.",
    "Write a song about walking through a party where your ex is there, using no emotional words like sad, hurt, miss, love, pain, broken, lonely, or empty; let the chorus still feel memorable.",
    "Write a song about a breakup in a small town, but do not use \"same old town,\" \"same streets,\" \"driving around,\" or \"everyone knows\"; make the hook original.",
    "Write a song where the chorus repeats one short line three times, but the repeated line must feel specific and striking, not vague or generic.",
    "Write a pop song about wanting to start over with someone, but do not use \"start again,\" \"start fresh,\" \"try again,\" \"come back,\" or \"find my way back.\"",
    "Write a song with no section labels at all that still clearly has verses and a chorus, and make the chorus easy to identify because of its shape and repetition.",
    "Write a singable non-pop song about sitting in a parked car after an argument; it should feel lyrical and musical, but it should not be classified as a pop-hook song.",
    "Write a chorus about two people drifting apart that avoids all abstract emotional filler and instead uses one concrete repeated image to make it memorable.",
    "Write a pop song where the verses are longer and the chorus is only 3 short lines, and make sure the chorus does not sound like a slogan, caption, or stock lyric.",
    "Write a breakup song that sounds like something a real 19-year-old would sing in 2026, not like classic polished songwriter language and not like AI trying to sound poetic.",
    "Write a song about hearing your ex's name mentioned casually in conversation, with a chorus that is catchy, simple, and human, but not broad or overdramatic.",
]


def main():
    import argparse
    n = len(BATCH_PROMPTS)
    p = argparse.ArgumentParser(description=f"Run {n} fixed prompts through the pipeline and save run_log for each.")
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
        print(f"[{i}/{n}] Running: {prompt[:70]}...")
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
