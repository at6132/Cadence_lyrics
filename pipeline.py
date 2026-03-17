"""
Anti-AI lyric generation pipeline.
Multi-step: draft → blacklist/eval → rewrite → score → retry (max 2 retries after first rewrite) → return best.
Uses existing model; no retraining.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Union, List

from prompts import (
    DRAFT_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    augment_user_prompt,
)
from phrase_blacklist import load_blacklist, match_phrases, heuristic_flags
from scoring import (
    human_realism_score,
    passed_threshold,
    DEFAULT_CONFIG as SCORE_CONFIG,
)

# Pipeline config (max_rewrite_passes = 1 initial + 2 retries = 3 total rewrites)
PIPELINE_CONFIG = {
    "max_rewrite_passes": 3,
    "pass_threshold": 80,
    "draft_max_tokens": 500,
    "rewrite_max_tokens": 550,
    "evaluator_max_tokens": 600,
    "draft_temperature": 0.85,
    "rewrite_temperature": 0.75,
    "evaluator_temperature": 0.3,
    # Optional future: "style_mode": None,  # e.g. "indie", "pop", "rap", "folk"
    # Optional future: "make_more_raw": False,  # final pass to roughen phrasing
}


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model reply (handles ```json ... ``` or raw JSON)."""
    text = text.strip()
    # Code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Raw JSON
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _slug(s: str, max_len: int = 50) -> str:
    """Safe filename slug from prompt."""
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s).strip("-").lower()
    return s[:max_len] if s else "run"


def _write_run_log(
    user_prompt: str,
    draft: str,
    draft_eval: Optional[dict],
    draft_score: float,
    rewrite_history: List[str],
    eval_history: List[Optional[dict]],
    score_history: List[float],
    final_lyrics: str,
    final_score: float,
    passes_used: int,
    banned_phrases: List[str],
    log_dir: Path,
) -> Path:
    """Write one text file per run with prompt, draft, every eval step, rewrites, and final."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slug(user_prompt)
    path = log_dir / f"{ts}_{slug}.txt"
    lines = [
        "=" * 60,
        "PROMPT",
        "=" * 60,
        user_prompt.strip(),
        "",
        "=" * 60,
        "DRAFT",
        "=" * 60,
        draft.strip(),
        "",
        "=" * 60,
        "EVALUATOR (draft)",
        "=" * 60,
        json.dumps(draft_eval, indent=2) if draft_eval else "(no evaluator output)",
        "",
        "Draft score (blended): " + str(round(draft_score, 1)),
        "",
    ]
    # eval_history[0]=draft eval, eval_history[1]=after rewrite 1, ...; same for score_history
    for i in range(len(rewrite_history)):
        eval_data = eval_history[i + 1] if i + 1 < len(eval_history) else None
        score_val = score_history[i + 1] if i + 1 < len(score_history) else 0.0
        lines.extend([
            "=" * 60,
            f"REWRITE {i + 1}",
            "=" * 60,
            rewrite_history[i].strip(),
            "",
            "=" * 60,
            f"EVALUATOR (rewrite {i + 1})",
            "=" * 60,
            json.dumps(eval_data, indent=2) if eval_data else "(no evaluator output)",
            "",
            f"Rewrite {i + 1} score (blended): " + str(round(score_val, 1)),
            "",
        ])
    lines.extend([
        "=" * 60,
        "FINAL LYRICS",
        "=" * 60,
        final_lyrics.strip(),
        "",
        "=" * 60,
        "RUN SUMMARY",
        "=" * 60,
        f"Final score: {round(final_score, 1)}",
        f"Passes used: {passes_used}",
        f"Banned phrases in final: {banned_phrases}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _dedupe_lines(text: str) -> str:
    """Remove duplicate consecutive lines and near-duplicates (same line repeated)."""
    lines = text.splitlines()
    seen = set()
    out = []
    for ln in lines:
        strip = ln.strip()
        norm = " ".join(strip.lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(ln)
        elif strip:
            out.append(ln)
    return "\n".join(out).strip()


def _call_model(
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Call the lyric model (generate module). If stream_callback(text_so_far) is set, stream output."""
    from generate import generate
    return generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        system_prompt=system_prompt,
        stream_callback=stream_callback,
    )


def run_draft(
    user_prompt: str,
    config: Optional[dict] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Step A: Initial draft generation. stream_callback(text_so_far) for streaming."""
    cfg = {**PIPELINE_CONFIG, **(config or {})}
    augmented = augment_user_prompt(user_prompt)
    return _call_model(
        augmented,
        DRAFT_SYSTEM_PROMPT,
        max_new_tokens=cfg["draft_max_tokens"],
        temperature=cfg["draft_temperature"],
        stream_callback=stream_callback,
    ).strip()


def run_evaluator(lyrics: str, config: Optional[dict] = None) -> Optional[dict]:
    """Step: Run critic/evaluator; return parsed JSON or None."""
    cfg = {**PIPELINE_CONFIG, **(config or {})}
    prompt = "Evaluate these lyrics:\n\n" + lyrics
    raw = _call_model(
        prompt,
        CRITIC_SYSTEM_PROMPT,
        max_new_tokens=cfg["evaluator_max_tokens"],
        temperature=cfg["evaluator_temperature"],
    )
    return _extract_json(raw)


def run_rewrite(
    lyrics: str,
    config: Optional[dict] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Step C: Humanization rewrite. stream_callback(text_so_far) for streaming."""
    cfg = {**PIPELINE_CONFIG, **(config or {})}
    prompt = "Rewrite these lyrics to sound more human:\n\n" + lyrics
    return _call_model(
        prompt,
        REWRITE_SYSTEM_PROMPT,
        max_new_tokens=cfg["rewrite_max_tokens"],
        temperature=cfg["rewrite_temperature"],
        stream_callback=stream_callback,
    ).strip()


def run_pipeline(
    user_prompt: str,
    *,
    debug: bool = False,
    config: Optional[dict] = None,
    stream_callback: Optional[Callable[[str, str], None]] = None,
    run_log_dir: Optional[Path] = None,
    run_log_path_out: Optional[List[str]] = None,
) -> Union[str, dict]:
    """
    Full pipeline: draft → blacklist/heuristic → evaluator → rewrite → score → retry.
    stream_callback(phase, text_so_far) for streaming; phase in ("draft", "evaluating", "rewrite_1", "rewrite_2", "rewrite_3").
    If run_log_dir is set, writes a text file per run with prompt, draft, every eval step, rewrites, and final.
    Returns only final lyrics (str), or if debug=True a dict with final_lyrics, score, passes_used, etc.
    """
    cfg = {**PIPELINE_CONFIG, **(config or {})}
    score_cfg = {**SCORE_CONFIG, "pass_threshold": cfg["pass_threshold"]}
    blacklist = load_blacklist()
    eval_history: List[Optional[dict]] = []
    score_history: List[float] = []

    def cb(phase: str, text: str = "") -> None:
        if stream_callback:
            stream_callback(phase, text)

    # Step A: Draft (streamed)
    draft = run_draft(
        user_prompt,
        config=cfg,
        stream_callback=(lambda t: cb("draft", t)) if stream_callback else None,
    )
    draft = _dedupe_lines(draft)

    banned = match_phrases(draft, blacklist)
    heur = heuristic_flags(draft)
    cb("evaluating", "")
    eval_result = run_evaluator(draft, config=cfg)
    eval_score = None
    if eval_result and isinstance(eval_result.get("score"), (int, float)):
        eval_score = float(eval_result["score"])
    score, rule_score = human_realism_score(draft, banned, heur, eval_score, score_cfg)
    eval_history.append(eval_result)
    score_history.append(score)

    best_lyrics = draft
    best_score = score
    best_eval_result = eval_result
    passes_used = 0
    rewrite_history: List[str] = []

    # Rewrite loop: up to max_rewrite_passes rewrites (each pass rewrites current best)
    for pass_num in range(cfg["max_rewrite_passes"]):
        phase = f"rewrite_{pass_num + 1}"
        current = run_rewrite(
            best_lyrics,
            config=cfg,
            stream_callback=(lambda t, p=phase: cb(p, t)) if stream_callback else None,
        )
        current = _dedupe_lines(current)
        rewrite_history.append(current)

        banned_cur = match_phrases(current, blacklist)
        heur_cur = heuristic_flags(current)
        eval_cur = run_evaluator(current, config=cfg)
        eval_score_cur = None
        if eval_cur and isinstance(eval_cur.get("score"), (int, float)):
            eval_score_cur = float(eval_cur["score"])
        score_cur, _ = human_realism_score(current, banned_cur, heur_cur, eval_score_cur, score_cfg)
        eval_history.append(eval_cur)
        score_history.append(score_cur)

        if score_cur > best_score:
            best_score = score_cur
            best_lyrics = current
            best_eval_result = eval_cur

        passes_used += 1
        if passed_threshold(best_score, score_cfg) and not match_phrases(best_lyrics, blacklist):
            break

    banned_final = match_phrases(best_lyrics, blacklist)
    if run_log_dir:
        log_path = _write_run_log(
            user_prompt=user_prompt,
            draft=draft,
            draft_eval=eval_history[0] if eval_history else None,
            draft_score=score_history[0] if score_history else 0.0,
            rewrite_history=rewrite_history,
            eval_history=eval_history,
            score_history=score_history,
            final_lyrics=best_lyrics,
            final_score=best_score,
            passes_used=passes_used,
            banned_phrases=banned_final,
            log_dir=Path(run_log_dir),
        )
        if run_log_path_out is not None:
            run_log_path_out.append(str(log_path))

    if debug:
        return {
            "final_lyrics": best_lyrics,
            "score": best_score,
            "passes_used": passes_used,
            "banned_phrases": banned_final,
            "evaluator_issues": (best_eval_result or {}).get("issues", []) if best_eval_result else [],
            "draft": draft,
            "rewrite_passes": rewrite_history,
        }
    return best_lyrics


def main():
    import argparse
    p = argparse.ArgumentParser(description="Anti-AI lyric pipeline: draft → detect → rewrite → score → retry")
    p.add_argument("prompt", nargs="?", default="Write a verse and chorus about heartbreak.", help="Lyric request")
    p.add_argument("--debug", action="store_true", help="Return structured object with score, passes, etc.")
    p.add_argument("--max-rewrites", type=int, default=3, help="Max total rewrite passes (1 initial + retries)")
    p.add_argument("--threshold", type=int, default=80, help="Score threshold to pass")
    p.add_argument("--log-dir", type=str, default=None, help="Write a run log (draft, evals, rewrites) to this directory")
    args = p.parse_args()

    config = {"max_rewrite_passes": args.max_rewrites, "pass_threshold": args.threshold}
    log_dir = Path(args.log_dir) if args.log_dir else None
    result = run_pipeline(args.prompt, debug=args.debug, config=config, run_log_dir=log_dir)

    if args.debug and isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
