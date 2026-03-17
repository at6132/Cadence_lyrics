"""
Human realism scoring for lyrics.
Combines blacklist hits, repetition, abstractness, specificity, evaluator LLM score,
chorus/hook quality, and musicality (anti-prose).
"""
import re
from collections import Counter
from typing import Optional, List, Dict, Any

# Default config (easy to tune)
DEFAULT_CONFIG = {
    "start_score": 100,
    "penalty_per_banned_phrase": 8,
    "penalty_repeated_bigram_per_hit": 2,
    "penalty_abstract_heavy_line": 5,
    "penalty_obvious_rhyme": 4,
    "penalty_repeated_structure": 6,
    "reward_concrete_detail": 2,
    "evaluator_weight": 0.5,
    "pass_threshold": 85,
    "max_reward_concrete": 10,
    # Total score blend (non-pop): realism + evaluator + chorus + musicality
    "realism_weight": 0.35,
    "evaluator_weight_total": 0.35,
    "chorus_hook_weight": 0.15,
    "musicality_weight": 0.15,
    # Pop prompt: chorus and musicality matter more
    "pop_realism_weight": 0.25,
    "pop_evaluator_weight": 0.25,
    "pop_chorus_hook_weight": 0.30,
    "pop_musicality_weight": 0.20,
    "min_chorus_score_for_pop": 40.0,
    "no_chorus_penalty_pop": 25.0,
    "no_memorable_hook_penalty_pop": 15.0,
    "penalty_per_chorus_blacklist_phrase": 6,
}


def _count_repeated_ngrams(text: str, n: int = 2) -> int:
    """Count how many times the same n-gram appears more than once."""
    words = re.findall(r"\S+", text.lower())
    if len(words) < n:
        return 0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    return sum(1 for c in counts.values() if c > 1)


def _concrete_detail_count(text: str) -> int:
    """Heuristic: lines with concrete nouns/places/actions (numbers, quoted things, specific nouns)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    count = 0
    for line in lines:
        # Modest reward for lines that look specific (contain digits, or multiple commas/periods suggesting detail)
        if re.search(r"\d", line):
            count += 1
        if '"' in line or "'" in line:
            count += 1
        if len(line) > 40 and "," in line:
            count += 1
    return min(count, 5)  # cap so one metric doesn't dominate


def rule_based_score(
    lyrics: str,
    banned_matches: List[str],
    heuristic_flags: dict,
    config: Optional[dict] = None,
) -> float:
    """
    Score 0–100 from rules only (no LLM).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    score = float(cfg["start_score"])

    # Banned phrases
    score -= len(banned_matches) * cfg["penalty_per_banned_phrase"]

    # Repeated n-grams (bigrams)
    repeat_hits = _count_repeated_ngrams(lyrics, 2)
    score -= repeat_hits * cfg["penalty_repeated_bigram_per_hit"]

    # Heuristics
    if heuristic_flags.get("abstract_heavy_line"):
        score -= cfg["penalty_abstract_heavy_line"]
    if heuristic_flags.get("obvious_rhyme_ending"):
        score -= cfg["penalty_obvious_rhyme"]
    if heuristic_flags.get("repeated_structure"):
        score -= cfg["penalty_repeated_structure"]

    # Reward concrete detail
    concrete = _concrete_detail_count(lyrics)
    score += min(concrete * cfg["reward_concrete_detail"], cfg["max_reward_concrete"])

    return max(0.0, min(100.0, score))


def blend_with_evaluator(
    rule_score: float,
    evaluator_score: Optional[float],
    config: Optional[dict] = None,
) -> float:
    """Blend rule-based score with evaluator LLM score (0–100)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if evaluator_score is None:
        return rule_score
    w = cfg["evaluator_weight"]
    return (1 - w) * rule_score + w * evaluator_score


def human_realism_score(
    lyrics: str,
    banned_matches: List[str],
    heuristic_flags: dict,
    evaluator_score: Optional[float] = None,
    config: Optional[dict] = None,
) -> tuple:
    """
    Returns (final_score, rule_only_score).
    Final score is blended with evaluator if provided.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rule = rule_based_score(lyrics, banned_matches, heuristic_flags, config)
    final = blend_with_evaluator(rule, evaluator_score, config)
    return final, rule


def passed_threshold(score: float, config: Optional[dict] = None) -> bool:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return score >= cfg["pass_threshold"]


def chorus_hook_score(
    chorus_analysis: Dict[str, Any],
    chorus_blacklist_matches: List[str],
    config: Optional[dict] = None,
    evaluator_line_notes: Optional[List[Dict[str, Any]]] = None,
) -> tuple:
    """
    Score 0–100 for chorus/hook quality. Cliché/generic choruses cannot reach elite scores.
    Returns (score, penalties_applied_list) for debug.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    penalties_applied: List[str] = []
    score = 45.0
    if not chorus_analysis.get("has_chorus"):
        return max(0.0, score - 25.0), ["no_chorus"]
    score += 15.0
    mem = chorus_analysis.get("memorability_score", 0)
    score += mem * 0.15  # memorability helps but is capped by generic_phrase_penalty inside memorability
    score += chorus_analysis.get("repeat_consistency_score", 0) * 0.08
    avg_len = chorus_analysis.get("avg_chorus_line_length", 99)
    if avg_len > 0 and avg_len <= 7:
        score += 8.0
    elif avg_len <= 10:
        score += 4.0
    if chorus_analysis.get("hook_lines"):
        score += 6.0
    # Strong penalty for generic/cliché phrases in chorus
    n_generic = len(chorus_blacklist_matches)
    if n_generic > 0:
        p = n_generic * cfg["penalty_per_chorus_blacklist_phrase"]
        score -= p
        penalties_applied.append(f"generic_hook_flags:{n_generic}")
    if chorus_analysis.get("is_prose_like"):
        score -= 15.0
        penalties_applied.append("prose_like_chorus")
    if not chorus_analysis.get("hook_lines") and chorus_analysis.get("has_chorus"):
        score -= cfg["no_memorable_hook_penalty_pop"]
        penalties_applied.append("no_memorable_hook")
    # Evaluator line notes: if any line in chorus is flagged as cliché, penalize
    if evaluator_line_notes:
        for note in evaluator_line_notes:
            probs = note.get("problems") or []
            if any("cliché" in str(p).lower() or "cliche" in str(p).lower() or "generic" in str(p).lower() for p in probs):
                score -= 8.0
                penalties_applied.append("evaluator_cliche_note")
                break
    # Cap: chorus with any generic flags cannot exceed 78 unless strong distinctiveness
    if n_generic > 0:
        distinctiveness = (chorus_analysis.get("memorability_score_components") or {}).get("distinctiveness_bonus", 0)
        if distinctiveness < 5.0:
            score = min(score, 78.0)
            if score >= 76:
                penalties_applied.append("capped_due_to_generic_phrases")
    return max(0.0, min(100.0, score)), penalties_applied


def total_score_with_chorus_and_musicality(
    realism_score: float,
    evaluator_score: Optional[float],
    chorus_hook_score_val: float,
    musicality_score_val: float,
    is_pop_prompt: bool,
    chorus_analysis: Dict[str, Any],
    config: Optional[dict] = None,
) -> tuple:
    """
    Returns (total_score, breakdown_dict).
    breakdown_dict has: realism_score, evaluator_score, chorus_hook_score, musicality_score, total_score.
    For pop prompts, applies no-chorus penalty and uses pop weights.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ev = evaluator_score if evaluator_score is not None else realism_score
    if is_pop_prompt:
        rw = cfg["pop_realism_weight"]
        ew = cfg["pop_evaluator_weight"]
        cw = cfg["pop_chorus_hook_weight"]
        mw = cfg["pop_musicality_weight"]
        pop_penalty = 0.0
        if not chorus_analysis.get("has_chorus"):
            pop_penalty = cfg["no_chorus_penalty_pop"]
        if chorus_hook_score_val < cfg["min_chorus_score_for_pop"]:
            pop_penalty += cfg["min_chorus_score_for_pop"] - chorus_hook_score_val
        total = rw * realism_score + ew * ev + cw * chorus_hook_score_val + mw * musicality_score_val - pop_penalty
    else:
        rw = cfg["realism_weight"]
        ew = cfg["evaluator_weight_total"]
        cw = cfg["chorus_hook_weight"]
        mw = cfg["musicality_weight"]
        total = rw * realism_score + ew * ev + cw * chorus_hook_score_val + mw * musicality_score_val
    total = max(0.0, min(100.0, total))
    breakdown = {
        "realism_score": round(realism_score, 1),
        "evaluator_score": round(ev, 1) if ev is not None else None,
        "chorus_hook_score": round(chorus_hook_score_val, 1),
        "musicality_score": round(musicality_score_val, 1),
        "total_score": round(total, 1),
    }
    return total, breakdown


def passed_threshold_with_pop(
    total_score: float,
    is_pop_prompt: bool,
    chorus_analysis: Dict[str, Any],
    chorus_hook_score_val: float,
    config: Optional[dict] = None,
) -> bool:
    """Pass/fail that for pop prompts also requires minimum chorus/hook quality."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not (total_score >= cfg["pass_threshold"]):
        return False
    if not is_pop_prompt:
        return True
    if not chorus_analysis.get("has_chorus"):
        return False
    return chorus_hook_score_val >= cfg["min_chorus_score_for_pop"]
