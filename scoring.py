"""
Human realism scoring for lyrics.
Combines blacklist hits, repetition, abstractness, specificity, and optional evaluator LLM score.
"""
import re
from collections import Counter

# Default config (easy to tune)
DEFAULT_CONFIG = {
    "start_score": 100,
    "penalty_per_banned_phrase": 8,
    "penalty_repeated_bigram_per_hit": 2,
    "penalty_abstract_heavy_line": 5,
    "penalty_obvious_rhyme": 4,
    "penalty_repeated_structure": 6,
    "reward_concrete_detail": 2,   # cap applied
    "evaluator_weight": 0.5,       # blend: (1-w)*rule_score + w*evaluator_score
    "pass_threshold": 80,
    "max_reward_concrete": 10,
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
    banned_matches: list[str],
    heuristic_flags: dict,
    config: dict | None = None,
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
    evaluator_score: float | None,
    config: dict | None = None,
) -> float:
    """Blend rule-based score with evaluator LLM score (0–100)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if evaluator_score is None:
        return rule_score
    w = cfg["evaluator_weight"]
    return (1 - w) * rule_score + w * evaluator_score


def human_realism_score(
    lyrics: str,
    banned_matches: list[str],
    heuristic_flags: dict,
    evaluator_score: float | None = None,
    config: dict | None = None,
) -> tuple[float, float]:
    """
    Returns (final_score, rule_only_score).
    Final score is blended with evaluator if provided.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rule = rule_based_score(lyrics, banned_matches, heuristic_flags, config)
    final = blend_with_evaluator(rule, evaluator_score, config)
    return final, rule


def passed_threshold(score: float, config: dict | None = None) -> bool:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return score >= cfg["pass_threshold"]
