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
    "chorus_cap_one_generic": 70.0,
    "chorus_cap_multiple_or_evaluator_cliche": 60.0,
    "pop_prompt_low_evaluator_cap": 65.0,
    "pop_prompt_low_evaluator_cap_with_generic_chorus": 55.0,
    "chorus_cap_when_evaluator_low_pop": 55.0,
    "evaluator_chorus_cliche_penalty": 12.0,
    "evaluator_low_bonus_multiplier": 0.4,
    "chorus_cap_multiple_template": 55.0,
    "penalty_per_template_hook_flag": 5,
    "chorus_only_generic_cap": 50.0,
    "chorus_only_low_evaluator_cap": 55.0,
    "chorus_only_template_penalty_multiplier": 1.5,
}


def _normalize_line_for_match(line: str) -> str:
    """Normalize line text for matching evaluator notes to chorus lines."""
    return re.sub(r"[^\w\s]", "", (line or "").strip().lower())


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


def _evaluator_cliche_keywords(problem_text: str) -> bool:
    """True if problem text indicates cliché/stock/generic/lacks specificity/familiar trope."""
    t = (problem_text or "").lower()
    return any(
        k in t
        for k in (
            "cliché", "cliche", "generic", "stock", "overused",
            "lacks specificity", "familiar trope", "generic sentiment",
            "stock pop", "predictable", "repetitive chorus", "lacks originality",
        )
    )


def _evaluator_chorus_level_issues(issues_text: str) -> bool:
    """True if evaluator issues text likely refers to chorus (stock pop lyric, clichéd chorus, etc.)."""
    t = (issues_text or "").lower()
    return any(
        k in t
        for k in (
            "stock pop lyric", "stock pop", "clichéd chorus", "cliche chorus",
            "repetitive chorus", "generic sentiment", "lacks originality",
            "familiar trope", "generic chorus",
        )
    )


def _word_overlap_ratio(note_words: List[str], chorus_line_norm: str) -> float:
    """Return share of note words that appear in chorus line (0-1)."""
    if not note_words:
        return 0.0
    chorus_words = set(re.findall(r"\w+", chorus_line_norm))
    return sum(1 for w in note_words if len(w) > 2 and w in chorus_words) / len(note_words)


def _fuzzy_match_note_to_chorus(
    note_line: str, chorus_lines: List[str], chorus_norm: Dict[str, str], min_overlap: float = 0.4
) -> Optional[str]:
    """
    If note line matches a chorus line (exact norm, substring, or word overlap), return that chorus line.
    """
    norm = _normalize_line_for_match(note_line)
    if not norm:
        return None
    if norm in chorus_norm:
        return chorus_norm[norm]
    for c_norm, c_orig in chorus_norm.items():
        if norm in c_norm or c_norm in norm:
            return c_orig
    note_words = re.findall(r"\w+", norm)
    for c_norm, c_orig in chorus_norm.items():
        if _word_overlap_ratio(note_words, c_norm) >= min_overlap:
            return c_orig
    return None


def chorus_hook_score(
    chorus_analysis: Dict[str, Any],
    chorus_blacklist_matches: List[str],
    config: Optional[dict] = None,
    evaluator_line_notes: Optional[List[Dict[str, Any]]] = None,
    evaluator_score: Optional[float] = None,
    is_pop_prompt: bool = False,
    evaluator_issues: Optional[Any] = None,
    chorus_only_prompt: bool = False,
) -> tuple:
    """
    Score 0–100 for chorus/hook quality. Short repeated cliché choruses are capped well below elite.
    Uses generic_hook_flags + template_hook_flags for penalties/caps. Fuzzy-matches evaluator notes to chorus.
    Returns (score, debug_dict) including chorus_penalty_reason_details, evaluator_fuzzy_chorus_matches.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    penalties_applied: List[str] = []
    penalty_reason_details: List[str] = []
    debug: Dict[str, Any] = {
        "chorus_lines_flagged_by_evaluator": [],
        "evaluator_fuzzy_chorus_matches": [],
        "evaluator_chorus_penalty_applied": False,
        "chorus_score_cap_reason": "",
        "evaluator_low_score_cap_applied": False,
        "evaluator_low_score_cap_value": None,
        "chorus_penalty_reason_details": [],
        "chorus_only_cap_applied": False,
    }
    score = 45.0
    if not chorus_analysis.get("has_chorus"):
        debug["penalties_applied"] = ["no_chorus"]
        return max(0.0, score - 25.0), debug
    score += 15.0

    evaluator_low = evaluator_score is not None and evaluator_score <= 50
    bonus_mult = cfg["evaluator_low_bonus_multiplier"] if (evaluator_low and is_pop_prompt) else 1.0

    mem = chorus_analysis.get("memorability_score", 0)
    score += mem * 0.15 * bonus_mult
    score += chorus_analysis.get("repeat_consistency_score", 0) * 0.08 * bonus_mult
    avg_len = chorus_analysis.get("avg_chorus_line_length", 99)
    if avg_len > 0 and avg_len <= 7:
        score += 8.0 * bonus_mult
    elif avg_len <= 10:
        score += 4.0 * bonus_mult
    if chorus_analysis.get("hook_lines"):
        score += 6.0 * bonus_mult
    if bonus_mult < 1.0:
        penalties_applied.append("evaluator_low_bonus_reduction")

    n_generic = len(chorus_blacklist_matches)
    template_hook_flags = chorus_analysis.get("template_hook_flags") or []
    n_template = len(template_hook_flags)
    if n_generic > 0:
        p = n_generic * cfg["penalty_per_chorus_blacklist_phrase"]
        score -= p
        penalties_applied.append(f"generic_hook_flags:{n_generic}")
        penalty_reason_details.append("generic_hook_flags")
    if n_template > 0:
        mult = cfg["chorus_only_template_penalty_multiplier"] if chorus_only_prompt else 1.0
        p_template = n_template * cfg["penalty_per_template_hook_flag"] * mult
        score -= p_template
        penalties_applied.append(f"template_hook_flags:{n_template}")
        penalty_reason_details.append("stock_pop_hook_family")
    if chorus_analysis.get("is_prose_like"):
        score -= 15.0
        penalties_applied.append("prose_like_chorus")
    if not chorus_analysis.get("hook_lines") and chorus_analysis.get("has_chorus"):
        score -= cfg["no_memorable_hook_penalty_pop"]
        penalties_applied.append("no_memorable_hook")

    chorus_lines = chorus_analysis.get("chorus_lines") or []
    chorus_norm = {_normalize_line_for_match(l): l for l in chorus_lines}
    flagged_lines: List[str] = []
    fuzzy_matches: List[str] = []
    for note in (evaluator_line_notes or []):
        line_text = note.get("line") or note.get("text") or ""
        probs = note.get("problems") or []
        if not any(_evaluator_cliche_keywords(str(p)) for p in probs):
            continue
        matched = _fuzzy_match_note_to_chorus(line_text, chorus_lines, chorus_norm)
        if matched:
            flagged_lines.append(matched)
            snippet = (line_text.strip() or matched)[:60]
            if snippet not in fuzzy_matches:
                fuzzy_matches.append(snippet)
    if flagged_lines:
        flagged_lines = list(dict.fromkeys(flagged_lines))
        debug["chorus_lines_flagged_by_evaluator"] = flagged_lines
        debug["evaluator_fuzzy_chorus_matches"] = fuzzy_matches
        debug["evaluator_chorus_penalty_applied"] = True
        score -= cfg["evaluator_chorus_cliche_penalty"]
        penalties_applied.append("evaluator_chorus_cliche")
        penalty_reason_details.append("evaluator_fuzzy_chorus_cliche_match")

    issues_text = ""
    if isinstance(evaluator_issues, str):
        issues_text = evaluator_issues
    elif isinstance(evaluator_issues, dict):
        issues_text = (evaluator_issues.get("issues") or evaluator_issues.get("summary") or "").strip() or str(evaluator_issues.get("problems", ""))
    if _evaluator_chorus_level_issues(issues_text):
        if not debug["evaluator_chorus_penalty_applied"]:
            score -= cfg["evaluator_chorus_cliche_penalty"]
            penalties_applied.append("evaluator_chorus_level_criticism")
        debug["evaluator_chorus_penalty_applied"] = True
        penalty_reason_details.append("evaluator_chorus_level_criticism")

    total_suspicious = n_generic + n_template
    cap_55 = cfg["chorus_cap_multiple_template"]
    distinctiveness = (chorus_analysis.get("memorability_score_components") or {}).get("distinctiveness_bonus", 0)
    cap_70 = cfg["chorus_cap_one_generic"]
    cap_60 = cfg["chorus_cap_multiple_or_evaluator_cliche"]
    cap_low_ev = cfg["chorus_cap_when_evaluator_low_pop"]
    if is_pop_prompt and evaluator_low:
        score = min(score, cap_low_ev)
        debug["chorus_score_cap_reason"] = "evaluator_low_pop_chorus_cap"
        debug["evaluator_low_score_cap_applied"] = True
        debug["evaluator_low_score_cap_value"] = cap_low_ev
        penalties_applied.append(debug["chorus_score_cap_reason"])
    elif total_suspicious >= 2 or debug["evaluator_chorus_penalty_applied"]:
        score = min(score, cap_60)
        if total_suspicious >= 2:
            score = min(score, cap_55)
            debug["chorus_score_cap_reason"] = "generic_and_template_multiple"
        else:
            debug["chorus_score_cap_reason"] = "generic_hook_flags_multiple_or_evaluator_cliche"
    elif n_generic == 1 and n_template == 0:
        if distinctiveness >= 5.0:
            pass
        else:
            score = min(score, cap_70)
            debug["chorus_score_cap_reason"] = "generic_hook_flags_single"
    elif n_template >= 1 or n_generic >= 1:
        score = min(score, cap_55)
        debug["chorus_score_cap_reason"] = "template_or_generic_single"

    if chorus_only_prompt and (n_generic > 0 or n_template > 0):
        cap_co = cfg["chorus_only_generic_cap"]
        score = min(score, cap_co)
        debug["chorus_score_cap_reason"] = debug.get("chorus_score_cap_reason") or "chorus_only_generic_cap"
        debug["chorus_only_cap_applied"] = True

    debug["chorus_penalty_reason_details"] = list(dict.fromkeys(penalty_reason_details))
    if debug["chorus_score_cap_reason"] and not debug["evaluator_low_score_cap_applied"]:
        penalties_applied.append(debug["chorus_score_cap_reason"])
    debug["penalties_applied"] = penalties_applied
    return max(0.0, min(100.0, score)), debug


def total_score_with_chorus_and_musicality(
    realism_score: float,
    evaluator_score: Optional[float],
    chorus_hook_score_val: float,
    musicality_score_val: float,
    is_pop_prompt: bool,
    chorus_analysis: Dict[str, Any],
    config: Optional[dict] = None,
    chorus_only_prompt: bool = False,
) -> tuple:
    """
    Returns (total_score, breakdown_dict).
    breakdown_dict includes: realism_score, evaluator_score, chorus_hook_score, musicality_score,
    total_score, final_score_cap_reason (when pop + low evaluator).
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
    final_score_cap_reason = ""
    evaluator_low_cap_applied = False
    evaluator_low_cap_value = None
    if is_pop_prompt and evaluator_score is not None and evaluator_score <= 50:
        has_generic_chorus = (chorus_analysis.get("generic_hook_flags") or []) or (chorus_analysis.get("template_hook_flags") or [])
        if has_generic_chorus:
            cap = cfg["pop_prompt_low_evaluator_cap_with_generic_chorus"]
        else:
            cap = cfg["pop_prompt_low_evaluator_cap"]
        if total > cap:
            total = cap
            final_score_cap_reason = "pop_prompt_low_evaluator_cap_with_generic_chorus" if has_generic_chorus else "pop_prompt_low_evaluator_cap"
            evaluator_low_cap_applied = True
            evaluator_low_cap_value = cap
    if chorus_only_prompt and evaluator_score is not None and evaluator_score <= 50:
        cap_co = cfg["chorus_only_low_evaluator_cap"]
        if total > cap_co:
            total = cap_co
            final_score_cap_reason = final_score_cap_reason or "chorus_only_low_evaluator_cap"
            evaluator_low_cap_applied = True
            evaluator_low_cap_value = min(evaluator_low_cap_value or 100, cap_co)
    breakdown = {
        "realism_score": round(realism_score, 1),
        "evaluator_score": round(ev, 1) if ev is not None else None,
        "chorus_hook_score": round(chorus_hook_score_val, 1),
        "musicality_score": round(musicality_score_val, 1),
        "total_score": round(total, 1),
        "final_score_cap_reason": final_score_cap_reason or None,
        "evaluator_low_score_cap_applied": evaluator_low_cap_applied,
        "evaluator_low_score_cap_value": evaluator_low_cap_value,
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
