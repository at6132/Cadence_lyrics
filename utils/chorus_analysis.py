"""
Chorus and hook awareness for the lyric pipeline.
Detects chorus sections, hook lines, repeat consistency, prose-like chorus, generic hooks.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple

# Section headers: [Chorus], Chorus:, Verse 1:, VERSE 1, Pre-Chorus:, etc.
# Optional [ prefix, optional number after verse, optional ] and :, case-insensitive.
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*\[?\s*"
    r"(verse\s*\d*|chorus|hook|bridge|pre\s*[- ]?\s*chorus|outro|intro)"
    r"\s*\]?\s*:?\s*$",
    re.IGNORECASE,
)


def _normalize_section_type(raw: str) -> str:
    """Map 'Verse 1', 'Pre-Chorus', etc. to canonical 'verse', 'prechorus'."""
    s = raw.lower().strip()
    if s.startswith("verse"):
        return "verse"
    if s.startswith("pre") and "chorus" in s:
        return "prechorus"
    for name in ("chorus", "hook", "bridge", "outro", "intro"):
        if name in s or s == name:
            return name
    return s.replace(" ", "").replace("-", "")


def _normalize_line(s: str) -> str:
    """Normalize for comparison: lowercase, collapse whitespace, strip punctuation."""
    s = " ".join(s.strip().split()).lower()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation for refrain matching
    return s.strip()


def get_sections(
    lyrics: str,
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """
    Split lyrics into (section_type, lines) and list of detected header strings.
    Section type is canonical: verse, chorus, hook, bridge, prechorus, outro, intro, or block.
    Supports: Verse 1:, Chorus:, [Chorus], VERSE 1, Pre-Chorus:, etc. Case-insensitive, optional numbering and colon.
    Returns (sections, detected_section_headers).
    """
    lines = lyrics.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_type: Optional[str] = None
    current_lines: List[str] = []
    detected_headers: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            if current_lines:
                sections.append((current_type or "block", current_lines))
                current_lines = []
            current_type = None
            continue
        match = SECTION_HEADER_PATTERN.match(line)
        if match:
            if current_lines:
                sections.append((current_type or "block", current_lines))
                current_lines = []
            raw_label = match.group(1).strip()
            detected_headers.append(raw_label)
            current_type = _normalize_section_type(raw_label)
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_type or "block", current_lines))
    return sections, detected_headers


def _get_sections(lyrics: str) -> List[tuple]:
    """Back-compat wrapper; returns sections only."""
    sections, _ = get_sections(lyrics)
    return sections


def _find_repeated_refrains(
    all_lines: List[str], min_repeats: int = 2
) -> Tuple[List[List[str]], List[dict], str]:
    """
    Find 1/2/3/4-line blocks that repeat at least min_repeats times (non-adjacent ok).
    Prefer longer repeated blocks (2 or 3 lines) over a single repeated line when both exist.
    Returns (chorus_sections, candidates, selection_reason).
    """
    if not all_lines:
        return [], [], ""
    lines = [l.strip() for l in all_lines if l.strip()]
    if len(lines) < 2:
        return [], [], ""

    candidates: List[dict] = []

    for block_len in range(1, 5):
        if block_len > len(lines):
            break
        sig_to_lines: Dict[str, List[str]] = {}
        sig_to_count: Dict[str, int] = {}
        for i in range(len(lines) - block_len + 1):
            block = lines[i : i + block_len]
            sig = "\n".join(_normalize_line(l) for l in block)
            if not sig:
                continue
            sig_to_lines[sig] = block
            sig_to_count[sig] = sig_to_count.get(sig, 0) + 1
        for sig, count in sig_to_count.items():
            if count >= min_repeats and sig in sig_to_lines:
                candidates.append({
                    "sig": sig,
                    "lines": list(sig_to_lines[sig]),
                    "count": count,
                    "block_len": block_len,
                })
    if not candidates:
        return [], [], ""

    # Prefer: at least 2 repeats, then LONGER block (2 or 3 line over 1-line), then more repeats
    best = max(
        candidates,
        key=lambda c: (c["count"] >= 2, c["block_len"], c["count"]),
    )
    block_lines = best["lines"]
    reason = (
        f"preferred repeated {best['block_len']}-line block (count={best['count']}) over shorter alternatives"
    )
    return [list(block_lines) for _ in range(best["count"])], candidates, reason


def _infer_chorus_sections(
    sections: List[tuple],
    all_lines_flat: Optional[List[str]] = None,
) -> Tuple[List[List[str]], List[dict], List[str], str]:
    """
    Return (chorus_sections, fallback_candidates, chosen_fallback_refrain, fallback_block_selection_reason).
    Use explicit Chorus/Hook labels as primary; fallback to repeated 1-4 line refrains when no labels.
    Preserve full multi-line block as chosen_fallback_refrain; prefer 2/3-line blocks over 1-line.
    """
    explicit = [lines for stype, lines in sections if stype in ("chorus", "hook") and lines]
    if explicit:
        return explicit, [], [], ""

    blocks = [lines for _, lines in sections if len(lines) >= 1]
    block_sigs = ["\n".join(_normalize_line(l) for l in b) for b in blocks]
    counts = Counter(block_sigs)
    best_sig = counts.most_common(1)[0][0] if counts else None
    if best_sig and counts[best_sig] >= 2:
        for b in blocks:
            if "\n".join(_normalize_line(l) for l in b) == best_sig:
                n = counts[best_sig]
                reason = f"section-level repeated {len(b)}-line block (count={n})"
                return [b] * n, [{"block": b, "count": n}], list(b), reason

    flat = all_lines_flat or [l for _, lines in sections for l in lines]
    chorus_sections, fallback_candidates, selection_reason = _find_repeated_refrains(flat, min_repeats=2)
    if chorus_sections:
        chosen = list(chorus_sections[0])  # full block, not collapsed
        return chorus_sections, fallback_candidates, chosen, selection_reason
    candidates = [b for b in blocks if 2 <= len(b) <= 8]
    if candidates:
        b = min(candidates, key=lambda x: (sum(len(l) for l in x), -len(x)))
        return [b], [], list(b), "last_resort_shortest_2_to_6_line_block"
    return [], fallback_candidates, [], selection_reason or "no_repeated_block_found"


def _hook_candidates(chorus_lines: List[str], all_lines: List[str]) -> List[str]:
    """
    Likely hook lines: short, repeated, or central to chorus.
    """
    if not chorus_lines:
        return []
    # Short lines in chorus (<= 8 words) are hook candidates
    short = [ln for ln in chorus_lines if len(ln.split()) <= 8 and len(ln.strip()) >= 3]
    # Repeated lines anywhere
    norm_all = [_normalize_line(l) for l in all_lines if l.strip()]
    counts = Counter(norm_all)
    repeated = [ln for ln in short if counts.get(_normalize_line(ln), 0) >= 2]
    if repeated:
        return list(dict.fromkeys(repeated[:5]))  # dedupe, cap 5
    return list(dict.fromkeys(short[:3]))


def _repeat_consistency_score(chorus_sections: List[List[str]]) -> float:
    """0–100: how consistently the chorus repeats."""
    if len(chorus_sections) < 2:
        return 50.0  # single occurrence
    sigs = ["\n".join(_normalize_line(l) for l in c) for c in chorus_sections]
    if len(set(sigs)) == 1:
        return 100.0
    # Partial match
    base_len = min(len(s) for s in sigs)
    matches = sum(1 for s in sigs if s[:base_len] == sigs[0][:base_len])
    return min(100.0, 50.0 + 50.0 * matches / len(sigs))


def _memorability_score(
    chorus_lines: List[str],
    hook_lines: List[str],
    generic_hook_flags: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    0–100: shortness/singability, distinctiveness, freshness; repetition helps but isn't enough.
    Returns (score, components_dict) for debug.
    """
    components: Dict[str, float] = {
        "base": 40.0,
        "repetition_bonus": 0.0,
        "short_line_bonus": 0.0,
        "distinctiveness_bonus": 0.0,
        "generic_phrase_penalty": 0.0,
    }
    if not chorus_lines:
        return 0.0, components
    score = 40.0
    avg_words = sum(len(l.split()) for l in chorus_lines) / len(chorus_lines)
    if avg_words <= 6:
        score += 15.0
        components["short_line_bonus"] = 15.0
    elif avg_words <= 9:
        score += 8.0
        components["short_line_bonus"] = 8.0
    if hook_lines:
        score += 12.0
        components["repetition_bonus"] = 12.0
    if any(len(l.split()) <= 5 for l in chorus_lines):
        score += 5.0
        components["short_line_bonus"] += 5.0
    # Distinctiveness: no obvious template words (you, me, love, heart, away, etc.)
    template_words = {"you", "me", "love", "heart", "away", "stay", "night", "day", "know", "feel"}
    chorus_words = set()
    for l in chorus_lines:
        chorus_words.update(re.findall(r"[a-z]+", _normalize_line(l)))
    overlap = len(chorus_words & template_words)
    if overlap <= 2 and len(chorus_words) >= 5:
        score += 8.0
        components["distinctiveness_bonus"] = 8.0
    # Generic/cliché penalty: material reduction when flags present
    if generic_hook_flags:
        penalty = min(35.0, 10.0 + len(generic_hook_flags) * 8.0)
        score -= penalty
        components["generic_phrase_penalty"] = -penalty
    components["total"] = score
    return max(0.0, min(100.0, score)), components


def _chorus_template_flags(chorus_lines: List[str]) -> List[str]:
    """
    Lightweight heuristics for familiar chorus templates (abstract emotion + city/night,
    without you / missing you / alone tonight style). Case-insensitive.
    """
    if not chorus_lines:
        return []
    text = " ".join(chorus_lines).lower()
    flags: List[str] = []
    # Abstract emotional + city/streetlight/night image
    if any(w in text for w in ("streetlight", "streetlights", "city at night", "city lights")):
        if any(w in text for w in ("undone", "missing you", "without you", "alone", "falling", "breaking")):
            flags.append("template: emotion + city/streetlight/night")
    # Without you / missing you / alone tonight style
    if ("without you" in text or "missing you" in text) and ("alone" in text or "tonight" in text):
        flags.append("template: without you / missing you / alone tonight")
    # Familiar breakup rhyme/emotion combos
    if "losing the fight" in text or "can't escape" in text or "coming apart" in text:
        flags.append("template: breakup/escape trope")
    return flags


def _prose_like(chorus_lines: List[str]) -> bool:
    """True if chorus reads like prose (long lines, sentence-like)."""
    if not chorus_lines:
        return True
    avg_len = sum(len(l.split()) for l in chorus_lines) / len(chorus_lines)
    long_ratio = sum(1 for l in chorus_lines if len(l.split()) > 12) / len(chorus_lines)
    return avg_len > 10 or long_ratio > 0.5


def analyze_chorus(
    lyrics: str,
    chorus_blacklist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze lyrics for chorus/hook presence and quality.
    chorus_blacklist: phrases that count as generic in chorus (stronger penalty).
    """
    sections, detected_section_headers = get_sections(lyrics)
    all_lines = [l for _, lines in sections for l in lines]
    chorus_sections, fallback_refrain_candidates, chosen_fallback_refrain, fallback_block_selection_reason = _infer_chorus_sections(
        sections, all_lines_flat=all_lines
    )
    has_explicit_chorus = any(
        stype in ("chorus", "hook") for stype, _ in sections
    )
    chorus_source = "explicit_labels" if has_explicit_chorus else "fallback_repeated_blocks"

    chorus_lines_flat = (chorus_sections[0] if chorus_sections else []) or list(chosen_fallback_refrain or [])
    hook_lines = _hook_candidates(chorus_lines_flat, all_lines) if chorus_lines_flat else []

    avg_chorus_line_length = (
        sum(len(l.split()) for l in chorus_lines_flat) / len(chorus_lines_flat)
        if chorus_lines_flat else 0.0
    )
    repeat_consistency = _repeat_consistency_score(chorus_sections)
    generic_hook_flags = []
    if chorus_blacklist and chorus_lines_flat:
        chorus_text = " ".join(chorus_lines_flat).lower()
        for phrase in chorus_blacklist:
            if phrase.lower() in chorus_text:
                generic_hook_flags.append(phrase)
    # Lightweight template heuristics: familiar breakup/chorus patterns
    template_flags = _chorus_template_flags(chorus_lines_flat)
    generic_hook_flags = list(dict.fromkeys(generic_hook_flags + template_flags))
    memorability, memorability_components = _memorability_score(
        chorus_lines_flat, hook_lines, generic_hook_flags
    )
    is_prose_like = _prose_like(chorus_lines_flat)

    # Debug: fallback candidates (simplified for JSON)
    fallback_candidates_debug = []
    for c in fallback_refrain_candidates:
        fallback_candidates_debug.append({
            "block_len": c.get("block_len"),
            "count": c.get("count"),
            "preview": (c.get("lines", [])[:2] or []) if isinstance(c.get("lines"), list) else [],
        })

    parsed_sections = [{"type": stype, "line_count": len(lines)} for stype, lines in sections]

    return {
        "has_chorus": len(chorus_sections) > 0,
        "chorus_sections_found": len(chorus_sections),
        "chorus_lines": chorus_lines_flat,
        "hook_lines": hook_lines,
        "avg_chorus_line_length": round(avg_chorus_line_length, 1),
        "repeat_consistency_score": round(repeat_consistency, 1),
        "memorability_score": round(memorability, 1),
        "generic_hook_flags": generic_hook_flags,
        "is_prose_like": is_prose_like,
        "parsed_sections": parsed_sections,
        "detected_section_headers": detected_section_headers,
        "chorus_source": chorus_source,
        "fallback_refrain_candidates": fallback_candidates_debug,
        "chosen_fallback_refrain": chosen_fallback_refrain if chorus_source == "fallback_repeated_blocks" else [],
        "fallback_block_selection_reason": fallback_block_selection_reason if chorus_source == "fallback_repeated_blocks" else "",
        "memorability_score_components": memorability_components,
    }
