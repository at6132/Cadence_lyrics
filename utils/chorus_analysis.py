"""
Chorus and hook awareness for the lyric pipeline.
Detects chorus sections, hook lines, repeat consistency, prose-like chorus, generic hooks.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Dict, Any

# Section labels (case-insensitive)
SECTION_PATTERN = re.compile(
    r"^\s*\[?\s*(verse|chorus|hook|bridge|pre.?chorus|outro|intro)\s*\]?\s*$",
    re.IGNORECASE,
)


def _normalize_line(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def _get_sections(lyrics: str) -> List[tuple]:
    """
    Split lyrics into (section_type, lines) where section_type is 'verse', 'chorus', etc. or 'block'.
    """
    lines = lyrics.splitlines()
    sections: List[tuple] = []
    current_type: Optional[str] = None
    current_lines: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            if current_lines:
                sections.append((current_type or "block", current_lines))
                current_lines = []
            current_type = None
            continue
        match = SECTION_PATTERN.match(line)
        if match:
            if current_lines:
                sections.append((current_type or "block", current_lines))
                current_lines = []
            current_type = match.group(1).lower().replace(" ", "").replace("-", "")
            if "prechorus" in current_type or "prechorus" in current_type:
                current_type = "prechorus"
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_type or "block", current_lines))
    return sections


def _infer_chorus_sections(sections: List[tuple]) -> List[List[str]]:
    """
    Return list of chorus line lists. Use explicit [Chorus] or infer by repetition.
    """
    explicit = [lines for stype, lines in sections if stype == "chorus" and lines]
    if explicit:
        return explicit
    # No labels: find the block that repeats most (or is repeated verbatim)
    blocks = [lines for _, lines in sections if len(lines) >= 2]
    if not blocks:
        return []
    # Normalized text of each block
    block_sigs = ["\n".join(_normalize_line(l) for l in b) for b in blocks]
    counts = Counter(block_sigs)
    best_sig = counts.most_common(1)[0][0] if counts else None
    if not best_sig or counts[best_sig] < 2:
        # No repetition: treat shortest block with 2–6 lines as possible chorus
        candidates = [b for b in blocks if 2 <= len(b) <= 8]
        if candidates:
            return [min(candidates, key=lambda x: (sum(len(l) for l in x), -len(x)))]
        return []
    # Return the block that repeats (first occurrence's lines)
    for b in blocks:
        if "\n".join(_normalize_line(l) for l in b) == best_sig:
            return [b]
    return []


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


def _memorability_score(chorus_lines: List[str], hook_lines: List[str]) -> float:
    """0–100: presence of short, repeatable hook and concise chorus."""
    if not chorus_lines:
        return 0.0
    score = 50.0
    avg_words = sum(len(l.split()) for l in chorus_lines) / len(chorus_lines)
    if avg_words <= 6:
        score += 20.0
    elif avg_words <= 9:
        score += 10.0
    if hook_lines:
        score += 25.0
    if any(len(l.split()) <= 5 for l in chorus_lines):
        score += 5.0
    return min(100.0, score)


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
    sections = _get_sections(lyrics)
    chorus_sections = _infer_chorus_sections(sections)
    all_lines = [l for _, lines in sections for l in lines]
    chorus_lines_flat = [l for c in chorus_sections for l in c] if chorus_sections else []
    hook_lines = _hook_candidates(chorus_lines_flat, all_lines) if chorus_lines_flat else []

    avg_chorus_line_length = (
        sum(len(l.split()) for l in chorus_lines_flat) / len(chorus_lines_flat)
        if chorus_lines_flat else 0.0
    )
    repeat_consistency = _repeat_consistency_score(chorus_sections)
    memorability = _memorability_score(chorus_lines_flat, hook_lines)
    is_prose_like = _prose_like(chorus_lines_flat)

    generic_hook_flags: List[str] = []
    if chorus_blacklist and chorus_lines_flat:
        chorus_text = " ".join(chorus_lines_flat).lower()
        for phrase in chorus_blacklist:
            if phrase.lower() in chorus_text:
                generic_hook_flags.append(phrase)

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
    }
