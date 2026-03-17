"""
Musicality / anti-prose analysis for the lyric pipeline.
Detects prose-like lines, storytelling overload, verse/chorus distinction, singability.
Uses shared section parsing from chorus_analysis for consistent Verse/Chorus detection.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any

from utils.chorus_analysis import get_sections

# Approximate "full sentence" = ends with . ! ? or has a comma and is long
SENTENCE_END = re.compile(r"[.!?]\s*$")
LONG_LINE_WORD_THRESHOLD = 14
PROSE_LIKE_WORD_THRESHOLD = 11


def _word_count(line: str) -> int:
    return len(line.split())


def _is_prose_like_line(line: str) -> bool:
    """Single line that reads like a sentence/paragraph rather than a lyric."""
    w = _word_count(line)
    if w >= LONG_LINE_WORD_THRESHOLD:
        return True
    if w >= PROSE_LIKE_WORD_THRESHOLD and SENTENCE_END.search(line):
        return True
    if w >= 10 and line.count(",") >= 2:
        return True
    return False


def _is_storytelling_line(line: str) -> bool:
    """Explanatory/narrative: starts with Then/So/When/After/Before/Because/And/But and is long."""
    lower = line.strip().lower()
    starters = ("then ", "so ", "when ", "after ", "before ", "because ", "and ", "but ")
    if not any(lower.startswith(s) for s in starters):
        return False
    return _word_count(line) >= 8


def analyze_musicality(lyrics: str) -> Dict[str, Any]:
    """
    Analyze lyrics for musicality vs prose: line length, prose ratio, verse/chorus distinction.
    """
    sections, _ = get_sections(lyrics)
    all_lines = [l for _, lines in sections for l in lines if l.strip()]
    if not all_lines:
        return {
            "avg_line_length": 0.0,
            "long_line_ratio": 0.0,
            "prose_like_ratio": 0.0,
            "verse_chorus_distinction_score": 0.0,
            "musicality_score": 0.0,
            "storytelling_overload": False,
        }

    word_counts = [_word_count(l) for l in all_lines]
    avg_line_length = sum(word_counts) / len(word_counts)
    long_line_ratio = sum(1 for w in word_counts if w > LONG_LINE_WORD_THRESHOLD) / len(word_counts)
    prose_like_count = sum(1 for l in all_lines if _is_prose_like_line(l))
    prose_like_ratio = prose_like_count / len(all_lines)
    storytelling_count = sum(1 for l in all_lines if _is_storytelling_line(l))
    storytelling_overload = storytelling_count >= max(2, len(all_lines) * 0.25)

    # Verse vs chorus distinction: compare avg line length of verse blocks vs chorus blocks
    verse_lines = [l for stype, lines in sections if stype in ("verse", "block") for l in lines]
    chorus_lines = [l for stype, lines in sections if stype in ("chorus", "hook") for l in lines]
    verse_chorus_distinction_score = 50.0
    if verse_lines and chorus_lines:
        v_avg = sum(_word_count(l) for l in verse_lines) / len(verse_lines)
        c_avg = sum(_word_count(l) for l in chorus_lines) / len(chorus_lines)
        diff = abs(v_avg - c_avg)
        if diff >= 3:
            verse_chorus_distinction_score = min(100.0, 50.0 + diff * 10)
        elif c_avg < v_avg:
            verse_chorus_distinction_score = 70.0  # chorus shorter = good
    elif chorus_lines:
        verse_chorus_distinction_score = 60.0

    # Musicality score: reward short lines, low prose ratio, distinction; penalize storytelling
    musicality_score = 70.0
    if avg_line_length <= 7:
        musicality_score += 15.0
    elif avg_line_length <= 10:
        musicality_score += 5.0
    musicality_score -= prose_like_ratio * 40.0
    musicality_score -= long_line_ratio * 20.0
    if storytelling_overload:
        musicality_score -= 15.0
    musicality_score += (verse_chorus_distinction_score - 50.0) * 0.3
    musicality_score = max(0.0, min(100.0, musicality_score))

    return {
        "avg_line_length": round(avg_line_length, 1),
        "long_line_ratio": round(long_line_ratio, 3),
        "prose_like_ratio": round(prose_like_ratio, 3),
        "verse_chorus_distinction_score": round(verse_chorus_distinction_score, 1),
        "musicality_score": round(musicality_score, 1),
        "storytelling_overload": storytelling_overload,
    }
