"""
Extract and validate hard constraints from user prompts.
Used for inference-time enforcement: banned phrases/words, structure, section rules.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple

# Section header pattern (match Verse:, Chorus:, [Chorus], etc.)
SECTION_HEADER = re.compile(
    r"^\s*\[?\s*(verse\s*\d*|chorus|hook|bridge|pre\s*[- ]?\s*chorus|outro|intro)\s*\]?\s*:?\s*$",
    re.IGNORECASE,
)


def extract_constraints(user_prompt: str) -> Dict[str, Any]:
    """
    Parse user prompt for explicit constraints.
    Returns structured object: banned_phrases, banned_words, banned_pattern_families,
    required_structure, section_rules, line_limits, special_constraints.
    """
    text = (user_prompt or "").strip().lower()
    out: Dict[str, Any] = {
        "banned_phrases": [],
        "banned_words": [],
        "banned_pattern_families": [],
        "required_structure": {},
        "section_rules": {},
        "line_limits": {},
        "special_constraints": [],
        "chorus_only_prompt": False,
    }

    # --- Banned phrases: "do not use X", "no X", "avoid X", "without X"
    for pattern, group in [
        (r"do not use\s+([^.]+?)(?:\.|$)", 1),
        (r"don't use\s+([^.]+?)(?:\.|$)", 1),
        (r"\bno\s+([^.;]+?)(?:\.|;|$)", 1),
        (r"avoid\s+([^.]+?)(?:\.|$)", 1),
        (r"without\s+([^.]+?)(?:\.|$)", 1),
        (r"not use\s+([^.]+?)(?:\.|$)", 1),
    ]:
        for m in re.finditer(pattern, text, re.I | re.S):
            raw = m.group(group).strip()
            # Skip structural descriptions (refrain, repeat, section labels, etc.)
            if re.search(r"refrain|repeat\s+\d+|section\s+labels|lines?\s+max", raw, re.I):
                continue
            # Split on / or comma or " or "
            for part in re.split(r"\s*/\s*|,\s*|\s+or\s+", raw):
                part = part.strip(' "\'')
                if len(part) > 2 and len(part) < 80:
                    out["banned_phrases"].append(part)

    # "no emotional words like sad, hurt, miss, love, pain"
    m = re.search(r"no\s+emotional\s+words\s+(?:like|such as|:)\s*([^.]+)", text)
    if m:
        for w in re.findall(r"[\w']+", m.group(1)):
            if len(w) > 2:
                out["banned_words"].append(w)
    for word in ["sad", "hurt", "miss", "love", "pain", "broken", "lonely", "empty"]:
        if f"no emotional words" in text or "emotional words like" in text:
            if word in text and word not in out["banned_words"]:
                out["banned_words"].append(word)

    # --- Banned imagery / pattern families
    if re.search(r"no\s+city\s*/\s*streetlight\s*/\s*night\s+imagery", text) or "city/streetlight" in text:
        out["banned_pattern_families"].append("city_streetlight_night_imagery")
    if "no section labels" in text or "without section labels" in text or "unlabeled" in text and "refrain" in text:
        out["section_rules"]["no_section_labels"] = True

    # --- Chorus only / hook only
    if re.search(r"chorus\s+only|hook\s+only|only\s+a\s+chorus|only\s+a\s+hook", text):
        out["chorus_only_prompt"] = True
        out["section_rules"]["chorus_only"] = True
    if re.search(r"\d+\s*lines?\s*max\s*(?:for\s+)?(?:chorus|hook)", text) or re.search(r"(?:chorus|hook)\s*only[,.]?\s*(\d+)\s*lines?\s*max", text):
        out["chorus_only_prompt"] = True
        m = re.search(r"(\d+)\s*lines?\s*max", text)
        if m:
            out["line_limits"]["chorus_max_lines"] = int(m.group(1))
    if "6 lines max" in text and ("chorus" in text or "hook" in text):
        out["line_limits"]["chorus_max_lines"] = 6
        out["chorus_only_prompt"] = True
    if "3 short lines" in text or "3-line chorus" in text or "chorus is only 3" in text:
        out["line_limits"]["chorus_max_lines"] = 3
    if "repeat one short line three times" in text or "one short line three times" in text:
        out["required_structure"]["repeated_one_line_count"] = 3

    # --- Refrain / structure
    m = re.search(r"(\d+)[- ]?line\s+(?:repeated\s+)?refrain\s+(?:repeated\s+)?(\d+)\s+times", text)
    if m:
        out["required_structure"]["refrain_lines"] = int(m.group(1))
        out["required_structure"]["repeat_count"] = int(m.group(2))
    m = re.search(r"refrain\s+(\d+)\s+lines?\s+long\s+and\s+(?:naturally\s+)?recur\s+(\d+)\s+times", text)
    if m:
        out["required_structure"]["refrain_lines"] = int(m.group(1))
        out["required_structure"]["repeat_count"] = int(m.group(2))
    if "no section labels" in text and "verses and a chorus" in text:
        out["section_rules"]["no_section_labels"] = True
        out["special_constraints"].append("chorus_identifiable_by_shape_and_repetition")

    # --- Verses longer than chorus / chorus simpler
    if "verses are longer" in text or "verses longer" in text:
        out["required_structure"]["verses_longer_than_chorus"] = True
    if "chorus simpler" in text or "chorus is simpler" in text:
        out["required_structure"]["chorus_simpler_than_verses"] = True

    # Dedupe
    out["banned_phrases"] = list(dict.fromkeys(p.strip() for p in out["banned_phrases"] if p.strip()))
    out["banned_words"] = list(dict.fromkeys(w.strip() for w in out["banned_words"] if w.strip()))
    return out


def _normalize_for_match(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").strip().lower())


def validate_lyrics_against_constraints(
    lyrics: str,
    constraints: Dict[str, Any],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate lyrics against extracted constraints.
    Returns (constraint_passed, constraint_violations, structure_validation_result).
    """
    violations: List[str] = []
    structure_result: Dict[str, Any] = {"passed": True, "details": {}}
    lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]

    # --- Banned phrases (and close variants: phrase contained in text)
    text_lower = lyrics.lower()
    text_norm = _normalize_for_match(lyrics)
    for phrase in constraints.get("banned_phrases", []):
        if not phrase:
            continue
        p_lower = phrase.lower()
        p_norm = _normalize_for_match(phrase)
        if p_lower in text_lower or p_norm in text_norm:
            violations.append(f"banned_phrase: {phrase}")

    # --- Banned words (whole-word or as standalone token)
    banned_words = constraints.get("banned_words", [])
    for w in banned_words:
        if not w:
            continue
        # Word boundary match
        if re.search(r"\b" + re.escape(w.lower()) + r"\b", text_lower):
            violations.append(f"banned_word: {w}")

    # --- Banned pattern families
    if "city_streetlight_night_imagery" in constraints.get("banned_pattern_families", []):
        if any(k in text_lower for k in ("city at night", "streetlight", "streetlights", "city lights", "city at night")):
            violations.append("banned_pattern_family: city/streetlight/night imagery")

    # --- No section labels
    if constraints.get("section_rules", {}).get("no_section_labels"):
        for line in lines:
            if SECTION_HEADER.match(line):
                violations.append("section_rule: section labels forbidden but found (e.g. Verse:/Chorus:)")
                break

    # --- Chorus only + line limit
    chorus_max = constraints.get("line_limits", {}).get("chorus_max_lines")
    if chorus_max is not None and constraints.get("section_rules", {}).get("chorus_only"):
        # Count non-empty lines; if it's chorus-only, total lines should be <= chorus_max
        if len(lines) > chorus_max:
            violations.append(f"line_limits: chorus only {chorus_max} lines max, found {len(lines)} lines")
            structure_result["details"]["chorus_max_lines"] = {"expected": chorus_max, "found": len(lines)}
            structure_result["passed"] = False

    # --- Required structure: 2-line refrain repeated 3 times
    req = constraints.get("required_structure", {})
    refrain_lines = req.get("refrain_lines")
    repeat_count = req.get("repeat_count")
    if refrain_lines is not None and repeat_count is not None:
        from .chorus_analysis import _find_repeated_refrains
        sections, _, _ = _find_repeated_refrains(lines, min_repeats=2)
        if not sections:
            violations.append(f"structure: expected {refrain_lines}-line repeated refrain (×{repeat_count}), no repeated block found")
            structure_result["passed"] = False
            structure_result["details"]["refrain"] = {"expected_lines": refrain_lines, "expected_repeats": repeat_count, "found": "no repeated block"}
        else:
            block = sections[0]
            if len(block) != refrain_lines:
                violations.append(f"structure: expected {refrain_lines}-line repeated refrain, found {len(block)}-line block")
                structure_result["passed"] = False
                structure_result["details"]["refrain"] = {"expected_lines": refrain_lines, "found_lines": len(block)}

    violations = list(dict.fromkeys(violations))
    constraint_passed = len(violations) == 0
    return constraint_passed, violations, structure_result


def is_chorus_only_prompt(constraints: Dict[str, Any]) -> bool:
    """True if the prompt asks for chorus/hook only."""
    return bool(constraints.get("chorus_only_prompt"))
