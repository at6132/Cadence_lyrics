"""
Unit tests for chorus/section parsing in utils.chorus_analysis.
"""
import sys
from pathlib import Path

# Add Lyric_model to path when running tests from project root or Lyric_model
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
from utils.chorus_analysis import (
    get_sections,
    analyze_chorus,
    _get_sections,
    _infer_chorus_sections,
)


# --- Example A: Verse 1: / Chorus: / Verse 2: / Chorus: ---
EXAMPLE_A = """Verse 1:
line one
line two
Chorus:
hook line
another hook
Verse 2:
line three
line four
Chorus:
hook line
another hook
"""


def test_example_a_sections():
    sections, headers = get_sections(EXAMPLE_A)
    assert len(sections) == 4
    assert headers == ["Verse 1", "Chorus", "Verse 2", "Chorus"]
    assert sections[0][0] == "verse"
    assert sections[0][1] == ["line one", "line two"]
    assert sections[1][0] == "chorus"
    assert sections[1][1] == ["hook line", "another hook"]
    assert sections[2][0] == "verse"
    assert sections[2][1] == ["line three", "line four"]
    assert sections[3][0] == "chorus"
    assert sections[3][1] == ["hook line", "another hook"]


def test_example_a_analyze_chorus():
    out = analyze_chorus(EXAMPLE_A)
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] == 2
    assert out["chorus_source"] == "explicit_labels"
    assert "hook line" in out["chorus_lines"] or "another hook" in out["chorus_lines"]
    assert out["memorability_score"] > 0
    assert "Verse 1" in out["detected_section_headers"]
    assert "Chorus" in out["detected_section_headers"]
    assert len(out["parsed_sections"]) == 4


# --- Example B: [Verse] / [Chorus] ---
EXAMPLE_B = """[Verse]
line a
line b
[Chorus]
line c
line d
"""


def test_example_b_sections():
    sections, headers = get_sections(EXAMPLE_B)
    assert len(sections) == 2
    assert sections[0][0] == "verse"
    assert sections[0][1] == ["line a", "line b"]
    assert sections[1][0] == "chorus"
    assert sections[1][1] == ["line c", "line d"]


def test_example_b_analyze_chorus():
    out = analyze_chorus(EXAMPLE_B)
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] == 1
    assert out["chorus_source"] == "explicit_labels"
    assert out["chorus_lines"] == ["line c", "line d"]


# --- Example C: VERSE 1 / CHORUS (no colon, uppercase) ---
EXAMPLE_C = """VERSE 1
line one
CHORUS
line two
"""


def test_example_c_sections():
    sections, headers = get_sections(EXAMPLE_C)
    assert len(sections) == 2
    assert sections[0][0] == "verse"
    assert sections[0][1] == ["line one"]
    assert sections[1][0] == "chorus"
    assert sections[1][1] == ["line two"]


def test_example_c_analyze_chorus():
    out = analyze_chorus(EXAMPLE_C)
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] == 1
    assert out["chorus_source"] == "explicit_labels"


# --- Example D: Verse 1: / Chorus: / Bridge: ---
EXAMPLE_D = """Verse 1:
line

Chorus:
line
line

Bridge:
line
"""


def test_example_d_sections():
    sections, headers = get_sections(EXAMPLE_D)
    assert len(sections) == 3
    assert sections[0][0] == "verse"
    assert sections[1][0] == "chorus"
    assert sections[2][0] == "bridge"


def test_example_d_analyze_chorus():
    out = analyze_chorus(EXAMPLE_D)
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] == 1
    assert out["chorus_source"] == "explicit_labels"
    assert len(out["chorus_lines"]) == 2


# --- Pre-Chorus and Hook ---
EXAMPLE_PRE_CHORUS = """Verse 1:
a
b
Pre-Chorus:
build up
Chorus:
payoff
"""


def test_pre_chorus_and_chorus():
    sections, headers = get_sections(EXAMPLE_PRE_CHORUS)
    types = [s[0] for s in sections]
    assert "verse" in types
    assert "prechorus" in types
    assert "chorus" in types


EXAMPLE_HOOK = """[Verse]
x
[Hook]
catchy
line
"""


def test_hook_treated_as_chorus():
    out = analyze_chorus(EXAMPLE_HOOK)
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] == 1
    assert out["chorus_source"] == "explicit_labels"
    assert "catchy" in out["chorus_lines"]


# --- No labels: fallback (two identical blocks separated by blank line) ---
EXAMPLE_NO_LABELS = """line a
line b

line a
line b
"""


def test_fallback_when_no_labels():
    sections, headers = get_sections(EXAMPLE_NO_LABELS)
    assert headers == []
    assert len(sections) == 2  # two blocks
    chorus_sections = _infer_chorus_sections(sections)
    # Repeated block should be inferred as chorus
    assert len(chorus_sections) >= 1
    out = analyze_chorus(EXAMPLE_NO_LABELS)
    assert out["chorus_source"] == "fallback_repeated_blocks"


# --- Back-compat _get_sections ---
def test_get_sections_back_compat():
    sections = _get_sections(EXAMPLE_A)
    assert len(sections) == 4
    assert sections[0][0] == "verse"
    assert sections[1][0] == "chorus"
