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
    _find_repeated_refrains,
    _trim_chorus_boundary,
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
    chorus_sections, _, _chosen, _reason = _infer_chorus_sections(sections)
    # Repeated block should be inferred as chorus
    assert len(chorus_sections) >= 1
    out = analyze_chorus(EXAMPLE_NO_LABELS)
    assert out["chorus_source"] == "fallback_repeated_blocks"
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] >= 1
    assert len(out["chorus_lines"]) >= 2


# --- Unlabeled repeated refrain (non-adjacent) ---
EXAMPLE_REFRAIN_SEPARATED = """I went down to the corner store
You were not there anymore

We used to dance
We used to dance

I drove past your street at night
The lights were off and that's right

We used to dance
We used to dance
"""


def test_fallback_refrain_non_adjacent():
    out = analyze_chorus(EXAMPLE_REFRAIN_SEPARATED)
    assert out["has_chorus"] is True
    assert out["chorus_sections_found"] >= 2
    assert out["chorus_source"] == "fallback_repeated_blocks"
    assert "We used to dance" in out["chorus_lines"] or any("dance" in l for l in out["chorus_lines"])
    assert out.get("chosen_fallback_refrain") or out["chorus_lines"]


def test_fallback_refrain_punctuation_variant():
    # Slight punctuation/case difference should still match
    lyrics = """First verse line
Second verse line

Don't go away
dont go away

Third verse
Fourth verse

Don't go away
dont go away
"""
    out = analyze_chorus(lyrics)
    assert out["has_chorus"] is True
    assert out["chorus_source"] == "fallback_repeated_blocks"
    # Full 2-line block should be chosen, not collapsed to one line
    assert len(out.get("chosen_fallback_refrain", [])) == 2
    assert "fallback_block_selection_reason" in out


def test_fallback_prefer_2line_block_over_single_line():
    # Same line repeated 4 times vs 2-line block repeated 2 times: prefer 2-line block
    lines = [
        "Verse here",
        "We're undone",
        "Like a wire cut",
        "Verse there",
        "We're undone",
        "Like a wire cut",
        "Bridge",
        "We're undone",
        "Like a wire cut",
    ]
    sections, candidates, reason = _find_repeated_refrains(lines, min_repeats=2)
    assert sections
    # Best should be the 2-line block "We're undone" / "Like a wire cut" (repeats 3 times)
    assert len(sections[0]) == 2
    assert "We're undone" in sections[0][0] or "undone" in sections[0][0].lower()
    assert "wire" in sections[0][1].lower() or "wire" in sections[0][0].lower()
    assert "preferred repeated 2-line" in reason or "2-line" in reason


def test_fallback_3line_chorus_chosen():
    lyrics = """Intro line

Hook one
Hook two
Hook three

Verse
Hook one
Hook two
Hook three

Outro
Hook one
Hook two
Hook three
"""
    out = analyze_chorus(lyrics)
    assert out["has_chorus"] is True
    assert out["chorus_source"] == "fallback_repeated_blocks"
    assert len(out["chorus_lines"]) == 3
    assert out["chorus_lines"] == ["Hook one", "Hook two", "Hook three"]
    assert len(out.get("chosen_fallback_refrain", [])) == 3
    assert "3-line" in out.get("fallback_block_selection_reason", "") or "repeated" in out.get("fallback_block_selection_reason", "")


def test_fallback_full_block_punctuation_case():
    # Full 2-line block with punctuation/case variants still matches as one block
    lyrics = """A
B
Oh we're undone
Like a wire cut the signals gone

C
D
Oh, we're undone
Like a wire cut, the signal's gone
"""
    sections, _, reason = _find_repeated_refrains(lyrics.splitlines(), min_repeats=2)
    assert sections
    assert len(sections[0]) >= 2
    assert "undone" in sections[0][0].lower()
    assert "wire" in sections[0][1].lower() or "signal" in sections[0][1].lower()


# --- Unlabeled verse after chorus: trim so chorus does not swallow verse ---
# No blank line between refrain and verse so parser initially assigns all to chorus
EXAMPLE_UNLABELED_VERSE_AFTER_CHORUS = """First verse line here
Another verse line

Chorus:
Oh we're undone
Like a wire cut we're undone
I drove past your street last Tuesday night
The lights were off and we never said goodbye
She told me something that I can't forget

Chorus:
Oh we're undone
Like a wire cut we're undone
"""


def test_unlabeled_verse_after_chorus_not_in_chorus_lines():
    out = analyze_chorus(EXAMPLE_UNLABELED_VERSE_AFTER_CHORUS)
    assert out["has_chorus"] is True
    assert out["chorus_source"] == "explicit_labels"
    # Chorus should be only the 2-line refrain, not the 3 narrative lines
    assert len(out["chorus_lines"]) == 2
    assert "Oh we're undone" in out["chorus_lines"][0] or "undone" in out["chorus_lines"][0].lower()
    assert "wire cut" in " ".join(out["chorus_lines"]).lower()
    assert "I drove past" not in out["chorus_lines"]
    assert "She told me" not in out["chorus_lines"]
    assert out.get("unlabeled_verse_after_chorus_detected") is True
    assert out.get("chorus_boundary_reason", "")
    assert out.get("chorus_compact_block_chosen") == out["chorus_lines"]


def test_trim_chorus_boundary_splits_chorus_then_verse():
    sections, _ = get_sections(EXAMPLE_UNLABELED_VERSE_AFTER_CHORUS)
    trimmed, debug = _trim_chorus_boundary(sections)
    assert debug["unlabeled_verse_after_chorus_detected"] is True
    assert len(debug["chorus_compact_block_chosen"]) == 2
    # First chorus section in trimmed should be 2 lines; then a block with the 3 verse lines
    chorus_sections = [lines for stype, lines in trimmed if stype == "chorus"]
    assert len(chorus_sections) == 2
    assert chorus_sections[0] == ["Oh we're undone", "Like a wire cut we're undone"]
    assert chorus_sections[1] == ["Oh we're undone", "Like a wire cut we're undone"]
    blocks = [lines for stype, lines in trimmed if stype == "block"]
    assert any("I drove past" in " ".join(b) for b in blocks)


# --- Back-compat _get_sections ---
def test_get_sections_back_compat():
    sections = _get_sections(EXAMPLE_A)
    assert len(sections) == 4
    assert sections[0][0] == "verse"
    assert sections[1][0] == "chorus"
