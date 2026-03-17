"""
Configurable phrase blacklist for anti-AI lyric pipeline.
Case-insensitive matching; partial phrase detection; returns all matches.
"""
from pathlib import Path
import re
from typing import Optional, List, Dict, Union

# Default phrases if no file is found (subset; full list in data/phrase_blacklist.txt)
DEFAULT_PHRASES = [
    "broken heart", "shattered heart", "lost in the dark", "fading away", "fade away",
    "echo of your name", "ghost of you", "empty space", "hollow place", "slipping away",
    "holding on", "letting go", "drowning in my thoughts", "haunting me", "piece of me",
    "without you by my side", "ashes of", "light in the dark", "scars remain", "torn apart",
    "grasping at air", "fading like a photograph", "shadow of your smile",
]

# Rhyme words that often signal generic pop (too many in one song = suspicious)
GENERIC_RHYME_WORDS = {"name", "pain", "rain", "remain", "away", "day", "way", "stay", "night", "light", "fight", "right"}

# Abstract emotion nouns (too many in one line = suspicious)
ABSTRACT_EMOTION = {"love", "pain", "heart", "soul", "dream", "hope", "fear", "tears", "joy", "sadness", "loneliness", "darkness", "light", "forever", "never"}


def load_blacklist(path: Optional[Path] = None) -> List[str]:
    """Load blacklist from file (one phrase per line, # = comment). Fallback to DEFAULT_PHRASES."""
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "phrase_blacklist.txt"
    if not path.exists():
        return list(DEFAULT_PHRASES)
    phrases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                phrases.append(line.strip().lower())
    return phrases if phrases else list(DEFAULT_PHRASES)


# Chorus-specific blacklist (stronger penalty when phrase appears in chorus)
DEFAULT_CHORUS_PHRASES = [
    "turn back time", "just strangers now", "crowded room", "feels the same", "don't feel the same",
    "falling apart", "coming undone", "unraveling", "frayed and worn", "without you here",
    "every little thing reminds me", "everyone's a stranger", "city's still awake",
    "streetlights don't shine as bright", "memory of your touch",
]


def load_chorus_blacklist(path: Optional[Path] = None) -> List[str]:
    """Load chorus-specific blacklist (one phrase per line, # = comment)."""
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "chorus_blacklist.txt"
    if not path.exists():
        return list(DEFAULT_CHORUS_PHRASES)
    phrases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                phrases.append(line.strip().lower())
    return phrases if phrases else list(DEFAULT_CHORUS_PHRASES)


def match_phrases(text: str, blacklist: Optional[List[str]] = None) -> List[str]:
    """
    Return all blacklist phrases that appear in text (case-insensitive, partial match).
    """
    if blacklist is None:
        blacklist = load_blacklist()
    text_lower = text.lower()
    found = []
    for phrase in blacklist:
        if phrase in text_lower:
            found.append(phrase)
    return found


def heuristic_flags(text: str) -> Dict[str, Union[bool, list]]:
    """
    Softer heuristic flags for suspicious wording.
    Returns dict with keys: abstract_heavy_line, obvious_rhyme_ending, repeated_structure.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    flags = {
        "abstract_heavy_line": False,
        "obvious_rhyme_ending": False,
        "repeated_structure": False,
        "abstract_heavy_lines": [],
        "rhyme_ending_lines": [],
    }
    if not lines:
        return flags

    # Abstract-heavy: line with 3+ abstract emotion words
    for i, line in enumerate(lines):
        words = set(re.findall(r"[a-z]+", line.lower()))
        count = len(words & ABSTRACT_EMOTION)
        if count >= 3:
            flags["abstract_heavy_line"] = True
            flags["abstract_heavy_lines"].append((i + 1, line[:50]))

    # Obvious rhyme: many lines ending in generic rhyme words
    endings = []
    for line in lines:
        last_word = re.findall(r"[a-z]+", line.lower())
        if last_word and last_word[-1] in GENERIC_RHYME_WORDS:
            endings.append(last_word[-1])
    if len(endings) >= 3 and len(set(endings)) <= 2:
        flags["obvious_rhyme_ending"] = True
        flags["rhyme_ending_lines"] = endings

    # Repeated structure: many lines with same pattern (e.g. "I [verb] [noun]")
    if len(lines) >= 4:
        # Simple check: lines starting with same word (e.g. "I" or "You")
        starts = [re.match(r"^(\w+)", ln) for ln in lines]
        starts = [m.group(1).lower() for m in starts if m]
        if starts:
            from collections import Counter
            most_common = Counter(starts).most_common(1)[0]
            if most_common[1] >= len(lines) * 0.6:
                flags["repeated_structure"] = True

    return flags
