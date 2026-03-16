"""
Prepare human lyrics for fine-tuning.
- Load from data/raw_lyrics/*.txt or *.jsonl
- Or download from Hugging Face (e.g. genius-lyrics-cleaned)
- Output: data/processed/train.jsonl in chat/completion format for the model
"""
import json
import re
from pathlib import Path

from config import RAW_LYRICS_DIR, PROCESSED_DIR, DATA_DIR

# Max chars per example to avoid super long songs blowing VRAM
MAX_CHARS_PER_EXAMPLE = 2000
MIN_CHARS_PER_EXAMPLE = 80

# Prompt template: model learns to output human-like lyrics when asked
SYSTEM_PROMPT = "You are a songwriter. Write authentic, human-sounding lyrics. No clichés or generic AI phrases."
USER_PROMPT = "Write song lyrics."


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def chunk_lyrics(text: str, max_chars: int = MAX_CHARS_PER_EXAMPLE) -> list[str]:
    """Split long lyrics into chunks that fit in context."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if len(text) >= MIN_CHARS_PER_EXAMPLE else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # Break at paragraph or line
            break_at = text.rfind("\n\n", start, end + 1)
            if break_at == -1:
                break_at = text.rfind("\n", start, end + 1)
            if break_at != -1:
                end = break_at + 1
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHARS_PER_EXAMPLE:
            chunks.append(chunk)
        start = end
    return chunks


def load_raw_local() -> list[str]:
    out = []
    for path in sorted(RAW_LYRICS_DIR.glob("**/*.txt")):
        out.append(path.read_text(encoding="utf-8", errors="replace"))
    for path in sorted(RAW_LYRICS_DIR.glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # Support {"text": "..."} or {"lyrics": "..."} or {"content": "..."}
                text = obj.get("text") or obj.get("lyrics") or obj.get("content") or ""
                if isinstance(text, list):
                    text = "\n".join(str(t) for t in text)
                if text.strip():
                    out.append(text)
            except json.JSONDecodeError:
                pass
    return out


def build_examples(lyric_texts: list[str]) -> list[dict]:
    examples = []
    for text in lyric_texts:
        for chunk in chunk_lyrics(text):
            # Chat format for Qwen/Mistral/Llama instruct
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": chunk},
                ]
            })
    return examples


def save_train_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


def main():
    lyric_texts = load_raw_local()
    if not lyric_texts:
        print("No local lyrics found in", RAW_LYRICS_DIR)
        print("Add .txt or .jsonl files there, or run with --download-hf to use Genius dataset.")
        print("Example: echo 'Verse 1\\n...' > data/raw_lyrics/my_song.txt")
        return

    examples = build_examples(lyric_texts)
    if not examples:
        print("No examples after chunking (maybe all too short?). Increase MIN_CHARS or add longer lyrics.")
        return

    out_path = PROCESSED_DIR / "train.jsonl"
    save_train_jsonl(examples, out_path)


def download_genius_stratified(
    min_songs: int = 10_000,
    max_per_genre: int | None = None,
    lyric_col: str = "lyrics",
    genre_col: str = "tag",
) -> None:
    """
    Download Genius lyrics from HF, stratified across genres.
    Dataset: theelderemo/genius-lyrics-cleaned — 3.17M songs, 15 genres (rap, trap, pop, r&b, rock, country, metal, folk, jazz, indie, electronic, reggae, soul, blues, latin).
    """
    from datasets import load_dataset
    from collections import defaultdict

    print("Loading theelderemo/genius-lyrics-cleaned (3.17M songs, 15 genres)...")
    ds = load_dataset("theelderemo/genius-lyrics-cleaned", split="train")
    if lyric_col not in ds.column_names:
        lyric_col = "text" if "text" in ds.column_names else ds.column_names[0]
    if genre_col not in ds.column_names:
        genre_col = None

    # Collect by genre (or single "all" if no genre column)
    by_genre: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        text = (row.get(lyric_col) or "").strip()
        if len(text) < MIN_CHARS_PER_EXAMPLE:
            continue
        genre = (row.get(genre_col) or "unknown").strip().lower() if genre_col else "all"
        if not genre:
            genre = "unknown"
        by_genre[genre].append({"lyrics": text, "genre": genre})

    genres = sorted(by_genre.keys())
    print(f"Genres found: {genres} ({len(genres)} total)")

    # Stratified sample: at least min_songs total, balanced across genres
    import random
    rng = random.Random(42)
    n_genres = max(1, len(genres))
    per_genre_target = max(1, (min_songs + n_genres - 1) // n_genres)
    if max_per_genre is not None:
        per_genre_target = min(per_genre_target, max_per_genre)

    # First pass: take up to per_genre_target from each genre
    chosen: list[tuple[str, str]] = []
    taken_per_genre: dict[str, set[int]] = {g: set() for g in genres}
    for g in genres:
        pool = by_genre[g]
        n_take = min(per_genre_target, len(pool))
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        for i in indices[:n_take]:
            taken_per_genre[g].add(i)
            chosen.append((g, pool[i]["lyrics"]))
    # Top up to min_songs from genres that have more
    if len(chosen) < min_songs:
        shortfall = min_songs - len(chosen)
        for g in sorted(genres, key=lambda x: len(by_genre[x]), reverse=True):
            if shortfall <= 0:
                break
            pool = by_genre[g]
            remaining = [i for i in range(len(pool)) if i not in taken_per_genre[g]]
            rng.shuffle(remaining)
            for i in remaining[:shortfall]:
                taken_per_genre[g].add(i)
                chosen.append((g, pool[i]["lyrics"]))
                shortfall -= 1
    rng.shuffle(chosen)
    for idx, (g, text) in enumerate(chosen):
        (RAW_LYRICS_DIR / f"genius_{g}_{idx}.txt").write_text(text, encoding="utf-8")
    from collections import Counter
    counts = Counter(g for g, _ in chosen)
    for g in sorted(counts.keys()):
        print(f"  {g}: {counts[g]}")
    print(f"  Total: {len(chosen)} songs")
    print(f"Saved to {RAW_LYRICS_DIR}. Run without --download-hf to build train.jsonl.")


def build_full_dataset_to_jsonl(
    dataset_name: str = "theelderemo/genius-lyrics-cleaned",
    lyric_col: str = "lyrics",
    out_path: Path | None = None,
) -> None:
    """
    Use the whole HF dataset: stream every song into train.jsonl (no per-song files).
    One big file, all genres, ~3.17M examples. Disk: a few GB for the jsonl.
    """
    from datasets import load_dataset

    out_path = out_path or PROCESSED_DIR / "train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading full dataset {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    if lyric_col not in ds.column_names:
        lyric_col = "text" if "text" in ds.column_names else ds.column_names[0]

    total_examples = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i > 0 and i % 100_000 == 0:
                print(f"  ... {i} songs, {total_examples} examples so far")
            text = (row.get(lyric_col) or "").strip()
            for chunk in chunk_lyrics(text):
                ex = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        {"role": "assistant", "content": chunk},
                    ]
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                total_examples += 1
    print(f"Wrote {total_examples} examples to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-hf", action="store_true", help="Download Genius lyrics from HF (stratified by genre)")
    parser.add_argument("--full", action="store_true", help="Use whole dataset: stream all 3.17M songs into train.jsonl (no file per song)")
    parser.add_argument("--min-songs", type=int, default=10_000, help="Minimum songs when not --full (default 10000)")
    parser.add_argument("--max-per-genre", type=int, default=None, help="Cap per genre when not --full")
    args = parser.parse_args()

    if args.download_hf:
        try:
            if args.full:
                build_full_dataset_to_jsonl()
            else:
                download_genius_stratified(
                    min_songs=args.min_songs,
                    max_per_genre=args.max_per_genre,
                )
        except Exception as e:
            print("HF download failed:", e)
            raise
    else:
        main()
