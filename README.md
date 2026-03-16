# Lyric model: local LLM fine-tuned on human lyrics

Run the **largest model that fits your GPU** and fine-tune it on real human lyrics so it stops writing generic AI slop.

**Your setup:** RTX 5070 (12GB VRAM) → default is **Qwen2.5-7B-Instruct** with 4-bit QLoRA (~6–8GB). You can try a 14B model by setting `LYRIC_MODEL_ID` (see below) and using `batch_size=1`.

## Quick start

```bash
cd Lyric_model
pip install -r requirements.txt
```

### 1. Download the base model (one-time)

```bash
python download_model.py
```

This caches the model from Hugging Face (several GB). For gated models (e.g. Llama), set:

```bash
set HUGGING_FACE_HUB_TOKEN=your_token
```

### 2. Add human lyrics

Lyrics come from **theelderemo/genius-lyrics-cleaned** on Hugging Face: 3.17M human-cleaned songs with **15 genres** (rap, trap, pop, r&b, rock, country, metal, folk, jazz, indie, electronic, reggae, soul, blues, latin). The download step samples **at least 10,000 songs** stratified across all genres.

- **Option A – Download 10k+ songs (many genres)**  
  Default: 10,000 songs, balanced across genres.

```bash
python prepare_lyrics.py --download-hf
python prepare_lyrics.py
```

  More songs (e.g. 25k) or cap per genre:

```bash
python prepare_lyrics.py --download-hf --min-songs 25000
python prepare_lyrics.py --download-hf --min-songs 15000 --max-per-genre 1500
```

- **Use the whole dataset (3.17M songs)**  
  Stream everything into one `train.jsonl` (no per-song files; one big file, a few GB):

```bash
python prepare_lyrics.py --download-hf --full
```

  Then run `finetune.py` as usual. Training will take longer but see all genres and styles.

- **Option B – Your own lyrics**  
  Put `.txt` or `.jsonl` files in `data/raw_lyrics/`. They’re merged with whatever you downloaded.

- **Option C – Both**  
  Run `--download-hf` first, add your own files in `data/raw_lyrics/`, then run `prepare_lyrics.py` (no `--download-hf`) to build `data/processed/train.jsonl`.

### 3. Fine-tune (QLoRA)

```bash
python finetune.py
```

Training uses 4-bit QLoRA so it fits in ~8GB VRAM. Checkpoints go to `checkpoints/lyric-lora/`, and the final adapter is saved under `adapters/lyric-human/`.

### 4. Generate lyrics

```bash
python generate.py "Write a verse and chorus about rain"
python generate.py "Write a bridge in a sad, sparse style" --max-tokens 300
```

---

## Pushing 12GB: bigger models

In the project root `.env` (or in the shell), set:

```env
LYRIC_MODEL_ID=Qwen/Qwen2.5-14B-Instruct
```

Then in `config.py` set `PER_DEVICE_TRAIN_BATCH_SIZE = 1` and keep `GRADIENT_CHECKPOINTING = True`. Training will be slower but the model will be larger.

Other options:

- `meta-llama/Llama-3.2-3B-Instruct` – smaller, faster, less VRAM
- `mistralai/Mistral-7B-Instruct-v0.3` – 7B alternative to Qwen

---

## Layout

```
Lyric_model/
  config.py          # Model id, paths, LoRA & training settings
  download_model.py  # Download base model from HF
  prepare_lyrics.py  # raw_lyrics → train.jsonl (or --download-hf)
  finetune.py        # QLoRA fine-tune
  generate.py        # Inference with adapter
  data/
    raw_lyrics/     # Your .txt / .jsonl lyrics
    processed/      # train.jsonl
  checkpoints/      # Training checkpoints
  adapters/         # Final LoRA adapter (lyric-human/)
```

---

## Why this works

- **QLoRA** keeps the base model in 4-bit and only trains a small LoRA adapter, so a 7B model fits on 12GB and still learns your data.
- **Human lyrics** in the training set teach the model real phrasing, structure, and tone instead of generic “AI verse” patterns.
- **Instruct format** (system + user + assistant) lets you steer generation with prompts while the model stays in a “songwriter” style.
