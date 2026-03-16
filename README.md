# Cadence Lyrics – Lyric model for Cadence AI

Fine-tuned LLM for authentic, human-sounding song lyrics. Trained on 3.17M real songs from Genius across 15 genres.

**Two modes:**
- **Local** (RTX 5070 12GB): Qwen2.5-7B-Instruct with QLoRA
- **Server** (4× A100 80GB): **Llama 3.3 70B** with QLoRA — best open model for creative writing

## Quick start (local)

```bash
cd Lyric_model
pip install -r requirements.txt
python finetune.py --quick          # sanity check (~10 min)
python generate.py "Write a verse about rain"
```

## Full pipeline

### 1. Download the base model

```bash
python download_model.py
```

For **Llama 3.3** (gated model — server mode):
1. Accept the license at https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
2. Set your token: `export HUGGING_FACE_HUB_TOKEN=hf_xxxxx`

### 2. Prepare lyrics data

**Full dataset (3.17M songs, recommended for server):**
```bash
python prepare_lyrics.py --download-hf --full
```

**Smaller sample (10k+ songs, good for local):**
```bash
python prepare_lyrics.py --download-hf
python prepare_lyrics.py
```

**Your own lyrics:** Drop `.txt` or `.jsonl` files in `data/raw_lyrics/`, then run `python prepare_lyrics.py`.

### 3. Fine-tune

**Local (single GPU):**
```bash
python finetune.py
```

**4× A100 80GB server (Llama 3.3 70B, ≤5h):**
```bash
export LYRIC_DEVICE=4xa100
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
./run_4xa100.sh
```

### 4. Generate lyrics

```bash
python generate.py "Write a verse and chorus about heartbreak"
python generate.py "Write a bridge in a sad, sparse style" --max-tokens 300
```

---

## 4× A100 80GB server setup

### Why Llama 3.3 70B?

After researching all major open models for creative writing:

| Model | Creative quality | Issue |
|-------|-----------------|-------|
| Qwen2.5 7B–14B | Decent | Sounds formal, fights creative tone |
| Mistral Nemo 12B | Good | Smaller, less nuance than 70B |
| **Llama 3.3 70B** | **Best** | Best instruction-following, proven lyric/poetry fine-tunes, massive creative capacity |

Llama 3.3 70B matches Llama 3.1 405B on many tasks. With QLoRA 4-bit it uses ~41GB per GPU — fits on A100 80GB with room for batch.

### Server configuration

`LYRIC_DEVICE=4xa100` activates:
- **Model:** `meta-llama/Llama-3.3-70B-Instruct`
- **QLoRA 4-bit** on all linear layers, rank 32
- **Batch:** 4 per GPU × 4 GPUs × 4 grad accum = **64 global batch**
- **Gradient checkpointing** ON (needed for 70B + batch to fit 80GB)
- **Flash Attention 2** when available
- **max_steps = 7200** (~5 hour cap at ~2.5 sec/step)
- **1 epoch**, bf16, AdamW

### Running on the server

```bash
# 1. Install (PyTorch 2.5+ required for 4-bit load)
pip install -r requirements.txt
# If container has old PyTorch: upgrade torch + torchvision + torchaudio together (else: operator torchvision::nms does not exist)
pip install -U torch torchvision torchaudio transformers bitsandbytes accelerate
pip install flash-attn --no-build-isolation   # strongly recommended: ~2–3x faster; without it 7200 steps can take 50+ hours

# 2. Set env
export LYRIC_DEVICE=4xa100
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx

# 3. Copy train.jsonl or build it on the server
python prepare_lyrics.py --download-hf --full

# 4. Launch (uses accelerate for 4-GPU data parallel)
./run_4xa100.sh
```

Adapter is saved to `adapters/lyric-human/` (rank 0 only).

---

## Layout

```
Lyric_model/
  config.py              # Model, LoRA, training settings (local vs 4xA100)
  configs/
    accelerate_4xa100.yaml  # Accelerate config for 4 GPUs
  run_4xa100.sh          # Launch script for server
  download_model.py      # Download base model from HF
  prepare_lyrics.py      # Build train.jsonl from HF dataset or local files
  finetune.py            # QLoRA fine-tune (single or multi-GPU)
  generate.py            # Inference with adapter
  data/
    raw_lyrics/          # Your .txt / .jsonl lyrics
    processed/           # train.jsonl (4M+ examples from 3.17M songs)
  checkpoints/           # Training checkpoints
  adapters/              # Final LoRA adapter (lyric-human/)
```
