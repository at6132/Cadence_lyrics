#!/usr/bin/env bash
# Fine-tune Llama 3.3 70B on human lyrics — 4× A100 80GB
# Training capped at 5 hours.
#
# Prerequisites:
#   pip install -r requirements.txt
#   pip install flash-attn --no-build-isolation   # optional, 2x faster
#   data/processed/train.jsonl exists
#
# Llama 3.3 is gated — set your HF token:
#   export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
# And accept the license at https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

set -e
cd "$(dirname "$0")"
export LYRIC_DEVICE=4xa100

accelerate launch \
  --config_file configs/accelerate_4xa100.yaml \
  finetune.py
