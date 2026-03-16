"""
Config for lyric model: pick the biggest model that fits your GPU.
RTX 5070 12GB: 7B–8B at 4-bit QLoRA is safe; 14B possible with batch_size=1.
"""
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_LYRICS_DIR = DATA_DIR / "raw_lyrics"   # drop .txt/.jsonl human lyrics here
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = ROOT / "checkpoints"
ADAPTERS_DIR = ROOT / "adapters"

for d in (DATA_DIR, RAW_LYRICS_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, ADAPTERS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Model: biggest you can run on 12GB VRAM with QLoRA
# Options (set LYRIC_MODEL_ID in .env to override):
#   - Qwen2.5-7B-Instruct     ~6–8GB VRAM, best quality/size for 12GB
#   - mistralai/Mistral-7B-Instruct-v0.3
#   - meta-llama/Llama-3.2-3B-Instruct  (smaller, faster)
#   - Qwen2.5-14B (push 12GB: batch_size=1, gradient_checkpointing=True)
# -----------------------------------------------------------------------------
MODEL_ID = os.getenv("LYRIC_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# QLoRA
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"

# LoRA
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training (tuned for 12GB; increase batch_size if you have more VRAM)
OUTPUT_DIR = str(CHECKPOINTS_DIR / "lyric-lora")
NUM_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STRATEGY = "steps"
SAVE_STEPS = 100
FP16 = False
BF16 = True
GRADIENT_CHECKPOINTING = True

# Adapter save name (after training)
ADAPTER_NAME = "lyric-human"
