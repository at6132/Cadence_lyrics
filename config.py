"""
Config for lyric model: local (RTX 5070 12GB) vs 4xA100 80GB server.
Set LYRIC_DEVICE=4xa100 for server mode.

4xA100 80GB default: Llama 3.3 70B — best open model for creative writing.
  QLoRA 4-bit uses ~41GB per GPU, fits 80GB with room for batch.
Local default: Qwen2.5-7B-Instruct — fits 12GB with QLoRA.
"""
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_LYRICS_DIR = DATA_DIR / "raw_lyrics"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = ROOT / "checkpoints"
ADAPTERS_DIR = ROOT / "adapters"

for d in (DATA_DIR, RAW_LYRICS_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, ADAPTERS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Device preset
# -----------------------------------------------------------------------------
LYRIC_DEVICE = os.getenv("LYRIC_DEVICE", "").strip().lower()
IS_4XA100 = LYRIC_DEVICE == "4xa100"

if IS_4XA100:
    # =========================================================================
    # 4x A100 80GB — Llama 3.3 70B with QLoRA
    # Best creative writing quality of any open model. ~41GB VRAM per GPU.
    # Data parallel across 4 GPUs; each GPU holds the full 4-bit model.
    # Training capped at 5 hours.
    # =========================================================================
    # Pre-quantized 4-bit = lower peak VRAM during load (~40GB vs ~80GB); same license as Meta Llama
    MODEL_ID = os.getenv("LYRIC_MODEL_ID", "unsloth/Llama-3.3-70B-Instruct-bnb-4bit")

    # 70B 4-bit ~41GB + batch overhead; batch 4 per GPU fits 80GB comfortably
    # Global batch = 4 GPUs × 4 batch × 4 accum = 64
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 1
    # 5-hour budget ($6/h): default steps stay under 5h at ~30 s/step (no flash-attn). With flash-attn set LYRIC_MAX_STEPS=7200.
    MAX_TRAIN_HOURS = 5.0
    MAX_STEPS = int(os.getenv("LYRIC_MAX_STEPS", "600"))   # 600 × 30s ≈ 5h; with flash-attn use LYRIC_MAX_STEPS=7200
    SAVE_STRATEGY = "steps"
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.03
    # Gradient checkpointing ON for 70B to fit batch + model in 80GB
    GRADIENT_CHECKPOINTING = True
    USE_FLASH_ATTN_2 = True
    # LoRA for 70B: rank 32 is sweet spot (Unsloth default); all linear layers
    LORA_R = 32
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
else:
    # =========================================================================
    # Local / single GPU (e.g. RTX 5070 12GB)
    # =========================================================================
    MODEL_ID = os.getenv("LYRIC_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 2
    MAX_TRAIN_HOURS = None
    MAX_STEPS = -1
    SAVE_STRATEGY = "steps"
    SAVE_STEPS = 100
    LOGGING_STEPS = 10
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.03
    GRADIENT_CHECKPOINTING = True
    USE_FLASH_ATTN_2 = False
    LORA_R = 64
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

# QLoRA (both presets)
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"

# LoRA target modules (same for Llama and Qwen)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Shared
OUTPUT_DIR = str(CHECKPOINTS_DIR / "lyric-lora")
MAX_SEQ_LENGTH = 1024
FP16 = False
BF16 = True
ADAPTER_NAME = "lyric-human"

# Early stopping: stop if eval loss doesn't improve for N evaluations (saves $ when loss plateaus)
EARLY_STOPPING_PATIENCE = 3
EVAL_STEPS = 500
EVAL_FRACTION = 0.02  # use 2% of data for eval

# Optimizer / gradients (shared)
MAX_GRAD_NORM = 1.0  # gradient clipping; prevents explosion
WEIGHT_DECAY = 0.01  # L2 regularization
LR_SCHEDULER_TYPE = "cosine"  # decay LR smoothly to 0
SEED = 42  # reproducibility
