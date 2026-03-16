"""
Download the base model from Hugging Face (biggest that fits your VRAM).
Uses HF cache; run once before fine-tuning.
"""
import os
from pathlib import Path

from config import MODEL_ID, ROOT

# Optional: set HF token if using gated models (e.g. Llama)
# export HUGGING_FACE_HUB_TOKEN=your_token
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")


def main():
    print(f"Downloading base model: {MODEL_ID}")
    print("This may take a while (several GB). Model is cached in HF cache.")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("Install deps first: pip install -r requirements.txt")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(ROOT / "tokenizer_cache")
    print("Tokenizer saved to tokenizer_cache/")

    # Download full model to cache (used later by training script)
    _ = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    print("Model weights cached. You can run finetune.py next.")


if __name__ == "__main__":
    main()
