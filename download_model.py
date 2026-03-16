"""
Download the base model from Hugging Face (cached for subsequent runs).
For Llama 3.3 70B: accept the license first at
  https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
Then set HUGGING_FACE_HUB_TOKEN=hf_xxxxx
"""
import os
from pathlib import Path

from config import MODEL_ID, ROOT

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")


def main():
    if "llama" in MODEL_ID.lower() and not HF_TOKEN:
        print("WARNING: Llama models are gated. Set HUGGING_FACE_HUB_TOKEN first.")
        print("  1. Go to https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct")
        print("  2. Accept the license")
        print("  3. Create a token at https://huggingface.co/settings/tokens")
        print("  4. export HUGGING_FACE_HUB_TOKEN=hf_xxxxx")

    print(f"Downloading: {MODEL_ID}")
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
    print("Tokenizer cached.")

    _ = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    print("Model weights cached. Ready to fine-tune.")


if __name__ == "__main__":
    main()
