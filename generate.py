"""
Generate lyrics with the fine-tuned model (base + LoRA adapter).
Works with both Llama 3.3 70B (server) and Qwen 7B (local).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_ID, ADAPTERS_DIR, ADAPTER_NAME

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

_model = None
_tokenizer = None


def load_model_and_tokenizer():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = ADAPTERS_DIR / ADAPTER_NAME
    if not adapter_path.exists():
        raise SystemExit(
            f"Adapter not found at {adapter_path}. Run finetune.py first."
        )

    _tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    _model = PeftModel.from_pretrained(_model, str(adapter_path))
    _model.eval()
    return _model, _tokenizer


def generate(
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.85,
    do_sample: bool = True,
    system_prompt: str | None = None,
) -> str:
    import torch

    system_prompt = system_prompt or (
        "You are a songwriter. Write authentic, human-sounding lyrics. "
        "No clichés or generic AI phrases."
    )
    model, tokenizer = load_model_and_tokenizer()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    reply = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return reply.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate lyrics with fine-tuned model")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Write a verse and chorus about heartbreak.",
        help="Lyric prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--no-sample", action="store_true", help="Greedy decode")
    args = parser.parse_args()

    out = generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=0.0 if args.no_sample else args.temperature,
        do_sample=not args.no_sample,
    )
    print(out)


if __name__ == "__main__":
    main()
