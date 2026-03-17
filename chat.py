"""
CLI chatbot for the lyric model.
Uses local adapter if adapters/lyric-human/ exists; otherwise loads from Hugging Face.
Use --hub to force loading from HF even when local adapter is present.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ADAPTERS_DIR, ADAPTER_NAME, MODEL_ID

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
ADAPTER_HF_ID = "at6132/lyric-human-70b-lora"

SYSTEM_PROMPT = (
    "You are a songwriter. Write authentic, human-sounding lyrics. "
    "No clichés or generic AI phrases."
)

_model = None
_tokenizer = None


def _get_base_model_id(adapter_path: Path) -> str:
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path")
            if base:
                return base
        except (json.JSONDecodeError, KeyError):
            pass
    return MODEL_ID


def load_model_and_tokenizer(use_hub: bool = False):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = ADAPTERS_DIR / ADAPTER_NAME
    use_local = adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists()
    if use_hub or not use_local:
        if use_local and use_hub:
            print("Loading adapter from Hugging Face (--hub): %s" % ADAPTER_HF_ID)
        else:
            print("Local adapter not found; loading from Hugging Face: %s" % ADAPTER_HF_ID)
        if not HF_TOKEN:
            raise SystemExit("Set HUGGING_FACE_HUB_TOKEN to load adapter from Hub.")
        _tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_HF_ID,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        base_model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        _model = PeftModel.from_pretrained(_model, ADAPTER_HF_ID, token=HF_TOKEN)
    else:
        print("Using local adapter: %s" % adapter_path.resolve())
        base_model_id = _get_base_model_id(adapter_path)
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
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        _model = PeftModel.from_pretrained(_model, str(adapter_path))

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model.eval()
    return _model, _tokenizer


def generate_reply(model, tokenizer, messages: list, max_new_tokens: int = 400, temperature: float = 0.85):
    import torch
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
            do_sample=True,
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
    p = argparse.ArgumentParser(description="Chat with the lyric model (CLI)")
    p.add_argument("--hub", action="store_true", help="Force load adapter from Hugging Face (ignore local adapters/)")
    p.add_argument("--max-tokens", type=int, default=400, help="Max new tokens per reply")
    p.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    args = p.parse_args()

    print("Loading model and adapter...")
    model, tokenizer = load_model_and_tokenizer(use_hub=args.hub)
    print("Ready. Type your lyric prompts (e.g. 'Write a verse about rain'). /quit to exit, /clear to clear history.\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    max_tokens = args.max_tokens
    temperature = args.temperature

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "/q"):
            print("Bye.")
            break
        if line.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[History cleared.]\n")
            continue
        if line.lower().startswith("/max-tokens "):
            try:
                max_tokens = int(line.split(maxsplit=1)[1])
                print("[max_new_tokens = %d]\n" % max_tokens)
            except (IndexError, ValueError):
                print("Usage: /max-tokens 300\n")
            continue
        if line.lower().startswith("/temp "):
            try:
                temperature = float(line.split(maxsplit=1)[1])
                print("[temperature = %s]\n" % temperature)
            except (IndexError, ValueError):
                print("Usage: /temp 0.8\n")
            continue
        if line.lower() == "/help":
            print("Commands: /quit, /clear, /max-tokens N, /temp F, /help\n")
            continue

        messages.append({"role": "user", "content": line})
        print("Lyric: ", end="", flush=True)
        reply = generate_reply(model, tokenizer, messages, max_new_tokens=max_tokens, temperature=temperature)
        messages.append({"role": "assistant", "content": reply})
        print(reply)
        print()


if __name__ == "__main__":
    main()
