"""
QLoRA fine-tuning on human lyrics. Fits in ~8GB VRAM for 7B model.
Run: python download_model.py && python prepare_lyrics.py && python finetune.py
"""
import os
from pathlib import Path

from config import (
    MODEL_ID,
    USE_4BIT,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    NUM_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    WARMUP_RATIO,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    SAVE_STEPS,
    BF16,
    FP16,
    GRADIENT_CHECKPOINTING,
    PROCESSED_DIR,
    ADAPTER_NAME,
    ADAPTERS_DIR,
)

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Quick test: ~10 min local run to confirm pipeline before moving to 4xA100
QUICK_MAX_SAMPLES = 500
QUICK_MAX_STEPS = 50


def main(quick: bool = False):
    train_path = PROCESSED_DIR / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(
            "No train.jsonl. Run: python prepare_lyrics.py (and add lyrics to data/raw_lyrics/ or use --download-hf)"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    output_dir = OUTPUT_DIR
    adapter_name = ADAPTER_NAME
    if quick:
        output_dir = str(CHECKPOINTS_DIR / "lyric-lora-quick")
        adapter_name = "lyric-human-quick"
        print("Quick run: max %d samples, max %d steps (~10 min)" % (QUICK_MAX_SAMPLES, QUICK_MAX_STEPS))

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=True,
    ) if USE_4BIT else None

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=compute_dtype if USE_4BIT else torch.bfloat16,
        attn_implementation="sdpa",
    )
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(train_path), split="train")
    if quick:
        n = min(QUICK_MAX_SAMPLES, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(n))
        print("Using %d samples for quick run" % len(dataset))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if quick else NUM_EPOCHS,
        max_steps=QUICK_MAX_STEPS if quick else -1,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=min(5, LOGGING_STEPS) if quick else LOGGING_STEPS,
        save_strategy="steps" if quick else SAVE_STRATEGY,
        save_steps=25 if quick else SAVE_STEPS,
        bf16=BF16,
        fp16=FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="messages",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save adapter only for inference
    adapter_path = ADAPTERS_DIR / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("Adapter saved to", adapter_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Quick local test: 500 samples, 50 steps, ~10 min. Use before full run on 4xA100.")
    args = p.parse_args()
    main(quick=args.quick)
