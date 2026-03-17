"""
QLoRA fine-tuning on human lyrics.
Local: 7B model on 12GB GPU.
Server: Llama 3.3 70B on 4× A100 80GB (data parallel, ~41GB/GPU).
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
    IS_4XA100,
    MAX_STEPS,
    USE_FLASH_ATTN_2,
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    EVAL_FRACTION,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    LR_SCHEDULER_TYPE,
    SEED,
)

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

QUICK_MAX_SAMPLES = 500
QUICK_MAX_STEPS = 50
# Use 3B for quick test so it fits on 12GB without OOM during load; full run uses config MODEL_ID
QUICK_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main(quick: bool = False):
    train_path = PROCESSED_DIR / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(
            "No train.jsonl found. Run prepare_lyrics.py first."
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
        print("Quick test: %d samples, %d steps" % (QUICK_MAX_SAMPLES, QUICK_MAX_STEPS))

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=True,
    ) if USE_4BIT else None

    # Device map: per-rank GPU in multi-GPU, auto otherwise
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Attention: try FA3 (prebuilt wheel) then FA2 then SDPA
    if IS_4XA100:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    print("Loading %s ..." % MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = dict(
        quantization_config=bnb_config,
        device_map=device_map,
        token=HF_TOKEN,
        trust_remote_code=True,
        dtype=compute_dtype if USE_4BIT else torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    attn_impl = None
    if USE_FLASH_ATTN_2:
        for candidate in ("flash_attention_3", "flash_attention_2"):
            load_kw["attn_implementation"] = candidate
            try:
                model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kw)
                attn_impl = candidate
                if local_rank == 0:
                    print("Using %s" % candidate.replace("_", " ").title())
                break
            except Exception as e1:
                if local_rank == 0 and "flash" in str(e1).lower():
                    print("%s not available, trying next..." % candidate)
                continue
    if attn_impl is None:
        load_kw["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kw)
        if local_rank == 0 and USE_FLASH_ATTN_2:
            print("flash-attn not available, falling back to SDPA")
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
        train_dataset = dataset
        eval_dataset = None
        print("Using %d samples" % len(dataset))
    else:
        split = dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        if len(eval_dataset) < 10:
            eval_dataset = None  # too small to be useful
        if eval_dataset is not None:
            print("Train: %d, eval: %d (early stop if eval loss plateaus %d evals)" % (
                len(train_dataset), len(eval_dataset), EARLY_STOPPING_PATIENCE,
            ))
        else:
            print("Train: %d (no eval set; early stopping disabled)" % len(train_dataset))

    max_steps = QUICK_MAX_STEPS if quick else (MAX_STEPS if IS_4XA100 and MAX_STEPS > 0 else -1)
    if IS_4XA100 and not quick:
        print("4×A100 80GB mode: %s, max_steps=%d (≤5h cap)" % (MODEL_ID, max_steps))

    from transformers import EarlyStoppingCallback
    callbacks = []
    if eval_dataset is not None and EARLY_STOPPING_PATIENCE > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    training_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=1 if quick else NUM_EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=SEED,
        logging_steps=min(5, LOGGING_STEPS) if quick else LOGGING_STEPS,
        save_strategy="steps" if quick else SAVE_STRATEGY,
        save_steps=25 if quick else SAVE_STEPS,
        bf16=BF16,
        fp16=FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4 if IS_4XA100 else 0,
        dataloader_pin_memory=True,
    )
    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = EVAL_STEPS
        training_kwargs["load_best_model_at_end"] = False
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False

    # Build SFTConfig with only base training args (server TRL rejects SFT-specific params in __init__)
    training_args = SFTConfig(**training_kwargs)
    # Set SFT data params on config so trainer uses them when building data collator (server TRL reads from args)
    setattr(training_args, "max_length", MAX_SEQ_LENGTH)
    setattr(training_args, "max_seq_length", MAX_SEQ_LENGTH)
    setattr(training_args, "dataset_text_field", "messages")
    setattr(training_args, "dataset_num_proc", 4 if IS_4XA100 else 2)
    setattr(training_args, "packing", False)

    sft_trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    # TRL version compatibility: some versions want these on SFTTrainer, not config
    extra = {
        "dataset_text_field": "messages",
        "max_seq_length": MAX_SEQ_LENGTH,
        "dataset_num_proc": 4 if IS_4XA100 else 2,
        "packing": False,
    }
    try:
        trainer = SFTTrainer(**sft_trainer_kw, **extra)
    except TypeError:
        trainer = SFTTrainer(**sft_trainer_kw)

    def _save_final():
        if local_rank == 0:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            adapter_path = ADAPTERS_DIR / adapter_name
            adapter_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))
            print("Adapter saved to", adapter_path)
        if IS_4XA100 and torch.distributed.is_initialized():
            torch.distributed.barrier()

    print("Starting training... (Ctrl+C saves and exits)")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nCtrl+C — saving checkpoint and adapter then exiting...")
        _save_final()
        raise SystemExit(0)

    _save_final()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Quick local test: 500 samples, 50 steps")
    args = p.parse_args()
    main(quick=args.quick)
