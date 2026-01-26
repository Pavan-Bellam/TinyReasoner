"""
Supervised Fine-Tuning (SFT) script for TinyReasoner.

Loads pre-tokenized data from data/train and data/val, then trains.
Run process_data.py first to prepare the data.

Usage:
    python src/sft.py --config config.yml
    python src/sft.py --config config.yml --resume ./checkpoints/checkpoint-500
"""

import argparse
import os

import torch
import yaml
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, TrainerCallback
from trl import SFTConfig, SFTTrainer
import time
import torch.distributed as dist
import wandb

def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank == 0


def log_info(msg: str):
    if is_main_process():
        logger.info(msg)


def setup_quantization(config: dict) -> BitsAndBytesConfig | None:
    if not config.get("enabled", False):
        return None

    if config.get("load_in_4bit", False):
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        )

    if config.get("load_in_8bit", False):
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def get_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
        bias=config["bias"],
    )


def load_model(config: dict, adapter_path: str | None = None):
    model_path = config["model"]["path"]
    quant_config = setup_quantization(config["quant"])
    use_grad_ckpt = config["train"].get("use_gradient_checkpointing", False)

    log_info(f"Loading model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map=None,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )


    # Prepare for k-bit training if quantized
    if quant_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        )
    elif use_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load or create LoRA adapter
    if adapter_path:
        log_info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        if use_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        log_info("Creating new LoRA adapter")
        model = get_peft_model(model, get_lora_config(config["lora"]))

    if use_grad_ckpt:
        model.enable_input_require_grads()

    return model

def setup_wandb(config: dict, full_config: dict):
    if not config.get("enabled", False) or not is_main_process():
        return None  # Just return, don't disable

    run = wandb.init(
        project=config.get("project", "tinyreasoner"),
        name=config.get("run_name"),
        config=full_config,
        resume="allow",
    )
    log_info(f"Wandb: {run.url}")
    return run

def _write_eval_samples(
    model,
    tokenizer,
    eval_dataset,
    output_dir: str,
    epoch: float | None,
    global_step: int,
    num_examples: int = 2,
    max_new_tokens: int = 256,
):
    if not is_main_process():
        return
    if eval_dataset is None or len(eval_dataset) == 0:
        return

    num_examples = min(num_examples, len(eval_dataset))
    samples = [eval_dataset[i] for i in range(num_examples)]

    model.eval()
    device = next(model.parameters()).device

    lines = []
    lines.append("=" * 80)
    lines.append(f"Eval samples | epoch={epoch} | step={global_step}")
    lines.append("=" * 80)

    with torch.no_grad():
        for idx, sample in enumerate(samples, start=1):
            input_ids = torch.tensor(sample["input_ids"], device=device).unsqueeze(0)
            attention_mask = torch.tensor(sample["attention_mask"], device=device).unsqueeze(0)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            lines.append(f"[Example {idx}]")
            lines.append("--- Prompt ---")
            lines.append(prompt_text)
            lines.append("--- Output ---")
            lines.append(output_text)
            lines.append("")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_samples.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log_info(f"Wrote eval samples to {out_path}")

class EvalSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, output_dir: str):
        self._tokenizer = tokenizer
        self._eval_dataset = eval_dataset
        self._output_dir = output_dir

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        _write_eval_samples(
            model=model,
            tokenizer=self._tokenizer,
            eval_dataset=self._eval_dataset,
            output_dir=self._output_dir,
            epoch=state.epoch,
            global_step=state.global_step,
        )
        return control

def main(config_path: str, resume: str | None = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    wandb_config = config.get("wandb", {})
    setup_wandb(wandb_config, config)

    model = load_model(config, adapter_path=resume)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load pre-tokenized data
    log_info("Loading pre-tokenized data...")
    train_data = load_from_disk(config["data"]["path"])
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in ["input_ids", "attention_mask", "labels"]])

    val_data = load_from_disk(config["data"]["eval_path"])
    val_data = val_data.remove_columns([c for c in val_data.column_names if c not in ["input_ids", "attention_mask", "labels"]])

    log_info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_cfg = config["train"]
    ckpt_cfg = config["ckpt"]
    log_cfg = config["logging"]

    training_args = TrainingArguments(
        output_dir=ckpt_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["use_gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if train_cfg.get("use_gradient_checkpointing") else None,
        logging_steps=log_cfg["log_steps"],
        save_steps=ckpt_cfg["save_steps"],
        save_total_limit=ckpt_cfg["save_total_limit"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config["seed"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        save_safetensors=True,
        report_to="wandb" if wandb_config.get("enabled", False) else "none",
        run_name=wandb_config.get("run_name"),
        logging_first_step=True,
        disable_tqdm=False,
        log_level="info",
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[EvalSampleCallback(tokenizer, val_data, ckpt_cfg["output_dir"])],
    )
    
    log_info("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    log_info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    main(args.config, resume=args.resume)
