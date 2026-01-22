"""
Supervised Fine-Tuning (SFT) script for TinyReasoner.

This script trains a language model to output responses in a structured
<think>...</think><answer>...</answer> format using the GSM8K dataset.

Usage:
    # Start fresh training
    python src/sft.py --config config.yml

    # Resume from checkpoint (continues optimizer state, step count)
    python src/sft.py --config config.yml --resume ./checkpointing/checkpoint-500

    # Initialize from checkpoint but start fresh (new optimizer, step 0)
    python src/sft.py --config config.yml --init-from ./checkpointing/checkpoint-500
"""

import argparse
import os

import torch
import wandb
import yaml
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = """
You must reply in exactly this format and output nothing else:

<think>
Step-by-step reasoning here.
</think>
<answer>
Final answer only.
</answer>

Rules:
- Do not include any text before <think> or after </answer>.
- Put all reasoning in <think>.
- Put only the final answer in <answer> (no explanation).
- If the answer is numeric, output only the number (no commas, no units).

"""


def get_formatting_func(tokenizer):
    """
    Create a formatting function for SFT training examples.

    Converts dataset examples into chat-formatted strings with the model's
    expected input/output format.

    Args:
        tokenizer: The tokenizer to use for applying chat templates.

    Returns:
        A function that takes an example dict and returns a formatted string.

    Example:
        >>> func = get_formatting_func(tokenizer)
        >>> formatted = func({"question": "What is 2+2?", "cot": "2+2=4", "answer": "4"})
    """
    def formatting_func(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {
                "role": "assistant",
                "content": (
                    f"<think>\n{example['cot']}\n</think>\n"
                    f"<answer>{example['answer']}</answer>"
                ),
            },
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return formatting_func


def setup_quantization(config: dict) -> BitsAndBytesConfig | None:
    """
    Configure quantization settings for memory-efficient training.

    Supports both 4-bit (QLoRA) and 8-bit quantization via bitsandbytes.

    Args:
        config: Quantization config dict with keys:
            - enabled: Whether to enable quantization
            - load_in_4bit: Use 4-bit quantization
            - load_in_8bit: Use 8-bit quantization
            - bnb_4bit_compute_dtype: Compute dtype for 4-bit (e.g., "bfloat16")
            - bnb_4bit_quant_type: Quantization type (e.g., "nf4")
            - bnb_4bit_use_double_quant: Whether to use nested quantization

    Returns:
        BitsAndBytesConfig if quantization is enabled, None otherwise.
    """
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

    if config.get("load_in_8bit", None):
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def get_lora_config(config: dict) -> LoraConfig:
    """
    Create a LoRA configuration for parameter-efficient fine-tuning.

    Args:
        config: LoRA config dict with keys:
            - r: LoRA rank (e.g., 4, 8, 16)
            - alpha: LoRA alpha scaling factor
            - dropout: Dropout probability for LoRA layers
            - target_modules: List of module names to apply LoRA to
            - bias: Bias training mode ("none", "all", "lora_only")

    Returns:
        LoraConfig object for PEFT.
    """
    return LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
        bias=config["bias"],
    )


def load_model_and_tokenizer(config: dict, adapter_path: str | None = None):
    """
    Load the base model and tokenizer, optionally with a pretrained adapter.

    Handles quantization setup, gradient checkpointing, and LoRA initialization.

    Args:
        config: Full configuration dict containing 'model', 'quant', 'train', and 'lora' sections.
        adapter_path: Optional path to a pretrained LoRA adapter to load.
            If None, creates a new LoRA adapter from config.

    Returns:
        Tuple of (model, tokenizer) ready for training.
    """
    model_path = config["model"]["path"]
    quant_config = setup_quantization(config["quant"])
    use_gradient_checkpointing = config["train"].get("use_gradient_checkpointing", False)

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Gradient checkpointing: {use_gradient_checkpointing}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer pad_token is None; pad-based batching may break.")

    # Prepare for k-bit training if quantized
    if quant_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if use_gradient_checkpointing
            else None,
        )
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Load existing adapter or create a new one
    if adapter_path:
        logger.info(f"Loading LoRA weights from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
        )
        # Re-enable gradient checkpointing on the adapter-wrapped model
        if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        logger.info("Creating new LoRA adapter")
        lora_config = get_lora_config(config=config["lora"])
        model = get_peft_model(model, lora_config)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()

    return model, tokenizer


def load_data(config: dict, tokenizer, formatting_func, max_length: int = 1024):
    """
    Load train and val datasets, filtering out examples exceeding max_length.

    Args:
        config: Data config dict with keys:
            - path: Path to training dataset
            - eval_path: Path to test dataset (10% sampled for validation)
        tokenizer: Tokenizer for computing sequence lengths.
        formatting_func: Function to format examples into training text.
        max_length: Maximum token length (examples exceeding this are filtered out).

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    def is_within_limit(example):
        text = formatting_func(example)
        return len(tokenizer.encode(text)) <= max_length

    # Load and filter train dataset
    train_path = config["path"]
    logger.info(f"Loading train dataset from {train_path}")
    train_dataset = load_from_disk(train_path)
    train_original = len(train_dataset)
    train_dataset = train_dataset.filter(is_within_limit)
    logger.info(f"Train filtered: {train_original} -> {len(train_dataset)} (<= {max_length} tokens)")

    # Load and filter val dataset (10% sample from test set)
    eval_path = config["eval_path"]
    logger.info(f"Loading val dataset from {eval_path}")
    val_dataset = load_from_disk(eval_path)
    val_original = len(val_dataset)
    val_dataset = val_dataset.filter(is_within_limit)
    val_dataset = val_dataset.shuffle(seed=42).select(range(int(len(val_dataset) * 0.1)))
    logger.info(f"Val filtered: {val_original} -> {len(val_dataset)} (10% sample, <= {max_length} tokens)")

    return train_dataset, val_dataset


def setup_wandb(config: dict, full_config: dict):
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        config: Wandb config dict with keys:
            - enabled: Whether to enable wandb logging
            - project: Wandb project name
            - run_name: Optional run name (auto-generated if None)
        full_config: Full training config to log as wandb config.

    Returns:
        Wandb run object if enabled, None otherwise.
    """
    if not config.get("enabled", False):
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run = wandb.init(
        project=config.get("project", "tinyreasoner-sft"),
        name=config.get("run_name"),
        config=full_config,
        resume="allow",
    )
    logger.info(f"Wandb initialized: {run.url}")
    return run


def main(config_path: str, init_from: str | None = None, resume: str | None = None):
    """
    Main training function.

    Args:
        config_path: Path to the YAML configuration file.
        init_from: Optional path to checkpoint for weight initialization only.
            Starts training from step 0 with a fresh optimizer.
        resume: Optional path to checkpoint for full training resumption.
            Continues from saved step with restored optimizer state.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb_config = config.get("wandb", {})
    setup_wandb(wandb_config, config)

    # Load model (from checkpoint if resuming or init-from)
    adapter_path = resume if resume else init_from
    model, tokenizer = load_model_and_tokenizer(config, adapter_path)

    # Load data
    formatting_func = get_formatting_func(tokenizer)
    max_length = config["train"].get("max_length", 1024)
    train_data, val_data = load_data(
        config=config["data"],
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_length=max_length,
    )

    # Extract config sections
    train_config = config["train"]
    ckpt_config = config["ckpt"]
    log_config = config["logging"]

    # Build training arguments
    training_args = SFTConfig(
        output_dir=ckpt_config["output_dir"],
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        max_grad_norm=train_config["max_grad_norm"],
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["use_gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if train_config.get("use_gradient_checkpointing")
        else None,
        logging_steps=log_config["log_steps"],
        save_steps=ckpt_config["save_steps"],
        save_total_limit=ckpt_config["save_total_limit"],
        eval_strategy="steps",
        eval_steps=train_config["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config["seed"],
        dataloader_num_workers=train_config.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        max_length=train_config.get("max_length", 1024),
        save_safetensors=True,
        report_to="wandb" if wandb_config.get("enabled", False) else "none",
        run_name=wandb_config.get("run_name"),
    )

    # Watch model for gradient logging
    if wandb_config.get("enabled", False) and wandb_config.get("watch_model", False):
        wandb.watch(model, log="all", log_freq=log_config["log_steps"])

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        formatting_func=formatting_func,
    )

    # Train (resume from checkpoint if specified)
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    # Cleanup wandb
    if wandb_config.get("enabled", False):
        if wandb_config.get("log_model", False):
            wandb.save(f"{ckpt_config['output_dir']}/*")
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for TinyReasoner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh training
  python src/sft.py --config config.yml

  # Resume from checkpoint
  python src/sft.py --config config.yml --resume ./checkpointing/checkpoint-500

  # Initialize weights but start fresh
  python src/sft.py --config config.yml --init-from ./checkpointing/checkpoint-500
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize weights from checkpoint but start fresh (step 0, new optimizer)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint (continues step count, optimizer state)",
    )
    args = parser.parse_args()

    if args.resume and args.init_from:
        raise ValueError(
            "Cannot use both --resume and --init-from. "
            "Use --resume to continue training, or --init-from to start fresh with pretrained weights."
        )

    main(args.config, init_from=args.init_from, resume=args.resume)
