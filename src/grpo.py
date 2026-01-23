import os
import re
import math
import yaml
import argparse
from dataclasses import dataclass

import torch
import wandb
from loguru import logger
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from sympy.parsing.latex import parse_latex
from sympy import N


_INTERVAL_RE = re.compile(r'^\s*[\[\(].*[,;].*[\]\)]\s*$')


def _canonicalize_interval(s: str) -> str:
    """Normalize interval string: remove whitespace, normalize separators."""
    s = re.sub(r"\s+", "", s)
    s = s.replace(";", ",")
    return s


def _canonicalize_set(s: str) -> str:
    """Normalize set string: convert LaTeX braces to {}, remove whitespace."""
    s = s.strip()
    s = s.replace(r"\left\{", "{").replace(r"\right\}", "}")
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = re.sub(r"\s+", "", s)
    return s


def canonicalize_answer(ans: str) -> tuple[str | None, any]:
    """
    Returns a tagged canonical form:
      ("num", float) for scalars
      ("interval", str) for intervals
      ("set", str) for sets
      (None, None) for unhandled
    """
    if ans is None:
        return (None, None)

    s = ans.strip()

    # Interval like [-2,7], (0,1], etc.
    if _INTERVAL_RE.match(s) and (s[0] in "[(") and (s[-1] in "])"):
        return ("interval", _canonicalize_interval(s))

    # Set: LaTeX braces
    if (r"\{" in s) or (s.startswith("{") and s.endswith("}")):
        return ("set", _canonicalize_set(s))

    # Percentage
    if r"\%" in s:
        s2 = s.replace(r"\%", "")
        try:
            return ("num", float(N(parse_latex(s2))) / 100.0)
        except Exception:
            return (None, None)

    # Numeric / fraction / dfrac
    try:
        return ("num", float(N(parse_latex(s))))
    except Exception:
        return (None, None)


def answers_match(pred: str, gt: str) -> bool:
    """Check if predicted answer matches ground truth, handling different types."""
    p_tag, p_val = canonicalize_answer(pred)
    g_tag, g_val = canonicalize_answer(gt)

    if p_tag is None or g_tag is None:
        return False
    if p_tag != g_tag:
        return False
    if p_tag in ("interval", "set"):
        return p_val == g_val
    # numeric
    return math.isclose(p_val, g_val, rel_tol=1e-6)



def parse_response(response: str) -> tuple[bool, str | None]:
    pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True, match.group(1).strip()
    return False, None


def make_reward_funcs(format_weight: float = 0.2, correctness_weight: float = 1.0):

    def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        format_valid_count = 0

        for completion in completions:
            valid, _ = parse_response(completion)
            if valid:
                rewards.append(format_weight)
                format_valid_count += 1
            else:
                rewards.append(0.0)

        if wandb.run is not None:
            wandb.log({
                "custom/format_accuracy": format_valid_count / max(1, len(completions)),
                "custom/format_valid_count": format_valid_count,
                "custom/n_samples": len(completions),
            })

        return rewards

    def correctness_reward_fn(
        completions: list[str],
        answer: list[str],
        **kwargs
    ) -> list[float]:

        rewards = []
        correct_count = 0
        format_valid_count = 0

        for completion, gt in zip(completions, answer):
            valid, extracted = parse_response(completion)

            if not valid:
                rewards.append(0.0)
                continue

            format_valid_count += 1

            if answers_match(extracted, gt):
                rewards.append(correctness_weight)
                correct_count += 1
            else:
                rewards.append(0.0)

        if wandb.run is not None:
            wandb.log({
                "custom/answer_accuracy": correct_count / max(1, len(completions)),
                "custom/correct_count": correct_count,
                "custom/accuracy_given_valid_format": (
                    correct_count / format_valid_count if format_valid_count > 0 else 0.0
                ),
                "custom/n_samples": len(completions),
            })

        return rewards

    return format_reward_fn, correctness_reward_fn



def load_model_and_tokenizer(config: dict, resume_checkpoint: str | None = None):
    model_path = config["model"]["path"]
    grpo_config = config["grpo"]
    lora_config_dict = config["lora"]
    sft_adapter_path = config["sft_adapter_path"]
    use_gradient_checkpointing = grpo_config.get("use_gradient_checkpointing", False)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="sdpa",
    )

    # Sync tokenizer special tokens with model config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id

    # Always merge SFT adapter first (GRPO adapter was trained on merged model)
    logger.info(f"Loading SFT adapter from {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, sft_adapter_path)

    logger.info("Merging SFT adapter into base model")
    model = model.merge_and_unload()

    # Remove leftover PEFT metadata to prevent multi-adapter warning
    for attr in ("peft_config", "active_adapter"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass

    logger.info(f"After merge: is PeftModel={isinstance(model, PeftModel)}, has peft_config={hasattr(model, 'peft_config')}")

    if resume_checkpoint:
        # Resuming from GRPO checkpoint - load existing GRPO adapter
        logger.info(f"Loading GRPO adapter from checkpoint: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        # Fresh GRPO training - create new GRPO adapter
        logger.info("Creating new LoRA adapter for GRPO")
        lora_config = LoraConfig(
            r=lora_config_dict["r"],
            lora_alpha=lora_config_dict["alpha"],
            lora_dropout=lora_config_dict["dropout"],
            target_modules=lora_config_dict["target_modules"],
            bias=lora_config_dict["bias"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    return model, tokenizer



def setup_wandb(config: dict, full_config: dict):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        os.environ["WANDB_DISABLED"] = "true"
        return None

    run = wandb.init(
        project=wandb_config.get("project", "tinyreasoner-grpo"),
        name=wandb_config.get("run_name"),
        config=full_config,
        resume="allow",
    )
    logger.info(f"Wandb initialized: {run.url}")
    return run


@torch.no_grad()
def evaluate_pass_metrics(
    model,
    tokenizer,
    eval_dataset,
    *,
    max_prompt_length: int,
    max_completion_length: int,
    temperature_train: float,
    k: int,
    device: torch.device | None = None,
    max_examples: int = 256,
):
    """
    Logs:
      - eval/pass1_accuracy: greedy/near-deterministic (do_sample=False)
      - eval/passk_accuracy: sampled with k returns (do_sample=True, temperature=temperature_train)
      - eval/format_pass1_accuracy
      - eval/format_passk_accuracy
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    n = min(max_examples, len(eval_dataset))
    subset = eval_dataset.shuffle(seed=42).select(range(n))

    pass1_correct = 0
    passk_correct = 0
    pass1_format = 0
    passk_format = 0

    for ex in subset:
        prompt = ex["prompt"]
        gt = ex["answer"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
        ).to(device)
        prompt_len = inputs['input_ids'].shape[1]

        # PASS@1 (greedy)
        out1 = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            do_sample=False,
        )
        # Only decode the completion, not the prompt
        text1 = tokenizer.decode(out1[0][prompt_len:], skip_special_tokens=True)
        valid1, ans1 = parse_response(text1)
        if valid1:
            pass1_format += 1
            if answers_match(ans1, gt):
                pass1_correct += 1

        # PASS@K (sampled)
        outk = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=temperature_train,
            num_return_sequences=k,
        )

        any_valid = False
        any_correct = False
        for seq in outk:
            # Only decode the completion, not the prompt
            textk = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            validk, ansk = parse_response(textk)
            if validk:
                any_valid = True
                if answers_match(ansk, gt):
                    any_correct = True
                    break

        if any_valid:
            passk_format += 1
        if any_correct:
            passk_correct += 1

    metrics = {
        "eval/n_examples": n,
        "eval/pass1_accuracy": pass1_correct / max(1, n),
        "eval/passk_accuracy": passk_correct / max(1, n),
        "eval/format_pass1_accuracy": pass1_format / max(1, n),
        "eval/format_passk_accuracy": passk_format / max(1, n),
    }

    if wandb.run is not None:
        wandb.log(metrics)

    if model_was_training:
        model.train()

    return metrics


@dataclass
class EvalConfig:
    eval_steps: int = 100
    eval_max_examples: int = 256


class PassEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        *,
        grpo_cfg: dict,
        eval_cfg: EvalConfig,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.grpo_cfg = grpo_cfg
        self.eval_cfg = eval_cfg

    def on_step_end(self, args, state, control, **kwargs):
        # state.global_step is updated at step end
        if state.global_step == 0:
            return control
        if state.global_step % self.eval_cfg.eval_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        metrics = evaluate_pass_metrics(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            max_prompt_length=self.grpo_cfg["max_prompt_length"],
            max_completion_length=self.grpo_cfg["max_completion_length"],
            temperature_train=self.grpo_cfg["temperature"],
            k=self.grpo_cfg["num_generations"],
            max_examples=self.eval_cfg.eval_max_examples,
        )
        logger.info(f"[eval @ step {state.global_step}] {metrics}")
        return control


def main(config_path: str, resume: str | None = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    grpo_config = config["grpo"]
    wandb_config = config.get("wandb", {})

    setup_wandb(config, config)

    model, tokenizer = load_model_and_tokenizer(config, resume_checkpoint=resume)

    logger.info(f"Loading dataset from {config['data']['path']}")
    train_data = load_from_disk(config["data"]["path"])
    keep_columns = ["prompt", "answer", "level", "subject"]
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in keep_columns])
    logger.info(f"Full dataset size: {len(train_data)}")

    # Filter to only Level 3-5 (harder problems) for GRPO
    train_data = train_data.filter(lambda x: x["level"] in ["Level 3", "Level 4", "Level 5"])
    logger.info(f"Filtered to Level 3-5: {len(train_data)}")

    eval_data = load_from_disk(config['data']['eval_path'])
    eval_data = eval_data.remove_columns([c for c in eval_data.column_names if c not in keep_columns])

    format_reward_fn, correctness_reward_fn = make_reward_funcs(
        format_weight=grpo_config.get("format_reward", 0.2),
        correctness_weight=grpo_config.get("correctness_reward", 1.0),
    )

    training_args = GRPOConfig(
        output_dir=grpo_config["output_dir"],
        # Generation
        num_generations=grpo_config["num_generations"],
        max_prompt_length=grpo_config["max_prompt_length"],
        max_completion_length=grpo_config["max_completion_length"],
        temperature=grpo_config["temperature"],
        # Training
        learning_rate=grpo_config["learning_rate"],
        per_device_train_batch_size=grpo_config["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_config["gradient_accumulation_steps"],
        max_steps=grpo_config["max_steps"],
        max_grad_norm=grpo_config["max_grad_norm"],
        bf16=grpo_config["bf16"],
        gradient_checkpointing=grpo_config.get("use_gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if grpo_config.get("use_gradient_checkpointing")
        else None,
        # Logging & saving
        logging_steps=grpo_config["logging_steps"],
        save_steps=grpo_config["save_steps"],
        seed=config.get("seed", 42),
        # Wandb
        report_to="wandb" if wandb_config.get("enabled", False) else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        reward_funcs=[format_reward_fn, correctness_reward_fn],
    )

    # Add PASS@1 and PASS@K eval callback
    eval_cfg = EvalConfig(
        eval_steps=grpo_config.get("eval_steps", 100),
        eval_max_examples=grpo_config.get("eval_max_examples", 256),
    )
    trainer.add_callback(
        PassEvalCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_data,
            grpo_cfg=grpo_config,
            eval_cfg=eval_cfg,
        )
    )

    logger.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume)

    final_path = f"{grpo_config['output_dir']}/final"
    trainer.save_model(final_path)
    logger.info(f"Model saved to {final_path}")

    if wandb_config.get("enabled", False):
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args.config, resume=args.resume)
