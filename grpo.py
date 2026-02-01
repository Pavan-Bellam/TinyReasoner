import argparse
import json
import re
import subprocess
import threading

import torch
import wandb
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

from utils import parse_answer, parse_single_value, compare_parsed, extract_answer


def upload_to_s3(local_path: str, s3_path: str, blocking: bool = False):
    """Upload a directory to S3 using aws cli."""
    def _upload():
        cmd = ["aws", "s3", "sync", local_path, s3_path, "--quiet"]
        print(f"Uploading {local_path} -> {s3_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"S3 upload failed: {result.stderr}")
        else:
            print(f"Upload complete: {local_path}")

    if blocking:
        _upload()
    else:
        threading.Thread(target=_upload, daemon=True).start()


class S3UploadCallback(TrainerCallback):
    def __init__(self, s3_base_path: str):
        self.s3_base_path = s3_base_path

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        ckpt_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        s3_dest = f"{self.s3_base_path}checkpoint-{state.global_step}/"
        upload_to_s3(ckpt_dir, s3_dest)
        return control



def get_length_penalty(token_length: int, level: int) -> float:
    """
    Linear penalty from 0 to 0.5 over [0, budget] range.
    
    - 0 tokens: 0.0 penalty
    - At budget: 0.5 penalty (capped)
    """
    budgets = {1: 2048, 2: 4096, 3: 8192, 4: 16384, 5: 24576}
    
    level = max(1, min(5, level))
    budget = budgets[level]
    
    return min(0.5, 0.5 * token_length / budget)

def _parse_level(level) -> int:
    """Parse level from various formats: 'Level 3', '3', 3, etc."""
    if isinstance(level, int):
        return level
    s = str(level).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 1


def make_reward_fn(
    tokenizer,
    level_col: str,
    correct_reward: float = 1.0,
    wrong_reward: float = -0.1,
    invalid_format_reward: float = -0.05,
    debug_path: str | None = None,
):
    """
    Gated reward: format validity gates correctness, minus length penalty.
    - Invalid format: invalid_format_reward (-0.05)
    - Valid format + correct: correct_reward - length_penalty
    - Valid format + wrong: wrong_reward - length_penalty
    """
    batch_idx = [0]

    def reward_fn(completions: list[str], answer: list[str], **kwargs) -> list[float]:
        rewards = []
        correct_count = 0
        format_valid_count = 0
        total_length_penalty = 0.0
        total_token_length = 0
        debug_records = []

        prompts = kwargs.get("prompt", [None] * len(completions))
        levels = kwargs.get(level_col, [1] * len(completions))

        for i, (completion, gt) in enumerate(zip(completions, answer)):
            parsed_pred = parse_answer(completion)
            parsed_gt = parse_single_value(gt)
            has_think = "<think>" in completion
            is_correct = False

            level = _parse_level(levels[i])
            token_length = len(tokenizer.encode(completion, add_special_tokens=False))
            length_penalty = get_length_penalty(token_length, level)
            total_length_penalty += length_penalty
            total_token_length += token_length

            if parsed_pred is None or not has_think:
                reward = invalid_format_reward
            else:
                format_valid_count += 1
                if parsed_gt is not None and compare_parsed(parsed_pred, parsed_gt):
                    reward = correct_reward - length_penalty
                    correct_count += 1
                    is_correct = True
                else:
                    reward = wrong_reward
            rewards.append(reward)

            if debug_path:
                debug_records.append({
                    "batch": batch_idx[0],
                    "idx": i,
                    "prompt": prompts[i],
                    "completion": completion,
                    "expected": gt,
                    "parsed_pred": str(parsed_pred),
                    "parsed_gt": str(parsed_gt),
                    "has_think": has_think,
                    "has_boxed": parsed_pred is not None,
                    "format_valid": parsed_pred is not None and has_think,
                    "correct": is_correct,
                    "reward": reward,
                    "level": level,
                    "token_length": token_length,
                    "length_penalty": length_penalty,
                })

        if debug_path and debug_records:
            with open(debug_path, "a", encoding="utf-8") as f:
                for rec in debug_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        batch_idx[0] += 1

        n = max(1, len(completions))
        if wandb.run is not None:
            wandb.log({
                "custom/format_accuracy": format_valid_count / n,
                "custom/answer_accuracy": correct_count / n,
                "custom/accuracy_given_valid_format": (
                    correct_count / format_valid_count if format_valid_count > 0 else 0.0
                ),
                "custom/mean_length_penalty": total_length_penalty / n,
                "custom/mean_token_length": total_token_length / n,
            })

        return rewards

    return reward_fn


def main(resume_from: str | None = None, config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    gcfg = cfg["grpo"]

    model_name = gcfg["model_path"]
    output_dir = gcfg["output_dir"]
    s3_ckpt_path = gcfg.get("s3_checkpoint_path", "")

    print(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load dataset ---
    ds_cfg = gcfg["dataset"]

    print("Loading dataset...")
    subsets = ds_cfg.get("subsets")
    if subsets:
        datasets = [load_dataset(ds_cfg["name"], s, split=ds_cfg["split"]) for s in subsets]
        train_dataset = concatenate_datasets(datasets)
    else:
        train_dataset = load_dataset(ds_cfg["name"], ds_cfg.get("config"), split=ds_cfg["split"])
    print(f"Raw dataset size: {len(train_dataset)} examples")

    # Extract boxed answers and filter unparseable
    solution_col = ds_cfg["solution_col"]
    def add_answer(example):
        example["answer"] = extract_answer(example[solution_col])
        return example
    train_dataset = train_dataset.map(add_answer)
    before = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x["answer"] is not None)
    print(f"After filtering unparseable: {len(train_dataset)} / {before} examples")

    # Rename and format prompt column for GRPOTrainer
    prompt_col = ds_cfg["prompt_col"]
    if prompt_col != "prompt":
        train_dataset = train_dataset.rename_column(prompt_col, "prompt")

    def format_prompt(example):
        example["prompt"] = [{"role": "user", "content": example["prompt"]}]
        return example
    train_dataset = train_dataset.map(format_prompt)
    print(f"Final dataset size: {len(train_dataset)} examples")

    # --- Training args ---
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=gcfg["num_generations"],
        max_prompt_length=gcfg["max_prompt_length"],
        max_completion_length=gcfg["max_completion_length"],
        temperature=gcfg["temperature"],
        learning_rate=gcfg["learning_rate"],
        per_device_train_batch_size=gcfg["per_device_batch_size"],
        gradient_accumulation_steps=gcfg["gradient_accumulation_steps"],
        max_steps=gcfg["max_steps"],
        max_grad_norm=gcfg.get("max_grad_norm", 1.0),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=gcfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=gcfg["warmup_steps"],
        logging_steps=gcfg["logging_steps"],
        save_strategy="steps",
        save_steps=gcfg["save_steps"],
        save_total_limit=gcfg.get("save_total_limit", 3),
        seed=gcfg.get("seed", 42),
        report_to="wandb",
        run_name=gcfg["run_name"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4, 
    )

    reward_fn = make_reward_fn(
        tokenizer=tokenizer,
        level_col=ds_cfg.get("level_col", "level"),
        correct_reward=gcfg.get("correct_reward", 1.0),
        wrong_reward=gcfg.get("wrong_reward", -0.1),
        invalid_format_reward=gcfg.get("invalid_format_reward", -0.05),
        debug_path=gcfg.get("debug_generations_path"),
    )

    callbacks = []
    if s3_ckpt_path:
        callbacks.append(S3UploadCallback(s3_ckpt_path))

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        callbacks=callbacks,
    )

    print("Starting GRPO Training")
    trainer.train(resume_from_checkpoint=resume_from)

    print("Saving Model")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    if s3_ckpt_path:
        upload_to_s3(f"{output_dir}/final", f"{s3_ckpt_path}final/", blocking=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    main(resume_from=args.resume, config_path=args.config)
