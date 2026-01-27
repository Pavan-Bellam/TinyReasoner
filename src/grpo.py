import os
import yaml
import argparse
from dataclasses import dataclass

import regex as re
import torch
import wandb
from loguru import logger
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from sympy import simplify, Expr, Equality
from sympy.parsing.latex import parse_latex

# --- PATTERNS ---
BOXED_PAT = re.compile(r"\\boxed\s*\{((?:[^{}]+|\{(?1)\})*)\}")
TUPLE_PAT = re.compile(r"^\\?(?:left)?[\(\[]\s*(.+?)\s*,\s*(.+?)\s*\\?(?:right)?[\)\]]$")
NUM_PAT = re.compile(r"^\s*\\?\$?\s*([+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?)\s*$", re.VERBOSE)
PMATRIX_PAT = re.compile(r"\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}", re.DOTALL)


def _parse_num(s: str):
    m = NUM_PAT.fullmatch(s)
    if not m:
        return None
    num = m.group(1)
    return float(num) if ("." in num or "e" in num.lower()) else int(num)


def _strip_latex_cruft(s: str) -> str:
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\\circ", "", s)
    s = re.sub(r"\\circ", "", s)
    s = re.sub(r"\b(euros?|dollars?|meters?|cm|kg)\b", "", s, flags=re.I)
    return s.strip()


def extract_raw_boxed(text: str) -> str | None:
    text = str(text)
    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None
    return _strip_latex_cruft(matches[-1].group(1))


def parse_vector(s: str):
    """Parse \\begin{pmatrix}...\\end{pmatrix} into tuple of values."""
    match = PMATRIX_PAT.fullmatch(s.strip())
    if not match:
        return None
    content = match.group(1)
    elements = re.split(r"\\\\", content)
    parsed = []
    for el in elements:
        val = parse_single_value(el.strip())
        if val is None:
            return None
        parsed.append(val)
    return tuple(parsed) if parsed else None


def parse_single_value(s: str):
    s = s.strip()
    if not s:
        return None

    # Try numeric first
    num = _parse_num(s)
    if num is not None:
        return num

    # Try sympy latex
    try:
        expr = parse_latex(s)
        if isinstance(expr, Equality):
            expr = expr.rhs
        expr = simplify(expr)
        if expr.free_symbols:
            return expr
        evaluated = expr.evalf()
        if evaluated.is_Integer:
            return int(evaluated)
        if evaluated.is_real:
            return float(evaluated)
        return expr
    except Exception:
        pass

    # Fallback: return cleaned string for text answers
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", s).strip()
    return cleaned if cleaned else None


def parse_answer(text: str | None):
    if text is None:
        return None

    text = str(text)
    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None
    answer = _strip_latex_cruft(matches[-1].group(1))
    if not answer:
        return None

    # Try vector/matrix first
    vec = parse_vector(answer)
    if vec is not None:
        return vec

    # Try tuple
    tuple_match = TUPLE_PAT.fullmatch(answer)
    if tuple_match:
        left = parse_single_value(tuple_match.group(1))
        right = parse_single_value(tuple_match.group(2))
        if left is not None and right is not None:
            return (left, right)
        return None

    return parse_single_value(answer)


def compare_values(a, b) -> bool:
    """Compare two parsed values."""
    # Both numeric
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9

    # Both sympy expressions
    if isinstance(a, Expr) and isinstance(b, Expr):
        try:
            return simplify(a - b) == 0
        except Exception:
            return False

    # Mixed types - try numeric comparison
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (ValueError, TypeError):
        return str(a).strip().lower() == str(b).strip().lower()


def compare_answers(text_a, text_b) -> bool:
    parsed_a = parse_answer(text_a)
    parsed_b = parse_answer(text_b)

    # Both parsed successfully
    if parsed_a is not None and parsed_b is not None:
        # Tuple comparison
        if isinstance(parsed_a, tuple) and isinstance(parsed_b, tuple):
            if len(parsed_a) != len(parsed_b):
                return False
            return all(compare_values(a, b) for a, b in zip(parsed_a, parsed_b))

        if isinstance(parsed_a, tuple) or isinstance(parsed_b, tuple):
            return False

        return compare_values(parsed_a, parsed_b)

    # Fallback: compare raw boxed content as strings
    raw_a = extract_raw_boxed(text_a)
    raw_b = extract_raw_boxed(text_b)

    if raw_a is None or raw_b is None:
        return False

    return raw_a.strip().lower() == raw_b.strip().lower()


SYSTEM_PROMPT = """Solve this math problem. Show your work step by step, then put your final answer in \\boxed{}.
"""



def make_reward_fn(
    correct_reward: float = 1.0,
    wrong_reward: float = 0.0,
    no_boxed_reward: float = 0.0,
    debug_path: str | None = None,
):
    """
    Reward function for free-form output with \\boxed{} answer.
    - No \\boxed{} found: no_boxed_reward (0.0)
    - \\boxed{} correct: correct_reward (+1.0)
    - \\boxed{} wrong: wrong_reward (0.0)
    """
    import json
    batch_idx = [0]

    def reward_fn(completions: list[str], answer: list[str], prompt: list[str] | None = None, **kwargs) -> list[float]:
        rewards = []
        correct_count = 0
        has_boxed_count = 0
        debug_records = []

        for i, (completion, gt) in enumerate(zip(completions, answer)):
            extracted = extract_raw_boxed(completion)
            has_boxed = extracted is not None
            is_correct = False

            if not has_boxed:
                reward = no_boxed_reward
            else:
                has_boxed_count += 1
                if compare_answers(completion, gt):
                    reward = correct_reward
                    correct_count += 1
                    is_correct = True
                else:
                    reward = wrong_reward
            rewards.append(reward)

            if debug_path:
                debug_records.append({
                    "batch": batch_idx[0],
                    "idx": i,
                    "prompt": prompt[i] if prompt else None,
                    "completion": completion,
                    "expected": gt,
                    "extracted": extracted,
                    "has_boxed": has_boxed,
                    "correct": is_correct,
                    "reward": reward,
                })

        if debug_path and debug_records:
            with open(debug_path, "a", encoding="utf-8") as f:
                for rec in debug_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        batch_idx[0] += 1

        if wandb.run is not None:
            wandb.log({
                "custom/boxed_rate": has_boxed_count / max(1, len(completions)),
                "custom/answer_accuracy": correct_count / max(1, len(completions)),
                "custom/accuracy_given_boxed": (
                    correct_count / has_boxed_count if has_boxed_count > 0 else 0.0
                ),
            })

        return rewards

    return reward_fn



def load_model_and_tokenizer(config: dict, resume_checkpoint: str | None = None):
    model_path = config["model"]["path"]
    grpo_config = config["grpo"]
    lora_config_dict = config["lora"]
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
        attn_implementation="flash_attention_2",
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

    if resume_checkpoint:
        # Resuming from checkpoint - load existing LoRA adapter
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from checkpoint: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        # Fresh training - create new LoRA adapter on base model
        logger.info("Creating new LoRA adapter on base model")
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
    examples_save_path: str | None = None,
    num_examples_to_save: int = 10,
    global_step: int | None = None,
):
    """
    Logs:
      - eval/pass1_accuracy: greedy/near-deterministic (do_sample=False)
      - eval/passk_accuracy: sampled with k returns (do_sample=True, temperature=temperature_train)
      - eval/boxed_pass1_rate: rate of \\boxed{} in greedy outputs
      - eval/boxed_passk_rate: rate of any \\boxed{} in sampled outputs

    Optionally saves example generations to a file.
    """
    import json

    model_was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    n = min(max_examples, len(eval_dataset))
    subset = eval_dataset.shuffle(seed=42).select(range(n))

    pass1_correct = 0
    passk_correct = 0
    pass1_boxed = 0
    passk_boxed = 0

    # Collect examples to save
    examples_to_save = []

    for idx, ex in enumerate(subset):
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
        text1 = tokenizer.decode(out1[0][prompt_len:], skip_special_tokens=True)
        boxed1 = extract_raw_boxed(text1)
        is_correct = False
        if boxed1 is not None:
            pass1_boxed += 1
            if compare_answers(text1, gt):
                pass1_correct += 1
                is_correct = True

        # Save example if within limit
        if examples_save_path and idx < num_examples_to_save:
            examples_to_save.append({
                "idx": idx,
                "prompt": prompt,
                "ground_truth": gt,
                "generation": text1,
                "extracted_boxed": boxed1,
                "correct": is_correct,
            })

        # PASS@K (sampled)
        outk = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=temperature_train,
            num_return_sequences=k,
        )

        any_boxed = False
        any_correct = False
        for seq in outk:
            textk = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            boxedk = extract_raw_boxed(textk)
            if boxedk is not None:
                any_boxed = True
                if compare_answers(textk, gt):
                    any_correct = True
                    break

        if any_boxed:
            passk_boxed += 1
        if any_correct:
            passk_correct += 1

    # Save examples to file
    if examples_save_path and examples_to_save:
        with open(examples_save_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"EVAL EXAMPLES @ step {global_step}\n")
            f.write(f"{'='*80}\n\n")
            for ex in examples_to_save:
                f.write(f"--- Example {ex['idx']} ---\n")
                f.write(f"Correct: {ex['correct']}\n")
                f.write(f"Ground Truth: {ex['ground_truth']}\n")
                f.write(f"Extracted: {ex['extracted_boxed']}\n")
                f.write(f"Prompt:\n{ex['prompt']}\n\n")
                f.write(f"Generation:\n{ex['generation']}\n")
                f.write(f"\n{'-'*40}\n\n")

    metrics = {
        "eval/n_examples": n,
        "eval/pass1_accuracy": pass1_correct / max(1, n),
        "eval/passk_accuracy": passk_correct / max(1, n),
        "eval/boxed_pass1_rate": pass1_boxed / max(1, n),
        "eval/boxed_passk_rate": passk_boxed / max(1, n),
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
    examples_save_path: str | None = None
    num_examples_to_save: int = 10


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
            examples_save_path=self.eval_cfg.examples_save_path,
            num_examples_to_save=self.eval_cfg.num_examples_to_save,
            global_step=state.global_step,
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
    logger.info(f"Full dataset size: {len(train_data)}")

    # Filter to only Level 3-5 (harder problems) for GRPO
    train_data = train_data.filter(lambda x: x["level"] in ["Level 3", "Level 4", "Level 5"])
    logger.info(f"Filtered to Level 3-5: {len(train_data)}")

    # Format prompts with chat template and system prompt
    def format_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return example

    train_data = train_data.map(format_prompt)
    train_data = train_data.select_columns(["prompt", "answer"])

    eval_data = load_from_disk(config['data']['eval_path'])
    eval_data = eval_data.map(format_prompt)
    eval_data = eval_data.select_columns(["prompt", "answer"])

    debug_path = grpo_config.get("debug_generations_path")
    if debug_path:
        logger.info(f"Debug mode: saving generations to {debug_path}")

    reward_fn = make_reward_fn(
        correct_reward=grpo_config.get("correct_reward", 1.0),
        wrong_reward=grpo_config.get("wrong_reward", 0.0),
        no_boxed_reward=grpo_config.get("no_boxed_reward", 0.0),
        debug_path=debug_path,
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
        reward_funcs=[reward_fn],
    )

    # Add PASS@1 and PASS@K eval callback
    eval_cfg = EvalConfig(
        eval_steps=grpo_config.get("eval_steps", 100),
        eval_max_examples=grpo_config.get("eval_max_examples", 256),
        examples_save_path=grpo_config.get("eval_examples_path"),
        num_examples_to_save=grpo_config.get("eval_num_examples_to_save", 10),
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
