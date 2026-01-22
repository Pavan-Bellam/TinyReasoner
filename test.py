"""
Evaluation script for TinyReasoner.

Evaluates format compliance and answer accuracy on the test set.

Usage:
    python src/eval.py --config config.yml --checkpoint ./checkpoints/checkpoint-500
    python src/eval.py --config config.yml --checkpoint ./checkpoints/checkpoint-500 --num-samples 100
"""

import argparse
import re
import json
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def load_model_for_inference(config: dict, checkpoint_path: str):
    """Load base model + LoRA adapter for inference."""
    model_path = config["model"]["path"]
    quant_config = setup_quantization(config["quant"])

    logger.info(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"Loading adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


SYSTEM_PROMPT = """You must reply in exactly this format and output nothing else:

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


def extract_format_components(text: str) -> dict:
    """Extract <think> and <answer> blocks from generated text."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    return {
        "has_think": think_match is not None,
        "has_answer": answer_match is not None,
        "think_content": think_match.group(1).strip() if think_match else None,
        "answer_content": answer_match.group(1).strip() if answer_match else None,
    }


def check_format_compliance(text: str) -> dict:
    """Check if output follows the required format."""
    components = extract_format_components(text)

    # Strict format: starts with <think>, has both tags, nothing after </answer>
    text_stripped = text.strip()
    starts_with_think = text_stripped.startswith("<think>")
    ends_with_answer = text_stripped.endswith("</answer>")

    # Check ordering: <think> before </think> before <answer> before </answer>
    think_open = text.find("<think>")
    think_close = text.find("</think>")
    answer_open = text.find("<answer>")
    answer_close = text.find("</answer>")

    correct_order = (
        think_open != -1
        and think_close != -1
        and answer_open != -1
        and answer_close != -1
        and think_open < think_close < answer_open < answer_close
    )

    return {
        "has_think_tag": components["has_think"],
        "has_answer_tag": components["has_answer"],
        "starts_correctly": starts_with_think,
        "ends_correctly": ends_with_answer,
        "correct_order": correct_order,
        "fully_compliant": all([
            components["has_think"],
            components["has_answer"],
            starts_with_think,
            ends_with_answer,
            correct_order,
        ]),
        "extracted_answer": components["answer_content"],
    }


def normalize_answer(answer: str | None) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""

    # Remove whitespace, convert to lowercase
    ans = answer.strip().lower()

    # Remove common LaTeX wrappers
    ans = re.sub(r"^\$+|\$+$", "", ans)
    ans = re.sub(r"^\\text\{(.*)\}$", r"\1", ans)
    ans = re.sub(r"^\\boxed\{(.*)\}$", r"\1", ans)

    # Normalize fractions: \frac{a}{b} -> a/b
    ans = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", ans)

    # Remove spaces
    ans = ans.replace(" ", "")

    return ans


def check_answer_correctness(predicted: str | None, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if not pred_norm:
        return False

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        pred_float = float(pred_norm.replace(",", ""))
        gt_float = float(gt_norm.replace(",", ""))
        return abs(pred_float - gt_float) < 1e-6
    except ValueError:
        pass

    return False


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} from solution."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return string[idx:right_brace_idx + 1]


def extract_ground_truth(solution: str) -> str | None:
    """Extract ground truth answer from solution."""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    if boxed.startswith("\\boxed{"):
        return boxed[7:-1]
    if boxed.startswith("\\fbox{"):
        return boxed[6:-1]
    return boxed


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    problem: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate model response for a problem."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if temperature == 0.0:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response


def evaluate(
    model,
    tokenizer,
    test_data,
    num_samples: int | None = None,
    max_new_tokens: int = 512,
    save_path: str | None = None,
):
    """Run evaluation on test set."""

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    results = []
    format_stats = {
        "has_think_tag": 0,
        "has_answer_tag": 0,
        "starts_correctly": 0,
        "ends_correctly": 0,
        "correct_order": 0,
        "fully_compliant": 0,
    }
    correct_answers = 0

    # We need original problems and solutions - reload raw data
    from datasets import load_dataset, concatenate_datasets

    SUBSETS = [
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    test_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in SUBSETS]
    raw_test = concatenate_datasets(test_datasets)

    if num_samples:
        raw_test = raw_test.select(range(min(num_samples, len(raw_test))))

    logger.info(f"Evaluating on {len(raw_test)} samples...")

    for i, example in enumerate(tqdm(raw_test, desc="Evaluating")):
        problem = example["problem"]
        solution = example["solution"]
        ground_truth = extract_ground_truth(solution)

        # Generate response
        response = generate_response(model, tokenizer, problem, max_new_tokens)

        # Check format
        format_check = check_format_compliance(response)
        for key in format_stats:
            if format_check.get(key, False):
                format_stats[key] += 1

        # Check answer
        is_correct = check_answer_correctness(format_check["extracted_answer"], ground_truth)
        if is_correct:
            correct_answers += 1

        results.append({
            "problem": problem,
            "ground_truth": ground_truth,
            "response": response,
            "extracted_answer": format_check["extracted_answer"],
            "format_compliant": format_check["fully_compliant"],
            "answer_correct": is_correct,
        })

        # Log progress every 50 samples
        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i+1}/{len(raw_test)} | "
                f"Format: {format_stats['fully_compliant']/(i+1)*100:.1f}% | "
                f"Accuracy: {correct_answers/(i+1)*100:.1f}%"
            )

    # Compute final metrics
    total = len(results)
    metrics = {
        "total_samples": total,
        "format": {k: {"count": v, "percent": v / total * 100} for k, v in format_stats.items()},
        "accuracy": {
            "correct": correct_answers,
            "percent": correct_answers / total * 100,
        },
        # Accuracy only on format-compliant samples
        "accuracy_given_format": {
            "correct": sum(1 for r in results if r["format_compliant"] and r["answer_correct"]),
            "total_compliant": format_stats["fully_compliant"],
            "percent": (
                sum(1 for r in results if r["format_compliant"] and r["answer_correct"])
                / max(format_stats["fully_compliant"], 1) * 100
            ),
        },
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total}")
    logger.info(f"\nFormat Compliance:")
    logger.info(f"  Has <think> tag:    {format_stats['has_think_tag']:4d} ({format_stats['has_think_tag']/total*100:.1f}%)")
    logger.info(f"  Has <answer> tag:   {format_stats['has_answer_tag']:4d} ({format_stats['has_answer_tag']/total*100:.1f}%)")
    logger.info(f"  Starts correctly:   {format_stats['starts_correctly']:4d} ({format_stats['starts_correctly']/total*100:.1f}%)")
    logger.info(f"  Ends correctly:     {format_stats['ends_correctly']:4d} ({format_stats['ends_correctly']/total*100:.1f}%)")
    logger.info(f"  Correct order:      {format_stats['correct_order']:4d} ({format_stats['correct_order']/total*100:.1f}%)")
    logger.info(f"  Fully compliant:    {format_stats['fully_compliant']:4d} ({format_stats['fully_compliant']/total*100:.1f}%)")
    logger.info(f"\nAnswer Accuracy:")
    logger.info(f"  Overall:            {correct_answers:4d} ({correct_answers/total*100:.1f}%)")
    logger.info(f"  Given format OK:    {metrics['accuracy_given_format']['correct']:4d} / {format_stats['fully_compliant']} ({metrics['accuracy_given_format']['percent']:.1f}%)")

    # Save results
    if save_path:
        output = {"metrics": metrics, "results": results}
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to: {save_path}")

    return metrics, results


def main(config_path: str, checkpoint_path: str, num_samples: int | None, output_path: str | None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model_for_inference(config, checkpoint_path)

    metrics, results = evaluate(
        model,
        tokenizer,
        test_data=None,  # We load raw data inside
        num_samples=num_samples,
        save_path=output_path,
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TinyReasoner")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output path for results")
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.num_samples, args.output)
