import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sympy import simplify, Expr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from process_data import parse_answer

SYSTEM_PROMPT = """You must reply in exactly this format and output nothing else:

<think>Step-by-step reasoning here.</think><answer>Final answer only.</answer>

Rules:
- Do not include any text before <think> or after </answer>.
- Put all reasoning in <think>.
- Put only the final answer in <answer> (no explanation).
- If the answer is numeric, output only the number (no commas, no units).
"""


def load_model(base_model_path: str, sft_adapter_path: str, grpo_adapter_path: str):
    """Load base model with SFT adapter merged, then GRPO adapter on top."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "left"

    # Merge SFT adapter
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model = model.merge_and_unload()

    # Remove leftover PEFT metadata to prevent multi-adapter warning
    for attr in ("peft_config", "active_adapter"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass

    # Load GRPO adapter
    model = PeftModel.from_pretrained(model, grpo_adapter_path)
    model.eval()
    return model, tokenizer


def parse_response(response: str) -> tuple[bool, str | None]:
    pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True, match.group(1).strip()
    return False, None


def compare_answers(extracted: str | None, expected: str | None) -> bool:
    """Compare answers using sympy parsing."""
    if extracted is None or expected is None:
        return False

    parsed_extracted = parse_answer(extracted)
    parsed_expected = parse_answer(expected)

    if parsed_extracted is None or parsed_expected is None:
        return False

    if isinstance(parsed_extracted, (int, float)) and isinstance(parsed_expected, (int, float)):
        return abs(parsed_extracted - parsed_expected) < 1e-9

    if isinstance(parsed_extracted, Expr) and isinstance(parsed_expected, Expr):
        try:
            return simplify(parsed_extracted - parsed_expected) == 0
        except Exception:
            return False

    try:
        return abs(float(parsed_extracted) - float(parsed_expected)) < 1e-9
    except (ValueError, TypeError):
        return str(parsed_extracted) == str(parsed_expected)


def generate_response(model, tokenizer, problem: str, max_new_tokens: int = 1024) -> str:
    """Generate a single greedy response (pass@1)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def generate_responses_sampled(
    model, tokenizer, problem: str, k: int, temperature: float, max_new_tokens: int = 1024
) -> list[str]:
    """Generate k sampled responses for pass@k evaluation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    for seq in outputs:
        responses.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
    return responses


def evaluate(
    model,
    tokenizer,
    test_dataset,
    output_path: str | None = None,
    k: int = 4,
    temperature: float = 0.7,
    pass1_only: bool = False,
):
    """Evaluate model with pass@1 (greedy) and optionally pass@k (sampled) metrics."""
    total = len(test_dataset)
    results = []

    # Stats by subject and level
    stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "pass1_correct": 0, "passk_correct": 0,
        "pass1_format": 0, "passk_format": 0
    }))

    for i, example in enumerate(tqdm(test_dataset)):
        problem = example["problem"]
        expected = example["answer"].strip()
        level = example["level"]
        subject = example["type"]

        # === PASS@1 (greedy) ===
        response_greedy = generate_response(model, tokenizer, problem)
        valid1, extracted1 = parse_response(response_greedy)

        is_correct_pass1 = False
        if valid1:
            if compare_answers(extracted1, expected):
                is_correct_pass1 = True

        # === PASS@K (sampled) ===
        any_valid_k = False
        any_correct_k = False
        sampled_extractions = []

        if not pass1_only:
            responses_sampled = generate_responses_sampled(model, tokenizer, problem, k, temperature)

            for resp in responses_sampled:
                valid_k, extracted_k = parse_response(resp)
                is_correct_k = valid_k and compare_answers(extracted_k, expected)
                sampled_extractions.append({
                    "response": resp[:500],
                    "format_valid": valid_k,
                    "extracted": extracted_k,
                    "correct": is_correct_k,
                })
                if valid_k:
                    any_valid_k = True
                    if is_correct_k:
                        any_correct_k = True

        # Update stats
        stats[subject][level]["total"] += 1
        if valid1:
            stats[subject][level]["pass1_format"] += 1
        if is_correct_pass1:
            stats[subject][level]["pass1_correct"] += 1
        if any_valid_k:
            stats[subject][level]["passk_format"] += 1
        if any_correct_k:
            stats[subject][level]["passk_correct"] += 1

        result = {
            "id": i,
            "problem": problem,
            "expected": expected,
            "level": level,
            "type": subject,
            "pass1_response": response_greedy,
            "pass1_format_valid": valid1,
            "pass1_extracted": extracted1,
            "pass1_correct": is_correct_pass1,
            "passk_any_format_valid": any_valid_k,
            "passk_any_correct": any_correct_k,
            "passk_samples": sampled_extractions,
        }
        results.append(result)

        if i < 3:
            print(f"\n[{i}] Level: {level}, Type: {subject}")
            print(f"[{i}] Expected: {expected}")
            print(f"[{i}] Pass@1 - Valid: {valid1}, Extracted: {extracted1}, Correct: {is_correct_pass1}")
            print(f"[{i}] Pass@{k} - Any Valid: {any_valid_k}, Any Correct: {any_correct_k}")

    # Build results table
    levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    subjects = sorted(stats.keys())

    # Pass@1 table
    pass1_table = []
    for subject in subjects:
        row = {"Subject": subject}
        subj_correct, subj_total = 0, 0
        for level in levels:
            s = stats[subject][level]
            if s["total"] > 0:
                acc = s["pass1_correct"] / s["total"] * 100
                row[level] = f"{acc:.1f}%"
            else:
                row[level] = "-"
            subj_correct += s["pass1_correct"]
            subj_total += s["total"]
        row["Overall"] = f"{subj_correct / subj_total * 100:.1f}%" if subj_total > 0 else "-"
        pass1_table.append(row)

    # Overall row for pass@1
    overall_pass1 = {"Subject": "OVERALL"}
    for level in levels:
        lc = sum(stats[s][level]["pass1_correct"] for s in subjects)
        lt = sum(stats[s][level]["total"] for s in subjects)
        overall_pass1[level] = f"{lc / lt * 100:.1f}%" if lt > 0 else "-"
    total_pass1_correct = sum(r["pass1_correct"] for r in results)
    overall_pass1["Overall"] = f"{total_pass1_correct / total * 100:.1f}%"
    pass1_table.append(overall_pass1)

    # Pass@k table
    passk_table = []
    for subject in subjects:
        row = {"Subject": subject}
        subj_correct, subj_total = 0, 0
        for level in levels:
            s = stats[subject][level]
            if s["total"] > 0:
                acc = s["passk_correct"] / s["total"] * 100
                row[level] = f"{acc:.1f}%"
            else:
                row[level] = "-"
            subj_correct += s["passk_correct"]
            subj_total += s["total"]
        row["Overall"] = f"{subj_correct / subj_total * 100:.1f}%" if subj_total > 0 else "-"
        passk_table.append(row)

    # Overall row for pass@k
    overall_passk = {"Subject": "OVERALL"}
    for level in levels:
        lc = sum(stats[s][level]["passk_correct"] for s in subjects)
        lt = sum(stats[s][level]["total"] for s in subjects)
        overall_passk[level] = f"{lc / lt * 100:.1f}%" if lt > 0 else "-"
    total_passk_correct = sum(r["passk_any_correct"] for r in results)
    overall_passk["Overall"] = f"{total_passk_correct / total * 100:.1f}%"
    passk_table.append(overall_passk)

    df_pass1 = pd.DataFrame(pass1_table).set_index("Subject")

    # Print results
    print("\n" + "=" * 80)
    print(f"GRPO EVALUATION RESULTS (n={total})")
    print("=" * 80)

    print(f"\nPass@1 Accuracy by Subject and Level:")
    print(df_pass1.to_string())
    print(f"\nPass@1 Overall: {total_pass1_correct}/{total} = {total_pass1_correct/total*100:.2f}%")

    # Summary
    summary = {
        "total": total,
        "pass1_correct": total_pass1_correct,
        "pass1_accuracy": total_pass1_correct / total,
        "timestamp": datetime.now().isoformat(),
    }

    if not pass1_only:
        df_passk = pd.DataFrame(passk_table).set_index("Subject")
        print(f"\nPass@{k} Accuracy by Subject and Level (k={k}, temp={temperature}):")
        print(df_passk.to_string())
        print(f"\nPass@{k} Overall: {total_passk_correct}/{total} = {total_passk_correct/total*100:.2f}%")
        summary["k"] = k
        summary["temperature"] = temperature
        summary["passk_correct"] = total_passk_correct
        summary["passk_accuracy"] = total_passk_correct / total

    # Save results
    if output_path:
        output_data = {"summary": summary, "results": results}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

        csv_path = output_path.replace(".json", "_pass1.csv")
        df_pass1.to_csv(csv_path)
        print(f"Table saved to: {csv_path}")

        if not pass1_only:
            csv_path_k = output_path.replace(".json", f"_pass{k}.csv")
            df_passk.to_csv(csv_path_k)
            print(f"Pass@{k} table saved to: {csv_path_k}")

    return summary, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GRPO model with pass@1 and pass@k metrics")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sft-adapter", type=str, required=True, help="SFT adapter path (will be merged)")
    parser.add_argument("--grpo-adapter", type=str, required=True, help="GRPO adapter path")
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--pass1-only", action="store_true", help="Only evaluate pass@1, skip pass@k")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.sft_adapter, args.grpo_adapter)
    test_data = load_from_disk(args.test_data)

    if args.limit:
        test_data = test_data.select(range(min(args.limit, len(test_data))))
        print(f"Evaluating on {len(test_data)} examples")

    evaluate(
        model, tokenizer, test_data,
        output_path=args.output,
        k=args.k,
        temperature=args.temperature,
        pass1_only=args.pass1_only,
    )
