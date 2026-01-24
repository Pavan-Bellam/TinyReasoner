import re
import json
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
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

def load_model(base_model_path: str, adapter_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 1024) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

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

    # Direct comparison for int/float
    if isinstance(parsed_extracted, (int, float)) and isinstance(parsed_expected, (int, float)):
        return abs(parsed_extracted - parsed_expected) < 1e-9

    # Sympy expression comparison
    if isinstance(parsed_extracted, Expr) and isinstance(parsed_expected, Expr):
        try:
            return simplify(parsed_extracted - parsed_expected) == 0
        except Exception:
            return False

    # Mixed types - try numeric comparison
    try:
        return abs(float(parsed_extracted) - float(parsed_expected)) < 1e-9
    except (ValueError, TypeError):
        return str(parsed_extracted) == str(parsed_expected)

def evaluate(model, tokenizer, test_dataset, output_path: str | None = None):
    format_valid = 0
    correct = 0
    total = len(test_dataset)
    results = []

    for i, example in enumerate(tqdm(test_dataset)):
        response = generate_response(model, tokenizer, example["problem"])
        valid, extracted = parse_response(response)

        expected_raw = example["answer"].strip()
        is_correct = False

        if valid:
            format_valid += 1
            if compare_answers(extracted, expected_raw):
                is_correct = True
                correct += 1

        # Store result for debugging
        result = {
            "id": i,
            "problem": example["problem"],
            "expected_answer": expected_raw,
            "raw_response": response,
            "format_valid": valid,
            "extracted_answer": extracted,
            "is_correct": is_correct,
        }
        results.append(result)

        if i < 5:
            print(f"[{i}] Expected: {expected_raw}")
            print(f"[{i}] Extracted: {extracted}")
            print(f"[{i}] Correct: {is_correct}")
            print(response)
            print('-------------------------------------\n')

    # Summary
    summary = {
        "total": total,
        "format_valid": format_valid,
        "format_accuracy": format_valid / total,
        "correct": correct,
        "answer_accuracy": correct / total,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"Format Accuracy: {format_valid/total:.1%} ({format_valid}/{total})")
    print(f"Answer Accuracy: {correct/total:.1%} ({correct}/{total})")

    # Save results
    if output_path:
        output_data = {
            "summary": summary,
            "results": results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

        # Save human-readable text file
        txt_path = output_path.replace(".json", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Format Accuracy: {format_valid}/{total} ({format_valid/total:.1%})\n")
            f.write(f"Answer Accuracy: {correct}/{total} ({correct/total:.1%})\n")
            f.write(f"\n{'='*80}\n\n")

            for r in results:
                status = "✓ CORRECT" if r["is_correct"] else ("✗ WRONG" if r["format_valid"] else "✗ BAD FORMAT")
                f.write(f"[{r['id']}] {status}\n")
                f.write(f"Problem: {r['problem']}\n")
                f.write(f"Expected: {r['expected_answer']}\n")
                f.write(f"Extracted: {r['extracted_answer']}\n")
                f.write(f"--- Raw Response ---\n")
                f.write(f"{r['raw_response']}\n")
                f.write(f"\n{'-'*80}\n\n")

        print(f"Readable results saved to: {txt_path}")

    return summary, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Path to save detailed results JSON (e.g., eval_results/run1.json)")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N examples")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)
    test_data = load_from_disk(args.test_data)

    if args.limit:
        test_data = test_data.select(range(min(args.limit, len(test_data))))
        print(f"Evaluating on {len(test_data)} examples")

    evaluate(model, tokenizer, test_data, output_path=args.output)