import re
import json
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM_PROMPT = """
Return your response in this exact format:
<think>
...
</think>
<answer>...</answer>
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


def normalize_answer(answer: str) -> str | None:
    """Normalize to canonical numeric form."""
    if answer is None:
        return None
    answer = str(answer).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        num = float(answer)
        return str(int(num)) if num == int(num) else str(num)
    except ValueError:
        return None

def evaluate(model, tokenizer, test_dataset, output_path: str | None = None):
    format_valid = 0
    correct = 0
    total = len(test_dataset)
    results = []

    for i, example in enumerate(tqdm(test_dataset)):
        response = generate_response(model, tokenizer, example["question"])
        valid, extracted = parse_response(response)

        expected_raw = example["answer"].strip()
        expected_norm = normalize_answer(expected_raw)
        extracted_norm = normalize_answer(extracted) if extracted else None
        is_correct = False

        if valid:
            format_valid += 1
            # Compare normalized numeric answers
            if extracted_norm is not None and extracted_norm == expected_norm:
                is_correct = True
                correct += 1

        # Store result for debugging
        result = {
            "id": i,
            "question": example["question"],
            "expected_answer": expected_raw,
            "expected_normalized": expected_norm,
            "raw_response": response,
            "format_valid": valid,
            "extracted_answer": extracted,
            "extracted_normalized": extracted_norm,
            "is_correct": is_correct,
        }
        results.append(result)

        if i < 5:
            print(f"[{i}] Expected: {expected_raw} -> {expected_norm}")
            print(f"[{i}] Extracted: {extracted} -> {extracted_norm}")
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
                f.write(f"Question: {r['question']}\n")
                f.write(f"Expected: {r['expected_answer']} (normalized: {r['expected_normalized']})\n")
                f.write(f"Extracted: {r['extracted_answer']} (normalized: {r['extracted_normalized']})\n")
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