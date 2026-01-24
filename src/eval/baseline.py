import json
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "data/math_test"
OUTPUT_PATH = "results/baseline_results.jsonl"
BATCH_SIZE = 16  # Start here, increase if VRAM allows
MAX_NEW_TOKENS = 1024


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} element from a string."""
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


def remove_boxed(s: str) -> str | None:
    """Remove \\boxed{} wrapper, return inner content."""
    if s is None:
        return None
    if s.startswith("\\boxed{"):
        return s[7:-1]
    if s.startswith("\\fbox{"):
        return s[6:-1]
    return s


def extract_answer(text: str) -> str | None:
    """Extract final answer from model output."""
    boxed = last_boxed_only_string(text)
    return remove_boxed(boxed)


def normalize_answer(s: str | None) -> str:
    """Basic normalization for comparison."""
    if s is None:
        return ""
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\:", "")
    s = s.replace("\\quad", "")
    s = s.replace("\\qquad", "")
    return s.lower()


def is_correct(pred: str | None, target: str | None) -> bool:
    """Check if prediction matches target."""
    if pred is None or target is None:
        return False
    return normalize_answer(pred) == normalize_answer(target)


def build_prompt(question: str) -> str:
    """Build prompt for the model."""
    return f"""Solve the following math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {question}

Solution:"""


def main():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading dataset: {DATA_PATH}")
    dataset = load_from_disk(DATA_PATH)
    print(f"Dataset size: {len(dataset)}")

    results = []
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    print(f"\nRunning evaluation with batch size {BATCH_SIZE}...")
    
    num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_start in tqdm(range(0, len(dataset), BATCH_SIZE), total=num_batches):
        batch_end = min(batch_start + BATCH_SIZE, len(dataset))
        batch = dataset[batch_start:batch_end]
        
        prompts = []
        for question in batch["problem"]:
            prompt = build_prompt(question)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        for i in range(len(prompts)):
            response = tokenizer.decode(
                outputs[i][input_lengths[i]:],
                skip_special_tokens=True
            )
            
            idx = batch_start + i
            expected = batch["answer"][i]
            level = batch["level"][i]
            subject = batch["type"][i]
            
            extracted = extract_answer(response)
            correct = is_correct(extracted, expected)
            
            results.append({
                "idx": idx,
                "problem": batch["problem"][i],
                "expected": expected,
                "extracted": extracted,
                "response": response,
                "correct": correct,
                "level": level,
                "type": subject,
            })
            
            stats[subject][level]["total"] += 1
            if correct:
                stats[subject][level]["correct"] += 1

    # Save detailed results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved detailed results to {OUTPUT_PATH}")

    # Build results table
    levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    subjects = sorted(stats.keys())

    table_data = []
    for subject in subjects:
        row = {"Subject": subject}
        subject_correct = 0
        subject_total = 0
        for level in levels:
            s = stats[subject][level]
            if s["total"] > 0:
                acc = s["correct"] / s["total"] * 100
                row[level] = f"{acc:.1f}%"
            else:
                row[level] = "-"
            subject_correct += s["correct"]
            subject_total += s["total"]
        row["Overall"] = f"{subject_correct / subject_total * 100:.1f}%" if subject_total > 0 else "-"
        table_data.append(row)

    overall_row = {"Subject": "OVERALL"}
    for level in levels:
        level_correct = sum(stats[s][level]["correct"] for s in subjects)
        level_total = sum(stats[s][level]["total"] for s in subjects)
        if level_total > 0:
            overall_row[level] = f"{level_correct / level_total * 100:.1f}%"
        else:
            overall_row[level] = "-"
    
    total_correct = sum(r["correct"] for r in results)
    total = len(results)
    overall_row["Overall"] = f"{total_correct / total * 100:.1f}%"
    table_data.append(overall_row)

    df = pd.DataFrame(table_data)
    df = df.set_index("Subject")
    
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(f"\nModel: {MODEL_ID}")
    print(f"Dataset: {DATA_PATH} ({len(dataset)} examples)")
    print(f"\nAccuracy by Subject and Level:\n")
    print(df.to_string())
    
    print(f"\n{'=' * 80}")
    print(f"OVERALL ACCURACY: {total_correct}/{total} = {total_correct/total*100:.2f}%")
    print(f"{'=' * 80}")

    csv_path = OUTPUT_PATH.replace(".jsonl", "_table.csv")
    df.to_csv(csv_path)
    print(f"\nSaved results table to {csv_path}")


if __name__ == "__main__":
    main()