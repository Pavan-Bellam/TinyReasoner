#!/usr/bin/env python3
"""
GSM8K Baseline Evaluation for Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(answer: str) -> Optional[str]:
    """Normalize to canonical numeric form."""
    if answer is None:
        return None
    answer = str(answer).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        num = float(answer)
        return str(int(num)) if num == int(num) else str(num)
    except ValueError:
        return None


def extract_answer(response: str) -> Optional[str]:
    """
    Extract answer from model response.
    Priority: \boxed{} -> <answer> -> #### -> last number
    """
    if not response:
        return None
    
    # \boxed{} - Qwen's format
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return normalize_answer(match.group(1))
    
    # <answer> tags - our SFT format
    match = re.search(r"<answer>\s*([^<]+?)\s*</answer>", response, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    
    # #### - GSM8K format
    match = re.search(r"####\s*(-?[\d,.\s]+)", response)
    if match:
        return normalize_answer(match.group(1))
    
    # Fallback: last number
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response)
    return normalize_answer(numbers[-1]) if numbers else None


@dataclass 
class EvalResult:
    model: str
    timestamp: str
    num_examples: int
    accuracy: float
    correct: int


@torch.no_grad()
def evaluate(
    model_path: str,
    data_path: str = "data/gsm8k_test",
    output_dir: str = "eval_results",
    batch_size: int = 4,
    max_new_tokens: int = 512,
    subset: Optional[int] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-----------------device: {device}")
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    print(f"Loading data: {data_path}")
    dataset = load_from_disk(data_path)
    if subset:
        dataset = dataset.select(range(min(subset, len(dataset))))
    
    questions = dataset["question"]
    ground_truths = dataset["answer"]
    
    print(f"Evaluating {len(questions)} examples...")
    
    correct = 0
    results = []
    
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_q = questions[i:i + batch_size]
        batch_gt = ground_truths[i:i + batch_size]
        
        # Build chat messages with Qwen's expected format
        batch_messages = [
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q}
            ]
            for q in batch_q
        ]
        
        # Apply chat template
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        
        for j, (inp, out, gt) in enumerate(zip(inputs.input_ids, outputs, batch_gt)):
            response = tokenizer.decode(out[len(inp):], skip_special_tokens=True)
            pred = extract_answer(response)
            is_correct = pred == normalize_answer(gt)
            
            if is_correct:
                correct += 1
            
            results.append({
                "idx": i + j,
                "question": batch_q[j],
                "ground_truth": gt,
                "predicted": pred,
                "correct": is_correct,
                "response": response,
            })
    
    accuracy = correct / len(questions)
    
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(questions)})")
    print(f"{'='*50}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = Path(model_path).name or model_path.replace("/", "_")
    
    result = EvalResult(
        model=model_path,
        timestamp=timestamp,
        num_examples=len(questions),
        accuracy=accuracy,
        correct=correct,
    )
    
    with open(output_dir / f"{model_short}_{timestamp}.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    with open(output_dir / f"{model_short}_{timestamp}_preds.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Saved to {output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/gsm8k_test")
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        subset=args.subset,
    )