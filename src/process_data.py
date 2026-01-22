"""
Data processing script for TinyReasoner.

Downloads MATH dataset, formats for SFT training, tokenizes, and saves.
Run this ONCE before training.

Usage:
    python src/process_data.py --model Qwen/Qwen2.5-3B-Instruct --max-length 1024
"""

import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

SUBSETS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus'
]

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


def extract_answer(solution: str) -> str | None:
    """Extract final answer from solution."""
    boxed = last_boxed_only_string(solution)
    return remove_boxed(boxed)


def format_example(example: dict) -> dict | None:
    """Format a single example as chat text. Returns None if answer extraction fails."""
    answer = extract_answer(example['solution'])
    if answer is None:
        return None

    cot = example['solution'].strip()
    assistant_content = f"<think>\n{cot}\n</think>\n<answer>\n{answer}\n</answer>"

    return {
        "system": SYSTEM_PROMPT,
        "user": example['problem'],
        "assistant": assistant_content,
    }


def process_example(example: dict, tokenizer) -> dict:
    formatted = format_example(example)
    if formatted is None:
        return {
            "prompt": None,
            "answer": None,
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

    # ---- GRPO fields ----
    grpo_messages = [
        {"role": "system", "content": formatted["system"]},
        {"role": "user", "content": formatted["user"]},
    ]
    prompt = tokenizer.apply_chat_template(
        grpo_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ground truth answer (raw LaTeX extracted from boxed)
    answer = extract_answer(example["solution"])

    # ---- SFT fields (your current behavior) ----
    sft_messages = [
        {"role": "system", "content": formatted["system"]},
        {"role": "user", "content": formatted["user"]},
        {"role": "assistant", "content": formatted["assistant"]},
    ]
    full = tokenizer.apply_chat_template(sft_messages, tokenize=False)
    prefix = tokenizer.apply_chat_template(sft_messages[:-1], tokenize=False)

    out = tokenizer(full, add_special_tokens=False)
    prefix_tokens = tokenizer(prefix, add_special_tokens=False)

    assistant_start = len(prefix_tokens["input_ids"])
    eos = tokenizer.eos_token_id

    out["input_ids"] = out["input_ids"] + [eos]
    out["attention_mask"] = out["attention_mask"] + [1]
    out["labels"] = out["input_ids"].copy()
    out["labels"][:assistant_start] = [-100] * assistant_start

    return {
        # GRPO
        "prompt": prompt,
        "answer": answer,

        # SFT
        "input_ids": out["input_ids"],
        "attention_mask": out["attention_mask"],
        "labels": out["labels"],
    }



def load_math_dataset():
    """Load and concatenate all MATH subsets."""
    print("Loading MATH dataset...")
    train_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in SUBSETS]
    test_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in SUBSETS]

    train = concatenate_datasets(train_datasets)
    test = concatenate_datasets(test_datasets)

    print(f"Raw train: {len(train)}, Raw test: {len(test)}")
    return train, test


def process_split(dataset, tokenizer, max_length: int, desc: str):
    """Process a dataset split: tokenize, filter, return."""
    print(f"\nProcessing {desc}...")
    original_size = len(dataset)

    processed = dataset.map(
        lambda x: process_example(x, tokenizer),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {desc}"
    )

    # Filter out failed examples and those exceeding max_length
    processed = processed.filter(lambda x: x["input_ids"] is not None and x["prompt"] is not None and x["answer"] is not None)
    processed = processed.filter(lambda x: len(x["input_ids"]) <= max_length)


    print(f"{desc}: {original_size} -> {len(processed)}")
    return processed


def main(model_id: str, max_length: int, val_ratio: float):
    print(f"Model: {model_id}")
    print(f"Max length: {max_length}")
    print(f"Val ratio: {val_ratio}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw data
    train_raw, test_raw = load_math_dataset()

    # Process
    train = process_split(train_raw, tokenizer, max_length, "train")
    test = process_split(test_raw, tokenizer, max_length, "test")

    # Sample val from test
    val_size = int(len(test) * val_ratio)
    val = test.shuffle(seed=42).select(range(val_size))
    print(f"Val: {len(val)} ({val_ratio*100:.0f}% of test)")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE (decoded)")
    print("=" * 60)
    sample = tokenizer.decode(train[0]["input_ids"])
    print(sample[:500] + "..." if len(sample) > 500 else sample)

    # Save
    print("\nSaving...")
    train.save_to_disk("data/train")
    val.save_to_disk("data/val")
    test.save_to_disk("data/test")

    print(f"\nDone! Saved to data/{{train,val,test}}")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MATH dataset for SFT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    main(args.model, args.max_length, args.val_ratio)
