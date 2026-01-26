import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = """Solve the math problem. Show your reasoning step by step."""

def get_tokenize_fn(tokenizer):
    def tokenize_example(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["qwq"]},
        ]

        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        prefix_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        prefix_length = len(prefix_ids)

        labels = token_ids.copy()
        labels[:prefix_length] = [-100] * prefix_length

        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
            "labels": labels,
            "length": len(token_ids), # Track length for stats
        }

    return tokenize_example

def print_stats(lengths, title="Dataset"):
    print(f"\n--- {title} Statistics ---")
    print(f"Count:   {len(lengths)}")
    print(f"Max:     {np.max(lengths)}")
    print(f"Min:     {np.min(lengths)}")
    print(f"Average: {np.mean(lengths):.2f}")
    print(f"Median:  {np.median(lengths)}")
    print(f"95th %:  {np.quantile(lengths, 0.95)}")
    print(f"99th %:  {np.quantile(lengths, 0.99)}")

def main(model_name: str, val_ratio: float=0.05, max_length: int=2048):
    # Load dataset
    ds = load_dataset("qingy2024/QwQ-LongCoT-Verified-130K", "verified", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer)

    # Map and keep the 'length' column for analysis
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # Print overall stats
    all_lengths = ds["length"]
    print_stats(all_lengths, "Overall Dataset")
    before_count = len(ds)

    # Filter by max length and report counts
    ds = ds.filter(lambda x: x["length"] <= max_length)
    after_count = len(ds)
    print(f"Filtered by max_length={max_length}: {before_count} -> {after_count}")

    # Split
    split = ds.train_test_split(test_size=val_ratio, seed=42)
    train_ds = split['train']
    val_ds = split['test']

    # Final cleanup: remove 'length' column before saving if you don't want it in the final files
    train_ds = train_ds.remove_columns(["length"])
    val_ds = val_ds.remove_columns(["length"])

    val_ds.save_to_disk('./data/qwq/val')
    train_ds.save_to_disk('./data/qwq/train')

if __name__ == "__main__":
    main(model_name="Qwen/Qwen2.5-3B-Instruct")
