from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

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

        eos = tokenizer.eos_token_id
        token_ids = token_ids + [eos]

        labels = token_ids.copy()
        labels[:prefix_length] = [-100] * prefix_length

        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
            "labels": labels,
        }

    return tokenize_example


def main(model_name: str, val_ratio: float=0.05, max_length=4096):
    ds = load_dataset("qingy2024/QwQ-LongCoT-Verified-130K", "verified")
    ds = ds["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenize_fn = get_tokenize_fn(tokenizer=tokenizer)

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    before_count = len(ds)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length)
    after_count = len(ds)
    print(f"Filtered: {before_count} -> {after_count} ({before_count - after_count} removed)")

    split = ds.train_test_split(test_size=val_ratio, seed=42)
    train_ds = split['train']
    val_ds = split['test']

    val_ds.save_to_disk('./data/qwq/val')
    train_ds.save_to_disk('./data/qwq/train')



if __name__ == "__main__":
    main(model_name="Qwen/Qwen2.5-3B-Instruct")
