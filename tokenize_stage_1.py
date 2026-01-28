import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DATASET_NAME = "nvidia/OpenMathInstruct-2"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train_2M"

SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Solve the problem step by step."


def main(out_path: str, max_len: int, num_proc: int, batch_size: int):

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print("Dataset size:", len(ds))


    # --------------------------------------------------
    # 1) Build full + prefix text
    # --------------------------------------------------

    def build_text(batch):

        full_texts = []
        prefix_texts = []

        for p, s in zip(batch["problem"], batch["generated_solution"]):

            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
                {"role": "assistant", "content": s},
            ]

            # Full conversation
            full = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )

            # System + user only (boundary)
            prefix = tok.apply_chat_template(
                msgs[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_texts.append(full)
            prefix_texts.append(prefix)

        return {
            "full_text": full_texts,
            "prefix_text": prefix_texts,
        }


    print("Building chat text...")
    ds = ds.map(
        build_text,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Formatting",
    )


    # --------------------------------------------------
    # 2) Tokenize + build labels
    # --------------------------------------------------

    def tokenize_and_mask(batch):

        full = tok(
            batch["full_text"],
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=False,
        )

        prefix = tok(
            batch["prefix_text"],
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = []
        attention_mask = []
        labels = []

        for ids, pids in zip(full["input_ids"], prefix["input_ids"]):

            boundary = min(len(pids), len(ids))

            lab = ids.copy()
            for i in range(boundary):
                lab[i] = -100

            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            labels.append(lab)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    print("Tokenizing + masking...")
    ds = ds.map(
        tokenize_and_mask,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )


    # --------------------------------------------------
    # 3) Save
    # --------------------------------------------------

    print("Final dataset:", ds)
    print("Columns:", ds.column_names)

    os.makedirs(out_path, exist_ok=True)
    ds.save_to_disk(out_path)

    print("Saved to:", out_path)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--out", type=str, default="./openmath_tok_simple")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--num_proc", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--batch_size", type=int, default=512)

    args = ap.parse_args()

    main(
        out_path=args.out,
        max_len=args.max_len,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
    )