import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./qwen3b-math-sft-stage1"
DATASET_NAME = "nvidia/OpenMathInstruct-2"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train_1M"

SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Solve the problem step by step."

MAX_SEQ_LENGTH = 4096
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32  # 4 GPUs × 4 batch × 32 accum = 512
LEARNING_RATE = 2e-6  
WEIGHT_DECAY = 0.01  
NUM_EPOCHS = 1
WARMUP_STEPS  = 100  # ~3% warmup

def main(resume_from: str | None = None):
    print('Loading Model')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("loading dataset....")
    train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"Dataset size: {len(train_dataset)} examples")

    def format_to_messages(example):
        example["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["generated_solution"]},
        ]
        return example

    train_dataset = train_dataset.map(format_to_messages)
    print(f"Sample messages: {train_dataset[0]['messages']}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        gradient_checkpointing=True,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        seed=42,
        report_to="wandb",
        run_name="qwen3b-math-sft-stage1",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting Training")
    trainer.train(resume_from_checkpoint=resume_from)

    print("Saving Model")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    main(resume_from=args.resume)
