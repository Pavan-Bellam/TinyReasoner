import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./qwen3b-math-sft-stage1"
TOKENIZED_DATASET_PATH = "./openmath_tok_simple"

MAX_SEQ_LENGTH = 4096
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATE = 2e-6
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1
WARMUP_STEPS = 10

def main(resume_from: str | None = None):
    print('Loading Model')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading pre-tokenized dataset...")
    train_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
    print(f"Dataset size: {len(train_dataset)} examples")
    
    # Sanity check
    print(f"Sample input_ids shape: {len(train_dataset[0]['input_ids'])}")
    print(f"Sample decoded: {tokenizer.decode(train_dataset[0]['input_ids'][:100])}")
    
    # Check labels exist
    if "labels" not in train_dataset.column_names:
        print("Adding labels column (copy of input_ids)...")
        train_dataset = train_dataset.map(
            lambda x: {"labels": x["input_ids"].copy()},
            num_proc=32
        )
    
    # DataCollator handles padding to max length in batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # Efficient for tensor cores
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        seed=42,
        report_to="wandb",
        run_name="qwen3b-math-sft-stage1",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting Training")
    trainer.train(resume_from_checkpoint=resume_from)
    
    print("Saving Model")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(resume_from=args.resume)