import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./qwen3b-math-sft"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
DATASET_CONFIG = "default"

MAX_SEQ_LENGTH = 8192
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16 as i plan to use 2 gpus
LEARNING_RATE = 5e-6  # Conservative for full fine-tune
NUM_EPOCHS = 1



print('Loading Model')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False
)


print("loading dataset....")
train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
print(f"Dataset size: {len(train_dataset)} examples")



print(f"Sample messages format: {train_dataset[0]['messages'][:1]}")


training_args = SFTConfig(
    output_dir = OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_length=MAX_SEQ_LENGTH,
    packing=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    seed=42,
    report_to="wandb",
    run_name="qwen3b-math-sft",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,

)


print("Starting Training")
trainer.train()

print("Saving Model ")

trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")