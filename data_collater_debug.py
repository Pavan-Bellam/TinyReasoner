from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

ds = load_from_disk("./openmath_tok_simple")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

# Simulate a batch
batch_samples = [ds[i] for i in range(4)]
batch = data_collator(batch_samples)

print(f"input_ids shape: {batch['input_ids'].shape}")
print(f"labels shape: {batch['labels'].shape}")
print(f"Labels min: {batch['labels'].min()}")  # Should be -100
print(f"Labels max: {batch['labels'].max()}")
print(f"Pad token id: {tokenizer.pad_token_id}")
print(f"-100 count in labels: {(batch['labels'] == -100).sum()}")
print(f"pad_token count in input_ids: {(batch['input_ids'] == tokenizer.pad_token_id).sum()}")

# Forward pass with batched data
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

with torch.no_grad():
    outputs = model(
        input_ids=batch['input_ids'].to(model.device),
        attention_mask=batch['attention_mask'].to(model.device),
        labels=batch['labels'].to(model.device)
    )
    print(f"Batched loss: {outputs.loss.item()}")