from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

ds = load_from_disk("./openmath_tok_simple")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Enable gradient checkpointing like your training script
model.gradient_checkpointing_enable()

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)

batch = data_collator([ds[i] for i in range(4)])
input_ids = batch['input_ids'].to(model.device)
attention_mask = batch['attention_mask'].to(model.device)
labels = batch['labels'].to(model.device)

# Forward
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(f"Forward loss: {outputs.loss.item()}")

# Backward
outputs.loss.backward()

# Check gradients
nan_params = []
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            nan_params.append(name)

if nan_params:
    print(f"NaN/Inf gradients in {len(nan_params)} params:")
    for p in nan_params[:5]:
        print(f"  {p}")
else:
    print("All gradients finite ✓")

optimizer.step()
print("Optimizer step completed")