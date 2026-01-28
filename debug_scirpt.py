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

# THE FIX: use_reentrant=False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)

batch = data_collator([ds[i] for i in range(4)])
input_ids = batch['input_ids'].to(model.device)
attention_mask = batch['attention_mask'].to(model.device)
labels = batch['labels'].to(model.device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(f"Forward loss: {outputs.loss.item()}")

outputs.loss.backward()

nan_count = sum(1 for n, p in model.named_parameters() 
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))
print(f"NaN/Inf gradient params: {nan_count}")

if nan_count == 0:
    print("✓ Gradients are clean")