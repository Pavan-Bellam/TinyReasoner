from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
ds = load_from_disk("./openmath_tok_simple")

# Check 1: Valid token IDs?
sample = ds[0]
max_id = max(sample["input_ids"])
vocab_size = tokenizer.vocab_size
print(f"Max token ID: {max_id}, Vocab size: {vocab_size}")
print(f"Valid: {max_id < vocab_size}")

# Check 2: Any weird values?
print(f"Min token ID: {min(sample['input_ids'])}")
print(f"Any negative: {any(x < 0 for x in sample['input_ids'])}")

# Check 3: Decode looks right?
print(tokenizer.decode(sample["input_ids"][:200]))

# Check 4: Labels exist and match?
if "labels" in ds.column_names:
    print(f"Labels == input_ids: {sample['labels'] == sample['input_ids']}")

# Check 5: Try a forward pass
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

input_ids = torch.tensor([sample["input_ids"][:512]]).to(model.device)
labels = input_ids.clone()

with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)
    print(f"Test loss: {outputs.loss.item()}")  # Should be 1-3, not 10^14