from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
ds = load_from_disk("./openmath_tok_simple")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

def test_config(name, model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    
    batch = data_collator([ds[i] for i in range(4)])
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    labels = batch['labels'].to(model.device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    
    nan_count = sum(1 for n, p in model.named_parameters() 
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))
    print(f"{name}: loss={outputs.loss.item():.4f}, NaN params={nan_count}")
    
    del model
    torch.cuda.empty_cache()

# Test 1: No flash attention
print("Test 1: bf16, no flash attention")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
test_config("bf16 + sdpa", model)

# Test 2: Single GPU
print("\nTest 2: bf16, single GPU")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
test_config("bf16 + single GPU", model)

# Test 3: fp32
print("\nTest 3: fp32")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.float32).to("cuda")
test_config("fp32", model)