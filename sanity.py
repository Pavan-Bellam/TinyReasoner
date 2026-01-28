"""
Pure HuggingFace inference - no vLLM
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load checkpoint
CHECKPOINT_PATH = "./qwen3b-math-sft/checkpoint-2800"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Test question
question = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"

# Format as chat
messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("=" * 50)
print("INPUT:")
print(text)
print("=" * 50)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("OUTPUT:")
print(response)
