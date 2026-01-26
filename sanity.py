from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "/workspace/TinyReasoner/checkpointing/checkpoint-500")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
streamer = TextStreamer(tokenizer)

messages = [
    {"role": "system", "content": "Solve the math problem. Show your reasoning step by step."},
    {"role": "user", "content": "What is 15 + 27?"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=200, do_sample=False, streamer=streamer)
