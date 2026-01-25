"""
Merge LoRA adapter into base model for vLLM serving.

Usage:
    python merge_adapter.py \
        --base-model <path_or_hf_id> \
        --adapter-path <path_to_adapter> \
        --output-path <merged_output_path> \
        [--dtype float16|bfloat16]
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16",
) -> None:
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
    )
    
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", required=True, help="Base model path or HF ID")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-path", required=True, help="Output path for merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter_path, args.output_path, args.dtype)