"""Load SFT checkpoint weights into base model and save as a standalone model."""
import argparse
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint dir (contains model.safetensors)")
    parser.add_argument("--output", type=str, default="./r1-distill-qwen-1.5b-math", help="Output path for the exported model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    model_name = tcfg["model_name"]

    print(f"Loading base model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    config.max_position_embeddings = tcfg["max_position_embeddings"]
    config.rope_theta = tcfg["rope_theta"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        config=config,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = f"{args.checkpoint}/model.safetensors"
    print(f"Loading SFT weights from: {ckpt_path}")
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)

    print(f"Saving model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
