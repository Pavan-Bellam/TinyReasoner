import argparse
import subprocess
import threading
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, AutoConfig
from trl import SFTTrainer, SFTConfig


def upload_to_s3(local_path: str, s3_path: str, blocking: bool = False):
    """Upload a directory to S3 using aws cli."""
    def _upload():
        cmd = ["aws", "s3", "sync", local_path, s3_path, "--quiet"]
        print(f"Uploading {local_path} -> {s3_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"S3 upload failed: {result.stderr}")
        else:
            print(f"Upload complete: {local_path}")

    if blocking:
        _upload()
    else:
        threading.Thread(target=_upload, daemon=True).start()


class S3UploadCallback(TrainerCallback):
    def __init__(self, s3_base_path: str):
        self.s3_base_path = s3_base_path

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        ckpt_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        s3_dest = f"{self.s3_base_path}checkpoint-{state.global_step}/"
        upload_to_s3(ckpt_dir, s3_dest)
        return control


def main(resume_from: str | None = None, config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    system_prompt = cfg["system_prompt"]
    tcfg = cfg["training"]

    model_name = tcfg["model_name"]
    output_dir = tcfg["output_dir"]
    s3_ckpt_path = tcfg["s3_checkpoint_path"]

    print('Loading Model')
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

    print("loading dataset....")
    train_dataset = load_dataset(tcfg["dataset_name"], tcfg["dataset_config"], split=tcfg["dataset_split"])
    print(f"Dataset size: {len(train_dataset)} examples")

    def add_system_prompt(example):
        system_msg = {"role": "system", "content": system_prompt}
        example["messages"] = [system_msg] + example["messages"]
        return example

    train_dataset = train_dataset.map(add_system_prompt, num_proc=tcfg["dataset_num_proc"])
    print(f"Sample messages: {train_dataset[0]['messages'][:1]}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        max_length=tcfg["max_seq_length"],
        packing=tcfg["packing"],
        gradient_checkpointing=True,
        bf16=True,
        lr_scheduler_type="linear",
        warmup_steps=tcfg["warmup_steps"],
        logging_steps=tcfg["logging_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        seed=tcfg["seed"],
        report_to="wandb",
        run_name=tcfg["run_name"],
        dataset_num_proc=tcfg["dataset_num_proc"],
        weight_decay=tcfg["weight_decay"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[S3UploadCallback(s3_ckpt_path)],
    )

    print("Starting Training")
    trainer.train(resume_from_checkpoint=resume_from)

    print("Saving Model")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    upload_to_s3(f"{output_dir}/final", f"{s3_ckpt_path}final/", blocking=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    main(resume_from=args.resume, config_path=args.config)
