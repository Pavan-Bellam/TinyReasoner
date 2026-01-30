import argparse
import subprocess
import threading
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "open-r1/Qwen2.5-Math-7B-RoPE-300k"
OUTPUT_DIR = "./ckpts/"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"

SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Solve the problem step by step."

MAX_SEQ_LENGTH = 32768
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8  # 4 GPUs × 4 batch × 32 accum = 512
LEARNING_RATE = 4e-5  
WEIGHT_DECAY = 0.01  
NUM_EPOCHS = 1
WARMUP_STEPS  = 10  # ~3% warmup

S3_CKPT_PATH = "s3://tinyreasoner-1437/stage-1/ckpts/"


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


def main(resume_from: str | None = None):
    print('Loading Model')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
	 trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("loading dataset....")
    train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"Dataset size: {len(train_dataset)} examples")
    print(f"Sample messages: {train_dataset[0]['messages'][:1]}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        gradient_checkpointing=True,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        seed=42,
        report_to="wandb",
        run_name="qwen3b-math-sft-stage1",
	dataset_num_proc = 128,
	weight_decay=WEIGHT_DECAY
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[S3UploadCallback(S3_CKPT_PATH)],
    )

    print("Starting Training")
    trainer.train(resume_from_checkpoint=resume_from)

    print("Saving Model")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    upload_to_s3(f"{OUTPUT_DIR}/final", f"{S3_CKPT_PATH}final/", blocking=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    main(resume_from=args.resume)
