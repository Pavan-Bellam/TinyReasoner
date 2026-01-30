import argparse
import json
import time
import torch
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

# Configuration
BASE_MODEL = "amd/Instella-3B-Instruct"
SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Solve the problem step by step."

BENCHMARKS = {
    "math500": {
        "dataset": ("HuggingFaceH4/MATH-500", None, "test"),
        "prompt_col": "problem",
        "output_path": "./math500_results",
        "checkpoint_path": "./math500_checkpoint.json",
    },
    "gsm8k": {
        "dataset": ("openai/gsm8k", "main", "test"),
        "prompt_col": "question",
        "output_path": "./gsm8k_results",
        "checkpoint_path": "./gsm8k_checkpoint.json",
    },
}

NUM_GENERATIONS = 1
MAX_NEW_TOKENS = 30768
CHECKPOINT_INTERVAL = 20
BATCH_SIZE = 4


@dataclass
class AggregateStats:
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens_generated: int = 0
    latencies: list = field(default_factory=list)
    wall_time: float = 0.0

    def add(self, latency: float, tokens: int, success: bool = True):
        self.total_requests += 1
        if success and tokens > 0:
            self.successful_requests += 1
            self.total_tokens_generated += tokens
            self.latencies.append(latency)

    def add_batch(self, latency: float, token_counts: list[int]):
        batch_size = len(token_counts)
        self.total_requests += batch_size
        for tokens in token_counts:
            if tokens > 0:
                self.successful_requests += 1
                self.total_tokens_generated += tokens
        self.latencies.extend([latency / batch_size] * batch_size)

    def summary(self) -> dict:
        if not self.latencies:
            return {"error": "No successful requests"}

        latencies_ms = [t * 1000 for t in self.latencies]
        return {
            "wall_time_s": self.wall_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate_pct": self.successful_requests / self.total_requests * 100,
            "throughput_req_per_s": self.successful_requests / self.wall_time if self.wall_time > 0 else 0,
            "tokens_per_s": self.total_tokens_generated / self.wall_time if self.wall_time > 0 else 0,
            "latency_mean_ms": np.mean(latencies_ms),
            "latency_p50_ms": np.percentile(latencies_ms, 50),
            "latency_p90_ms": np.percentile(latencies_ms, 90),
            "latency_p99_ms": np.percentile(latencies_ms, 99),
            "avg_output_tokens": self.total_tokens_generated / self.successful_requests if self.successful_requests > 0 else 0,
        }

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "latencies": self.latencies,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AggregateStats":
        stats = cls()
        stats.total_requests = d["total_requests"]
        stats.successful_requests = d["successful_requests"]
        stats.total_tokens_generated = d["total_tokens_generated"]
        stats.latencies = d["latencies"]
        return stats

    def merge(self, other: "AggregateStats"):
        self.total_requests += other.total_requests
        self.successful_requests += other.successful_requests
        self.total_tokens_generated += other.total_tokens_generated
        self.latencies.extend(other.latencies)


def load_checkpoint(path: str) -> dict:
    checkpoint_file = Path(path)
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            data["completed_set"] = set(data.get("completed_indices", []))
            return data
    return {"completed_indices": [], "completed_set": set(), "results": {}}


def save_checkpoint(checkpoint: dict, path: str):
    save_data = {
        "completed_indices": checkpoint["completed_indices"],
        "results": checkpoint["results"],
    }
    with open(path, "w") as f:
        json.dump(save_data, f)


def build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], device) -> tuple[list[str], float, list[int]]:
    """Generate responses for a batch of prompts."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    elapsed = time.perf_counter() - start

    responses = []
    token_counts = []
    for output, input_len in zip(outputs, input_lens):
        new_tokens = output[input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
        token_counts.append(len(new_tokens))

    return responses, elapsed, token_counts


def worker_fn(
    rank: int,
    world_size: int,
    model_path: str,
    indices: list[int],
    prompts: list[str],
    return_dict: dict,
    batch_size: int,
    checkpoint_interval: int,
):
    """Worker function that runs on each GPU."""
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"[GPU {rank}] Model loaded. Processing {len(indices)} problems.")

    results = {}
    stats = AggregateStats()

    # Create batches for this worker
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

    try:
        for batch_indices in tqdm(batches, desc=f"GPU {rank}", position=rank, leave=True):
            batch_prompts = [build_prompt(tokenizer, prompts[idx]) for idx in batch_indices]

            # Handle NUM_GENERATIONS > 1
            for _ in range(NUM_GENERATIONS):
                responses, elapsed, token_counts = generate_batch(model, tokenizer, batch_prompts, device)
                stats.add_batch(elapsed, token_counts)

                for idx, response in zip(batch_indices, responses):
                    if idx not in results:
                        results[idx] = []
                    results[idx].append(response)

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        # Still return partial results
        return_dict[rank] = {"results": results, "stats": stats.to_dict(), "error": str(e)}
        return

    return_dict[rank] = {"results": results, "stats": stats.to_dict(), "error": None}
    print(f"[GPU {rank}] Done. Processed {len(results)} problems.")


def run_benchmark(benchmark_name: str, bench: dict, model_path: str, num_gpus: int, batch_size: int):
    ds_name, ds_config, ds_split = bench["dataset"]
    OUTPUT_PATH = bench["output_path"]
    CKPT_PATH = bench["checkpoint_path"]

    print(f"\n{'='*50}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"{'='*50}")

    print(f"Loading {benchmark_name} dataset...")
    dataset = load_dataset(ds_name, ds_config, split=ds_split) if ds_config else load_dataset(ds_name, split=ds_split)
    prompts = list(dataset[bench["prompt_col"]])  # Convert to plain list for pickling
    total_problems = len(prompts)
    print(f"Loaded {total_problems} problems")

    # Load checkpoint
    ckpt = load_checkpoint(CKPT_PATH)
    completed_set = ckpt["completed_set"]

    if completed_set:
        print(f"Resuming: {len(completed_set)}/{total_problems} already done")

    remaining_indices = [i for i in range(total_problems) if i not in completed_set]

    if not remaining_indices:
        print("All problems already completed!")
    else:
        print(f"Processing {len(remaining_indices)} problems across {num_gpus} GPUs")

        # Split indices across GPUs (interleaved for balance)
        chunks = [[] for _ in range(num_gpus)]
        for i, idx in enumerate(remaining_indices):
            chunks[i % num_gpus].append(idx)

        for i, chunk in enumerate(chunks):
            print(f"  GPU {i}: {len(chunk)} problems")

        # Spawn workers
        manager = mp.Manager()
        return_dict = manager.dict()

        start_time = time.perf_counter()

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker_fn,
                args=(rank, num_gpus, model_path, chunks[rank], prompts, return_dict, batch_size, CHECKPOINT_INTERVAL),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n\nInterrupted! Terminating workers...")
            for p in processes:
                p.terminate()
            print("Saving partial results...")

        wall_time = time.perf_counter() - start_time

        # Merge results from all workers
        merged_stats = AggregateStats()
        errors = []

        for rank in range(num_gpus):
            if rank not in return_dict:
                print(f"[WARNING] No results from GPU {rank}")
                continue

            worker_data = return_dict[rank]
            worker_results = worker_data["results"]
            worker_stats = AggregateStats.from_dict(worker_data["stats"])

            if worker_data["error"]:
                errors.append(f"GPU {rank}: {worker_data['error']}")

            merged_stats.merge(worker_stats)

            for idx, responses in worker_results.items():
                ckpt["results"][str(idx)] = responses
                if idx not in ckpt["completed_set"]:
                    ckpt["completed_indices"].append(idx)
                    ckpt["completed_set"].add(idx)

        # Save checkpoint
        save_checkpoint(ckpt, CKPT_PATH)

        merged_stats.wall_time = wall_time

        # Print results
        print("\n" + "-" * 50)
        print(f"RESULTS ({benchmark_name})")
        print("-" * 50)

        if errors:
            print("\nErrors encountered:")
            for err in errors:
                print(f"  {err}")
            print()

        summary = merged_stats.summary()
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

        print(f"\nCompleted: {len(ckpt['completed_set'])}/{total_problems}")

    # Build final dataset
    print(f"\nBuilding final dataset for {benchmark_name}...")
    results = ckpt["results"]
    all_responses = [results.get(str(i), [""]) for i in range(total_problems)]

    dataset = dataset.add_column("responses", all_responses)
    dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

    checkpoint_file = Path(CKPT_PATH)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Cleaned up checkpoint file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a training checkpoint to load instead of the base model"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()

    model_path = args.checkpoint if args.checkpoint else BASE_MODEL
    num_gpus = args.num_gpus
    batch_size = args.batch_size

    print(f"Model: {model_path}")
    print(f"Using {num_gpus} GPUs with batch size {batch_size} per GPU")

    mp.set_start_method("spawn", force=True)

    for benchmark_name, bench in BENCHMARKS.items():
        try:
            run_benchmark(benchmark_name, bench, model_path, num_gpus, batch_size)
        except KeyboardInterrupt:
            print("\nStopping early due to interrupt.")
            return


if __name__ == "__main__":
    main()