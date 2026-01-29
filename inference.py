import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from openai import AsyncOpenAI
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# Configuration
VLLM_URL = "https://snk0db6yg4zrdj-8000.proxy.runpod.net/v1"  # Update this
MODEL_NAME = "/workspace/TinyReasoner/checkpoint/ck-2800"

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

# Tuning parameters
MAX_CONCURRENT = 100  # With DP=4 and long sequences, start conservative
NUM_GENERATIONS = 1
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 20  # Save checkpoint every N completions


@dataclass
class RequestMetrics:
    total_time: float
    tokens_generated: int
    prompt_tokens: int = 0
    success: bool = True


@dataclass
class AggregateStats:
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    latencies: list = field(default_factory=list)
    wall_time: float = 0.0

    def add(self, metrics: RequestMetrics):
        self.total_requests += 1
        if metrics.success and metrics.tokens_generated > 0:
            self.successful_requests += 1
            self.total_tokens_generated += metrics.tokens_generated
            self.total_prompt_tokens += metrics.prompt_tokens
            self.latencies.append(metrics.total_time)

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
            "avg_output_tokens": self.total_tokens_generated / self.successful_requests,
        }


client = AsyncOpenAI(base_url=VLLM_URL, api_key="dummy", timeout=300)


async def generate_single(prompt: str, max_tokens: int = 30768) -> tuple[str, RequestMetrics]:
    """Generate a single completion with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            start = time.perf_counter()

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )

            total_time = time.perf_counter() - start
            result = response.choices[0].message.content or ""
            usage = response.usage

            return result, RequestMetrics(
                total_time=total_time,
                tokens_generated=usage.completion_tokens if usage else 0,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                success=True,
            )

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"\nRetry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"\nFailed after {MAX_RETRIES} attempts: {e}")
                return "", RequestMetrics(
                    total_time=0,
                    tokens_generated=0,
                    success=False,
                )


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


async def process_all(
    prompts: list[str],
    indices: list[int],
    checkpoint: dict,
    stats: AggregateStats,
    checkpoint_path: str,
) -> None:
    """Process all prompts using a continuous queue approach."""

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    checkpoint_lock = asyncio.Lock()
    completed_count = 0
    last_checkpoint_count = 0

    async def process_one(idx: int) -> RequestMetrics:
        nonlocal completed_count, last_checkpoint_count

        prompt = prompts[idx]
        async with semaphore:
            # Generate all responses for this prompt
            responses = []
            metrics_list = []

            for _ in range(NUM_GENERATIONS):
                result, metrics = await generate_single(prompt)
                responses.append(result)
                metrics_list.append(metrics)

            # Use the first generation's metrics as representative
            rep_metrics = metrics_list[0]

            # Update checkpoint
            async with checkpoint_lock:
                checkpoint["results"][str(idx)] = responses
                checkpoint["completed_indices"].append(idx)
                checkpoint["completed_set"].add(idx)
                completed_count += 1

                # Periodic checkpoint save
                if completed_count - last_checkpoint_count >= CHECKPOINT_INTERVAL:
                    save_checkpoint(checkpoint, checkpoint_path)
                    last_checkpoint_count = completed_count

            return rep_metrics

    # Create all tasks
    tasks = [process_one(idx) for idx in indices]

    # Process with progress bar using as_completed for real-time updates
    with tqdm(total=len(tasks), desc="Generating", unit="problem") as pbar:
        for coro in asyncio.as_completed(tasks):
            metrics = await coro
            stats.add(metrics)
            pbar.update(1)

            # Update progress bar postfix with live stats
            if stats.successful_requests > 0:
                elapsed = time.perf_counter() - pbar.start_t
                pbar.set_postfix({
                    "ok": f"{stats.successful_requests}/{stats.total_requests}",
                    "tok/s": f"{stats.total_tokens_generated / elapsed:.0f}",
                })

    # Final checkpoint save
    save_checkpoint(checkpoint, checkpoint_path)


async def main(benchmark: str):
    bench = BENCHMARKS[benchmark]
    ds_name, ds_config, ds_split = bench["dataset"]
    OUTPUT_PATH = bench["output_path"]
    CHECKPOINT_PATH = bench["checkpoint_path"]

    print(f"Loading {benchmark} dataset...")
    dataset = load_dataset(ds_name, ds_config, split=ds_split) if ds_config else load_dataset(ds_name, split=ds_split)
    prompts = dataset[bench["prompt_col"]]
    total_problems = len(prompts)

    print(f"Loaded {total_problems} problems")
    print(f"Config: MAX_CONCURRENT={MAX_CONCURRENT}, NUM_GENERATIONS={NUM_GENERATIONS}")

    # Load checkpoint
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    completed_set = checkpoint["completed_set"]

    if completed_set:
        print(f"Resuming: {len(completed_set)}/{total_problems} already done")

    # Find remaining work
    remaining_indices = [i for i in range(total_problems) if i not in completed_set]

    if not remaining_indices:
        print("All problems already completed!")
    else:
        total_requests = len(remaining_indices) * NUM_GENERATIONS
        print(f"Processing {len(remaining_indices)} problems ({total_requests} total requests)")

        stats = AggregateStats()
        start_time = time.perf_counter()

        try:
            await process_all(prompts, remaining_indices, checkpoint, stats, CHECKPOINT_PATH)
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            print(f"Progress saved: {len(checkpoint['completed_set'])}/{total_problems} completed")
            return
        except Exception as e:
            print(f"\n\nError: {e}")
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            print(f"Progress saved: {len(checkpoint['completed_set'])}/{total_problems} completed")
            raise

        stats.wall_time = time.perf_counter() - start_time

        # Print stats
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        summary = stats.summary()
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    # Build final dataset
    print("\nBuilding final dataset...")
    results = checkpoint["results"]
    all_responses = [results.get(str(i), [""]) for i in range(total_problems)]

    dataset = dataset.add_column("responses", all_responses)
    dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

    # Cleanup checkpoint
    checkpoint_file = Path(CHECKPOINT_PATH)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Cleaned up checkpoint file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math500", choices=BENCHMARKS.keys())
    args = parser.parse_args()
    asyncio.run(main(args.benchmark))