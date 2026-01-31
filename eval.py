import argparse
import asyncio
import json
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import regex as re
from datasets import load_dataset
from openai import AsyncOpenAI
from sympy import Expr, Equality, simplify
from sympy.parsing.latex import parse_latex
from tqdm import tqdm

# ============================================================
# Answer parsing (from boxed latex)
# ============================================================
BOXED_PAT = re.compile(r"\\boxed\s*\{((?:[^{}]+|\{(?1)\})*)\}")
TUPLE_PAT = re.compile(r"^\\?(?:left)?[\(\[]\s*(.+?)\s*,\s*(.+?)\s*\\?(?:right)?[\)\]]$")
NUM_PAT = re.compile(r"^\s*\\?\$?\s*([+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?)\s*$", re.VERBOSE)
PMATRIX_PAT = re.compile(r"\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}", re.DOTALL)


def _parse_num(s: str):
    m = NUM_PAT.fullmatch(s)
    if not m:
        return None
    num = m.group(1)
    return float(num) if ("." in num or "e" in num.lower()) else int(num)


def _strip_latex_cruft(s: str) -> str:
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\\circ", "", s)
    s = re.sub(r"\\circ", "", s)
    s = re.sub(r"\b(euros?|dollars?|meters?|cm|kg)\b", "", s, flags=re.I)
    return s.strip()


def parse_vector(s: str):
    match = PMATRIX_PAT.fullmatch(s.strip())
    if not match:
        return None
    content = match.group(1)
    elements = re.split(r"\\\\", content)
    parsed = []
    for el in elements:
        val = parse_single_value(el.strip())
        if val is None:
            return None
        parsed.append(val)
    return tuple(parsed) if parsed else None


def parse_single_value(s: str):
    s = s.strip()
    if not s:
        return None
    num = _parse_num(s)
    if num is not None:
        return num
    try:
        expr = parse_latex(s)
        if isinstance(expr, Equality):
            expr = expr.rhs
        expr = simplify(expr)
        if expr.free_symbols:
            return expr
        evaluated = expr.evalf()
        if evaluated.is_Integer:
            return int(evaluated)
        if evaluated.is_real:
            return float(evaluated)
        return expr
    except Exception:
        pass
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", s).strip()
    return cleaned if cleaned else None


def parse_answer(text: str | None):
    if text is None:
        return None
    text = str(text)
    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None
    answer = _strip_latex_cruft(matches[-1].group(1))
    if not answer:
        return None
    vec = parse_vector(answer)
    if vec is not None:
        return vec
    tuple_match = TUPLE_PAT.fullmatch(answer)
    if tuple_match:
        left = parse_single_value(tuple_match.group(1))
        right = parse_single_value(tuple_match.group(2))
        if left is not None and right is not None:
            return (left, right)
        return None
    return parse_single_value(answer)


def extract_raw_boxed(text: str) -> str | None:
    text = str(text)
    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None
    return _strip_latex_cruft(matches[-1].group(1))


# ============================================================
# Answer comparison
# ============================================================
def compare_values(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6
    if isinstance(a, Expr) and isinstance(b, Expr):
        try:
            return simplify(a - b) == 0
        except Exception:
            return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError, AttributeError):
        pass
    return str(a).strip().lower() == str(b).strip().lower()


def compare_parsed(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(compare_values(x, y) for x, y in zip(a, b))
    if isinstance(a, tuple) or isinstance(b, tuple):
        return False
    return compare_values(a, b)


def compare_answers(text_a, text_b) -> bool:
    parsed_a = parse_answer(text_a)
    parsed_b = parse_answer(text_b)
    if parsed_a is not None and parsed_b is not None:
        return compare_parsed(parsed_a, parsed_b)
    raw_a = extract_raw_boxed(text_a)
    raw_b = extract_raw_boxed(text_b)
    if raw_a is None or raw_b is None:
        return False

    def normalize(s):
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\left|\\right", "", s)
        s = re.sub(r"\s+", "", s)
        return s.lower()

    return normalize(raw_a) == normalize(raw_b)


# ============================================================
# GSM8K ground truth parsing
# ============================================================
def parse_gsm8k_gt(answer_text: str):
    """Parse GSM8K ground truth from '#### <number>' format."""
    s = answer_text.split("####")[-1].strip()
    s = s.replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


# ============================================================
# Evaluation logic
# ============================================================
def evaluate_math500(responses: list[str], gt_answers: list[str], levels: list[str] | None = None) -> dict:
    """Evaluate MATH-500 responses against ground truth."""
    total = len(responses)
    parsed_preds = []
    parsed_gts = []
    correct = []

    for resp, gt in zip(responses, gt_answers):
        pred = parse_answer(resp)
        parsed_preds.append(pred)

        # GT may or may not have \boxed{}
        gt_parsed = parse_answer(gt)
        if gt_parsed is None and "\\boxed" not in str(gt):
            gt_parsed = parse_answer(f"\\boxed{{{gt}}}")
        parsed_gts.append(gt_parsed)

        correct.append(compare_parsed(pred, gt_parsed))

    accuracy = sum(correct) / total if total > 0 else 0
    null_preds = sum(1 for p in parsed_preds if p is None)
    null_gts = sum(1 for g in parsed_gts if g is None)
    lengths = [len(r) for r in responses]

    result = {
        "accuracy": accuracy,
        "total": total,
        "correct": sum(correct),
        "null_parsed_preds": null_preds,
        "null_parsed_gts": null_gts,
        "avg_response_length": np.mean(lengths) if lengths else 0,
        "median_response_length": np.median(lengths) if lengths else 0,
        "p90_response_length": np.percentile(lengths, 90) if lengths else 0,
    }

    # Accuracy by level if available
    if levels is not None:
        level_correct = {}
        level_total = {}
        for lvl, c in zip(levels, correct):
            level_total[lvl] = level_total.get(lvl, 0) + 1
            level_correct[lvl] = level_correct.get(lvl, 0) + (1 if c else 0)
        result["accuracy_by_level"] = {
            lvl: level_correct.get(lvl, 0) / level_total[lvl]
            for lvl in sorted(level_total.keys())
        }

    return result, correct


def evaluate_gsm8k(responses: list[str], gt_answers: list[str]) -> dict:
    """Evaluate GSM8K responses against ground truth."""
    total = len(responses)
    correct = []
    null_preds = 0
    null_gts = 0

    for resp, gt in zip(responses, gt_answers):
        pred = parse_answer(resp)
        gt_parsed = parse_gsm8k_gt(gt)

        if pred is None:
            null_preds += 1
        if gt_parsed is None:
            null_gts += 1

        # Convert pred to int for comparison
        try:
            pred_int = int(pred) if pred is not None else None
        except (ValueError, TypeError):
            pred_int = None

        correct.append(pred_int is not None and gt_parsed is not None and pred_int == gt_parsed)

    accuracy = sum(correct) / total if total > 0 else 0
    lengths = [len(r) for r in responses]

    result = {
        "accuracy": accuracy,
        "total": total,
        "correct": sum(correct),
        "null_parsed_preds": null_preds,
        "null_parsed_gts": null_gts,
        "avg_response_length": np.mean(lengths) if lengths else 0,
        "median_response_length": np.median(lengths) if lengths else 0,
        "p90_response_length": np.percentile(lengths, 90) if lengths else 0,
    }

    return result, correct


# ============================================================
# vLLM inference
# ============================================================
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


def load_checkpoint(path: str) -> dict:
    checkpoint_file = Path(path)
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            data["completed_set"] = set(data.get("completed_indices", []))
            return data
    return {"completed_indices": [], "completed_set": set(), "results": {}}


def save_checkpoint(checkpoint: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "completed_indices": checkpoint["completed_indices"],
        "results": checkpoint["results"],
    }
    with open(path, "w") as f:
        json.dump(save_data, f)


async def generate_single(client, model_name: str, prompt: str, max_retries: int, max_tokens: int = 2096) -> tuple[str, RequestMetrics]:
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
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
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\nRetry {attempt + 1}/{max_retries} after error: {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return "", RequestMetrics(total_time=0, tokens_generated=0, success=False)


async def process_all(
    client,
    model_name: str,
    prompts: list[str],
    indices: list[int],
    checkpoint: dict,
    stats: AggregateStats,
    checkpoint_path: str,
    max_concurrent: int,
    max_retries: int,
    checkpoint_interval: int,
) -> None:
    semaphore = asyncio.Semaphore(max_concurrent)
    checkpoint_lock = asyncio.Lock()
    completed_count = 0
    last_checkpoint_count = 0

    async def process_one(idx: int) -> RequestMetrics:
        nonlocal completed_count, last_checkpoint_count
        prompt = prompts[idx]
        async with semaphore:
            result, metrics = await generate_single(client, model_name, prompt, max_retries)

            async with checkpoint_lock:
                checkpoint["results"][str(idx)] = result
                checkpoint["completed_indices"].append(idx)
                checkpoint["completed_set"].add(idx)
                completed_count += 1

                if completed_count - last_checkpoint_count >= checkpoint_interval:
                    save_checkpoint(checkpoint, checkpoint_path)
                    last_checkpoint_count = completed_count

            return metrics

    tasks = [process_one(idx) for idx in indices]

    with tqdm(total=len(tasks), desc="Generating", unit="problem") as pbar:
        for coro in asyncio.as_completed(tasks):
            metrics = await coro
            stats.add(metrics)
            pbar.update(1)
            if stats.successful_requests > 0:
                elapsed = time.perf_counter() - pbar.start_t
                pbar.set_postfix({
                    "ok": f"{stats.successful_requests}/{stats.total_requests}",
                    "tok/s": f"{stats.total_tokens_generated / elapsed:.0f}",
                })

    save_checkpoint(checkpoint, checkpoint_path)


# ============================================================
# Report management
# ============================================================
def load_report(path: str) -> list:
    report_file = Path(path)
    if report_file.exists():
        with open(report_file, "r") as f:
            return json.load(f)
    return []


def save_report(report: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


# ============================================================
# Main
# ============================================================
async def run_benchmark(benchmark_name: str, bench: dict, client, model_name: str, eval_cfg: dict, run_name: str):
    ds_name = bench["dataset"]
    ds_config = bench.get("dataset_config")
    ds_split = bench["dataset_split"]
    OUTPUT_PATH = bench["output_path"]
    CKPT_PATH = bench["checkpoint_path"]

    print(f"\n{'='*50}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"{'='*50}")

    # Load dataset
    print(f"Loading {benchmark_name} dataset...")
    dataset = load_dataset(ds_name, ds_config, split=ds_split) if ds_config else load_dataset(ds_name, split=ds_split)
    prompts = dataset[bench["prompt_col"]]
    gt_answers = dataset[bench["gt_col"]]
    total_problems = len(prompts)
    print(f"Loaded {total_problems} problems")

    # --- Generation phase ---
    checkpoint = load_checkpoint(CKPT_PATH)
    completed_set = checkpoint["completed_set"]

    if completed_set:
        print(f"Resuming: {len(completed_set)}/{total_problems} already done")

    remaining_indices = [i for i in range(total_problems) if i not in completed_set]

    gen_summary = {}
    if not remaining_indices:
        print("All problems already generated!")
    else:
        print(f"Generating {len(remaining_indices)} problems")
        stats = AggregateStats()
        start_time = time.perf_counter()

        try:
            await process_all(
                client, model_name, prompts, remaining_indices,
                checkpoint, stats, CKPT_PATH,
                max_concurrent=eval_cfg["max_concurrent"],
                max_retries=eval_cfg["max_retries"],
                checkpoint_interval=eval_cfg["checkpoint_interval"],
            )
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            save_checkpoint(checkpoint, CKPT_PATH)
            print(f"Progress saved: {len(checkpoint['completed_set'])}/{total_problems}")
            raise
        except Exception as e:
            print(f"\n\nError: {e}")
            save_checkpoint(checkpoint, CKPT_PATH)
            print(f"Progress saved: {len(checkpoint['completed_set'])}/{total_problems}")
            raise

        stats.wall_time = time.perf_counter() - start_time
        gen_summary = stats.summary()

        print("\n" + "-" * 50)
        print("Generation Stats")
        print("-" * 50)
        for key, value in gen_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    # Collect all responses (single string per problem now)
    results = checkpoint["results"]
    all_responses = [results.get(str(i), "") for i in range(total_problems)]

    # Save dataset with responses
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out_dataset = dataset.add_column("response", all_responses)

    # --- Evaluation phase ---
    print(f"\nEvaluating {benchmark_name}...")

    if benchmark_name == "math500":
        levels = list(dataset["level"]) if "level" in dataset.column_names else None
        eval_result, correct_list = evaluate_math500(all_responses, list(gt_answers), levels)
    elif benchmark_name == "gsm8k":
        eval_result, correct_list = evaluate_gsm8k(all_responses, list(gt_answers))
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Add eval columns to dataset
    out_dataset = out_dataset.add_column("correct", correct_list)

    print(f"\n{'='*50}")
    print(f"EVAL RESULTS: {benchmark_name}")
    print(f"{'='*50}")
    print(f"  Accuracy: {eval_result['accuracy']:.4f} ({eval_result['correct']}/{eval_result['total']})")
    print(f"  Null parsed predictions: {eval_result['null_parsed_preds']}/{eval_result['total']}")
    print(f"  Null parsed ground truths: {eval_result['null_parsed_gts']}/{eval_result['total']}")
    print(f"  Avg response length (chars): {eval_result['avg_response_length']:.0f}")
    print(f"  Median response length: {eval_result['median_response_length']:.0f}")
    print(f"  P90 response length: {eval_result['p90_response_length']:.0f}")

    if "accuracy_by_level" in eval_result:
        print(f"\n  Accuracy by level:")
        for lvl, acc in eval_result["accuracy_by_level"].items():
            print(f"    {lvl}: {acc:.4f}")

    # Save dataset
    out_dataset.save_to_disk(OUTPUT_PATH)
    print(f"\n  Dataset saved to {OUTPUT_PATH}")

    # Cleanup generation checkpoint
    checkpoint_file = Path(CKPT_PATH)
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Build report entry
    entry = {
        "benchmark": benchmark_name,
        "model": model_name,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "eval": eval_result,
    }
    if gen_summary:
        entry["generation"] = gen_summary

    return entry


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Name for this eval run (e.g. 'ck-2800' or 'base')")
    parser.add_argument("--vllm_url", type=str, default=None, help="vLLM server URL (overrides config)")
    parser.add_argument("--model", type=str, default=None, help="Model name on vLLM server (overrides config)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg["eval"]

    vllm_url = args.vllm_url if args.vllm_url else eval_cfg["vllm_url"]
    model_name = args.model if args.model else eval_cfg["model_name"]
    report_path = eval_cfg["report_path"]
    benchmarks = eval_cfg["benchmarks"]

    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy", timeout=300)

    print(f"Run: {args.run_name}")
    print(f"Model: {model_name}")
    print(f"vLLM: {vllm_url}")

    # Sample test: verify vLLM server is reachable
    print("\nSample test: asking 'What is 2+3?'...")
    try:
        test_response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "What is 2+3?"},
            ],
            max_tokens=256,
        )
        test_answer = test_response.choices[0].message.content or ""
        print(f"Sample response: {test_answer[:200]}")
        print("Server is reachable. Proceeding with benchmarks.\n")
    except Exception as e:
        import traceback
        print(f"ERROR: Sample test failed: {e}")
        traceback.print_exc()
        print("Check that vLLM server is running and model name is correct.")
        return

    # Load existing report
    report = load_report(report_path)

    for benchmark_name, bench in benchmarks.items():
        try:
            entry = await run_benchmark(benchmark_name, bench, client, model_name, eval_cfg, args.run_name)
            report.append(entry)
            save_report(report, report_path)
        except KeyboardInterrupt:
            print("\nStopping early due to interrupt.")
            save_report(report, report_path)
            return

    print(f"\n{'='*50}")
    print(f"Report saved to {report_path}")
    print(f"Total entries in report: {len(report)}")


if __name__ == "__main__":
    asyncio.run(main())
