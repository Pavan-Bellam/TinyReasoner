"""Debug script to see what's happening with answer extraction on hendrycks_math."""
from datasets import load_dataset, concatenate_datasets
from utils import extract_raw_boxed, extract_answer, parse_answer, parse_single_value

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

def main():
    datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in SUBSETS]
    ds = concatenate_datasets(datasets)
    print(f"Total examples: {len(ds)}")

    no_boxed = 0
    boxed_but_no_parse = 0
    parsed_ok = 0
    failures = []

    for i, ex in enumerate(ds):
        solution = ex["solution"]
        raw = extract_raw_boxed(solution)

        if raw is None:
            no_boxed += 1
            if len(failures) < 5:
                failures.append({"idx": i, "reason": "no_boxed", "solution": solution[:300]})
            continue

        # parse_single_value works on bare content (no \boxed{} wrapper)
        parsed = parse_single_value(raw)
        if parsed is None:
            boxed_but_no_parse += 1
            if len(failures) < 30:
                failures.append({"idx": i, "reason": "parse_failed", "raw_boxed": raw})
            continue

        parsed_ok += 1

    total = len(ds)
    print(f"\n{'='*60}")
    print(f"No \\boxed{{}} found:        {no_boxed:>6} / {total} ({100*no_boxed/total:.1f}%)")
    print(f"Boxed but parse failed:   {boxed_but_no_parse:>6} / {total} ({100*boxed_but_no_parse/total:.1f}%)")
    print(f"Parsed OK:                {parsed_ok:>6} / {total} ({100*parsed_ok/total:.1f}%)")
    print(f"{'='*60}")

    # Also verify parse_answer on full solution text (like eval.py does)
    full_parse_ok = sum(1 for ex in ds if parse_answer(ex["solution"]) is not None)
    print(f"parse_answer(solution):   {full_parse_ok:>6} / {total} ({100*full_parse_ok/total:.1f}%)")

    # And extract_answer (the function grpo.py uses)
    extract_ok = sum(1 for ex in ds if extract_answer(ex["solution"]) is not None)
    print(f"extract_answer(solution): {extract_ok:>6} / {total} ({100*extract_ok/total:.1f}%)")

    if failures:
        print(f"\n--- Sample parse failures ({len(failures)}) ---")
        for f in failures:
            print(f"\n[{f['reason']}] idx={f['idx']}")
            if "raw_boxed" in f:
                print(f"  raw_boxed: {f['raw_boxed']}")
            if "solution" in f:
                print(f"  solution:  {f['solution'][:200]}...")


if __name__ == "__main__":
    main()
