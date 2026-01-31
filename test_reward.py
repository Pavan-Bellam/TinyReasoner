"""Test the reward function end-to-end with real dataset examples + fake completions."""
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import extract_answer
from grpo import make_reward_fn, get_length_penalty, _parse_level

SUBSET = "algebra"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    print("Loading a few examples from hendrycks_math...")
    ds = load_dataset("EleutherAI/hendrycks_math", SUBSET, split="train[:10]")

    # Show what the dataset looks like
    print(f"\n{'='*60}")
    print("RAW DATASET SAMPLE")
    print(f"{'='*60}")
    for i, ex in enumerate(ds.select(range(3))):
        raw_answer = extract_answer(ex["solution"])
        print(f"\n[{i}] level={ex['level']}")
        print(f"  problem:  {ex['problem'][:100]}...")
        print(f"  solution: {ex['solution'][:100]}...")
        print(f"  extracted answer: {raw_answer}")

    # Prep dataset like grpo.py does
    def add_answer(example):
        example["answer"] = extract_answer(example["solution"])
        return example
    ds = ds.map(add_answer)
    ds = ds.filter(lambda x: x["answer"] is not None)
    print(f"\nAfter filtering: {len(ds)} examples")

    # Take first 3 examples
    examples = [ds[i] for i in range(min(3, len(ds)))]

    # Build fake completions: correct, wrong, bad format
    gt_answers = [ex["answer"] for ex in examples]
    levels = [ex["level"] for ex in examples]

    print(f"\n{'='*60}")
    print("GROUND TRUTH ANSWERS")
    print(f"{'='*60}")
    for i, (gt, lvl) in enumerate(zip(gt_answers, levels)):
        print(f"  [{i}] gt={gt!r}  level={lvl}")

    # Simulate completions
    completions = [
        # correct: wrap gt in boxed
        f"Let me solve this step by step.\n\\boxed{{{gt_answers[0]}}}",
        # wrong answer
        f"The answer is \\boxed{{99999}}",
        # bad format (no boxed)
        f"I think the answer is 42",
    ]
    answers = gt_answers[:3]
    levels_for_reward = levels[:3]

    print(f"\n{'='*60}")
    print("FAKE COMPLETIONS")
    print(f"{'='*60}")
    for i, c in enumerate(completions):
        print(f"  [{i}] {c[:100]}")

    # Create reward function
    reward_fn = make_reward_fn(
        tokenizer=tokenizer,
        level_col="level",
        correct_reward=1.0,
        wrong_reward=0.0,
        invalid_format_reward=-0.05,
        debug_path=None,
    )

    # Call it
    rewards = reward_fn(
        completions=completions,
        answer=answers,
        prompt=["p1", "p2", "p3"],
        level=levels_for_reward,
    )

    print(f"\n{'='*60}")
    print("REWARDS")
    print(f"{'='*60}")
    expected = ["correct (1.0 - penalty)", "wrong (0.0 - penalty)", "invalid format (-0.05)"]
    for i, (r, e) in enumerate(zip(rewards, expected)):
        print(f"  [{i}] reward={r:.4f}  expected={e}")

    # Also show length penalties
    print(f"\n{'='*60}")
    print("LENGTH PENALTIES")
    print(f"{'='*60}")
    for i, c in enumerate(completions):
        toks = len(tokenizer.encode(c, add_special_tokens=False))
        lvl = _parse_level(levels_for_reward[i])
        penalty = get_length_penalty(toks, lvl)
        print(f"  [{i}] tokens={toks}  level={lvl}  penalty={penalty:.4f}")


if __name__ == "__main__":
    main()
