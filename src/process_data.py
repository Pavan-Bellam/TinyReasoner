import re
from datasets import load_dataset, concatenate_datasets

SUBSETS = [
    'algebra', 
    'counting_and_probability', 
    'geometry', 
    'intermediate_algebra', 
    'number_theory', 
    'prealgebra', 
    'precalculus'
]


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} element from a string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    """Remove \\boxed{} wrapper, return inner content."""
    if s is None:
        return None
    if s.startswith("\\boxed{"):
        return s[7:-1]
    if s.startswith("\\fbox{"):
        return s[6:-1]
    return s


def extract_answer(solution: str) -> str | None:
    """Extract final answer from solution."""
    boxed = last_boxed_only_string(solution)
    return remove_boxed(boxed)


def process_example(example: dict) -> dict:
    """Process a single MATH example."""
    answer = extract_answer(example['solution'])
    cot = example['solution'].strip()  # Keep full solution including boxed answer
    
    return {
        "question": example['problem'],
        "cot": cot,
        "answer": answer,
        "level": example['level'],
        "type": example['type']
    }


def main():
    print("Loading dataset...")
    
    # Load and concatenate all subsets
    train_datasets = [
        load_dataset("EleutherAI/hendrycks_math", s, split="train") 
        for s in SUBSETS
    ]
    test_datasets = [
        load_dataset("EleutherAI/hendrycks_math", s, split="test") 
        for s in SUBSETS
    ]
    
    train = concatenate_datasets(train_datasets)
    test = concatenate_datasets(test_datasets)
    
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    print("\nProcessing splits...")
    train_processed = train.map(process_example, remove_columns=train.column_names)
    test_processed = test.map(process_example, remove_columns=test.column_names)

    # Filter out any rows where answer extraction failed
    train_before = len(train_processed)
    test_before = len(test_processed)
    
    train_processed = train_processed.filter(lambda x: x['answer'] is not None)
    test_processed = test_processed.filter(lambda x: x['answer'] is not None)
    
    print(f"\nFiltered out rows without boxed answer:")
    print(f"  Train: {train_before} -> {len(train_processed)} (removed {train_before - len(train_processed)})")
    print(f"  Test: {test_before} -> {len(test_processed)} (removed {test_before - len(test_processed)})")

    # Show samples
    print("\n" + "="*60)
    print("SAMPLE EXAMPLES")
    print("="*60)
    for i in range(3):
        ex = train_processed[i]
        print(f"\n--- Example {i+1} [{ex['level']}, {ex['type']}] ---")
        print(f"Q: {ex['question'][:150]}...")
        print(f"COT: {ex['cot'][:200]}...")
        print(f"A: {ex['answer']}")

    # Save
    train_processed.save_to_disk("data/train")
    test_processed.save_to_disk("data/test")
    
    print(f"\nSaved to data/math_{{train,test}}")


if __name__ == "__main__":
    main()
