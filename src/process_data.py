import re
from datasets import load_dataset, Dataset

def clean_calculator_annotations(text: str) -> str:
    """Remove <<...>> calculator annotations from text."""
    return re.sub(r'<<.*?>>', '', text)

def extract_answer(text: str) -> str | None:
    """Extract final answer thats placed after ####"""
    if '####' in text:
        return text.split('####')[-1].strip()
    return None

def extract_cot(text: str) -> str:
    """Extract chain of thought (everything before ####)."""
    if '####' in text:
        return text.split('####')[0].strip()
    return text.strip()


def process_example(example: dict) -> dict:
    """Process a single GSM8K example."""
    
    if 'answer' not in example.keys():
        return {"question": None, "cot": None, "answer": None}
    
    raw_answer = example['answer']
    
    # Clean calculator annotations
    cleaned = clean_calculator_annotations(raw_answer)
    
    # Split into CoT and final answer
    cot = extract_cot(cleaned)
    answer = extract_answer(cleaned)

    return {
        "question": example['question'],
        "cot": cot,
        "answer": answer  # Can be None if no ####
    }


def main():
    print("Loading dataset")
    ds = load_dataset("openai/gsm8k", "main")

    print(f"Train size: {len(ds['train'])}")
    print(f"Test size: {len(ds['test'])}")

    print("\nProcessing train split...")
    train_processed = ds['train'].map(process_example, remove_columns=ds['train'].column_names)

    print("Processing test split...")
    test_processed = ds['test'].map(process_example, remove_columns=ds['test'].column_names)

    # Remove rows with no answer (no #### in original)
    train_before = len(train_processed)
    test_before = len(test_processed)
    
    train_processed = train_processed.filter(lambda x: x['answer'] is not None)
    test_processed = test_processed.filter(lambda x: x['answer'] is not None)

    print(f"\nFiltered out rows without ####:")
    print(f"  Train: {train_before} -> {len(train_processed)} (removed {train_before - len(train_processed)})")
    print(f"  Test: {test_before} -> {len(test_processed)} (removed {test_before - len(test_processed)})")

    # Show samples
    print("\n" + "="*50)
    print("SAMPLE EXAMPLES")
    print("="*50)
    for i in range(2):
        ex = train_processed[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {ex['question'][:80]}...")
        print(f"COT: {ex['cot'][:100]}...")
        print(f"A: {ex['answer']}")

    # Save
    train_processed.save_to_disk("data/gsm8k_train")
    test_processed.save_to_disk("data/gsm8k_test")
    
    print("\nSaved to data/gsm8k_train and data/gsm8k_test")


if __name__ == "__main__":
    main()