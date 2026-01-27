"""
Filter dataset to only keep rows with parseable answers.
Uses the same parsing functions as grpo.py for consistency.
"""

import argparse
import regex as re
from datasets import load_dataset, concatenate_datasets, load_from_disk
from loguru import logger
from sympy import simplify, Expr, Equality
from sympy.parsing.latex import parse_latex

# --- PATTERNS (same as grpo.py) ---
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
    """Parse \\begin{pmatrix}...\\end{pmatrix} into tuple of values."""
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

    # Try numeric first
    num = _parse_num(s)
    if num is not None:
        return num

    # Try sympy latex
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

    # Fallback: return cleaned string for text answers
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", s).strip()
    return cleaned if cleaned else None


def parse_answer(text: str | None):
    """Parse answer from text containing \\boxed{}."""
    if text is None:
        return None

    text = str(text)
    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None
    answer = _strip_latex_cruft(matches[-1].group(1))
    if not answer:
        return None

    # Try vector/matrix first
    vec = parse_vector(answer)
    if vec is not None:
        return vec

    # Try tuple
    tuple_match = TUPLE_PAT.fullmatch(answer)
    if tuple_match:
        left = parse_single_value(tuple_match.group(1))
        right = parse_single_value(tuple_match.group(2))
        if left is not None and right is not None:
            return (left, right)
        return None

    return parse_single_value(answer)


def extract_raw_boxed(text: str) -> str | None:
    """Extract raw content from \\boxed{}."""
    text = str(text)
    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None
    return _strip_latex_cruft(matches[-1].group(1))


def is_parseable(example) -> bool:
    """Check if the answer in the example is parseable."""
    # The answer field should contain \boxed{} or be the raw answer
    answer = example.get("answer") or example.get("solution")
    if answer is None:
        return False

    # If it's already wrapped in \boxed{}, parse directly
    if "\\boxed" in str(answer):
        return parse_answer(answer) is not None

    # Otherwise, wrap it and try to parse
    wrapped = f"\\boxed{{{answer}}}"
    return parse_answer(wrapped) is not None


SUBSETS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus'
]


def main(
    output_dir: str,
    source: str = "huggingface",
    input_dir: str | None = None,
):
    """
    Filter dataset to only keep parseable answers.

    Args:
        output_dir: Directory to save filtered datasets
        source: "huggingface" to load from HF, "disk" to load from local disk
        input_dir: Input directory if source is "disk"
    """
    if source == "huggingface":
        logger.info("Loading datasets from HuggingFace")
        train_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in SUBSETS]
        test_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in SUBSETS]

        train_ds = concatenate_datasets(train_datasets)
        test_ds = concatenate_datasets(test_datasets)

        logger.info(f"Raw sizes: Train: {len(train_ds)} | Test: {len(test_ds)}")

        # Extract answer from solution field
        def add_answer(example):
            solution = example.get("solution", "")
            raw_boxed = extract_raw_boxed(solution)
            if raw_boxed:
                example["answer"] = f"\\boxed{{{raw_boxed}}}"
            else:
                example["answer"] = None
            return example

        logger.info("Extracting answers from solutions")
        train_ds = train_ds.map(add_answer)
        test_ds = test_ds.map(add_answer)

    elif source == "disk":
        if input_dir is None:
            raise ValueError("input_dir required when source='disk'")

        logger.info(f"Loading datasets from {input_dir}")
        train_ds = load_from_disk(f"{input_dir}/train")
        test_ds = load_from_disk(f"{input_dir}/test")

        # Check if val exists
        try:
            val_ds = load_from_disk(f"{input_dir}/val")
            has_val = True
        except Exception:
            val_ds = None
            has_val = False

        logger.info(f"Loaded sizes: Train: {len(train_ds)} | Test: {len(test_ds)}" +
                    (f" | Val: {len(val_ds)}" if has_val else ""))
    else:
        raise ValueError(f"Unknown source: {source}")

    # Filter for parseable answers
    logger.info("Filtering for parseable answers")

    train_before = len(train_ds)
    train_ds = train_ds.filter(is_parseable)
    logger.info(f"Train: {train_before} -> {len(train_ds)} ({len(train_ds)/train_before*100:.1f}%)")

    test_before = len(test_ds)
    test_ds = test_ds.filter(is_parseable)
    logger.info(f"Test: {test_before} -> {len(test_ds)} ({len(test_ds)/test_before*100:.1f}%)")

    if source == "disk" and has_val:
        val_before = len(val_ds)
        val_ds = val_ds.filter(is_parseable)
        logger.info(f"Val: {val_before} -> {len(val_ds)} ({len(val_ds)/val_before*100:.1f}%)")

    # For HuggingFace source, split test into val/test
    if source == "huggingface":
        split = test_ds.train_test_split(test_size=0.9, seed=42)
        val_ds = split['train']
        test_ds = split['test']
        has_val = True

    train_ds = train_ds.shuffle(seed=42)

    logger.info(f"Final sizes: Train: {len(train_ds)} | Test: {len(test_ds)}" +
                (f" | Val: {len(val_ds)}" if has_val else ""))

    # Save
    logger.info(f"Saving to {output_dir}")
    train_ds.save_to_disk(f"{output_dir}/train")
    test_ds.save_to_disk(f"{output_dir}/test")
    if has_val:
        val_ds.save_to_disk(f"{output_dir}/val")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset for parseable answers")
    parser.add_argument("--output-dir", type=str, default="data/filtered",
                        help="Output directory for filtered data")
    parser.add_argument("--source", type=str, default="huggingface",
                        choices=["huggingface", "disk"],
                        help="Data source: 'huggingface' or 'disk'")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory if source is 'disk'")
    args = parser.parse_args()

    main(args.output_dir, args.source, args.input_dir)
