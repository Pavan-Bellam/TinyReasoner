import regex as re  # NOT the built-in re; needed for recursive patterns
from sympy import simplify, Expr
from sympy.parsing.latex import parse_latex
from sympy import Equality
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import argparse
from sympy import Rational, Mul, Add, Pow, Symbol
from loguru import logger

SUBSETS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus'
]

SYSTEM_PROMPT = """You must reply in exactly this format and output nothing else:

<think>Step-by-step reasoning here.</think><answer>Final answer only.</answer>

Rules:
- Do not include any text before <think> or after </answer>.
- Put all reasoning in <think>.
- Put only the final answer in <answer> (no explanation).
- If the answer is numeric, output only the number (no commas, no units).
"""

BOXED_PAT = re.compile(r"\\boxed\s*\{((?:[^{}]+|\{(?1)\})*)\}")

INTERVAL_PAT = re.compile(
    r"""
    ^[\(\[]                      # opening ( or [
    \s*[^,]+?\s*                 # left endpoint (anything, lazy)
    ,\s*[^,]+?\s*                # right endpoint
    [\)\]]$                      # closing ) or ]
    """,
    re.VERBOSE,
)

NUM_PAT = re.compile(
    r"""
    ^\s*
    \\?\$?\s*                    # optional $ or \$
    ([+-]?(
        (\d+(\.\d*)?) |          # 3, 3., 3.14
        (\.\d+)                  # .3
    )([eE][+-]?\d+)?)
    \s*$
    """,
    re.VERBOSE,
)



def _parse_num(s: str) -> int | float | None:
    """Parse a simple numeric string to int or float."""
    m = NUM_PAT.fullmatch(s)
    if not m:
        return None
    num = m.group(1)
    if "." in num or "e" in num.lower():
        return float(num)
    return int(num)


def _strip_latex_cruft(s: str) -> str:
    """Remove \\text{...} and common unit words."""
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\b(euros?|dollars?|meters?|cm|kg)\b", "", s, flags=re.I)
    return s.strip()





def parse_answer(answer: str | None) -> int | float | str | Expr | None:
    """
    Parse an extracted answer into a comparable form.

    Args:
        answer: Raw answer string (output of extract_answer)

    Returns:
        - int/float for numeric answers
        - str for interval notation (preserved as-is)
        - sympy.Expr for symbolic expressions (Rational, Mul, Add, Pow, Symbol only)
        - None if unparseable
    """
    if answer is None:
        return None

    valid_types = (int, float, Rational, Mul, Add, Pow, Symbol)

    answer = str(answer)

    if not answer:
        return None

    # Preserve interval notation as string
    if INTERVAL_PAT.fullmatch(answer):
        return None

    # Try LaTeX -> SymPy
    try:
        expr = parse_latex(answer)

        if isinstance(expr, Equality):
            expr = expr.rhs

        expr = simplify(expr)

        if expr.free_symbols:
            # Check if valid type
            if isinstance(expr, valid_types):
                return expr
            return None

        evaluated = expr.evalf()

        if evaluated.is_Integer:
            return int(evaluated)
        if evaluated.is_real:
            return float(evaluated)

        # Check if valid type before returning
        if isinstance(expr, valid_types):
            return expr
        return None

    except Exception:
        return _parse_num(answer)


def extract_answer(text: str) -> str | None:
    """
    Extract the last \\boxed{...} from text.
    Only returns answer if it's parseable.

    Returns:
        - Raw string content from boxed (with latex cruft stripped)
        - None if no boxed content found or unparseable
    """
    text = str(text)

    matches = list(BOXED_PAT.finditer(text))
    if not matches:
        return None

    answer = matches[-1].group(1)
    answer = _strip_latex_cruft(answer)
    
    if not answer:
        return None
    
    parsed =parse_answer(answer)
    # Only keep if parseable
    if parsed is not None:
        return answer
    
    return None

def get_tokenizer_fn(tokenizer):
    def tokenize_example(example):
        ass_response = f"<think>{example['solution']}</think><answer>{example['answer']}</answer> "
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example['problem']},
            {"role": "assistant", "content": ass_response}
        ]

        full_message = tokenizer.apply_chat_template(messages, tokenize=False)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
        out = tokenizer(full_message, add_special_tokens=False)
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)
        assistant_start = len(prefix_tokens["input_ids"])
        eos = tokenizer.eos_token_id

        out["input_ids"] = out["input_ids"] + [eos]
        out["attention_mask"] = out["attention_mask"] + [1]
        out["labels"] = out["input_ids"].copy()
        out["labels"][:assistant_start] = [-100] * assistant_start

        return out


    return tokenize_example



def main(model_name: str, max_length: int, val_ratio: float):
    
    logger.info("Loading datasets")
    train_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in SUBSETS]
    test_datasets = [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in SUBSETS]

    train_ds = concatenate_datasets(train_datasets)
    test_ds = concatenate_datasets(test_datasets)

    logger.info(f"Raw sizes: Train: {len(train_ds)} | Test: {len(test_ds)}")

    def add_answer(example):
        example['answer'] = extract_answer(example['solution'])
        return example

    logger.info("Processing Train Dataset")
    train_ds = train_ds.map(add_answer)
    train_ds = train_ds.filter(lambda x: x['answer'] is not None)

    logger.info("Processing Test Dataset")
    test_ds = test_ds.map(add_answer)
    test_ds = test_ds.filter(lambda x: x['answer'] is not None)

    logger.info(f"After answer extraction: Train: {len(train_ds)} | Test: {len(test_ds)}")

    logger.info("Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenize_fn = get_tokenizer_fn(tokenizer=tokenizer)

    train_ds = train_ds.map(tokenize_fn)
    train_ds = train_ds.filter(lambda x: len(x['input_ids']) <= max_length)

    test_ds = test_ds.map(tokenize_fn)
    test_ds = test_ds.filter(lambda x: len(x['input_ids']) <= max_length)

    logger.info(f"After length filter: Train: {len(train_ds)} | Test: {len(test_ds)}")

    # Split test into val/test
    split = test_ds.train_test_split(test_size=1-val_ratio, seed=42)
    val_ds = split['train']
    test_ds = split['test']

    train_ds = train_ds.shuffle(seed=42)

    logger.info(f"Final sizes: Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    logger.info("Saving datasets to data/")

    train_ds.save_to_disk("data/train")
    val_ds.save_to_disk("data/val")
    test_ds.save_to_disk("data/test")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process MATH dataset for SFT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    main(args.model, args.max_length, args.val_ratio)
