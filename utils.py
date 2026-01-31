import regex as re
from sympy import Expr, Equality, simplify
from sympy.parsing.latex import parse_latex

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


def extract_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} from text, only if it's parseable."""
    raw = extract_raw_boxed(text)
    if raw is None:
        return None
    if parse_single_value(raw) is not None:
        return raw
    return None


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


def answers_match(pred: str | None, gt: str | None) -> bool:
    """Check if predicted answer matches ground truth using sympy parsing."""
    if pred is None or gt is None:
        return False

    parsed_pred = parse_answer(pred)
    parsed_gt = parse_answer(gt)

    if parsed_pred is None or parsed_gt is None:
        return False

    if isinstance(parsed_pred, (int, float)) and isinstance(parsed_gt, (int, float)):
        return abs(parsed_pred - parsed_gt) < 1e-9

    if isinstance(parsed_pred, Expr) and isinstance(parsed_gt, Expr):
        try:
            return simplify(parsed_pred - parsed_gt) == 0
        except Exception:
            return False

    try:
        return abs(float(parsed_pred) - float(parsed_gt)) < 1e-9
    except (ValueError, TypeError):
        return str(parsed_pred) == str(parsed_gt)


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
