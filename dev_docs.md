# TinyReasoner Developer Documentation

---

## Overview

**Goal:** Train a small language model to develop reasoning capabilities through supervised fine-tuning, then enhance those capabilities using reinforcement learning.

**Training format:**
```
<think>
[reasoning]
</think>
<answer>[final answer]</answer>
```

---

## Data Processing (`src/process_data.py`)

Processes the MATH dataset into train/test splits for SFT training.

**Source:** `EleutherAI/hendrycks_math` (7 subsets: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus)

**Processing steps:**
1. Load all subsets and concatenate
2. Keep full solution as chain-of-thought (including `\boxed{}`)
3. Extract final answer (content inside `\boxed{}`)
4. Filter out examples without valid boxed answers

**Output format:**
```python
{
    "question": "...",
    "cot": "...",
    "answer": "...",
    "level": "Level 1-5",
    "type": "algebra/geometry/..."
}
```

**Storage:** `data/math_train/` and `data/math_test/`

**Usage:**
```bash
python src/process_data.py
```

---

## Baseline Evaluation (`src/eval/baseline.py`)

Evaluates the base model on the MATH dataset before any fine-tuning.

**Configuration:**
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Dataset: `data/math_test`
- Output: `results/baseline_results.jsonl`
- Batch size: 16

**Answer extraction:**
- Extracts content from `\boxed{}` or `\fbox{}`
- Handles nested braces correctly

**Answer normalization:**
- Strips whitespace
- Normalizes LaTeX fractions (`\dfrac` → `\frac`, `\tfrac` → `\frac`)
- Removes LaTeX spacing commands (`\left`, `\right`, `\!`, `\,`, etc.)
- Case-insensitive comparison

**Prompt format:**
```
Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: [question]

Solution:
```

**Output:**
- JSONL file with per-example results (question, expected, extracted, response, correct, level, type)
- CSV table with accuracy breakdown by subject and difficulty level
- Console summary with overall accuracy

**Usage:**
```bash
python src/eval/baseline.py
```

---

## File Structure

```
tinyreasoner/
├── src/
│   ├── process_data.py      # MATH dataset processing
│   ├── check_prompt_size.py # Token length analysis
│   └── eval/
│       └── baseline.py      # Base model evaluation
├── data/                    # Datasets (gitignored)
├── results/                 # Evaluation outputs (gitignored)
├── dev_docs.md
├── pyproject.toml
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| datasets | Loading datasets |
| transformers | Model loading and tokenization |
| torch | GPU inference |
| pandas | Results table formatting |
| tqdm | Progress bars |

---

## Useful Commands

```bash
# Activate virtual environment
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Process MATH dataset
python src/process_data.py

# Check prompt token sizes
python src/check_prompt_size.py

# Run baseline evaluation
python src/eval/baseline.py
```
