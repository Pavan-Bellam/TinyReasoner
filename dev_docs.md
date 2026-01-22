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
│   └── eval/
│       └── baseline.py   # Base model evaluation
├── data/                 # Datasets (gitignored)
├── results/              # Evaluation outputs (gitignored)
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

# Run baseline evaluation
python src/eval/baseline.py
```
