# TinyReasoner

Training a small language model to develop reasoning capabilities.

## Overview

TinyReasoner explores whether small language models can learn structured reasoning through fine-tuning. The target output format:

```
<think>...</think>
<answer>...</answer>
```

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv) package manager.

```bash
git clone <repo-url>
cd tinyreasoner
uv sync
```

## Baseline Evaluation

Evaluate the base model on MATH dataset:

```bash
python src/eval/baseline.py
```

Results are saved to `results/baseline_results.jsonl` with a summary table showing accuracy by subject and difficulty level.

## Project Structure

```
tinyreasoner/
├── src/
│   └── eval/
│       └── baseline.py   # Base model evaluation
├── data/                 # Datasets (gitignored)
├── results/              # Evaluation outputs (gitignored)
├── dev_docs.md           # Developer documentation
├── pyproject.toml
└── README.md
```

## License

MIT
