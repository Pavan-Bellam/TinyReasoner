# TinyReasoner

Replicating DeepSeek-R1's emergent reasoning behavior on a 0.5B parameter model using GRPO (Group Relative Policy Optimization) with verifiable rewards.

## Overview

TinyReasoner is a research project exploring whether small language models can develop emergent reasoning capabilities through reinforcement learning. Inspired by DeepSeek-R1's findings, we aim to train a tiny (0.5B) model to exhibit chain-of-thought reasoning without explicit supervision.

## Goals

- Train a 0.5B parameter model to solve grade-school math problems
- Use GRPO with verifiable rewards (correct/incorrect answers)
- Observe emergent reasoning patterns in the model's outputs
- Document the training process and findings

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv) package manager.

```bash
# Clone the repository
git clone <repo-url>
cd tinyreasoner

# Create virtual environment and install dependencies
uv sync
```

## Usage

### Data Processing

Process the GSM8K dataset for training:

```bash
python src/process_data.py
```

This will:
1. Download the GSM8K dataset from Hugging Face
2. Clean calculator annotations (e.g., `<<5*3=15>>`)
3. Extract question, chain-of-thought reasoning, and final answer
4. Save processed data to `data/gsm8k_train` and `data/gsm8k_test`

## Project Structure

```
tinyreasoner/
├── src/
│   └── process_data.py    # GSM8K dataset processing
├── data/                   # Processed datasets (gitignored)
│   ├── gsm8k_train/       # 7,473 training examples
│   └── gsm8k_test/        # 1,319 test examples
├── dev_docs.md            # Development documentation
├── pyproject.toml         # Project config and dependencies
└── README.md
```

## Dataset

Using [OpenAI's GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Grade School Math 8K, a dataset of 8,792 grade school math word problems with natural language solutions.

**Processed format:**
| Field | Description |
|-------|-------------|
| `question` | The math word problem |
| `cot` | Chain-of-thought reasoning steps |
| `answer` | Final numerical answer |

## Dependencies

- `datasets` - Hugging Face datasets library

## License

MIT
