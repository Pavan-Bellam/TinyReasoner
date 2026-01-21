# TinyReasoner

Exploring emergent reasoning behavior with supervised fine-tuning (SFT) on GSM8K-style chain-of-thought data.

## Overview

TinyReasoner investigates whether language models can learn to emit structured reasoning and answers via SFT. The training format enforces:

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

## Quick Start

### 1. Process Data

```bash
python src/process_data.py
```

### 2. Train SFT Model

```bash
python src/sft.py --config config.yml
```

### 3. Evaluate

Baseline evaluation on the base model:

```bash
python src/eval/baseline.py --model <base-model-id> --subset 100
```

Evaluate an SFT adapter:

```bash
python src/eval/sft.py \
  --base-model <base-model-id> \
  --adapter ./checkpointing/checkpoint-500 \
  --test-data data/gsm8k_test \
  --output eval_results/sft_eval.json \
  --limit 100
```

## Training

### SFT (Supervised Fine-Tuning)

Trains the model to output in the expected `<think>/<answer>` format using chain-of-thought examples.

```bash
# Start fresh
python src/sft.py --config config.yml

# Resume from checkpoint (continues optimizer state)
python src/sft.py --config config.yml --resume ./checkpointing/checkpoint-500

# Initialize weights but start fresh training
python src/sft.py --config config.yml --init-from ./checkpointing/checkpoint-500
```

## Configuration

All training parameters are in `config.yml`:

```yaml
model:
  path: "<base-model-id>"

quant:
  enabled: True
  load_in_4bit: True
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

data:
  path: "data/gsm8k_train"
  eval_path: "data/gsm8k_test"

train:
  epochs: 3
  learning_rate: 0.0002
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4

ckpt:
  output_dir: "./checkpointing"
```

## Results

| Model | GSM8K Accuracy |
|-------|----------------|
| Baseline (example) | 42.38% |
| + SFT | TBD |

## Project Structure

```
tinyreasoner/
|-- src/
|   |-- process_data.py   # GSM8K dataset processing
|   |-- sft.py            # SFT training script
|   `-- eval/
|       |-- baseline.py   # Base model evaluation
|       `-- sft.py        # SFT adapter evaluation
|-- data/                 # Processed datasets (gitignored)
|-- checkpointing/         # SFT checkpoints (gitignored)
|-- eval_results/          # Evaluation outputs (gitignored)
|-- config.yml             # Training configuration
|-- dev_docs.md            # Development documentation
|-- pyproject.toml         # Project config and dependencies
`-- README.md
```

## Monitoring

Training logs to Weights & Biases when enabled in config:

```yaml
wandb:
  enabled: True
  project: "tinyreasoner-sft"
```

## License

MIT
