# TinyReasoner Developer Documentation

---

## Overview

**Goal:** Train a model with supervised fine-tuning on GSM8K-style chain-of-thought data and evaluate it using consistent extraction and normalization rules.

**Training format:**
```
<think>
[reasoning]
</think>
<answer>[final answer]</answer>
```

---

## Data Processing Pipeline (`src/process_data.py`)

**Purpose:** Build train/test datasets from GSM8K with clean chain-of-thought and final answer fields.

**Processing steps implemented:**
1. **Clean calculator annotations**
   - Removes inline calculator annotations like `<<48/2=24>>`
   - Regex: `re.sub(r'<<.*?>>', '', text)`

2. **Extract chain-of-thought (CoT)**
   - Everything before `####` becomes the reasoning target

3. **Extract final answer**
   - The numerical answer after `####`

**Processed format:**
```python
{
    "question": "Natalia sold clips to 48 of her friends in April...",
    "cot": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.",
    "answer": "72"
}
```

**Storage:**
- Saved using Hugging Face `datasets` Arrow format
- Location: `data/gsm8k_train/` and `data/gsm8k_test/`

---

## Evaluation Scripts

### Baseline Evaluation (`src/eval/baseline.py`)

**Design:**
1. **Answer extraction pipeline** (priority order):
   - `\boxed{}` -> `<answer>` tags -> `####` -> last number fallback
2. **Answer normalization:**
   - Strips whitespace, commas, `$`, `%`
   - Converts to canonical numeric form
3. **Batched inference:**
   - Left-padding for efficient batch generation
   - Greedy decoding for reproducibility

**Prompt format:**
```
System: Please reason step by step, and put your final answer within \boxed{}.
User: [question]
```

### SFT Evaluation (`src/eval/sft.py`)

**Design:**
- Loads a base model and a LoRA adapter
- Generates deterministic completions (`do_sample=False`)
- Parses `<think>...</think><answer>...</answer>` and scores format validity + answer correctness
- Writes JSON results plus a readable TXT report when `--output` is provided

---

## SFT Training Pipeline (`src/sft.py`)

**Key features:**
- Optional 4-bit or 8-bit loading via BitsAndBytes
- LoRA configuration in `config.yml`
- Gradient checkpointing support
- Weights & Biases integration

**Flow:**
1. Load `config.yml` and initialize W&B (if enabled).
2. Load base model and tokenizer, optionally quantized.
3. Initialize LoRA adapter (fresh or from checkpoint).
4. Load dataset from disk and split train/validation.
5. Configure `SFTConfig` and train with `SFTTrainer`.

---

## Configuration (`config.yml`)

Key sections:
- `model.path`: base model identifier or path
- `quant`: BitsAndBytes quantization options
- `lora`: LoRA adapter hyperparameters
- `data.path`: dataset on disk (train split source)
- `data.eval_path`: optional evaluation dataset on disk
- `train`: SFT hyperparameters (batch size, LR, epochs, eval steps)
- `ckpt`: checkpoint output settings
- `logging`: logging frequency
- `wandb`: experiment tracking

### Training Config (`train`)

Common fields and how they are used in `src/sft.py`:
- `epochs`: number of full passes over the training split
- `per_device_train_batch_size`: micro-batch size per device
- `per_device_eval_batch_size`: eval batch size per device
- `gradient_accumulation_steps`: steps to accumulate before optimizer update
- `learning_rate`: base LR for the optimizer
- `weight_decay`: L2 regularization
- `warmup_ratio`: fraction of total steps used for LR warmup
- `max_grad_norm`: gradient clipping threshold
- `bf16`: use bfloat16 mixed precision
- `use_gradient_checkpointing`: saves memory at the cost of speed
- `eval_steps`: evaluation frequency (in steps)
- `dataloader_num_workers`: DataLoader worker count (defaults to 0 if unset)
- `max_length`: maximum sequence length for SFT training (defaults to 512 if unset)

---

## Code Reference

### `src/process_data.py`

| Function | Purpose |
|----------|---------|
| `clean_calculator_annotations(text)` | Removes `<<...>>` patterns |
| `extract_answer(text)` | Gets content after `####` |
| `extract_cot(text)` | Gets content before `####` |
| `process_example(example)` | Processes a single GSM8K example |
| `main()` | Orchestrates the pipeline |

**Usage:**
```bash
python src/process_data.py
```

### `src/sft.py`

| Function | Purpose |
|----------|---------|
| `setup_quantization(config)` | Builds BitsAndBytes config |
| `get_lora_config(config)` | Builds LoRA config |
| `load_model_and_tokenizer(...)` | Loads base model and adapters |
| `load_data(config)` | Loads and splits dataset |
| `setup_wandb(config, full_config)` | Initializes W&B |
| `main(...)` | Orchestrates SFT training |

**Usage:**
```bash
python src/sft.py --config config.yml
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| datasets | Loading and processing datasets |
| transformers | Model loading and tokenization |
| peft | LoRA adapters |
| trl | SFTTrainer |
| torch | GPU-accelerated training |
| wandb | Experiment tracking |

---

## File Structure

```
tinyreasoner/
|-- src/
|   |-- process_data.py   # Data processing
|   |-- sft.py            # SFT training
|   `-- eval/
|       |-- baseline.py   # Base model evaluation
|       `-- sft.py        # SFT adapter evaluation
|-- data/                 # Processed datasets (gitignored)
|-- checkpointing/         # SFT checkpoints (gitignored)
|-- eval_results/          # Evaluation outputs (gitignored)
|-- dev_docs.md
|-- pyproject.toml         # Project configuration
`-- README.md
```

---

## Useful Commands

```bash
# Activate virtual environment
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Process GSM8K data
python src/process_data.py

# Train SFT
python src/sft.py --config config.yml

# Evaluate base model
python src/eval/baseline.py --model <base-model-id>

# Evaluate SFT adapter
python src/eval/sft.py --base-model <base-model-id> --adapter ./checkpointing/checkpoint-500 --test-data data/gsm8k_test
```

---

## References

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - Emergent reasoning background
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k) - Grade School Math benchmark
