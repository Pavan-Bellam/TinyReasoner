# TinyReasoner Development Documentation

A developer diary documenting the design decisions, implementation details, and progress of the TinyReasoner project.

---

## Project Vision

**Goal:** Replicate the emergent reasoning behavior observed in DeepSeek-R1 on a much smaller (0.5B parameter) model using GRPO (Group Relative Policy Optimization) with verifiable rewards.

**Hypothesis:** By using reinforcement learning with a simple binary reward signal (correct/incorrect answer), a small model can learn to generate intermediate reasoning steps (chain-of-thought) without explicit supervision on the reasoning process itself.

**Why this matters:** If successful, this demonstrates that emergent reasoning doesn't require massive scale - opening doors for more accessible AI reasoning research.

---

## Development Log

### 2026-01-16: Project Initialization

**What was done:**
- Initialized project with `uv` package manager
- Set up Python 3.12 environment
- Created basic project structure

**Technical decisions:**
- **Package manager:** Chose `uv` over `pip` for faster dependency resolution and reproducible builds
- **Python version:** 3.12+ for modern type hints (`str | None` syntax) and performance improvements

---

### 2026-01-16: Data Processing Pipeline

**What was done:**
- Created `src/process_data.py` to process the GSM8K dataset
- Successfully processed all 8,792 examples (7,473 train + 1,319 test)

**Dataset: GSM8K**

GSM8K (Grade School Math 8K) is OpenAI's dataset of grade-school math word problems. Each example contains:
- A natural language math problem
- A solution with step-by-step reasoning
- A final numerical answer marked with `####`

**Original format:**
```
Question: Natalia sold clips to 48 of her friends in April...
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
#### 72
```

**Processing steps implemented:**

1. **Clean calculator annotations**
   - GSM8K includes inline calculator annotations like `<<48/2=24>>`
   - These are for verification but not needed for training
   - Regex: `re.sub(r'<<.*?>>', '', text)`

2. **Extract chain-of-thought (CoT)**
   - Everything before `####` is the reasoning process
   - This becomes the target for the model to learn to generate

3. **Extract final answer**
   - The numerical answer after `####`
   - This is used for the reward signal (verifiable)

**Processed format:**
```python
{
    "question": "Natalia sold clips to 48 of her friends in April...",
    "cot": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.",
    "answer": "72"
}
```

**Data statistics:**
- Train: 7,473 examples (100% retained after filtering)
- Test: 1,319 examples (100% retained after filtering)
- All examples contained the `####` delimiter

**Storage:**
- Saved using Hugging Face `datasets` library's Arrow format
- Location: `data/gsm8k_train/` and `data/gsm8k_test/`
- Total size: ~4.7 MB

---

## Architecture Decisions

### Why GRPO?

GRPO (Group Relative Policy Optimization) is a variant of policy gradient methods that:
- Compares multiple generations against each other (relative rewards)
- More sample-efficient than standard REINFORCE
- Used successfully in DeepSeek-R1 for reasoning

### Why verifiable rewards?

Math problems have objectively correct answers. This gives us:
- **Binary reward:** 1 if answer matches, 0 otherwise
- **No reward hacking:** Can't game a learned reward model
- **Clear evaluation:** Easy to measure progress

### Why 0.5B parameters?

- **Accessibility:** Trainable on consumer hardware
- **Research speed:** Faster iteration on experiments
- **Proof of concept:** If reasoning emerges at 0.5B, it's not just about scale

---

## Code Reference

### `src/process_data.py`

| Function | Purpose |
|----------|---------|
| `clean_calculator_annotations(text)` | Removes `<<...>>` patterns from text |
| `extract_answer(text)` | Gets content after `####` delimiter |
| `extract_cot(text)` | Gets content before `####` delimiter |
| `process_example(example)` | Processes a single GSM8K example |
| `main()` | Orchestrates the full pipeline |

**Usage:**
```bash
python src/process_data.py
```

---

## Next Steps

1. **Model Selection**
   - Choose a 0.5B base model (e.g., Qwen-0.5B, SmolLM-360M)
   - Set up model loading with transformers

2. **Training Infrastructure**
   - Implement GRPO training loop
   - Set up reward computation (answer matching)
   - Configure logging and checkpointing

3. **Evaluation**
   - Accuracy on GSM8K test set
   - Analysis of generated reasoning chains
   - Comparison with baseline (no RL)

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| datasets | >=4.5.0 | Loading and processing datasets |

*Future dependencies will be added as training infrastructure is built.*

---

## File Structure

```
tinyreasoner/
├── src/
│   └── process_data.py    # Data processing script
├── data/                   # Processed datasets (gitignored)
│   ├── gsm8k_train/       # Arrow format, 7,473 examples
│   └── gsm8k_test/        # Arrow format, 1,319 examples
├── dev_docs.md            # This file
├── pyproject.toml         # Project configuration
├── uv.lock                # Locked dependencies
└── README.md              # Project overview
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

# Load processed data in Python
from datasets import load_from_disk
train = load_from_disk("data/gsm8k_train")
test = load_from_disk("data/gsm8k_test")
```

---

## References

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - Original work on emergent reasoning
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k) - Grade School Math benchmark
- [GRPO](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
