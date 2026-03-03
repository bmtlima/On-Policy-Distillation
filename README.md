# OPD-Metis

On-Policy Distillation (OPD) to improve **Qwen3-1.7B**'s math reasoning using **Qwen3-8B** as a teacher, evaluated on the [MATH benchmark](https://arxiv.org/abs/2103.03874).

## How it works

Each training step:

1. **Rollout**: vLLM generates student completions using the latest LoRA adapter (hot-swapped)
2. **Teacher scoring**: HF forward pass computes per-token teacher log-probs on student trajectories
3. **Advantage**: `log π_teacher - log π_student` (negative reverse KL)
4. **Update**: Importance-sampled policy gradient with clipped ratios (PPO-style)
5. **Save**: Write updated LoRA adapter for the next vLLM rollout

LoRA enables this loop — vLLM keeps the base model loaded and swaps ~20MB adapters between steps instead of reloading full weights.

## Project structure

```
run_train.py              # Training entry point (Modal)
run_baselines.py          # Baseline eval on MATH test set
run_eval_distilled.py     # Evaluate a trained checkpoint

src/
  train.py                # OPD training loop, TrainConfig, sanity check
  rollout.py              # Trajectory dataclasses, vLLM rollout generation, batched logprob extraction
  teacher.py              # Teacher model loading + logprob scoring
  model.py                # Student model loading with LoRA, checkpoint save/merge
  inference.py            # Modal-remote vLLM functions for eval/baselines
  prompts.py              # Chat template formatting (/no_think mode)
  eval.py                 # Dataset loading, accuracy computation, eval reports
  modal_app.py            # Modal config: app, images, volumes, GPUs, model IDs

answer_extraction.py      # \boxed{} parsing, LaTeX normalization, answer comparison
data/                     # MATH dataset parquets (7500 train, 5000 test)
```

## Setup

```bash
pip install -r requirements.txt
```

You need a [Modal](https://modal.com) account and these secrets configured:

- `huggingface-secret` — HuggingFace token (for gated model access)
- `wandb-secret` — Weights & Biases API key (for training logs)

## Usage

### Baselines

```bash
# Full test set (5000 problems), both student and teacher
modal run run_baselines.py

# Student only, 200 random problems
modal run run_baselines.py --model student --sample 200
```

### Training

```bash
# Quick test (5 steps)
modal run run_train.py --num-steps 5 --skip-sanity-check --wandb-run-name "test-run"

# Full run (150 steps)
modal run run_train.py --skip-sanity-check --wandb-run-name "opd-v3-vllm"
```

### Evaluation

```bash
# Evaluate final merged checkpoint
modal run run_eval_distilled.py --run-name "opd-v3-vllm"

# Evaluate intermediate checkpoint
modal run run_eval_distilled.py --run-name "opd-v3-vllm" --checkpoint-name "step_50"
```

## Key details

- **Student**: Qwen3-1.7B + LoRA (rank 32, alpha 32, all-linear targets) — ~20MB adapter
- **Teacher**: Qwen3-8B in BF16
- **Compute**: Dual H200 for training (GPU 0: vLLM rollouts, GPU 1: HF training)
- **Non-thinking mode**: Both models use `/no_think` suffix + `enable_thinking=False`
