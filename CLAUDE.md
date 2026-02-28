# OPD-Metis: On-Policy Distillation for Math Reasoning

## What This Is
ML engineer work trial for Thinking Machines. Implementing on-policy distillation (OPD) to improve Qwen3-1.7B's math reasoning using Qwen3-32B as a teacher, evaluated on the MATH benchmark (Hendrycks et al.).

## Key Architecture Decisions
- **Student**: Qwen3-1.7B with LoRA (rank 64, alpha 128, q/k/v/o projections) — ~5GB in BF16
- **Teacher**: Qwen3-32B with NF4 quantization (~18GB) for co-located training on A100-80GB
- **Teacher for baselines**: Qwen3-32B-AWQ via vLLM, or OpenRouter for faster iteration
- **Non-thinking mode**: Both models use `/no_think` suffix + `enable_thinking=False`
- **Compute**: Modal for GPU orchestration

## Project Structure
```
run_baselines.py      # Baseline eval (student & teacher) on MATH test set
run_train.py          # OPD training entry point (Modal)
run_eval_distilled.py # Evaluate LoRA checkpoint after training
answer_extraction.py  # \boxed{} parsing + LaTeX normalization + answer comparison
src/
  modal_app.py        # Modal config: images, volumes, GPUs, model IDs
  prompts.py          # Chat template formatting (non-thinking mode)
  model.py            # Student model loading with LoRA
  inference.py        # vLLM inference functions (Modal remote)
  rollout.py          # On-policy trajectory sampling + logprob collection
  teacher.py          # Teacher logprob scoring (local HF + vLLM paths)
  train.py            # Core OPD loop: rollout → teacher score → PPO update
data/                 # MATH dataset parquet files (7500 train, 5000 test)
docs/
  plan.md             # Execution plan
  work-trial.md       # Task description
  thinky.md           # Thinking Machines blog post on OPD
```

## OPD Algorithm (from train.py)
1. Sample batch of prompts from MATH train set
2. Generate student rollouts (on-policy, collect π_old logprobs)
3. Score each trajectory with teacher (single forward pass → π_teacher logprobs)
4. Advantage = log π_teacher - log π_student (negative reverse KL)
5. Forward current student with gradients → π_current
6. Clipped PPO loss with importance sampling
7. Gradient clip + AdamW step

## Known Issues
- `run_baselines.py` and `run_eval_distilled.py` format prompts as `"role: content"` instead of using `apply_chat_template()` — hurts accuracy
- Existing baselines were only run on 50 problems, not the full 5000 test set
- Student truncation rate was 38% (19/50) — needs longer max_tokens or prompt tuning

## Commands
```bash
modal run run_baselines.py --limit 50          # Quick baseline (50 problems)
modal run run_baselines.py                      # Full baseline (5000 problems)
modal run run_train.py --num-steps 10           # Quick training test
modal run run_train.py                          # Full training (150 steps)
modal run run_eval_distilled.py                 # Evaluate final checkpoint
```
