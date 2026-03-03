# OPD-Metis: On-Policy Distillation for Math Reasoning

## What This Is
ML engineer work trial for Thinking Machines. Implementing on-policy distillation (OPD) to improve Qwen3-1.7B's math reasoning using Qwen3-32B as a teacher, evaluated on the MATH benchmark (Hendrycks et al.).

## Key Architecture Decisions
- **Student**: Qwen3-1.7B with PEFT LoRA (rank 32, alpha 32, all-linear targets) — ~3.6GB base + ~20MB adapter in BF16
- **Teacher**: Qwen3-8B in BF16 for co-located training on H200
- **Rollout generation**: vLLM with sleep mode + LoRA hot-swapping (colocated on same GPU)
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
1. Offload HF models to CPU, wake vLLM
2. Generate student rollouts via vLLM with LoRA hot-swap (π_old logprobs from vLLM output)
3. Sleep vLLM, restore HF models to GPU
4. Score each trajectory with teacher (single forward pass → π_teacher logprobs)
5. Advantage = log π_teacher - log π_student (negative reverse KL)
6. Forward current student with gradients → π_current
7. Importance-sampled policy gradient loss
8. Gradient clip + AdamW step
9. Save updated LoRA adapter for next step

## Known Issues
- `run_baselines.py` and `run_eval_distilled.py` format prompts as `"role: content"` instead of using `apply_chat_template()` — hurts accuracy
- Existing baselines were only run on 50 problems, not the full 5000 test set
- Student truncation rate was 38% (19/50) — needs longer max_tokens or prompt tuning

## Commands
```bash
# Baselines
modal run run_baselines.py --limit 50          # Quick baseline (50 problems)
modal run run_baselines.py                      # Full baseline (5000 problems)

# Training
modal run run_train.py --num-steps 5 --skip-sanity-check --wandb-run-name "test-run"   # Quick test
modal run run_train.py --skip-sanity-check --wandb-run-name "opd-v3-vllm"              # Full 150-step run

# Eval (uses merged checkpoint by default)
modal run run_eval_distilled.py --run-name "opd-v3-vllm"                               # Evaluate final_merged
modal run run_eval_distilled.py --run-name "opd-v3-vllm" --checkpoint-name "step_50"   # Evaluate intermediate
```
