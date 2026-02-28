# Execution Plan

## Current Status
- [x] Codebase scaffolded (model loading, rollouts, teacher scoring, PPO training loop)
- [x] Preliminary 50-problem baselines run (but with prompt formatting bug + small sample)
- [ ] Fix prompt formatting bug in baselines
- [ ] Reliable full baselines
- [ ] Training run with evidence it works
- [ ] Final eval of distilled model

---

## Phase 1: Fix Baselines & Get Reliable Numbers

### 1a. Fix the prompt formatting bug
`run_baselines.py` and `run_eval_distilled.py` format prompts as `"role: content"`
instead of using `apply_chat_template()`. This means models aren't getting their
proper Qwen3 chat format and perform worse than they should.

**Action**: Use `apply_chat_template_batch()` from `src/prompts.py` in both scripts
to produce properly formatted prompts for vLLM.

### 1b. Run teacher baseline via OpenRouter
Modal vLLM takes too long to spin up for the 32B teacher. Use OpenRouter API
for the teacher baseline instead — faster iteration, same model.

- Run on ~200-500 test problems (enough for reliable accuracy estimate)
- Greedy decoding (temperature=0)
- Track: accuracy overall, by level, by type, truncation rate
- Increase max_tokens if truncation is high (>10%)

### 1c. Run student baseline via Modal vLLM
Student (1.7B) is fast on A10G. Run on the same subset as teacher for apples-to-apples
comparison, then optionally full 5000.

### 1d. Establish expected accuracy ranges
Literature reference points:
- Qwen3-1.7B on MATH: probably ~30-45% with proper prompting
- Qwen3-32B on MATH: probably ~70-80%
- If numbers are way off, debug prompting before moving on

---

## Phase 2: Smoke-Test Training (1-5 steps)

### 2a. Run training for ~3-5 steps on Modal
```bash
modal run run_train.py --num-steps 5 --batch-size 4 --limit 100
```
Goal: verify the pipeline runs end-to-end without crashing.

Check:
- Teacher sanity check passes (teacher logprobs > student logprobs)
- Loss is computed and non-NaN
- Gradients are flowing (grad_norm > 0)
- Reverse KL is positive and finite
- Importance sampling ratio starts near 1.0
- VRAM usage fits in A100-80GB (~23GB expected)

### 2b. Debug any issues
Common problems to watch for:
- OOM: reduce batch_size or max_new_tokens
- Teacher logprobs all -100 (token not in top-k): check prompt format alignment
- Ratio exploding: reduce learning rate or check logprob alignment between old/current

---

## Phase 3: Full Training Run (~150 steps)

### 3a. Launch full training
```bash
modal run run_train.py --num-steps 150 --wandb-run-name "opd-v1"
```
Default config: batch_size=8, num_samples_per_prompt=4, lr=1e-5, cosine schedule.

### 3b. Monitor on wandb
Key metrics to watch:
- **train/reverse_kl**: should decrease over time (student approaching teacher)
- **train/loss**: should decrease
- **train/mean_ratio**: should stay near 1.0 (if it drifts far, policy is diverging)
- **train/clip_fraction**: should be low (<0.3); high means updates are being clipped a lot
- **train/grad_norm**: should be stable, not exploding
- **train/truncation_rate**: should stay reasonable

### 3c. Checkpoints
Saved every 25 steps. If wandb shows reverse KL plateauing early, can stop and eval
from best checkpoint rather than waiting for all 150 steps.

---

## Phase 4: Evaluate Distilled Model

### 4a. Eval distilled model on MATH test set
```bash
modal run run_eval_distilled.py --checkpoint-name final
# Or a specific step:
modal run run_eval_distilled.py --checkpoint-name step_100
```

### 4b. Compare to baselines
The key result table:

| Model | MATH Accuracy |
|-------|--------------|
| Qwen3-1.7B (student baseline) | ? |
| Qwen3-32B (teacher baseline) | ? |
| Qwen3-1.7B + OPD (distilled) | ? |

Success = distilled model meaningfully improves over student baseline.

### 4c. Deeper analysis
- Accuracy by difficulty level (Level 1-5) — does OPD help more on harder problems?
- Accuracy by type (Algebra, Geometry, etc.) — any category-specific gains?
- Truncation rate comparison — does OPD reduce incomplete outputs?

---

## Phase 5: Report & Presentation Prep

- Walk through code: explain the algorithm, architecture decisions, what you'd change
- Show wandb graphs: reverse KL decreasing, loss curves
- Present the results table
- Be ready to explain: why reverse KL, why PPO clipping, why LoRA rank 64,
  why NF4 for teacher, tradeoffs vs SFT and RL

---

## Stretch Goals (if time permits)
1. Compare to standard SFT via Unsloth
2. Single-task repeated training (train on one prompt many times, per the blog)
3. OPD with closed-source teacher (OpenRouter API for logprobs)
4. Compare Qwen3-32B vs QwQ-32B as teacher
5. Statistical analysis: variance across runs, confidence intervals
6. wandb integration with eval metrics (not just training metrics)
