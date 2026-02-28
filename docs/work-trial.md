**ML Engineer Work Trial**

_On-Policy Distillation for Math Reasoning_

# Task

Implement **on-policy distillation** to improve a small language model's math reasoning by transferring knowledge from a larger teacher model. Your implementation should follow the algorithm described in this blog post:

[On-Policy Distillation - Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)

_Please read it carefully before starting._

## Models

| **Role** | **Model** | **Size** | **Notes** |
| --- | --- | --- | --- |
| Student | Qwen/Qwen3-1.7B (instruct) | ~3.5 GB | You train this one |
| Teacher | Qwen/Qwen3-32B (instruct) | ~64 GB | Inference only (log-probs) |

Use both models in **non-thinking mode**. Prompt them to show reasoning step-by-step and put the final answer in \\boxed{}.

Evaluation benchmark: **MATH** (Hendrycks et al.) - 7,500 training problems, 5,000 test problems.

## Libraries

**Allowed:**

- PyTorch, transformers, peft (for LoRA)
- datasets (HuggingFace), vLLM, SGLang, accelerate
- Modal (for GPU orchestration)
- Standard utilities: numpy, tqdm, wandb, etc.

**Not allowed:**

- Pre-built distillation trainers (TRL GKDTrainer, GOLDTrainer, or similar)
- Any framework that abstracts away the core training loop

We want to see **_your implementation_** of the core loop: sampling from the student, computing teacher log-probs, calculating per-token reverse KL, and performing the policy gradient update.

## Compute

You will be given a **Modal API key** for GPU access. You choose which GPU to use. The spend limit is set well above what you should need. We give you access to any APIs you feel you need for any product.

You will need to decide how/whether to use LoRA, how/where to put the teacher and student models, and how to use open-source libraries effectively. How you scope and de-risk your work is part of what we're evaluating.

# MVP (Minimum Viable Product)

- Working evaluation harness around the MATH dataset
- Baseline scores for Qwen3-1.7B (instruct) and Qwen3-32B (instruct) on MATH
- Score for on-policy distilled Qwen3-1.7B (instruct) on MATH, with evidence that training is working (e.g., decreasing reverse KL)
- Report explaining what you did and why. We strongly recommend using [claude.ai/new](http://claude.ai/new) and prompt it to help you format your report.

# How to Be Successful

- Be able to explain your decisions and why you ruled out alternatives.
- Know what to focus on and what to treat as a solved problem. We're assessing your judgment about where the important decisions are **not** your ability to explain every line of library code.
- Both not reinventing the wheel unnecessarily and not blackboxing the whole process are important!
- Be able to explain on-policy distillation and its tradeoffs vs. SFT and RL.
- We expect heavy use of AI tools (Cursor, Claude Code, etc.). What matters is that you understand and can explain all the code, including parts AI helped write.
- Ask questions early. We'd rather you clarify something upfront than spend hours going down the wrong path.

# Work past MVP

- Connect to wandb and show reverse KL loss over time with eval and train set.
- Good statistical interpretation of your scores (variance across runs, confidence intervals, etc.).
- Compare results to standard SFT through Unsloth (no need to tweak hyperparameters, just get it implemented). This one is a bigger lift.
- Show how much you can gain from training on one task repeatedly (as explored in the Thinking Machines blog).
- Try to do OPD with closed-source teacher model
- Or your choice! If you come up with a good follow-up experiment outside this list, we'd love to see it.

# Logistics

- You will be given 2 days to work on this. Come in before 10:30 AM. There's no hard end time - stay as long as you need.
- Expect to grab a casual lunch with people both days and not work during that period.
- On the second day, you will give a presentation. No slides or doc needed - just walk through your code and graphs and verbally describe your results. Be prepared to answer questions.

Your point of contact is **Alejandro**. If unavailable, feel free to ask anyone else for questions, API keys, or access to the group meal orders.

**Good luck!**