"""Core on-policy distillation training loop.

Implements the OPD algorithm from the Thinking Machines blog:
1. Sample trajectories from the student (on-policy)
2. Score each token using the teacher's log-probs
3. Compute per-token reverse KL as advantage
4. Train via clipped PPO policy gradient
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from answer_extraction import extract_boxed_answer
from src.model import load_student_model, save_lora_checkpoint
from src.prompts import apply_chat_template, format_problem
from src.rollout import Trajectory, sample_rollouts_hf
from src.teacher import compute_teacher_logprobs_local, load_teacher_model

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    lr: float = 1e-5
    num_steps: int = 150
    batch_size: int = 8
    num_samples_per_prompt: int = 4
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    ppo_clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    checkpoint_every: int = 25
    teacher_batch_size: int = 2  # smaller batches for teacher scoring (memory)
    wandb_project: str = "opd-metis"
    wandb_run_name: str | None = None
    seed: int = SEED


def compute_advantages(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token advantage as negative reverse KL direction.

    advantage_t = -(log π_student(x_t) - log π_teacher(x_t))
                = log π_teacher(x_t) - log π_student(x_t)

    Higher advantage = student should increase probability toward teacher.
    """
    return teacher_logprobs - student_logprobs


def compute_ppo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute clipped PPO surrogate loss.

    Args:
        current_logprobs: log π_θ(x_t) from current policy (with gradients).
        old_logprobs: log π_old(x_t) from policy at sampling time (detached).
        advantages: Per-token advantages (detached).
        clip_eps: PPO clipping epsilon.
        mask: Optional boolean mask for valid tokens.

    Returns:
        Dict with loss, mean_ratio, max_ratio, clip_fraction.
    """
    # Importance sampling ratio
    ratio = torch.exp(current_logprobs - old_logprobs.detach())

    # Clipped surrogate
    adv = advantages.detach()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

    if mask is not None:
        # Only compute loss over valid (non-padding) tokens
        per_token_loss = -torch.min(surr1, surr2)
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
        clip_fraction = (((ratio - 1.0).abs() > clip_eps).float() * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = -torch.min(surr1, surr2).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

    return {
        "loss": loss,
        "mean_ratio": ratio.mean().detach(),
        "max_ratio": ratio.max().detach(),
        "clip_fraction": clip_fraction.detach(),
    }


def compute_current_logprobs(
    model,
    tokenizer,
    trajectory: Trajectory,
    device: torch.device,
) -> torch.Tensor:
    """Forward pass through current student to get log-probs with gradients.

    This is step (e) in the training algorithm — the only step that
    requires gradients flowing through the student.
    """
    full_ids = trajectory.prompt_token_ids + trajectory.completion_token_ids
    input_ids = torch.tensor([full_ids], device=device)
    prompt_len = len(trajectory.prompt_token_ids)

    outputs = model(input_ids)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    log_probs = F.log_softmax(logits[0].float(), dim=-1)

    # Extract log-probs for completion tokens
    token_logprobs = []
    for i, token_id in enumerate(trajectory.completion_token_ids):
        pos = prompt_len + i - 1  # logits[t] predicts token[t+1]
        if pos >= 0:
            token_logprobs.append(log_probs[pos, token_id])
        else:
            token_logprobs.append(torch.tensor(0.0, device=device))

    return torch.stack(token_logprobs)


def teacher_sanity_check(
    teacher_model,
    student_model,
    tokenizer,
    problems: list[str],
    ground_truths: list[str],
    device: torch.device,
    num_problems: int = 20,
) -> bool:
    """Verify quantized teacher log-probs are meaningfully higher than student.

    Picks problems, generates student rollouts, scores with both models,
    and checks that teacher log-probs dominate.

    Returns True if sanity check passes.
    """
    print("=" * 60)
    print("Teacher Sanity Check")
    print("=" * 60)

    subset_problems = problems[:num_problems]
    subset_gts = ground_truths[:num_problems]

    # Format prompts
    prompts = []
    for p in subset_problems:
        msgs = format_problem(p)
        prompts.append(apply_chat_template(tokenizer, msgs))

    # Generate student rollouts
    print(f"  Generating student rollouts for {len(prompts)} problems...")
    rollout_batch = sample_rollouts_hf(
        model=student_model,
        tokenizer=tokenizer,
        prompts=prompts,
        ground_truths=subset_gts,
        max_new_tokens=512,  # shorter for sanity check
        temperature=0.7,
        num_samples_per_prompt=1,
    )

    # Score with teacher
    print("  Scoring with teacher model...")
    compute_teacher_logprobs_local(
        model=teacher_model,
        tokenizer=tokenizer,
        trajectories=rollout_batch.trajectories,
        batch_size=1,
    )

    # Compare
    student_means = []
    teacher_means = []
    for traj in rollout_batch.trajectories:
        if len(traj.student_logprobs) > 0 and len(traj.teacher_logprobs) > 0:
            min_len = min(len(traj.student_logprobs), len(traj.teacher_logprobs))
            student_means.append(np.mean(traj.student_logprobs[:min_len]))
            teacher_means.append(np.mean(traj.teacher_logprobs[:min_len]))

    if not student_means:
        print("  FAIL: No valid trajectories to compare.")
        return False

    avg_student = np.mean(student_means)
    avg_teacher = np.mean(teacher_means)
    diff = avg_teacher - avg_student

    print(f"  Mean student log-prob: {avg_student:.4f}")
    print(f"  Mean teacher log-prob: {avg_teacher:.4f}")
    print(f"  Difference (teacher - student): {diff:.4f}")

    if diff > 0:
        print("  PASS: Teacher log-probs are higher than student (as expected).")
        return True
    else:
        print("  WARNING: Teacher log-probs are NOT higher than student.")
        print("  The quantized teacher may not be working correctly.")
        return False


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_opd(
    config: TrainConfig,
    train_df: pd.DataFrame,
    teacher_model,
    student_model,
    tokenizer,
    device: torch.device,
    checkpoint_dir: str = "/root/checkpoints",
):
    """Run the full on-policy distillation training loop.

    Args:
        config: Training hyperparameters.
        train_df: MATH training set DataFrame.
        teacher_model: Teacher model (NF4 quantized, eval mode).
        student_model: Student model (BF16 + LoRA, train mode).
        tokenizer: Shared tokenizer (both models are Qwen3).
        device: Torch device.
        checkpoint_dir: Where to save LoRA checkpoints.
    """
    import wandb

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "lr": config.lr,
            "num_steps": config.num_steps,
            "batch_size": config.batch_size,
            "num_samples_per_prompt": config.num_samples_per_prompt,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "ppo_clip_eps": config.ppo_clip_eps,
            "max_grad_norm": config.max_grad_norm,
            "warmup_steps": config.warmup_steps,
        },
    )

    # Optimizer — only LoRA params
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.num_steps,
    )

    problems = train_df["problem"].tolist()
    ground_truths = train_df["answer"].tolist()

    # Pre-format all prompts
    all_prompts = []
    for p in problems:
        msgs = format_problem(p)
        all_prompts.append(apply_chat_template(tokenizer, msgs))

    print(f"Starting OPD training for {config.num_steps} steps")
    print(f"  Batch size: {config.batch_size} prompts x {config.num_samples_per_prompt} samples = {config.batch_size * config.num_samples_per_prompt} trajectories/step")
    print(f"  LR: {config.lr}, PPO clip: {config.ppo_clip_eps}")
    print(f"  Total train problems: {len(problems)}")

    for step in range(config.num_steps):
        step_start = time.time()

        # (a) Sample batch of prompts from MATH train set
        indices = random.sample(range(len(problems)), min(config.batch_size, len(problems)))
        batch_prompts = [all_prompts[i] for i in indices]
        batch_gts = [ground_truths[i] for i in indices]

        # (b) Generate student rollouts (no grad)
        student_model.eval()
        rollout_batch = sample_rollouts_hf(
            model=student_model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            ground_truths=batch_gts,
            problem_indices=indices,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            num_samples_per_prompt=config.num_samples_per_prompt,
        )

        # (c) Score trajectories with teacher (no grad)
        compute_teacher_logprobs_local(
            model=teacher_model,
            tokenizer=tokenizer,
            trajectories=rollout_batch.trajectories,
            batch_size=config.teacher_batch_size,
        )

        # Track truncation
        n_truncated = sum(
            1
            for t in rollout_batch.trajectories
            if extract_boxed_answer(t.completion) is None
        )
        truncation_rate = n_truncated / len(rollout_batch.trajectories)

        # (d-g) Training step with gradient
        student_model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        total_max_ratio = 0.0
        total_clip_frac = 0.0
        valid_trajs = 0

        for traj in rollout_batch.trajectories:
            if len(traj.completion_token_ids) == 0:
                continue
            if len(traj.teacher_logprobs) == 0:
                continue

            min_len = min(
                len(traj.student_logprobs),
                len(traj.teacher_logprobs),
                len(traj.completion_token_ids),
            )
            if min_len == 0:
                continue

            # Old student log-probs (from sampling time, detached)
            old_logprobs = torch.tensor(
                traj.student_logprobs[:min_len], device=device, dtype=torch.float32
            )

            # Teacher log-probs (detached)
            teacher_logprobs = torch.tensor(
                traj.teacher_logprobs[:min_len], device=device, dtype=torch.float32
            )

            # (d) Compute advantages
            advantages = compute_advantages(old_logprobs, teacher_logprobs)

            # (e) Forward pass current student (WITH gradients)
            current_logprobs = compute_current_logprobs(
                model=student_model,
                tokenizer=tokenizer,
                trajectory=traj,
                device=device,
            )[:min_len]

            # (f-g) Compute clipped PPO loss
            loss_dict = compute_ppo_loss(
                current_logprobs=current_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_eps=config.ppo_clip_eps,
            )

            # Accumulate loss (divide by number of trajectories for mean)
            traj_loss = loss_dict["loss"] / len(rollout_batch.trajectories)
            traj_loss.backward()

            # Track metrics
            total_loss += loss_dict["loss"].item()
            total_kl += (old_logprobs - teacher_logprobs).mean().item()
            total_ratio += loss_dict["mean_ratio"].item()
            total_max_ratio = max(total_max_ratio, loss_dict["max_ratio"].item())
            total_clip_frac += loss_dict["clip_fraction"].item()
            valid_trajs += 1

        if valid_trajs > 0:
            # (h) Gradient clipping, optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            # Log metrics
            avg_loss = total_loss / valid_trajs
            avg_kl = total_kl / valid_trajs
            avg_ratio = total_ratio / valid_trajs
            avg_clip_frac = total_clip_frac / valid_trajs
            current_lr = scheduler.get_last_lr()[0]

            step_time = time.time() - step_start

            metrics = {
                "train/loss": avg_loss,
                "train/reverse_kl": avg_kl,
                "train/mean_ratio": avg_ratio,
                "train/max_ratio": total_max_ratio,
                "train/clip_fraction": avg_clip_frac,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/truncation_rate": truncation_rate,
                "train/lr": current_lr,
                "train/step_time_s": step_time,
                "train/valid_trajectories": valid_trajs,
                "train/total_trajectories": len(rollout_batch.trajectories),
            }
            wandb.log(metrics, step=step)

            print(
                f"Step {step:4d}/{config.num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"rev_kl={avg_kl:.4f} | "
                f"ratio={avg_ratio:.3f} | "
                f"grad={metrics['train/grad_norm']:.3f} | "
                f"trunc={truncation_rate:.1%} | "
                f"lr={current_lr:.2e} | "
                f"{step_time:.1f}s"
            )
        else:
            print(f"Step {step:4d}/{config.num_steps} | No valid trajectories (all truncated or empty)")
            optimizer.zero_grad()

        # Save checkpoint
        if (step + 1) % config.checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            save_lora_checkpoint(student_model, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    save_lora_checkpoint(student_model, final_path)
    print(f"Final checkpoint saved to {final_path}")

    wandb.finish()
    return student_model
