"""Core on-policy distillation training loop.

Implements the OPD algorithm from the Thinking Machines blog:
1. Sample trajectories from the student (on-policy)
2. Score each token using the teacher's log-probs
3. Compute per-token reverse KL as advantage
4. Train via clipped PPO policy gradient
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from answer_extraction import answers_match, extract_boxed_answer
from src.model import load_student_model, save_checkpoint
from src.prompts import apply_chat_template, format_problem
from src.rollout import Trajectory, batch_forward_logprobs, sample_rollouts_hf
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

    lr: float = 3e-6
    num_steps: int = 150
    batch_size: int = 32
    num_samples_per_prompt: int = 4
    max_new_tokens: int = 1500
    temperature: float = 0.7
    top_p: float = 0.95 # can change to 1.0 too
    ppo_clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    checkpoint_every: int = 25
    teacher_batch_size: int = 16  # teacher scoring sub-batch size
    generation_batch_size: int = 64  # batch size for student model.generate()
    train_fwd_batch_size: int = 4  # batch size for training forward pass (with grad)
    eval_every: int = 25  # run eval every N steps (0 = disabled)
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


def compute_is_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute importance-sampling policy gradient loss.

    Matches the Tinker reference: loss = -(ratio * advantages).sum()
    No PPO clipping — with num_substeps=1 the ratio is ~1.0 anyway.

    Args:
        current_logprobs: log π_θ(x_t) from current policy (with gradients).
        old_logprobs: log π_old(x_t) from policy at sampling time (detached).
        advantages: Per-token advantages (detached).

    Returns:
        Dict with loss and ratio diagnostics.
    """
    # Importance sampling ratio: p_θ(x) / q(x)
    ratio = torch.exp(current_logprobs - old_logprobs.detach())
    adv = advantages.detach()

    # Importance-weighted loss, averaged over tokens
    loss = -(ratio * adv).sum()

    return {
        "loss": loss,
        "mean_ratio": ratio.mean().detach(),
        "max_ratio": ratio.max().detach(),
    }

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
        temperature=1.0,
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


def evaluate_student(
    student_model,
    tokenizer,
    eval_prompts: list[str],
    eval_answers: list[str],
    device: torch.device,
    max_new_tokens: int = 1500,
    generation_batch_size: int = 64,
) -> dict[str, float]:
    """Run greedy eval on a fixed set of test problems.

    Returns dict with eval/accuracy, eval/answer_rate, eval/mean_length, eval/time_s.
    """
    student_model.eval()
    t_start = time.time()

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    all_completions: list[str] = []
    prompt_encodings = [tokenizer.encode(p, return_tensors="pt")[0] for p in eval_prompts]

    for chunk_start in range(0, len(eval_prompts), generation_batch_size):
        chunk_end = min(chunk_start + generation_batch_size, len(eval_prompts))
        chunk_encodings = prompt_encodings[chunk_start:chunk_end]
        chunk_lengths = [len(enc) for enc in chunk_encodings]

        max_prompt_len = max(chunk_lengths)
        padded_ids = []
        attn_masks = []
        for enc, length in zip(chunk_encodings, chunk_lengths):
            pad_len = max_prompt_len - length
            padded_ids.append(
                torch.cat([torch.full((pad_len,), pad_token_id, dtype=enc.dtype), enc])
            )
            attn_masks.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(length, dtype=torch.long)])
            )

        input_ids = torch.stack(padded_ids).to(device)
        attention_mask = torch.stack(attn_masks).to(device)

        with torch.no_grad():
            outputs = student_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

        for j in range(len(chunk_encodings)):
            gen_ids = outputs.sequences[j]
            prompt_len = chunk_lengths[j]
            left_pad_len = max_prompt_len - prompt_len
            comp_ids = gen_ids[left_pad_len + prompt_len:].tolist()
            while comp_ids and comp_ids[-1] in (pad_token_id, tokenizer.eos_token_id):
                comp_ids.pop()
            all_completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True))

    tokenizer.padding_side = original_padding_side

    # Score completions
    n_correct = 0
    n_with_answer = 0
    comp_lengths = []
    for comp, gt in zip(all_completions, eval_answers):
        comp_lengths.append(len(comp.split()))
        pred = extract_boxed_answer(comp)
        if pred is not None:
            n_with_answer += 1
            if answers_match(pred, gt):
                n_correct += 1

    n_total = len(eval_prompts)
    t_eval = time.time() - t_start

    metrics = {
        "eval/accuracy": n_correct / n_total if n_total else 0.0,
        "eval/answer_rate": n_with_answer / n_total if n_total else 0.0,
        "eval/mean_length_words": np.mean(comp_lengths) if comp_lengths else 0.0,
        "eval/time_s": t_eval,
    }

    print(
        f"  EVAL: accuracy={metrics['eval/accuracy']:.1%} "
        f"answer_rate={metrics['eval/answer_rate']:.1%} "
        f"({n_correct}/{n_total} correct) "
        f"[{t_eval:.0f}s]"
    )

    return metrics


def train_opd(
    config: TrainConfig,
    train_df: pd.DataFrame,
    teacher_model,
    student_model,
    tokenizer,
    device: torch.device,
    checkpoint_dir: str = "/root/checkpoints",
    eval_df: pd.DataFrame | None = None,
):
    """Run the full on-policy distillation training loop.

    Args:
        config: Training hyperparameters.
        train_df: MATH training set DataFrame.
        teacher_model: Teacher model (BF16, eval mode).
        student_model: Student model (BF16, train mode).
        tokenizer: Shared tokenizer (both models are Qwen3).
        device: Torch device.
        checkpoint_dir: Where to save LoRA checkpoints.
        eval_df: Optional MATH test set subset for periodic evaluation.
    """
    import wandb

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        id=wandb.util.generate_id(),
        resume="never",
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

    # Namespace checkpoint dir by run name so runs don't overwrite each other
    run_name = config.wandb_run_name or wandb.run.name
    checkpoint_dir = os.path.join(checkpoint_dir, run_name)

    # JSON-lines log file — persists to checkpoint volume
    log_path = os.path.join(checkpoint_dir, "train_log.jsonl")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = open(log_path, "a")
    # Write config header
    log_file.write(json.dumps({"event": "config", **asdict(config)}) + "\n")
    log_file.flush()

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

    # Pre-format eval prompts if eval set provided
    eval_prompts = None
    eval_answers = None
    if eval_df is not None and config.eval_every > 0:
        eval_answers = eval_df["answer"].tolist()
        eval_prompts = []
        for p in eval_df["problem"].tolist():
            msgs = format_problem(p)
            eval_prompts.append(apply_chat_template(tokenizer, msgs))
        print(f"Eval set: {len(eval_prompts)} problems, every {config.eval_every} steps")

    print(f"Starting OPD training for {config.num_steps} steps")
    print(f"  Batch size: {config.batch_size} prompts x {config.num_samples_per_prompt} samples = {config.batch_size * config.num_samples_per_prompt} trajectories/step")
    print(f"  LR: {config.lr}, PPO clip: {config.ppo_clip_eps}")
    print(f"  Total train problems: {len(problems)}")

    # Step-0 eval: baseline before any training
    if eval_prompts is not None and config.eval_every > 0:
        print("\nRunning baseline eval (step 0, before training)...")
        eval_metrics = evaluate_student(
            student_model=student_model,
            tokenizer=tokenizer,
            eval_prompts=eval_prompts,
            eval_answers=eval_answers,
            device=device,
            max_new_tokens=config.max_new_tokens,
            generation_batch_size=config.generation_batch_size,
        )
        wandb.log(eval_metrics, step=0)
        log_file.write(json.dumps({"event": "eval", "step": 0, **eval_metrics}) + "\n")
        log_file.flush()

    for step in range(config.num_steps):
        step_start = time.time()

        # (a) Sample batch of prompts from MATH train set
        indices = random.sample(range(len(problems)), min(config.batch_size, len(problems)))
        batch_prompts = [all_prompts[i] for i in indices]
        batch_gts = [ground_truths[i] for i in indices]

        # (b) Generate student rollouts (no grad)
        torch.cuda.reset_peak_memory_stats()
        t_rollout_start = time.time()
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
            generation_batch_size=config.generation_batch_size,
        )
        t_rollout = time.time() - t_rollout_start
        mem_after_rollout = torch.cuda.max_memory_allocated() / 1e9

        # (c) Score trajectories with teacher (no grad)
        torch.cuda.reset_peak_memory_stats()
        t_teacher_start = time.time()
        compute_teacher_logprobs_local(
            model=teacher_model,
            tokenizer=tokenizer,
            trajectories=rollout_batch.trajectories,
            batch_size=config.teacher_batch_size,
        )
        t_teacher = time.time() - t_teacher_start
        mem_after_teacher = torch.cuda.max_memory_allocated() / 1e9

        # Per-rollout metrics (computed on ALL trajectories before filtering)
        all_comp_lens = [len(t.completion_token_ids) for t in rollout_batch.trajectories]
        mean_completion_length = np.mean(all_comp_lens) if all_comp_lens else 0.0

        n_with_answer = 0
        n_correct = 0
        for t in rollout_batch.trajectories:
            pred = extract_boxed_answer(t.completion)
            if pred is not None:
                n_with_answer += 1
                if answers_match(pred, t.ground_truth):
                    n_correct += 1

        n_total_trajs = len(rollout_batch.trajectories)
        answer_rate = n_with_answer / n_total_trajs
        correct_rate = n_correct / n_total_trajs
        truncation_rate = 1.0 - answer_rate

        # (d-g) Training step with gradient
        torch.cuda.reset_peak_memory_stats()
        t_train_start = time.time()
        student_model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        total_max_ratio = 0.0
        total_student_lp = 0.0
        total_teacher_lp = 0.0
        total_adv = 0.0
        total_pct_pos_adv = 0.0
        valid_trajs = 0

        # Filter valid trajectories and precompute tensors
        valid_trajectories = []
        old_logprobs_list = []
        teacher_logprobs_list = []
        min_lens = []
        n_degenerate = 0

        for traj in rollout_batch.trajectories:
            if len(traj.completion_token_ids) == 0 or len(traj.teacher_logprobs) == 0:
                continue

            # Fix 2: Filter degenerate trajectories (likely repetition loops)
            if len(traj.completion_token_ids) >= config.max_new_tokens - 10:
                n_degenerate += 1
                continue

            min_len = min(
                len(traj.student_logprobs),
                len(traj.teacher_logprobs),
                len(traj.completion_token_ids),
            )
            if min_len == 0:
                continue
            valid_trajectories.append(traj)
            min_lens.append(min_len)
            old_logprobs_list.append(
                torch.tensor(traj.student_logprobs[:min_len], device=device, dtype=torch.float32)
            )
            teacher_logprobs_list.append(
                torch.tensor(traj.teacher_logprobs[:min_len], device=device, dtype=torch.float32)
            )

        total_tokens = sum(min_lens)

        if total_tokens == 0:
            print(f"Step {step:4d}/{config.num_steps} | No valid tokens (degenerate={n_degenerate})")
            optimizer.zero_grad()
            continue

        # Fix 1: Pre-compute and normalize advantages across all valid trajectories
        advantages_list = []
        for old_lps, teach_lps in zip(old_logprobs_list, teacher_logprobs_list):
            advantages_list.append(compute_advantages(old_lps, teach_lps))

        all_adv = torch.cat(advantages_list)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        all_adv = torch.clamp(all_adv, -5.0, 5.0)

        # Split back into per-trajectory tensors
        split_sizes = [a.shape[0] for a in advantages_list]
        advantages_list = list(torch.split(all_adv, split_sizes))

        # Pre-compute batch-level advantage and correlation metrics
        mean_adv_magnitude = all_adv.abs().mean().item()
        all_student = torch.cat(old_logprobs_list)
        all_teacher = torch.cat(teacher_logprobs_list)
        # Pearson correlation between teacher and student logprobs
        s_centered = all_student - all_student.mean()
        t_centered = all_teacher - all_teacher.mean()
        numer = (s_centered * t_centered).sum()
        denom = (s_centered.norm() * t_centered.norm()).clamp(min=1e-8)
        teacher_student_corr = (numer / denom).item()

        # (e-g) Forward + loss + backward per sub-batch (gradient accumulation).
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        fwd_bs = config.train_fwd_batch_size

        for chunk_start in range(0, len(valid_trajectories), fwd_bs):
            chunk_end = min(chunk_start + fwd_bs, len(valid_trajectories))
            chunk_trajs = valid_trajectories[chunk_start:chunk_end]
            chunk_old = old_logprobs_list[chunk_start:chunk_end]
            chunk_teacher = teacher_logprobs_list[chunk_start:chunk_end]
            chunk_advantages = advantages_list[chunk_start:chunk_end]
            chunk_mlens = min_lens[chunk_start:chunk_end]

            # Forward pass with gradients for this sub-batch
            try:
                chunk_lps = batch_forward_logprobs(
                    model=student_model,
                    trajectories=chunk_trajs,
                    device=device,
                    pad_token_id=pad_token_id,
                    enable_grad=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at chunk {chunk_start}, falling back to sequential")
                torch.cuda.empty_cache()
                chunk_lps = []
                for t in chunk_trajs:
                    lps = batch_forward_logprobs(
                        model=student_model,
                        trajectories=[t],
                        device=device,
                        pad_token_id=pad_token_id,
                        enable_grad=True,
                    )
                    chunk_lps.extend(lps)

            # Compute loss for this sub-batch and backward immediately
            chunk_loss = torch.tensor(0.0, device=device)
            for cur_lps, old_lps, teach_lps, advantages, ml in zip(
                chunk_lps, chunk_old, chunk_teacher, chunk_advantages, chunk_mlens
            ):
                cur_lps = cur_lps[:ml]

                loss_dict = compute_is_loss(
                    current_logprobs=cur_lps,
                    old_logprobs=old_lps,
                    advantages=advantages[:ml],
                )

                # Fix 3: Per-sequence loss normalization — equal weight per trajectory
                chunk_loss = chunk_loss + (loss_dict["loss"] / ml) / len(valid_trajectories)

                # Track metrics (detached) — log the same normalized loss used for backprop
                total_loss += (loss_dict["loss"].item() / ml) / len(valid_trajectories)
                total_kl += (old_lps - teach_lps).mean().item()
                total_ratio += loss_dict["mean_ratio"].item()
                total_max_ratio = max(total_max_ratio, loss_dict["max_ratio"].item())
                total_student_lp += old_lps.mean().item()
                total_teacher_lp += teach_lps.mean().item()
                total_adv += advantages[:ml].mean().item()
                total_pct_pos_adv += (advantages[:ml] > 0).float().mean().item()
                valid_trajs += 1

            # Backward for this sub-batch — frees the gradient graph
            chunk_loss.backward()

        if valid_trajs > 0:
            # (h) Gradient clipping, optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            t_train = time.time() - t_train_start
            mem_after_train = torch.cuda.max_memory_allocated() / 1e9

            # Log metrics
            avg_loss = total_loss  # already normalized per-token and per-trajectory during accumulation
            current_lr = scheduler.get_last_lr()[0]
            step_time = time.time() - step_start
            avg_adv = total_adv / valid_trajs
            avg_pct_pos_adv = total_pct_pos_adv / valid_trajs

            metrics = {
                "train/loss": avg_loss,
                "train/reverse_kl": total_kl / valid_trajs,
                "train/mean_ratio": total_ratio / valid_trajs,
                "train/max_ratio": total_max_ratio,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/truncation_rate": truncation_rate,
                "train/lr": current_lr,
                "train/step_time_s": step_time,
                "train/valid_trajectories": valid_trajs,
                "train/total_trajectories": n_total_trajs,
                "train/mean_student_logprob": total_student_lp / valid_trajs,
                "train/mean_teacher_logprob": total_teacher_lp / valid_trajs,
                "train/mean_advantage": avg_adv,
                "train/pct_positive_advantages": avg_pct_pos_adv,
                "train/total_tokens": total_tokens,
                "train/n_degenerate_filtered": n_degenerate,
                "train/mean_completion_length": mean_completion_length,
                "train/answer_rate": answer_rate,
                "train/correct_rate": correct_rate,
                "train/mean_advantage_magnitude": mean_adv_magnitude,
                "train/teacher_student_logprob_corr": teacher_student_corr,
                "profile/rollout_time_s": t_rollout,
                "profile/teacher_time_s": t_teacher,
                "profile/train_time_s": t_train,
                "profile/rollout_peak_gb": mem_after_rollout,
                "profile/teacher_peak_gb": mem_after_teacher,
                "profile/train_peak_gb": mem_after_train,
            }
            wandb.log(metrics, step=step)

            # Persist to JSONL log
            log_file.write(json.dumps({"event": "step", "step": step, **metrics}) + "\n")
            log_file.flush()

            print(
                f"Step {step:4d}/{config.num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"correct={correct_rate:.1%} | "
                f"answer={answer_rate:.1%} | "
                f"adv={avg_adv:.4f} | "
                f"pct_pos={avg_pct_pos_adv:.1%} | "
                f"grad={metrics['train/grad_norm']:.3f} | "
                f"degen={n_degenerate} | "
                f"corr={teacher_student_corr:.3f} | "
                f"lr={current_lr:.2e} | "
                f"{step_time:.1f}s "
                f"[rollout={t_rollout:.0f}s teacher={t_teacher:.0f}s train={t_train:.0f}s | "
                f"mem={mem_after_rollout:.1f}/{mem_after_teacher:.1f}/{mem_after_train:.1f}GB]"
            )
        else:
            print(f"Step {step:4d}/{config.num_steps} | No valid trajectories (all truncated or empty)")
            optimizer.zero_grad()

        # Save checkpoint
        if (step + 1) % config.checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            save_checkpoint(student_model, tokenizer, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

        # Periodic evaluation
        if (
            eval_prompts is not None
            and config.eval_every > 0
            and (step + 1) % config.eval_every == 0
        ):
            eval_metrics = evaluate_student(
                student_model=student_model,
                tokenizer=tokenizer,
                eval_prompts=eval_prompts,
                eval_answers=eval_answers,
                device=device,
                max_new_tokens=config.max_new_tokens,
                generation_batch_size=config.generation_batch_size,
            )
            wandb.log(eval_metrics, step=step)
            log_file.write(json.dumps({"event": "eval", "step": step, **eval_metrics}) + "\n")
            log_file.flush()

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    save_checkpoint(student_model, tokenizer, final_path)
    print(f"Final checkpoint saved to {final_path}")

    log_file.write(json.dumps({"event": "done", "num_steps": config.num_steps}) + "\n")
    log_file.close()
    print(f"Training log saved to {log_path}")

    wandb.finish()
    return student_model

