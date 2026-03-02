"""Sanity check: run student & teacher generation + forward passes on a single problem.

Mirrors exactly what train.py does:
1. Format prompt with chat template
2. Student generates a completion (sample_rollouts_hf)
3. Student forward pass for pi_old log-probs (already done inside rollout)
4. Teacher forward pass for pi_teacher log-probs (compute_teacher_logprobs_local)
5. Compute advantages = log pi_teacher - log pi_student
6. Teacher generates its own completion for comparison
7. Print per-token log-probs side by side and dump to JSON

Usage:
    modal run run_sanity_check.py
"""

from __future__ import annotations

import json

import modal

from src.modal_app import (
    MODEL_CACHE_DIR,
    STUDENT_MODEL_ID,
    TEACHER_MODEL_ID,
    TRAINING_GPU,
    app,
    model_cache,
    training_image,
)

# Problem the student got wrong in baseline eval
TEST_PROBLEM = (
    "The first term of an arithmetic sequence is 1, another term of the "
    "sequence is 91 and all of the terms of the sequence are integers. "
    "How many distinct arithmetic sequences meet these three conditions?"
)
GROUND_TRUTH = "12"


def _generate_teacher_completion(model, tokenizer, prompt, device, max_new_tokens=1024):
    """Generate a completion from the teacher model and collect log-probs via forward pass."""
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    completion_ids = generated_ids[prompt_len:].tolist()

    # Clean forward pass for log-probs (same as student rollout)
    full_ids = generated_ids.unsqueeze(0)
    with torch.no_grad():
        fwd_outputs = model(full_ids)
        logits = fwd_outputs.logits

    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    token_logprobs = []
    for i, token_id in enumerate(completion_ids):
        pos = prompt_len + i - 1
        token_logprobs.append(log_probs[pos, token_id].item())

    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_text, completion_ids, token_logprobs


@app.function(
    image=training_image,
    gpu=TRAINING_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def sanity_check():
    import torch

    from src.model import load_student_model
    from src.prompts import apply_chat_template, format_problem
    from src.rollout import sample_rollouts_hf
    from src.teacher import load_teacher_model, compute_teacher_logprobs_local
    from src.train import compute_advantages, compute_is_loss, compute_current_logprobs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    print("\nLoading teacher model (BF16)...")
    teacher_model, tokenizer = load_teacher_model(
        model_id=TEACHER_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
    )
    print(f"  Teacher loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading student model (BF16)...")
    student_model, _ = load_student_model(
        model_id=STUDENT_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        dtype="bfloat16",
    )
    print(f"  Student loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Format prompt ---
    msgs = format_problem(TEST_PROBLEM)
    prompt = apply_chat_template(tokenizer, msgs)
    print(f"\n{'='*60}")
    print("PROMPT")
    print(f"{'='*60}")
    print(prompt)

    # ================================================================
    # STUDENT: generation + forward passes
    # ================================================================
    print(f"\n{'='*60}")
    print("STUDENT GENERATION")
    print(f"{'='*60}")
    student_model.eval()
    rollout_batch = sample_rollouts_hf(
        model=student_model,
        tokenizer=tokenizer,
        prompts=[prompt],
        ground_truths=[GROUND_TRUTH],
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        num_samples_per_prompt=1,
    )
    traj = rollout_batch.trajectories[0]
    print(f"Completion ({len(traj.completion_token_ids)} tokens):")
    print(traj.completion)
    print(f"\nGround truth: {GROUND_TRUTH}")

    # Teacher scores student's trajectory
    print(f"\n{'='*60}")
    print("TEACHER SCORING (of student trajectory)")
    print(f"{'='*60}")
    compute_teacher_logprobs_local(
        model=teacher_model,
        tokenizer=tokenizer,
        trajectories=[traj],
        batch_size=1,
    )

    # Compute advantages
    min_len = min(len(traj.student_logprobs), len(traj.teacher_logprobs))
    student_lps = torch.tensor(traj.student_logprobs[:min_len], dtype=torch.float32)
    teacher_lps = torch.tensor(traj.teacher_logprobs[:min_len], dtype=torch.float32)
    advantages = compute_advantages(student_lps, teacher_lps)

    # Simulate training forward pass (pi_current)
    print(f"\n{'='*60}")
    print("CURRENT STUDENT FORWARD PASS (pi_current)")
    print(f"{'='*60}")
    student_model.train()
    current_logprobs = compute_current_logprobs(
        model=student_model,
        tokenizer=tokenizer,
        trajectory=traj,
        device=device,
    )[:min_len]

    # Compute IS loss
    old_logprobs = torch.tensor(traj.student_logprobs[:min_len], device=device, dtype=torch.float32)
    loss_dict = compute_is_loss(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages.to(device),
    )

    # Print per-token log-probs
    print(f"\n{'='*60}")
    print("PER-TOKEN LOG-PROBS (student trajectory)")
    print(f"{'='*60}")
    print(f"{'Pos':>4} {'Token':>15} {'Student':>10} {'Teacher':>10} {'Adv':>10}")
    print("-" * 55)

    tokens_data = []
    for i in range(min_len):
        token_id = traj.completion_token_ids[i]
        token_str = tokenizer.decode([token_id])
        s_lp = traj.student_logprobs[i]
        t_lp = traj.teacher_logprobs[i]
        adv = advantages[i].item()
        print(f"{i:4d} {token_str:>15} {s_lp:10.4f} {t_lp:10.4f} {adv:10.4f}")
        tokens_data.append({
            "pos": i,
            "token_id": token_id,
            "token": token_str,
            "student_logprob": round(s_lp, 6),
            "teacher_logprob": round(t_lp, 6),
            "advantage": round(adv, 6),
        })

    # Summary
    print(f"\n{'='*60}")
    print("STUDENT SUMMARY")
    print(f"{'='*60}")
    print(f"Problem: {TEST_PROBLEM}")
    print(f"Ground truth: {GROUND_TRUTH}")
    print(f"Completion tokens: {min_len}")
    print(f"Mean student log-prob:  {student_lps.mean().item():.4f}")
    print(f"Mean teacher log-prob:  {teacher_lps.mean().item():.4f}")
    print(f"Mean advantage:         {advantages.mean().item():.4f}")
    print(f"Std advantage:          {advantages.std().item():.4f}")
    print(f"% positive advantages:  {(advantages > 0).float().mean().item():.1%}")
    print(f"IS ratio (should be ~1):{loss_dict['mean_ratio'].item():.6f}")
    print(f"IS loss:                {loss_dict['loss'].item():.6f}")
    print(f"pi_current ~ pi_old:   {torch.allclose(current_logprobs.detach().cpu(), old_logprobs.cpu(), atol=1e-3)}")

    student_result = {
        "model": "student",
        "model_id": STUDENT_MODEL_ID,
        "problem": TEST_PROBLEM,
        "ground_truth": GROUND_TRUTH,
        "completion": traj.completion,
        "num_tokens": min_len,
        "mean_student_logprob": round(student_lps.mean().item(), 6),
        "mean_teacher_logprob": round(teacher_lps.mean().item(), 6),
        "mean_advantage": round(advantages.mean().item(), 6),
        "std_advantage": round(advantages.std().item(), 6),
        "pct_positive_advantages": round((advantages > 0).float().mean().item(), 4),
        "is_ratio": round(loss_dict["mean_ratio"].item(), 6),
        "is_loss": round(loss_dict["loss"].item(), 6),
        "tokens": tokens_data,
    }

    # ================================================================
    # TEACHER: generation + log-probs
    # ================================================================
    print(f"\n{'='*60}")
    print("TEACHER GENERATION")
    print(f"{'='*60}")
    teacher_completion, teacher_comp_ids, teacher_gen_lps = _generate_teacher_completion(
        model=teacher_model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=1024,
    )
    print(f"Completion ({len(teacher_comp_ids)} tokens):")
    print(teacher_completion)
    print(f"\nGround truth: {GROUND_TRUTH}")

    # Print teacher per-token log-probs
    print(f"\n{'='*60}")
    print("PER-TOKEN LOG-PROBS (teacher trajectory)")
    print(f"{'='*60}")
    print(f"{'Pos':>4} {'Token':>15} {'LogProb':>10}")
    print("-" * 33)

    teacher_tokens_data = []
    for i, (token_id, lp) in enumerate(zip(teacher_comp_ids, teacher_gen_lps)):
        token_str = tokenizer.decode([token_id])
        print(f"{i:4d} {token_str:>15} {lp:10.4f}")
        teacher_tokens_data.append({
            "pos": i,
            "token_id": token_id,
            "token": token_str,
            "teacher_logprob": round(lp, 6),
        })

    teacher_gen_lps_t = torch.tensor(teacher_gen_lps, dtype=torch.float32)

    print(f"\n{'='*60}")
    print("TEACHER SUMMARY")
    print(f"{'='*60}")
    print(f"Completion tokens: {len(teacher_comp_ids)}")
    print(f"Mean teacher log-prob (own tokens): {teacher_gen_lps_t.mean().item():.4f}")

    teacher_result = {
        "model": "teacher",
        "model_id": TEACHER_MODEL_ID,
        "problem": TEST_PROBLEM,
        "ground_truth": GROUND_TRUTH,
        "completion": teacher_completion,
        "num_tokens": len(teacher_comp_ids),
        "mean_teacher_logprob": round(teacher_gen_lps_t.mean().item(), 6),
        "tokens": teacher_tokens_data,
    }

    return {"student": student_result, "teacher": teacher_result}


@app.local_entrypoint()
def main():
    import os

    results = sanity_check.remote()

    os.makedirs("logs", exist_ok=True)

    # Save student result
    with open("logs/sanity_check_result.json", "w") as f:
        json.dump(results["student"], f, indent=2)
    print("Student results saved to logs/sanity_check_result.json")

    # Save teacher result
    with open("logs/sanity_check_teacher_result.json", "w") as f:
        json.dump(results["teacher"], f, indent=2)
    print("Teacher results saved to logs/sanity_check_teacher_result.json")
