"""Visualize per-token teacher vs student log-probs on a wrong student trajectory.

Picks a problem the student gets wrong, generates a student completion,
scores it with both student and teacher, and prints a token-by-token
comparison showing where the teacher disagrees with the student.

Usage:
    modal run run_token_viz.py
    modal run run_token_viz.py --problem-index 5  # pick a specific wrong problem
"""

from __future__ import annotations

import modal

from src.modal_app import (
    MODEL_CACHE_DIR,
    STUDENT_MODEL_ID,
    TEACHER_MODEL_ID,
    app,
    model_cache,
    training_image,
)

# The wrong problems from baseline_200_1.7B.json (index into wrong list)
WRONG_PROBLEMS = [
    {
        "problem": r"Given that $\binom{15}{8}=6435$, $\binom{16}{9}=11440$, and $\binom{16}{10}=8008$, find $\binom{15}{10}$.",
        "ground_truth": "3003",
        "level": "Level 2",
        "type": "Counting & Probability",
    },
]


@app.function(
    image=training_image,
    gpu="H200",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def visualize_token_logprobs(problem_index: int = 0):
    """Generate a student trajectory and score it with both models."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.prompts import apply_chat_template, format_problem
    from src.rollout import Trajectory, batch_forward_logprobs

    device = torch.device("cuda:0")

    # --- Pick the problem ---
    prob = WRONG_PROBLEMS[min(problem_index, len(WRONG_PROBLEMS) - 1)]
    print("=" * 80)
    print("PROBLEM")
    print("=" * 80)
    print(prob["problem"])
    print(f"\nGround truth: {prob['ground_truth']}")
    print(f"Level: {prob['level']}, Type: {prob['type']}")

    # --- Load tokenizer + student ---
    print("\nLoading student model (Qwen3-1.7B)...")
    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL_ID, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    student_model.eval()

    # --- Load teacher ---
    print("Loading teacher model (Qwen3-8B)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    teacher_model.eval()

    # --- Format prompt ---
    msgs = format_problem(prob["problem"])
    prompt_str = apply_chat_template(tokenizer, msgs)
    print("\n" + "=" * 80)
    print("FORMATTED PROMPT")
    print("=" * 80)
    print(prompt_str)

    # --- Generate student completion (temperature=1.0, like training) ---
    print("\nGenerating student completion...")
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = student_model.generate(
            input_ids,
            max_new_tokens=1500,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            return_dict_in_generate=True,
        )

    gen_ids = outputs.sequences[0]
    comp_ids = gen_ids[prompt_len:].tolist()

    # Strip trailing EOS
    eos_id = tokenizer.eos_token_id
    while comp_ids and comp_ids[-1] == eos_id:
        comp_ids.pop()

    completion_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
    prompt_token_ids = input_ids[0].tolist()

    print("\n" + "=" * 80)
    print("STUDENT COMPLETION")
    print("=" * 80)
    print(completion_text)

    # Extract answer
    from answer_extraction import extract_boxed_answer, answers_match

    pred = extract_boxed_answer(completion_text)
    correct = answers_match(pred, prob["ground_truth"]) if pred else False
    print(f"\nPredicted: {pred}")
    print(f"Correct: {correct}")

    # --- Score with both models via batch_forward_logprobs ---
    traj = Trajectory(
        prompt=prompt_str,
        completion=completion_text,
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=comp_ids,
        student_logprobs=[],
        teacher_logprobs=[],
        ground_truth=prob["ground_truth"],
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print("\nScoring with student model...")
    student_lps = batch_forward_logprobs(
        model=student_model, trajectories=[traj], device=device,
        pad_token_id=pad_token_id, enable_grad=False,
    )[0]

    print("Scoring with teacher model...")
    teacher_lps = batch_forward_logprobs(
        model=teacher_model, trajectories=[traj], device=device,
        pad_token_id=pad_token_id, enable_grad=False,
    )[0]

    student_lps = student_lps.float().cpu().numpy()
    teacher_lps = teacher_lps.float().cpu().numpy()

    # --- Token-by-token visualization ---
    n_tokens = min(len(student_lps), len(teacher_lps), len(comp_ids))
    tokens = [tokenizer.decode([tid]) for tid in comp_ids[:n_tokens]]

    # Compute advantage (same as training): teacher - student
    advantages = teacher_lps[:n_tokens] - student_lps[:n_tokens]

    print("\n" + "=" * 80)
    print("TOKEN-BY-TOKEN LOG-PROBS")
    print("=" * 80)
    print(f"{'Idx':>4}  {'Token':<20}  {'Student':>10}  {'Teacher':>10}  {'Advantage':>10}  {'Signal'}")
    print("-" * 85)

    for i in range(n_tokens):
        tok_display = repr(tokens[i])
        if len(tok_display) > 18:
            tok_display = tok_display[:15] + "..."

        adv = advantages[i]
        # Visual signal: how much the teacher disagrees
        if adv < -2.0:
            signal = "<<< STRONG PENALTY"
        elif adv < -1.0:
            signal = "<< penalty"
        elif adv < -0.5:
            signal = "< mild penalty"
        elif adv > 2.0:
            signal = ">>> STRONG REWARD"
        elif adv > 1.0:
            signal = ">> reward"
        elif adv > 0.5:
            signal = "> mild reward"
        else:
            signal = ""

        print(
            f"{i:4d}  {tok_display:<20}  {student_lps[i]:>10.4f}  {teacher_lps[i]:>10.4f}  {adv:>10.4f}  {signal}"
        )

    # --- Summary stats ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total completion tokens: {n_tokens}")
    print(f"Mean student log-prob:   {np.mean(student_lps[:n_tokens]):.4f}")
    print(f"Mean teacher log-prob:   {np.mean(teacher_lps[:n_tokens]):.4f}")
    print(f"Mean advantage:          {np.mean(advantages):.4f}")
    print(f"Reverse KL (student-teacher): {np.mean(student_lps[:n_tokens] - teacher_lps[:n_tokens]):.4f}")
    print()

    # Tokens with strongest penalties (teacher disagrees most)
    sorted_idx = np.argsort(advantages)
    print("Top 10 PENALIZED tokens (teacher assigns much lower prob):")
    for rank, idx in enumerate(sorted_idx[:10]):
        tok = repr(tokens[idx])
        print(f"  {rank+1:2d}. [{idx:3d}] {tok:<20} student={student_lps[idx]:.3f}  teacher={teacher_lps[idx]:.3f}  adv={advantages[idx]:.3f}")

    print()
    print("Top 10 REWARDED tokens (teacher assigns much higher prob):")
    for rank, idx in enumerate(sorted_idx[-10:][::-1]):
        tok = repr(tokens[idx])
        print(f"  {rank+1:2d}. [{idx:3d}] {tok:<20} student={student_lps[idx]:.3f}  teacher={teacher_lps[idx]:.3f}  adv={advantages[idx]:.3f}")

    # --- JSON output for downstream use (e.g., slideshow) ---
    import json
    output = {
        "problem": prob["problem"],
        "ground_truth": prob["ground_truth"],
        "predicted": pred,
        "correct": correct,
        "completion": completion_text,
        "tokens": [],
    }
    for i in range(n_tokens):
        output["tokens"].append({
            "index": i,
            "token": tokens[i],
            "token_id": comp_ids[i],
            "student_logprob": float(student_lps[i]),
            "teacher_logprob": float(teacher_lps[i]),
            "advantage": float(advantages[i]),
        })

    json_path = "/root/token_viz_output.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON output saved to {json_path}")

    return output


@app.local_entrypoint()
def main(problem_index: int = 0):
    result = visualize_token_logprobs.remote(problem_index=problem_index)

    # Save locally too
    import json
    import os

    os.makedirs("logs", exist_ok=True)
    local_path = "logs/token_viz_output.json"
    with open(local_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nLocal JSON saved to {local_path}")
