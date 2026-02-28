"""Student rollout sampling for on-policy distillation.

Generates completions from the student model and collects per-token
log-probs needed for the importance-sampling policy gradient update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Trajectory:
    """A single student trajectory with associated metadata."""

    prompt: str
    completion: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    # Per-token log-probs from the student at sampling time (π_old)
    student_logprobs: list[float]
    # Per-token log-probs from the teacher (filled in later)
    teacher_logprobs: list[float] = field(default_factory=list)
    # Ground truth answer for this problem
    ground_truth: str = ""
    # Problem index in the dataset
    problem_idx: int = -1


@dataclass
class RolloutBatch:
    """A batch of trajectories from student rollouts."""

    trajectories: list[Trajectory]

    @property
    def prompts(self) -> list[str]:
        return [t.prompt for t in self.trajectories]

    @property
    def completions(self) -> list[str]:
        return [t.completion for t in self.trajectories]

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)


def sample_rollouts_hf(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str] | None = None,
    problem_indices: list[int] | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_samples_per_prompt: int = 1,
) -> RolloutBatch:
    """Generate rollouts from the student model using HuggingFace generate.

    Collects per-token log-probs during generation for the policy gradient.

    Args:
        model: HuggingFace CausalLM (with or without LoRA).
        tokenizer: Corresponding tokenizer.
        prompts: List of formatted prompt strings.
        ground_truths: Optional ground truth answers.
        problem_indices: Optional problem indices in the dataset.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        num_samples_per_prompt: Number of completions per prompt.

    Returns:
        RolloutBatch containing all trajectories with per-token log-probs.
    """
    model.eval()
    device = next(model.parameters()).device
    trajectories = []

    if ground_truths is None:
        ground_truths = [""] * len(prompts)
    if problem_indices is None:
        problem_indices = list(range(len(prompts)))

    for prompt_idx, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
        for _ in range(num_samples_per_prompt):
            traj = _generate_single_trajectory(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            traj.ground_truth = gt
            traj.problem_idx = problem_indices[prompt_idx]
            trajectories.append(traj)

    return RolloutBatch(trajectories=trajectories)


def _generate_single_trajectory(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Trajectory:
    """Generate a single trajectory and collect per-token log-probs."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    # Generate with output scores to get per-token log-probs
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0]
    completion_ids = generated_ids[prompt_len:].tolist()
    scores = outputs.scores  # tuple of (vocab_size,) tensors per step

    # Compute log-probs for the tokens that were actually sampled
    token_logprobs = []
    for step_idx, (score, token_id) in enumerate(zip(scores, completion_ids)):
        # score shape: (1, vocab_size) — apply log-softmax
        log_probs = torch.log_softmax(score[0].float(), dim=-1)
        token_logprobs.append(log_probs[token_id].item())

    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return Trajectory(
        prompt=prompt,
        completion=completion_text,
        prompt_token_ids=input_ids[0].tolist(),
        completion_token_ids=completion_ids,
        student_logprobs=token_logprobs,
    )


def compute_student_logprobs_for_trajectory(
    model,
    tokenizer,
    trajectory: Trajectory,
    device: torch.device,
) -> list[float]:
    """Recompute student log-probs for an existing trajectory.

    Used to get π_current(x) for importance sampling when the model
    has been updated since the trajectory was originally sampled.

    Args:
        model: Current student model.
        tokenizer: Corresponding tokenizer.
        trajectory: Trajectory to score.
        device: Torch device.

    Returns:
        List of per-token log-probs under the current policy.
    """
    # Build full sequence: prompt + completion tokens
    full_ids = trajectory.prompt_token_ids + trajectory.completion_token_ids
    input_ids = torch.tensor([full_ids], device=device)
    prompt_len = len(trajectory.prompt_token_ids)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Log-probs for completion tokens
    # logits[t] predicts token[t+1], so for completion token at position p,
    # we use logits at position p-1
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)

    token_logprobs = []
    for i, token_id in enumerate(trajectory.completion_token_ids):
        pos = prompt_len + i - 1  # logits position that predicts this token
        if pos >= 0:
            token_logprobs.append(log_probs[pos, token_id].item())
        else:
            token_logprobs.append(0.0)

    return token_logprobs
