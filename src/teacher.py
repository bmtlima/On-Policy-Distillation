"""Teacher log-prob computation for on-policy distillation.

Computes the teacher's per-token log-probs on student-generated
trajectories via a single forward pass per trajectory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modal_app import TEACHER_MODEL_ID
from src.rollout import RolloutBatch, Trajectory, batch_forward_logprobs


def load_teacher_model(
    model_id: str = TEACHER_MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
):
    """Load the teacher model in BF16 (no quantization) for clean log-probs.

    Args:
        model_id: HuggingFace model ID for the teacher.
        cache_dir: Cache directory for model weights.
        device_map: Device placement strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def compute_teacher_logprobs_local(
    model,
    tokenizer,
    trajectories: list[Trajectory],
    device: torch.device | None = None,
    batch_size: int = 4,
) -> None:
    """Compute teacher log-probs for student trajectories (local HF model).

    Modifies trajectories in-place, filling in teacher_logprobs.

    Args:
        model: Teacher model (HuggingFace CausalLM).
        tokenizer: Teacher tokenizer.
        trajectories: List of student trajectories to score.
        device: Torch device.
        batch_size: Number of trajectories to process at once.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i : i + batch_size]
        _score_trajectory_batch(model, tokenizer, batch, device)


def _score_trajectory_batch(
    model,
    tokenizer,
    trajectories: list[Trajectory],
    device: torch.device,
) -> None:
    """Score a batch of trajectories with the teacher model.

    Uses batch_forward_logprobs to process all trajectories in a single
    forward pass (right-padded, no gradients).
    """
    # Filter out empty completions
    non_empty = [t for t in trajectories if len(t.completion_token_ids) > 0]
    if not non_empty:
        return

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    lp_tensors = batch_forward_logprobs(
        model=model,
        trajectories=non_empty,
        device=device,
        pad_token_id=pad_token_id,
        enable_grad=False,
    )

    for traj, lps in zip(non_empty, lp_tensors):
        traj.teacher_logprobs = lps.tolist()


def compute_teacher_logprobs_vllm(
    prompts: list[str],
    completions: list[str],
) -> list[list[float]]:
    """Compute teacher log-probs using vLLM on Modal (remote).

    This is the preferred method for production — uses the vLLM
    prompt_logprobs API for efficient batched scoring.

    Args:
        prompts: Formatted prompt strings.
        completions: Student-generated completion strings.

    Returns:
        List of per-token teacher log-prob lists.
    """
    from src.inference import compute_teacher_logprobs

    return compute_teacher_logprobs.remote(prompts, completions)
