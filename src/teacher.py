"""Teacher log-prob computation for on-policy distillation.

Computes the teacher's per-token log-probs on student-generated
trajectories via a single forward pass per trajectory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.modal_app import TEACHER_MODEL_ID
from src.rollout import RolloutBatch, Trajectory


def load_teacher_model(
    model_id: str = TEACHER_MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
    quantize_nf4: bool = False,
):
    """Load the teacher model for log-prob computation.

    Args:
        model_id: HuggingFace model ID for the teacher.
        cache_dir: Cache directory for model weights.
        device_map: Device placement strategy.
        quantize_nf4: If True, load with bitsandbytes NF4 4-bit quantization
            (~18GB for 32B model). Required for co-located training on A100-80GB.

    Returns:
        (model, tokenizer) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    load_kwargs = dict(
        cache_dir=cache_dir,
        device_map=device_map,
        trust_remote_code=True,
    )

    if quantize_nf4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
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

    For each trajectory, concatenates prompt + completion tokens and
    runs a single forward pass. Extracts the teacher's log-prob for
    each completion token.
    """
    for traj in trajectories:
        # Build full token sequence
        full_ids = traj.prompt_token_ids + traj.completion_token_ids
        input_ids = torch.tensor([full_ids], device=device)
        prompt_len = len(traj.prompt_token_ids)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Compute log-probs
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)

        # Extract teacher's log-prob for each completion token
        teacher_lps = []
        for i, token_id in enumerate(traj.completion_token_ids):
            # logits[t] predicts token[t+1]
            pos = prompt_len + i - 1
            if pos >= 0:
                teacher_lps.append(log_probs[pos, token_id].item())
            else:
                teacher_lps.append(0.0)

        traj.teacher_logprobs = teacher_lps


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
