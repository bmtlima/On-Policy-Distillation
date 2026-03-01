"""Student rollout sampling for on-policy distillation.

Generates completions from the student model and collects per-token
log-probs needed for the importance-sampling policy gradient update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


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


def batch_forward_logprobs(
    model,
    trajectories: list[Trajectory],
    device: torch.device,
    pad_token_id: int,
    enable_grad: bool = False,
) -> list[torch.Tensor]:
    """Batched forward pass to extract per-completion-token log-probs.

    Right-pads all (prompt+completion) sequences to the same length,
    runs a single model() call, and extracts the log-prob of each
    completion token via indexing (logits[t] predicts token[t+1]).

    Args:
        model: HuggingFace CausalLM.
        trajectories: Trajectories to score.
        device: Torch device.
        pad_token_id: Token ID used for right-padding.
        enable_grad: If True, preserve gradient graph (for training fwd pass).

    Returns:
        List of 1-D tensors, one per trajectory, containing per-completion-token
        log-probs. Gradients are preserved when enable_grad=True.
    """
    # Build padded batch
    all_ids = []
    seq_lengths = []
    for traj in trajectories:
        ids = traj.prompt_token_ids + traj.completion_token_ids
        all_ids.append(ids)
        seq_lengths.append(len(ids))

    max_len = max(seq_lengths)

    # Right-pad to max_len
    padded = []
    masks = []
    for ids, length in zip(all_ids, seq_lengths):
        pad_len = max_len - length
        padded.append(ids + [pad_token_id] * pad_len)
        masks.append([1] * length + [0] * pad_len)

    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(masks, device=device)

    # Forward pass
    ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with ctx:
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Extract per-completion-token log-probs for each trajectory.
        # Only compute log_softmax at completion positions (not prompt positions)
        # to avoid materializing a huge (batch, seq_len, vocab_size) float32 tensor.
        result = []
        for b, traj in enumerate(trajectories):
            prompt_len = len(traj.prompt_token_ids)
            comp_ids = traj.completion_token_ids
            n_comp = len(comp_ids)

            if n_comp == 0:
                result.append(torch.tensor([], device=device))
                continue

            # Positions in logits that predict each completion token:
            # logits[t] predicts token[t+1], so for completion token at
            # position (prompt_len + i), we read logits at (prompt_len + i - 1)
            positions = torch.arange(prompt_len - 1, prompt_len + n_comp - 1, device=device)
            selected_logits = logits[b, positions, :]  # (n_comp, vocab_size)
            selected_lps = F.log_softmax(selected_logits.float(), dim=-1)

            token_ids = torch.tensor(comp_ids, device=device)
            token_lps = selected_lps[torch.arange(n_comp, device=device), token_ids]
            result.append(token_lps)

    return result


def sample_rollouts_hf(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str] | None = None,
    problem_indices: list[int] | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    num_samples_per_prompt: int = 1,
    generation_batch_size: int = 32,
) -> RolloutBatch:
    """Generate rollouts from the student model using HuggingFace generate.

    Batches both generation and the follow-up log-prob forward pass.

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
        generation_batch_size: Batch size for model.generate().

    Returns:
        RolloutBatch containing all trajectories with per-token log-probs.
    """
    model.eval()
    device = next(model.parameters()).device

    if ground_truths is None:
        ground_truths = [""] * len(prompts)
    if problem_indices is None:
        problem_indices = list(range(len(prompts)))

    # Expand prompts × num_samples_per_prompt into flat lists
    flat_prompts = []
    flat_gts = []
    flat_indices = []
    for i, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
        for _ in range(num_samples_per_prompt):
            flat_prompts.append(prompt)
            flat_gts.append(gt)
            flat_indices.append(problem_indices[i])

    # Tokenize all prompts (no padding yet — need raw lengths)
    prompt_encodings = [tokenizer.encode(p, return_tensors="pt")[0] for p in flat_prompts]
    prompt_lengths = [len(enc) for enc in prompt_encodings]

    # --- Batched generation (left-padded, required by HF generate) ---
    original_padding_side = tokenizer.padding_side
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_token_id

    tokenizer.padding_side = "left"

    all_completion_ids: list[list[int]] = []
    all_prompt_ids: list[list[int]] = []

    for chunk_start in range(0, len(flat_prompts), generation_batch_size):
        chunk_end = min(chunk_start + generation_batch_size, len(flat_prompts))
        chunk_encodings = prompt_encodings[chunk_start:chunk_end]
        chunk_lengths = prompt_lengths[chunk_start:chunk_end]

        # Left-pad for generation
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
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
            )

        # Extract completions — strip left-padding and prompt
        for j in range(len(chunk_encodings)):
            gen_ids = outputs.sequences[j]
            prompt_len = chunk_lengths[j]

            # Strip left-pad: the actual prompt starts at (total_len - prompt_len - completion_len)
            # Easier: we know the original prompt token ids
            original_prompt_ids = chunk_encodings[j].tolist()

            # Everything after the prompt in the generated sequence
            # The generated sequence has left-padding + prompt + completion
            # Left-padding length = max_prompt_len - prompt_len
            left_pad_len = max_prompt_len - prompt_len
            completion_start = left_pad_len + prompt_len
            comp_ids = gen_ids[completion_start:].tolist()

            # Strip trailing pad/eos tokens
            while comp_ids and comp_ids[-1] in (pad_token_id, tokenizer.eos_token_id):
                comp_ids.pop()

            all_prompt_ids.append(original_prompt_ids)
            all_completion_ids.append(comp_ids)

    # Restore padding side
    tokenizer.padding_side = original_padding_side

    # Build Trajectory objects (without logprobs yet)
    trajectories = []
    for i in range(len(flat_prompts)):
        comp_text = tokenizer.decode(all_completion_ids[i], skip_special_tokens=True)
        traj = Trajectory(
            prompt=flat_prompts[i],
            completion=comp_text,
            prompt_token_ids=all_prompt_ids[i],
            completion_token_ids=all_completion_ids[i],
            student_logprobs=[],  # filled below
            ground_truth=flat_gts[i],
            problem_idx=flat_indices[i],
        )
        trajectories.append(traj)

    # --- Batched forward pass for student log-probs (right-padded, no grad) ---
    # Process in chunks to control memory
    for chunk_start in range(0, len(trajectories), generation_batch_size):
        chunk = trajectories[chunk_start : chunk_start + generation_batch_size]
        # Skip empty completions
        non_empty = [t for t in chunk if len(t.completion_token_ids) > 0]
        if not non_empty:
            continue
        lp_tensors = batch_forward_logprobs(
            model=model,
            trajectories=non_empty,
            device=device,
            pad_token_id=pad_token_id,
            enable_grad=False,
        )
        for traj, lps in zip(non_empty, lp_tensors):
            traj.student_logprobs = lps.tolist()

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
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    # Generate completion tokens (no output_scores — we use a clean forward pass)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    completion_ids = generated_ids[prompt_len:].tolist()

    # Clean forward pass for unfiltered log-probs (matches teacher scoring)
    full_ids = generated_ids.unsqueeze(0)  # (1, seq_len)
    with torch.no_grad():
        fwd_outputs = model(full_ids)
        logits = fwd_outputs.logits  # (1, seq_len, vocab_size)

    # Only compute log_softmax at completion positions, not the full sequence
    comp_positions = torch.arange(prompt_len - 1, prompt_len + len(completion_ids) - 1, device=logits.device)
    selected_logits = logits[0, comp_positions, :]
    selected_lps = torch.log_softmax(selected_logits.float(), dim=-1)

    token_logprobs = []
    for i, token_id in enumerate(completion_ids):
        token_logprobs.append(selected_lps[i, token_id].item())

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
    # Only compute log_softmax at completion positions, not the full sequence
    n_comp = len(trajectory.completion_token_ids)
    comp_positions = torch.arange(prompt_len - 1, prompt_len + n_comp - 1, device=logits.device)
    selected_logits = logits[0, comp_positions, :]
    selected_lps = torch.log_softmax(selected_logits.float(), dim=-1)

    token_logprobs = []
    for i, token_id in enumerate(trajectory.completion_token_ids):
        if prompt_len + i - 1 >= 0:
            token_logprobs.append(selected_lps[i, token_id].item())
        else:
            token_logprobs.append(0.0)

    return token_logprobs
