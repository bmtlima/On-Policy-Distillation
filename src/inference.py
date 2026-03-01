"""vLLM-based batch inference on Modal.

Provides functions for:
- Batch text generation (for evaluation)
- Batch generation with per-token log-probs (for OPD rollouts)
- Teacher log-prob computation on existing sequences
"""

from __future__ import annotations

import modal

from src.modal_app import (
    CHECKPOINT_DIR,
    MODEL_CACHE_DIR,
    STUDENT_GPU,
    STUDENT_MODEL_ID,
    TEACHER_GPU,
    TEACHER_MODEL_ID,
    TEACHER_MODEL_ID_AWQ,
    TEACHER_QUANTIZATION,
    app,
    checkpoint_vol,
    model_cache,
    vllm_image,
)

# ---------------------------------------------------------------------------
# vLLM generation for the student model
# ---------------------------------------------------------------------------


@app.function(
    image=vllm_image,
    gpu=STUDENT_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=5,
)
def generate_student(
    prompts: list[str],
    model_id: str = STUDENT_MODEL_ID,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    logprobs: int | None = None,
    lora_path: str | None = None,
) -> list[dict]:
    """Generate completions from the student model using vLLM.

    Args:
        prompts: List of formatted prompt strings.
        model_id: HuggingFace model ID.
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens to generate.
        top_p: Nucleus sampling threshold.
        logprobs: If set, return this many top log-probs per token.
        lora_path: Optional path to LoRA adapter on the checkpoint volume.

    Returns:
        List of dicts with keys: text, token_ids, logprobs (if requested).
    """
    from vllm import LLM, SamplingParams

    enable_lora = lora_path is not None
    llm = LLM(
        model=model_id,
        download_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        max_model_len=4096,
        enable_lora=enable_lora,
        dtype="float16",
    )

    from vllm.lora.request import LoRARequest

    lora_request = None
    if lora_path:
        lora_request = LoRARequest("student-lora", 1, lora_path)

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        logprobs=logprobs,
        prompt_logprobs=None,
    )

    outputs = llm.generate(prompts, params, lora_request=lora_request)

    results = []
    for output in outputs:
        completion = output.outputs[0]
        result = {
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
        }
        if logprobs is not None and completion.logprobs:
            # Extract the log-prob of the chosen token at each position
            token_logprobs = []
            for lp_dict in completion.logprobs:
                if lp_dict:
                    # Get the logprob of the token that was actually generated
                    chosen_token_id = None
                    chosen_lp = None
                    for tid, lp_obj in lp_dict.items():
                        if chosen_lp is None or lp_obj.rank == 1:
                            chosen_token_id = tid
                            chosen_lp = lp_obj.logprob
                    token_logprobs.append(chosen_lp if chosen_lp is not None else 0.0)
                else:
                    token_logprobs.append(0.0)
            result["logprobs"] = token_logprobs
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# vLLM generation for the teacher model
# ---------------------------------------------------------------------------


@app.function(
    image=vllm_image,
    gpu=TEACHER_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=2,
)
def generate_teacher(
    prompts: list[str],
    model_id: str = TEACHER_MODEL_ID_AWQ,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    logprobs: int | None = None,
) -> list[dict]:
    """Generate completions from the teacher model using vLLM.

    Same interface as generate_student but runs on A100-80GB.
    Uses AWQ-quantized model by default to fit in VRAM.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        download_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        max_model_len=4096,
        quantization=TEACHER_QUANTIZATION,
        dtype="float16",
        tensor_parallel_size=1,
    )

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        logprobs=logprobs,
    )

    outputs = llm.generate(prompts, params)

    results = []
    for output in outputs:
        completion = output.outputs[0]
        result = {
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
        }
        if logprobs is not None and completion.logprobs:
            token_logprobs = []
            for lp_dict in completion.logprobs:
                if lp_dict:
                    chosen_lp = None
                    for tid, lp_obj in lp_dict.items():
                        if chosen_lp is None or lp_obj.rank == 1:
                            chosen_lp = lp_obj.logprob
                    token_logprobs.append(chosen_lp if chosen_lp is not None else 0.0)
                else:
                    token_logprobs.append(0.0)
            result["logprobs"] = token_logprobs
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Teacher log-prob scoring on student-generated trajectories
# ---------------------------------------------------------------------------


@app.function(
    image=vllm_image,
    gpu=TEACHER_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=2,
)
def compute_teacher_logprobs(
    prompts: list[str],
    completions: list[str],
    model_id: str = TEACHER_MODEL_ID_AWQ,
) -> list[list[float]]:
    """Compute teacher log-probs for student-generated completions.

    For each (prompt, completion) pair, runs a single forward pass through the
    teacher and returns the teacher's log-prob for each token in the completion.

    Args:
        prompts: List of formatted prompt strings.
        completions: List of completion strings (from student rollouts).
        model_id: Teacher model HuggingFace ID.

    Returns:
        List of lists of per-token log-probs (one list per completion).
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        download_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        max_model_len=4096,
        quantization=TEACHER_QUANTIZATION,
        dtype="float16",
    )

    # Concatenate prompt + completion, then use prompt_logprobs to get
    # the teacher's log-probs on the completion portion.
    full_texts = [p + c for p, c in zip(prompts, completions)]
    tokenizer = llm.get_tokenizer()

    # Tokenize to find where the prompt ends and completion begins
    prompt_token_counts = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_counts.append(len(prompt_ids))

    # Use prompt_logprobs to score the full sequence
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # We don't need generation, just scoring
        prompt_logprobs=1,
    )

    outputs = llm.generate(full_texts, params)

    all_logprobs = []
    for output, prompt_len in zip(outputs, prompt_token_counts):
        if output.prompt_logprobs is None:
            all_logprobs.append([])
            continue

        # prompt_logprobs covers the full input sequence;
        # we want only the completion tokens (after prompt_len)
        completion_logprobs = []
        for i in range(prompt_len, len(output.prompt_logprobs)):
            lp_dict = output.prompt_logprobs[i]
            if lp_dict:
                # Get the logprob of the actual token at this position
                token_id = output.prompt_token_ids[i]
                if token_id in lp_dict:
                    lp_obj = lp_dict[token_id]
                    completion_logprobs.append(
                        lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
                    )
                else:
                    # Token not in top-k; use a floor value
                    completion_logprobs.append(-100.0)
            else:
                completion_logprobs.append(0.0)

        all_logprobs.append(completion_logprobs)

    return all_logprobs
