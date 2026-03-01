"""Evaluate the distilled student model (with LoRA checkpoint) on MATH.

Usage:
    modal run run_eval_distilled.py                                  # Final checkpoint
    modal run run_eval_distilled.py --checkpoint-name step_100       # Specific checkpoint
    modal run run_eval_distilled.py --limit 100                      # Quick test
"""

from __future__ import annotations

import random

import modal

from src.modal_app import (
    CHECKPOINT_DIR,
    MODEL_CACHE_DIR,
    STUDENT_GPU,
    STUDENT_MODEL_ID,
    app,
    checkpoint_vol,
    model_cache,
    vllm_image,
)
from src.prompts import apply_chat_template_batch, format_problems_batch

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


@app.function(
    image=vllm_image,
    gpu=STUDENT_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_distilled(
    prompts: list[str],
    model_id: str = STUDENT_MODEL_ID,
    lora_path: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> list[dict]:
    """Generate completions from distilled student using vLLM with LoRA."""
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

    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("student-lora", 1, lora_path)

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    outputs = llm.generate(prompts, params, lora_request=lora_request)

    results = []
    for output in outputs:
        completion = output.outputs[0]
        results.append({"text": completion.text})

    return results


@app.local_entrypoint()
def main(
    limit: int = 0,
    batch_size: int = 64,
    data_dir: str = "data",
    checkpoint_name: str = "final",
):
    """Evaluate the distilled model and compare to baseline.

    Args:
        limit: Number of problems (0 = all).
        batch_size: Inference batch size.
        data_dir: Path to data directory.
        checkpoint_name: Name of checkpoint subdirectory (e.g. "final", "step_100").
    """
    set_seeds()

    from src.eval import (
        compute_accuracy,
        compute_accuracy_by_group,
        evaluate_completions,
        format_eval_report,
        load_math_dataset,
        save_results,
    )

    limit_val = limit if limit > 0 else None
    df = load_math_dataset(split="test", data_dir=data_dir, limit=limit_val)
    problems = df["problem"].tolist()
    ground_truths = df["answer"].tolist()

    print(f"Loaded {len(problems)} test problems")
    print(f"Checkpoint: {checkpoint_name}")

    # Format prompts using proper chat template
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    messages_batch = format_problems_batch(problems)
    prompts = apply_chat_template_batch(tokenizer, messages_batch)

    # Path to LoRA checkpoint on the Modal volume
    lora_path = f"{CHECKPOINT_DIR}/{checkpoint_name}"

    # Process in batches
    all_completions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}")
        results = eval_distilled.remote(
            prompts=batch,
            lora_path=lora_path,
            temperature=0.0,
            max_tokens=3072,
        )
        all_completions.extend([r["text"] for r in results])

    # Track truncation
    from answer_extraction import extract_boxed_answer

    n_truncated = sum(1 for c in all_completions if extract_boxed_answer(c) is None)
    truncation_rate = n_truncated / len(all_completions) if all_completions else 0.0
    print(f"  Truncation rate (no \\boxed{{}}): {truncation_rate:.1%} ({n_truncated}/{len(all_completions)})")

    # Evaluate
    eval_results = evaluate_completions(all_completions, ground_truths)
    overall_acc = compute_accuracy(eval_results)
    by_level = compute_accuracy_by_group(eval_results, df, "level")
    by_type = compute_accuracy_by_group(eval_results, df, "type")

    report = format_eval_report(
        model_name=f"{STUDENT_MODEL_ID} + LoRA ({checkpoint_name})",
        overall_accuracy=overall_acc,
        by_level=by_level,
        by_type=by_type,
        num_problems=len(eval_results),
    )
    print(report)

    # Save detailed results
    output_path = f"results_distilled_{checkpoint_name}.json"
    save_results(eval_results, df, output_path)
    print(f"Detailed results saved to {output_path}")
