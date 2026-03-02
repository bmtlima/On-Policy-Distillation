"""Inspect model outputs side-by-side for debugging.

Runs a small number of problems through distilled checkpoints and/or the
baseline student, then prints the full completions for human review.

Usage:
    modal run inspect_outputs.py --checkpoint-name final --n 5
    modal run inspect_outputs.py --checkpoint-name step_50 --n 10
    modal run inspect_outputs.py --checkpoint-name final --compare-baseline --n 5
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


@app.function(
    image=vllm_image,
    gpu=STUDENT_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_completions(
    prompts: list[str],
    model_path: str,
    tokenizer_id: str,
    max_tokens: int = 3072,
) -> list[dict]:
    """Generate completions and return full text + metadata."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_id,
        download_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        max_model_len=8192,
        dtype="float16",
    )

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)
    outputs = llm.generate(prompts, params)

    results = []
    for output in outputs:
        completion = output.outputs[0]
        results.append({
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "num_tokens": len(completion.token_ids),
        })
    return results


@app.local_entrypoint()
def main(
    n: int = 5,
    checkpoint_name: str = "final",
    compare_baseline: bool = False,
    data_dir: str = "data",
    seed: int = SEED,
):
    """Inspect outputs from distilled model (and optionally baseline).

    Args:
        n: Number of problems to inspect.
        checkpoint_name: Checkpoint to evaluate (e.g. "final", "step_50").
        compare_baseline: Also run the undistilled baseline for comparison.
        data_dir: Path to data directory.
        seed: Random seed for problem selection.
    """
    from answer_extraction import extract_boxed_answer
    from src.eval import load_math_dataset

    random.seed(seed)

    # Load and sample problems
    df = load_math_dataset(split="test", data_dir=data_dir)
    indices = sorted(random.sample(range(len(df)), min(n, len(df))))
    df = df.iloc[indices].reset_index(drop=True)

    problems = df["problem"].tolist()
    ground_truths = df["answer"].tolist()
    levels = df["level"].tolist()
    types = df["type"].tolist()

    # Format prompts
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    messages_batch = format_problems_batch(problems)
    prompts = apply_chat_template_batch(tokenizer, messages_batch)

    # Run distilled model
    checkpoint_path = f"{CHECKPOINT_DIR}/{checkpoint_name}"
    print(f"Running {checkpoint_name} on {n} problems...")
    distilled_results = generate_completions.remote(
        prompts=prompts,
        model_path=checkpoint_path,
        tokenizer_id=STUDENT_MODEL_ID,
    )

    # Optionally run baseline
    baseline_results = None
    if compare_baseline:
        print(f"Running baseline ({STUDENT_MODEL_ID}) on {n} problems...")
        baseline_results = generate_completions.remote(
            prompts=prompts,
            model_path=STUDENT_MODEL_ID,
            tokenizer_id=STUDENT_MODEL_ID,
        )

    # Print results
    separator = "=" * 80
    for i in range(len(problems)):
        print(f"\n{separator}")
        print(f"PROBLEM {i+1}/{len(problems)} | {levels[i]} | {types[i]}")
        print(separator)
        print(f"\n{problems[i]}\n")
        print(f"GROUND TRUTH: {ground_truths[i]}")

        # Distilled output
        d = distilled_results[i]
        d_answer = extract_boxed_answer(d["text"])
        print(f"\n--- {checkpoint_name.upper()} ({d['num_tokens']} tokens, {d['finish_reason']}) ---")
        print(f"Extracted answer: {d_answer}")
        print(d["text"])

        # Baseline output
        if baseline_results:
            b = baseline_results[i]
            b_answer = extract_boxed_answer(b["text"])
            print(f"\n--- BASELINE ({b['num_tokens']} tokens, {b['finish_reason']}) ---")
            print(f"Extracted answer: {b_answer}")
            print(b["text"])

    # Summary
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)

    d_correct = sum(
        1 for i, d in enumerate(distilled_results)
        if extract_boxed_answer(d["text"]) == ground_truths[i]
        or (extract_boxed_answer(d["text"]) is not None and extract_boxed_answer(d["text"]) == ground_truths[i])
    )
    d_truncated = sum(1 for d in distilled_results if d["finish_reason"] == "length")
    d_no_boxed = sum(1 for d in distilled_results if extract_boxed_answer(d["text"]) is None)
    avg_tokens = sum(d["num_tokens"] for d in distilled_results) / len(distilled_results)

    print(f"\n{checkpoint_name}: {d_correct}/{n} correct, {d_truncated} truncated (length), {d_no_boxed} no \\boxed{{}}, avg {avg_tokens:.0f} tokens")

    if baseline_results:
        b_correct = sum(
            1 for i, b in enumerate(baseline_results)
            if extract_boxed_answer(b["text"]) == ground_truths[i]
            or (extract_boxed_answer(b["text"]) is not None and extract_boxed_answer(b["text"]) == ground_truths[i])
        )
        b_truncated = sum(1 for b in baseline_results if b["finish_reason"] == "length")
        b_no_boxed = sum(1 for b in baseline_results if extract_boxed_answer(b["text"]) is None)
        avg_tokens_b = sum(b["num_tokens"] for b in baseline_results) / len(baseline_results)
        print(f"baseline: {b_correct}/{n} correct, {b_truncated} truncated (length), {b_no_boxed} no \\boxed{{}}, avg {avg_tokens_b:.0f} tokens")
