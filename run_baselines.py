"""Run baseline evaluations for student and teacher models on MATH.

Usage:
    modal run run_baselines.py                          # Full test set, both models
    modal run run_baselines.py --model student           # Student only
    modal run run_baselines.py --model teacher            # Teacher only
    modal run run_baselines.py --sample 50               # Random 50 problems (seed 42)
    modal run run_baselines.py --limit 100               # First 100 problems
"""

from __future__ import annotations

import random

from src.modal_app import (
    STUDENT_MODEL_ID,
    TEACHER_MODEL_ID_AWQ,
    app,
)
from src.inference import generate_student, generate_teacher
from src.prompts import apply_chat_template_batch, format_problems_batch

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


@app.local_entrypoint()
def main(
    limit: int = 0,
    sample: int = 0,
    model: str = "both",
    batch_size: int = 64,
    data_dir: str = "data",
):
    """Run baseline evaluations.

    Args:
        limit: Number of problems to evaluate sequentially (0 = all).
        sample: Randomly sample N problems (seed 42). Overrides limit.
        model: Which model to evaluate: "student", "teacher", or "both".
        batch_size: Inference batch size.
        data_dir: Path to data directory.
    """
    set_seeds()

    from transformers import AutoTokenizer

    from src.eval import (
        compute_accuracy,
        compute_accuracy_by_group,
        evaluate_completions,
        format_eval_report,
        load_math_dataset,
        save_results,
    )
    from answer_extraction import extract_boxed_answer

    # Load full dataset first (for sampling)
    df = load_math_dataset(split="test", data_dir=data_dir)

    sampled_indices = None
    if sample > 0:
        indices = list(range(len(df)))
        random.seed(SEED)
        sampled_indices = sorted(random.sample(indices, min(sample, len(df))))
        df = df.iloc[sampled_indices].reset_index(drop=True)
        print(f"Sampled {len(df)} problems (seed={SEED})")
    elif limit > 0:
        df = df.head(limit)

    problems = df["problem"].tolist()
    ground_truths = df["answer"].tolist()
    print(f"Loaded {len(problems)} test problems")

    # Format prompts using proper chat template
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    messages_batch = format_problems_batch(problems)
    prompts = apply_chat_template_batch(tokenizer, messages_batch)

    n_problems = len(problems)

    if model in ("student", "both"):
        print(f"\n--- Evaluating Student: {STUDENT_MODEL_ID} ---")
        _run_eval(
            "student",
            STUDENT_MODEL_ID,
            prompts,
            ground_truths,
            df,
            batch_size,
            n_problems,
            sampled_indices,
            extract_boxed_answer,
            compute_accuracy,
            compute_accuracy_by_group,
            evaluate_completions,
            format_eval_report,
            save_results,
        )

    if model in ("teacher", "both"):
        print(f"\n--- Evaluating Teacher (AWQ): {TEACHER_MODEL_ID_AWQ} ---")
        _run_eval(
            "teacher",
            TEACHER_MODEL_ID_AWQ,
            prompts,
            ground_truths,
            df,
            batch_size,
            n_problems,
            sampled_indices,
            extract_boxed_answer,
            compute_accuracy,
            compute_accuracy_by_group,
            evaluate_completions,
            format_eval_report,
            save_results,
        )


def _run_eval(
    label,
    model_id,
    prompts,
    ground_truths,
    df,
    batch_size,
    n_problems,
    sampled_indices,
    extract_boxed_answer,
    compute_accuracy,
    compute_accuracy_by_group,
    evaluate_completions,
    format_eval_report,
    save_results,
):
    """Run evaluation for a single model."""
    generate_fn = generate_teacher if label == "teacher" else generate_student

    # Split prompts into batches
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    print(f"  Dispatching {len(batches)} batches across parallel containers...")

    # Fan out all batches in parallel via Modal .starmap()
    call_args = [(batch, model_id, 0.0, 3072) for batch in batches]
    all_completions = []
    all_finish_reasons = []
    for result_batch in generate_fn.starmap(call_args, order_outputs=True):
        all_completions.extend([r["text"] for r in result_batch])
        all_finish_reasons.extend([r.get("finish_reason", "unknown") for r in result_batch])

    # Track truncation stats
    n_null = sum(1 for c in all_completions if extract_boxed_answer(c) is None)
    n_length = sum(1 for r in all_finish_reasons if r == "length")
    n_stop = sum(1 for r in all_finish_reasons if r == "stop")
    truncation_rate = n_null / len(all_completions) if all_completions else 0.0
    print(f"  No \\boxed{{}}: {n_null}/{len(all_completions)} ({truncation_rate:.1%})")
    print(f"  Finish reasons: stop={n_stop}, length={n_length}")

    # Evaluate
    eval_results = evaluate_completions(all_completions, ground_truths)
    overall_acc = compute_accuracy(eval_results)
    by_level = compute_accuracy_by_group(eval_results, df, "level")
    by_type = compute_accuracy_by_group(eval_results, df, "type")

    report = format_eval_report(
        model_name=model_id,
        overall_accuracy=overall_acc,
        by_level=by_level,
        by_type=by_type,
        num_problems=len(eval_results),
    )
    print(report)

    # Save detailed results with metadata
    import json

    output_path = f"logs/{label}/baseline_{n_problems}.json"
    save_results(eval_results, df, output_path)

    with open(output_path, "r") as f:
        data = json.load(f)

    n_correct = sum(1 for r in eval_results if r["correct"])

    metadata = {
        "model_id": model_id,
        "n_problems": n_problems,
        "n_correct": n_correct,
        "n_null_predictions": n_null,
        "accuracy": overall_acc,
        "finish_reasons": {"stop": n_stop, "length": n_length},
    }
    if sampled_indices is not None:
        metadata["seed"] = SEED
        metadata["sampled_indices"] = sampled_indices

    wrapped = {"metadata": metadata, "results": data}
    with open(output_path, "w") as f:
        json.dump(wrapped, f, indent=2)
    print(f"Detailed results saved to {output_path}")
