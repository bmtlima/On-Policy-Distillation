"""Run baseline evaluations for student and teacher models on MATH.

Usage:
    modal run run_baselines.py                    # Full test set
    modal run run_baselines.py --limit 100        # Quick test with 100 problems
    modal run run_baselines.py --model student     # Student only
    modal run run_baselines.py --model teacher     # Teacher only
"""

from __future__ import annotations

import random

from src.modal_app import (
    STUDENT_MODEL_ID,
    TEACHER_MODEL_ID_AWQ,
    app,
)
from src.inference import generate_student, generate_teacher
from src.prompts import format_problems_batch

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


@app.local_entrypoint()
def main(
    limit: int = 0,
    model: str = "both",
    batch_size: int = 64,
    data_dir: str = "data",
):
    """Run baseline evaluations.

    Args:
        limit: Number of problems to evaluate (0 = all).
        model: Which model to evaluate: "student", "teacher", or "both".
        batch_size: Inference batch size.
        data_dir: Path to data directory.
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
    from answer_extraction import extract_boxed_answer

    limit_val = limit if limit > 0 else None
    df = load_math_dataset(split="test", data_dir=data_dir, limit=limit_val)
    problems = df["problem"].tolist()
    ground_truths = df["answer"].tolist()

    print(f"Loaded {len(problems)} test problems")

    # Format prompts for vLLM (plain text, not chat template)
    messages_batch = format_problems_batch(problems)

    if model in ("student", "both"):
        print(f"\n--- Evaluating Student: {STUDENT_MODEL_ID} ---")
        _run_eval(
            "student",
            STUDENT_MODEL_ID,
            messages_batch,
            ground_truths,
            df,
            batch_size,
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
            messages_batch,
            ground_truths,
            df,
            batch_size,
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
    messages_batch,
    ground_truths,
    df,
    batch_size,
    extract_boxed_answer,
    compute_accuracy,
    compute_accuracy_by_group,
    evaluate_completions,
    format_eval_report,
    save_results,
):
    """Run evaluation for a single model."""
    generate_fn = generate_teacher if label == "teacher" else generate_student

    # Format as plain text prompts for vLLM
    prompts = []
    for msgs in messages_batch:
        parts = []
        for msg in msgs:
            parts.append(f"{msg['role']}: {msg['content']}")
        prompts.append("\n\n".join(parts))

    # Process in batches via Modal remote calls
    all_completions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}")
        results = generate_fn.remote(
            prompts=batch,
            model_id=model_id,
            temperature=0.0,  # Greedy for eval
            max_tokens=2048,
        )
        all_completions.extend([r["text"] for r in results])

    # Track truncation rate (completions missing \boxed{})
    n_truncated = sum(1 for c in all_completions if extract_boxed_answer(c) is None)
    truncation_rate = n_truncated / len(all_completions) if all_completions else 0.0
    print(f"  Truncation rate (no \\boxed{{}}): {truncation_rate:.1%} ({n_truncated}/{len(all_completions)})")

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

    # Save detailed results
    output_path = f"results_{label}_baseline.json"
    save_results(eval_results, df, output_path)
    print(f"Detailed results saved to {output_path}")
