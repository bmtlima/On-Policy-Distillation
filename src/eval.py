"""Evaluation pipeline for MATH benchmark.

Loads test set, runs batch inference, extracts boxed answers,
and reports accuracy by problem level and type.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Import from the project root (answer_extraction.py is at repo root)
import answer_extraction


def load_math_dataset(
    split: str = "test",
    data_dir: str = "data",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load MATH dataset from parquet files.

    Args:
        split: "train" or "test".
        data_dir: Path to data directory.
        limit: If set, only load this many problems (for fast iteration).

    Returns:
        DataFrame with columns: problem, level, type, solution, answer.
    """
    path = Path(data_dir) / f"hendrycks_math_{split}.parquet"
    df = pd.read_parquet(path)

    if limit is not None:
        df = df.head(limit)

    return df


def evaluate_completions(
    completions: list[str],
    ground_truths: list[str],
) -> list[dict]:
    """Score a batch of model completions against ground truth.

    Args:
        completions: Model output strings.
        ground_truths: Ground truth answer strings.

    Returns:
        List of result dicts with predicted, ground_truth, correct.
    """
    results = []
    for completion, gt in zip(completions, ground_truths):
        result = answer_extraction.extract_and_compare(completion, gt)
        results.append(result)
    return results


def compute_accuracy(results: list[dict]) -> float:
    """Compute overall accuracy from evaluation results."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)


def compute_accuracy_by_group(
    results: list[dict],
    df: pd.DataFrame,
    group_col: str,
) -> dict[str, dict]:
    """Compute accuracy grouped by a column (level or type).

    Returns dict mapping group value to {correct, total, accuracy}.
    """
    grouped = {}
    for i, (result, (_, row)) in enumerate(zip(results, df.iterrows())):
        group = str(row[group_col])
        if group not in grouped:
            grouped[group] = {"correct": 0, "total": 0}
        grouped[group]["total"] += 1
        if result["correct"]:
            grouped[group]["correct"] += 1

    for group in grouped:
        g = grouped[group]
        g["accuracy"] = g["correct"] / g["total"] if g["total"] > 0 else 0.0

    return dict(sorted(grouped.items()))


def format_eval_report(
    model_name: str,
    overall_accuracy: float,
    by_level: dict[str, dict],
    by_type: dict[str, dict],
    num_problems: int,
) -> str:
    """Format evaluation results into a readable report."""
    lines = [
        f"{'=' * 60}",
        f"Evaluation Report: {model_name}",
        f"{'=' * 60}",
        f"Total problems: {num_problems}",
        f"Overall accuracy: {overall_accuracy:.1%}",
        "",
        "Accuracy by Level:",
        "-" * 40,
    ]
    for level, stats in by_level.items():
        lines.append(
            f"  {level}: {stats['accuracy']:.1%} "
            f"({stats['correct']}/{stats['total']})"
        )

    lines.extend(["", "Accuracy by Type:", "-" * 40])
    for ptype, stats in by_type.items():
        lines.append(
            f"  {ptype}: {stats['accuracy']:.1%} "
            f"({stats['correct']}/{stats['total']})"
        )

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def save_results(
    results: list[dict],
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save detailed per-problem results to JSON."""
    detailed = []
    for result, (_, row) in zip(results, df.iterrows()):
        detailed.append(
            {
                "problem": row["problem"][:200],  # truncate for readability
                "level": row.get("level", ""),
                "type": row.get("type", ""),
                "ground_truth": result["ground_truth"],
                "predicted": result["predicted"],
                "correct": result["correct"],
            }
        )

    with open(output_path, "w") as f:
        json.dump(detailed, f, indent=2)
