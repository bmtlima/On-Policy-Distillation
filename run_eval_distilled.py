"""Evaluate the distilled student model on MATH.

Usage:
    modal run run_eval_distilled.py                                  # Final checkpoint
    modal run run_eval_distilled.py --checkpoint-name step_100       # Specific checkpoint
    modal run run_eval_distilled.py --limit 100                      # First 100 problems
    modal run run_eval_distilled.py --sample 200                     # Random 200 (seed 42, matches baseline)
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
def patch_checkpoint_config(checkpoint_path: str) -> dict:
    """Fix checkpoint config.json for vLLM compatibility.

    Gradient checkpointing sets use_cache=False during training, which
    breaks vLLM inference. This patches it back to True.
    """
    import json
    import os

    config_path = os.path.join(checkpoint_path, "config.json")
    print(f"Checkpoint contents: {os.listdir(checkpoint_path)}")

    # Remove stale LoRA adapter files — vLLM treats all .safetensors as model
    # weight shards, so adapter_model.safetensors causes key mismatches.
    for stale in ("adapter_model.safetensors", "adapter_config.json"):
        stale_path = os.path.join(checkpoint_path, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"Removed stale LoRA file: {stale}")

    with open(config_path) as f:
        config = json.load(f)

    print(f"Original use_cache: {config.get('use_cache')}")
    config["use_cache"] = True

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Patched use_cache=True")
    checkpoint_vol.commit()
    return {"status": "ok", "files": os.listdir(checkpoint_path)}


@app.function(
    image=vllm_image,
    gpu=STUDENT_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_distilled(
    prompts: list[str],
    checkpoint_path: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> list[dict]:
    """Generate completions from distilled student using vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=checkpoint_path,
        tokenizer=STUDENT_MODEL_ID,
        download_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        max_model_len=8192,
        dtype="float16",
    )

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    outputs = llm.generate(prompts, params)

    results = []
    for output in outputs:
        completion = output.outputs[0]
        results.append({
            "text": completion.text,
            "finish_reason": completion.finish_reason,
        })

    return results


@app.local_entrypoint()
def main(
    limit: int = 0,
    sample: int = 0,
    batch_size: int = 64,
    data_dir: str = "data",
    run_name: str = "",
    checkpoint_name: str = "final_merged",
):
    """Evaluate the distilled model and compare to baseline.

    Args:
        limit: Number of problems sequentially (0 = all).
        sample: Randomly sample N problems (seed 42). Overrides limit.
        batch_size: Inference batch size.
        data_dir: Path to data directory.
        run_name: WandB run name (subdirectory under checkpoints). Required.
        checkpoint_name: Name of checkpoint subdirectory (e.g. "final_merged", "step_100").
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

    # Load dataset with same sampling logic as run_baselines.py
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
    print(f"Checkpoint: {checkpoint_name}")

    # Format prompts using proper chat template
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    messages_batch = format_problems_batch(problems)
    prompts = apply_chat_template_batch(tokenizer, messages_batch)

    # Path to full checkpoint on the Modal volume
    if run_name:
        checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}/{checkpoint_name}"
    else:
        # Legacy: flat checkpoint dir (pre-namespacing)
        checkpoint_path = f"{CHECKPOINT_DIR}/{checkpoint_name}"

    # Patch checkpoint config for vLLM compatibility
    print("Patching checkpoint config...")
    patch_result = patch_checkpoint_config.remote(checkpoint_path=checkpoint_path)
    print(f"  Checkpoint files: {patch_result['files']}")

    # Process in batches
    all_completions = []
    all_finish_reasons = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}")
        results = eval_distilled.remote(
            prompts=batch,
            checkpoint_path=checkpoint_path,
            temperature=0.0,
            max_tokens=3072,
        )
        all_completions.extend([r["text"] for r in results])
        all_finish_reasons.extend([r.get("finish_reason", "unknown") for r in results])

    # Track truncation
    from answer_extraction import extract_boxed_answer

    n_truncated = sum(1 for c in all_completions if extract_boxed_answer(c) is None)
    n_length = sum(1 for r in all_finish_reasons if r == "length")
    n_stop = sum(1 for r in all_finish_reasons if r == "stop")
    truncation_rate = n_truncated / len(all_completions) if all_completions else 0.0
    print(f"  No \\boxed{{}}: {n_truncated}/{len(all_completions)} ({truncation_rate:.1%})")
    print(f"  Finish reasons: stop={n_stop}, length={n_length}")

    # Evaluate
    eval_results = evaluate_completions(all_completions, ground_truths)
    overall_acc = compute_accuracy(eval_results)
    by_level = compute_accuracy_by_group(eval_results, df, "level")
    by_type = compute_accuracy_by_group(eval_results, df, "type")

    report = format_eval_report(
        model_name=f"{STUDENT_MODEL_ID} (distilled, {checkpoint_name})",
        overall_accuracy=overall_acc,
        by_level=by_level,
        by_type=by_type,
        num_problems=len(eval_results),
    )
    print(report)

    # Save detailed results with metadata (matches baseline format)
    import json
    import os

    run_label = f"{run_name}_" if run_name else ""
    output_path = f"logs/student/distilled_{run_label}{checkpoint_name}_{len(eval_results)}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(eval_results, df, output_path)

    with open(output_path, "r") as f:
        data = json.load(f)

    n_correct = sum(1 for r in eval_results if r["correct"])
    n_null = sum(1 for r in eval_results if r["predicted"] is None)

    metadata = {
        "model_id": STUDENT_MODEL_ID,
        "checkpoint": checkpoint_name,
        "n_problems": len(eval_results),
        "n_correct": n_correct,
        "n_null_predictions": n_null,
        "accuracy": overall_acc,
        "finish_reasons": {"stop": n_stop, "length": n_length},
    }
    if sample > 0:
        metadata["seed"] = SEED
        metadata["sampled_indices"] = sampled_indices

    wrapped = {"metadata": metadata, "results": data}
    with open(output_path, "w") as f:
        json.dump(wrapped, f, indent=2)
    print(f"Detailed results saved to {output_path}")
