"""Run on-policy distillation training on Modal.

Co-locates teacher (BF16 ~17GB) and student (BF16 ~1.2GB) on a single H200.

Usage:
    modal run run_train.py                                          # Full training run
    modal run run_train.py --num-steps 10                           # Quick test (10 steps)
    modal run run_train.py --teacher-model-name Qwen/Qwen3-14B     # Use a different teacher
    modal run run_train.py --skip-sanity-check                      # Skip teacher sanity check
"""

from __future__ import annotations

import modal

from src.modal_app import (
    CHECKPOINT_DIR,
    MODEL_CACHE_DIR,
    STUDENT_MODEL_ID,
    TEACHER_MODEL_ID,
    TRAINING_GPU,
    app,
    checkpoint_vol,
    model_cache,
    training_image,
)


@app.function(
    image=training_image,
    gpu=TRAINING_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_vol},
    timeout=21600,  # 6 hours
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret-bruno"),
    ],
)
def train(
    num_steps: int = 150,
    batch_size: int = 128,
    num_samples_per_prompt: int = 1,
    lr: float = 1e-5,
    max_new_tokens: int = 1500,
    warmup_steps: int = 10,
    checkpoint_every: int = 25,
    eval_every: int = 25,
    limit: int = 0,
    skip_sanity_check: bool = False,
    wandb_run_name: str | None = None,
    teacher_model_name: str = TEACHER_MODEL_ID,
):
    """Run OPD training on 2x H200.

    GPU 0: vLLM for fast rollout generation.
    GPU 1: HF teacher (scoring) + student (LoRA training).
    """
    import torch

    from src.eval import load_math_dataset
    from src.model import load_student_model
    from src.teacher import load_teacher_model
    from src.train import TrainConfig, set_seeds, teacher_sanity_check, train_opd

    set_seeds()
    train_device = torch.device("cuda:1")
    print(f"Training device: {train_device}")
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # Load teacher model on cuda:1 (BF16, no quantization — clean logprobs)
    print(f"\nLoading teacher model ({teacher_model_name}, BF16) on cuda:1...")
    teacher_model, tokenizer = load_teacher_model(
        model_id=teacher_model_name,
        cache_dir=MODEL_CACHE_DIR,
        device_map="cuda:1",
    )
    print(f"  Teacher loaded. GPU 1 mem: {torch.cuda.memory_allocated(1) / 1e9:.1f} GB")

    # Load student model on cuda:1 with LoRA (~3.6GB base + ~20MB adapter in BF16)
    print("Loading student model (BF16, LoRA) on cuda:1...")
    student_model, _ = load_student_model(
        model_id=STUDENT_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        device_map="cuda:1",
        dtype="bfloat16",
    )
    print(f"  Student loaded. GPU 1 mem: {torch.cuda.memory_allocated(1) / 1e9:.1f} GB")

    # Peak memory after loading both models
    peak_mem = torch.cuda.max_memory_allocated(1) / 1e9
    print(f"\n  Peak GPU 1 memory after loading both models: {peak_mem:.1f} GB")

    # Load training data
    limit_val = limit if limit > 0 else None
    train_df = load_math_dataset(split="train", data_dir="data", limit=limit_val)
    print(f"\nLoaded {len(train_df)} training problems")

    # Load eval set (stratified 200 problems from test split, seeded for reproducibility)
    full_test_df = load_math_dataset(split="test", data_dir="data")
    eval_df = full_test_df.sample(n=200, random_state=42)
    print(f"Loaded {len(eval_df)} eval problems (test split, stratified sample)")
    print(f"  Types: {eval_df['type'].value_counts().to_dict()}")
    print(f"  Levels: {eval_df['level'].value_counts().sort_index().to_dict()}")

    # Teacher sanity check
    if not skip_sanity_check:
        problems = train_df["problem"].tolist()
        gts = train_df["answer"].tolist()
        ok = teacher_sanity_check(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            problems=problems,
            ground_truths=gts,
            device=train_device,
            num_problems=20,
        )
        if not ok:
            print("\nTeacher sanity check WARNING: continuing training anyway.")

    # Configure training
    config = TrainConfig(
        lr=lr,
        num_steps=num_steps,
        batch_size=batch_size,
        num_samples_per_prompt=num_samples_per_prompt,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        wandb_run_name=wandb_run_name,
    )

    # Run training
    train_opd(
        config=config,
        train_df=train_df,
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        device=train_device,
        checkpoint_dir=CHECKPOINT_DIR,
        eval_df=eval_df,
    )

    # Commit volume to persist checkpoints
    checkpoint_vol.commit()
    print("\nTraining complete. Checkpoints saved to Modal volume.")


@app.local_entrypoint()
def main(
    num_steps: int = 150,
    batch_size: int = 128,
    num_samples_per_prompt: int = 1,
    lr: float = 1e-5,
    max_new_tokens: int = 1500,
    warmup_steps: int = 10,
    checkpoint_every: int = 25,
    eval_every: int = 25,
    limit: int = 0,
    skip_sanity_check: bool = False,
    wandb_run_name: str = "",
    teacher_model_name: str = "",
):
    """Local entry point that dispatches to Modal."""
    train.remote(
        num_steps=num_steps,
        batch_size=batch_size,
        num_samples_per_prompt=num_samples_per_prompt,
        lr=lr,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        limit=limit,
        skip_sanity_check=skip_sanity_check,
        wandb_run_name=wandb_run_name if wandb_run_name else None,
        teacher_model_name=teacher_model_name if teacher_model_name else TEACHER_MODEL_ID,
    )
