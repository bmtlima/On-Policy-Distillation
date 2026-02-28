"""Run on-policy distillation training on Modal.

Co-locates teacher (NF4 ~18GB) and student (BF16+LoRA ~5GB) on a single A100-80GB.

Usage:
    modal run run_train.py                         # Full training run
    modal run run_train.py --num-steps 10          # Quick test (10 steps)
    modal run run_train.py --skip-sanity-check     # Skip teacher sanity check
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
    timeout=14400,  # 4 hours
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train(
    num_steps: int = 150,
    batch_size: int = 8,
    num_samples_per_prompt: int = 4,
    lr: float = 1e-5,
    max_new_tokens: int = 2048,
    warmup_steps: int = 10,
    checkpoint_every: int = 25,
    limit: int = 0,
    skip_sanity_check: bool = False,
    wandb_run_name: str | None = None,
):
    """Run OPD training on a single A100-80GB.

    Loads both teacher (NF4) and student (BF16+LoRA) on the same GPU,
    then runs the on-policy distillation loop.
    """
    import torch

    from src.eval import load_math_dataset
    from src.model import load_student_model
    from src.teacher import load_teacher_model
    from src.train import TrainConfig, set_seeds, teacher_sanity_check, train_opd

    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load teacher model with NF4 quantization (~18GB)
    print("\nLoading teacher model (NF4 quantized)...")
    teacher_model, tokenizer = load_teacher_model(
        model_id=TEACHER_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        quantize_nf4=True,
    )
    print(f"  Teacher loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Load student model with LoRA (~5GB)
    print("Loading student model (BF16 + LoRA)...")
    student_model, _ = load_student_model(
        model_id=STUDENT_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        apply_lora=True,
        dtype="bfloat16",
    )
    print(f"  Student loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Load training data
    limit_val = limit if limit > 0 else None
    train_df = load_math_dataset(split="train", data_dir="data", limit=limit_val)
    print(f"\nLoaded {len(train_df)} training problems")

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
            device=device,
            num_problems=20,
        )
        if not ok:
            print("\nTeacher sanity check FAILED. Aborting training.")
            print("Debug the quantized teacher before proceeding.")
            return

    # Configure training
    config = TrainConfig(
        lr=lr,
        num_steps=num_steps,
        batch_size=batch_size,
        num_samples_per_prompt=num_samples_per_prompt,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        checkpoint_every=checkpoint_every,
        wandb_run_name=wandb_run_name,
    )

    # Run training
    train_opd(
        config=config,
        train_df=train_df,
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Commit volume to persist checkpoints
    checkpoint_vol.commit()
    print("\nTraining complete. Checkpoints saved to Modal volume.")


@app.local_entrypoint()
def main(
    num_steps: int = 150,
    batch_size: int = 8,
    num_samples_per_prompt: int = 4,
    lr: float = 1e-5,
    max_new_tokens: int = 2048,
    warmup_steps: int = 10,
    checkpoint_every: int = 25,
    limit: int = 0,
    skip_sanity_check: bool = False,
    wandb_run_name: str = "",
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
        limit=limit,
        skip_sanity_check=skip_sanity_check,
        wandb_run_name=wandb_run_name if wandb_run_name else None,
    )
