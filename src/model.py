"""Student model setup for Qwen3-0.6B.

Full fine-tuning — all 600M parameters are trainable.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modal_app import STUDENT_MODEL_ID


def load_student_model(
    model_id: str = STUDENT_MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
):
    """Load the student model for full fine-tuning.

    Args:
        model_id: HuggingFace model ID.
        cache_dir: Directory for cached model weights.
        device_map: Device placement strategy.
        dtype: Model dtype — "bfloat16" for training, "float16" for inference.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Gradient checkpointing: recompute activations during backward instead of
    # storing them for all 28 layers.  Cuts activation memory ~5x at ~30% more compute.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total:,} total, {trainable:,} trainable ({trainable/total:.1%})")
    print("Gradient checkpointing: enabled")

    return model, tokenizer


def load_student_from_checkpoint(
    checkpoint_path: str,
    model_id: str = STUDENT_MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
):
    """Load student model from a full checkpoint.

    Args:
        checkpoint_path: Path to saved model weights.
        model_id: Base model HuggingFace ID (used for tokenizer).
        cache_dir: Directory for cached model weights.
        device_map: Device placement strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    return model, tokenizer


def save_checkpoint(model, tokenizer, path: str) -> None:
    """Save full model weights and tokenizer.

    Restores use_cache=True in the saved config so vLLM can load the
    checkpoint for inference (gradient checkpointing sets it to False).
    """
    # Temporarily restore use_cache for inference-compatible config
    use_cache_orig = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.save_pretrained(path)
    model.config.use_cache = use_cache_orig
    tokenizer.save_pretrained(path)
