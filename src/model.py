"""LoRA model setup for Qwen3-1.7B student.

Applies LoRA adapters targeting attention projection layers
with rank 64 for a good balance of capacity vs efficiency.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modal_app import STUDENT_MODEL_ID

# LoRA configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,  # alpha = 2*r is a common choice
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)


def load_student_model(
    model_id: str = STUDENT_MODEL_ID,
    cache_dir: str | None = None,
    apply_lora: bool = True,
    device_map: str = "auto",
    dtype: str = "bfloat16",
):
    """Load the student model with optional LoRA adapters.

    Args:
        model_id: HuggingFace model ID.
        cache_dir: Directory for cached model weights.
        apply_lora: Whether to apply LoRA adapters.
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

    if apply_lora:
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    return model, tokenizer


def load_student_from_checkpoint(
    checkpoint_path: str,
    model_id: str = STUDENT_MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
):
    """Load student model with LoRA weights from a checkpoint.

    Args:
        checkpoint_path: Path to saved LoRA adapter weights.
        model_id: Base model HuggingFace ID.
        cache_dir: Directory for cached model weights.
        device_map: Device placement strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model, tokenizer


def save_lora_checkpoint(model, path: str) -> None:
    """Save only the LoRA adapter weights."""
    model.save_pretrained(path)
