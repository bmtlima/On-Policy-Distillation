"""Student model setup for Qwen3-1.7B with LoRA.

Uses PEFT LoRA (rank 32) — only ~0.5% of parameters are trainable.
"""

from __future__ import annotations

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modal_app import STUDENT_MODEL_ID

LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    bias="none",
)


def load_student_model(
    model_id: str = STUDENT_MODEL_ID,
    cache_dir: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
):
    """Load the student model with LoRA adapters.

    Args:
        model_id: HuggingFace model ID.
        cache_dir: Directory for cached model weights.
        device_map: Device placement strategy.
        dtype: Model dtype — "bfloat16" for training, "float16" for inference.

    Returns:
        (model, tokenizer) tuple. model is a PeftModel with LoRA.
    """
    import torch

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Apply LoRA — freezes base model, only adapter weights are trainable
    model = get_peft_model(base_model, LORA_CONFIG)

    # Gradient checkpointing: recompute activations during backward instead of
    # storing them for all layers. Cuts activation memory ~5x at ~30% more compute.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total:,} total, {trainable:,} trainable ({trainable/total:.1%})")
    print("Gradient checkpointing: enabled")

    return model, tokenizer


def save_checkpoint(model, tokenizer, path: str) -> None:
    """Save LoRA adapter weights and tokenizer.

    With a PeftModel, save_pretrained() only writes the adapter files
    (adapter_config.json + adapter_model.safetensors), which are ~20MB at rank 32.
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def save_full_checkpoint(model, tokenizer, path: str) -> None:
    """Merge LoRA into base model and save full weights for standalone inference.

    Used for the final checkpoint that can be loaded directly by vLLM
    without PEFT/LoRA support.
    """
    merged = model.merge_and_unload()
    # Ensure use_cache=True for inference compatibility
    merged.config.use_cache = True
    merged.save_pretrained(path)
    tokenizer.save_pretrained(path)
