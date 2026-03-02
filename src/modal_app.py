"""Modal app definition with images, volumes, and GPU configurations."""

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------
app = modal.App("opd-metis")

# ---------------------------------------------------------------------------
# Shared volume – caches model weights across runs
# ---------------------------------------------------------------------------
model_cache = modal.Volume.from_name("opd-metis-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("opd-metis-checkpoints", create_if_missing=True)

MODEL_CACHE_DIR = "/root/model-cache"
CHECKPOINT_DIR = "/root/checkpoints"

# ---------------------------------------------------------------------------
# Base image – shared deps for all functions
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.45.0",

        "datasets>=3.0.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "tqdm>=4.66.0",
        "pyarrow>=14.0.0",
        "wandb>=0.18.0",
        "huggingface_hub>=0.25.0",
        "accelerate>=0.34.0",
        "sentencepiece>=0.2.0",
    )
)

# ---------------------------------------------------------------------------
# Training image – base deps + local source/data for remote execution
# ---------------------------------------------------------------------------
training_image = (
    base_image
    .add_local_python_source("src")
    .add_local_file("answer_extraction.py", remote_path="/root/answer_extraction.py")
    .add_local_dir("data", remote_path="/root/data")
)

# ---------------------------------------------------------------------------
# vLLM image – for inference (eval + rollouts)
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "tqdm>=4.66.0",
        "pyarrow>=14.0.0",
    )
    .add_local_python_source("src")
    .add_local_file("answer_extraction.py", remote_path="/root/answer_extraction.py")
)

# ---------------------------------------------------------------------------
# GPU configurations
# ---------------------------------------------------------------------------
TEACHER_GPU = "H200"  # 32B model for baseline eval via vLLM
STUDENT_GPU = "A10G"  # 0.6B model, plenty of room
TRAINING_GPU = "H200"  # Co-located teacher+student for OPD

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
TEACHER_MODEL_ID = "Qwen/Qwen3-8B"
TEACHER_MODEL_ID_AWQ = "Qwen/Qwen3-8B"  # For vLLM baselines
STUDENT_MODEL_ID = "Qwen/Qwen3-0.6B"

# ---------------------------------------------------------------------------
# Quantization settings (vLLM baseline inference only)
# ---------------------------------------------------------------------------
TEACHER_QUANTIZATION = None  # No quantization needed for 8B
