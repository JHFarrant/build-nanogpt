#!/bin/bash
set -e

echo "=== nanoGPT Training Setup for vast.ai ==="
echo ""

# Safer defaults for 24GB-class GPUs like RTX 3090.
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Put Hugging Face caches on larger disk mount to avoid filling root (/)
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Install dependencies
echo "[1/3] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Download and tokenize data
echo "[2/3] Downloading and tokenizing FineWeb-Edu dataset (this may take 10-30 minutes)..."
python fineweb.py

# Train the model
echo "[3/3] Starting training..."
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE SEQUENCE_LENGTH=$SEQUENCE_LENGTH"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs, using DDP with torchrun"
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train_gpt2.py
else
    echo "Single GPU detected"
    python train_gpt2.py
fi

echo "Training complete!"
