#!/bin/bash
set -e

echo "=== nanoGPT Training Setup for vast.ai ==="
echo ""

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
python train_gpt2.py

echo "Training complete!"
