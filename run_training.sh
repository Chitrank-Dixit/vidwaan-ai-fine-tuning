#!/bin/bash
# Phase 4 — VidwaanAI Unsloth QLoRA Training Execution Script
# Optimized for NVIDIA RTX 5060 Ti (16GB VRAM, CUDA 12.x/13.x)

set -eo pipefail

export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

echo "========================================================="
echo "   VIDWAAN-AI UNSLOTH QLORA TRAINING PIPELINE            "
echo "========================================================="

# Step 1: Pre-Flight Readiness Checks
echo "[1/3] Running pre-flight system diagnostics..."
if ! uv run python check_readiness.py --non-interactive; then
    echo "❌ ERROR: Pre-flight readiness checks failed. Aborting training."
    exit 1
fi

# Step 2: Ensure Directories Exist
echo ""
echo "[2/3] Preparing workspace directories..."
mkdir -p data adapters tmp_checkpoints reports

# Step 3: Run the Unsloth QLoRA Training Engine
echo ""
echo "[3/3] Launching Unsloth QLoRA SFT training loop..."
echo "      Configuration: config.yaml"
echo "      Log file:      data/training.log"
echo "---------------------------------------------------------"

uv run python train.py --config config.yaml "$@"

echo ""
echo "========================================================="
echo "✔ Training run complete. Adapters saved to ./adapters/"
echo "========================================================="
exit 0
