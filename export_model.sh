#!/bin/bash
# Phase 6 — Model Fusion & GGUF Model Export Pipeline
# Merges trained adapter weights and base weights natively into standalone public directories

set -eo pipefail

# Ensure cache directory paths are resolved correctly
export HF_HOME="./models"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

echo "========================================================="
echo "        PHASE 6: MODEL FUSION & GGUF EXPORT              "
echo "========================================================="

# Create saving directories
mkdir -p fused_model

BASE_MODEL="./models/Meta-Llama-3-8B-Instruct-4bit"
if [ ! -d "$BASE_MODEL" ]; then
    BASE_MODEL="mlx-community/Meta-Llama-3-8B-Instruct-4bit"
fi

echo "Ingesting Base model:    $BASE_MODEL"
echo "Trained Adapters source: ./adapters/"
echo "Saving Destination:      ./fused_model/"
echo "Target GGUF file:        ./fused_model/scripture_model.gguf"
echo "---------------------------------------------------------"

echo "Starting model weights fusion..."
uv run python export_model.py

echo ""
echo "Model fusion completed successfully!"
echo "Standalone Model Folder: $(pwd)/fused_model"
echo "========================================================="
exit 0
