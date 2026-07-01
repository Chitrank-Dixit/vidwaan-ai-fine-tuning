#!/bin/bash
# Phase 4 — Low-Rank Adaptation (LoRA) Local Training Execution Script
# Mathematical memory safety guardrails for M1 32GB RAM local environments

set -eo pipefail

# Ensure Hugging Face utilizes local models directory for caching weights
export HF_HOME="./models"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

echo "========================================================="
# Step 1: Pre-Flight Sanity Checks (Bypassing interactive prompts)
echo "[1/4] Running automated pre-flight checks..."
if ! uv run python check_readiness.py --non-interactive; then
    echo "❌ ERROR: Pre-flight sanity checks failed. Aborting training run to protect hardware."
    exit 1
fi

# Step 2: Set up memory headroom monitor & OOM shield
echo ""
echo "[2/4] Initializing memory headroom monitor..."

# Cross-platform memory checking helper
check_free_memory_gb() {
    if [ -f "/sys/fs/cgroup/memory/memory.limit_in_bytes" ]; then
        # Linux container (cgroup v1 limit)
        local limit
        limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
        local usage
        usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
        if [ "$limit" -lt 9223372036854771712 ] 2>/dev/null; then
            echo "$(( (limit - usage) / 1024 / 1024 / 1024 ))"
            return
        fi
    fi
    
    if command -v free >/dev/null; then
        # General Linux
        local free_mem
        free_mem=$(free -m | awk '/^Mem:/{print $4}')
        echo "$(( free_mem / 1024 ))"
        return
    fi
    
    if command -v vm_stat >/dev/null; then
        # macOS host
        local page_size
        page_size=$(vm_stat | grep "page size" | awk '{print $8}' | tr -d '.')
        local free_pages
        free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
        local inactive_pages
        inactive_pages=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
        local speculative_pages
        speculative_pages=$(vm_stat | grep "Pages speculative" | awk '{print $3}' | tr -d '.')
        
        local total_pages=$(( free_pages + inactive_pages + speculative_pages ))
        echo "$(( (total_pages * page_size) / 1024 / 1024 / 1024 ))"
        return
    fi
    
    echo "16" # Safe default fallback
}

# Step 3: Run the training engine
echo ""
echo "[3/4] Launching MLX-LM LoRA training engine..."
echo "      Config: config.yaml"
echo "      Output logs: data/training.log"
echo "---------------------------------------------------------"

# Ensure output directory exists
mkdir -p data
mkdir -p adapters

# Launch training in background so we can monitor memory
uv run mlx_lm.lora --config ./config.yaml 2>&1 | tee data/training.log &
TRAINING_PID=$!

# Spawn background memory watchdog
(
    while kill -0 "$TRAINING_PID" 2>/dev/null; do
        sleep 5
        FREE_MEM=$(check_free_memory_gb)
        if [ "$FREE_MEM" -lt 1 ]; then
            echo -e "\n🚨 [WATCHDOG] CRITICAL: Low memory headroom detected (${FREE_MEM} GB free)."
            echo "             Terminating training cleanly to prevent kernel crash / OOM panic..."
            kill -15 "$TRAINING_PID" # Send SIGTERM for graceful exit
            exit 2
        fi
    done
) &
WATCHDOG_PID=$!

# Wait for training to complete
wait "$TRAINING_PID"
TRAIN_EXIT_CODE=$?

# Stop memory watchdog
kill "$WATCHDOG_PID" 2>/dev/null || true

# Handle exit codes
if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo "❌ ERROR: Training engine exited with error code $TRAIN_EXIT_CODE."
    exit "$TRAIN_EXIT_CODE"
fi

# Step 4: Final Compilation Report
echo ""
echo "[4/4] Generating Training Compilation Report..."
echo "========================================================="
echo "             TRAINING RUN COMPLETE                       "
echo "========================================================="

# Extract final loss from training logs if available
if [ -f "data/training.log" ]; then
    echo "Summary of Final Training Iterations (Last 5 lines):"
    grep -E "Iter|Loss|Val" data/training.log | tail -n 5 || true
fi

# Verify compiled adapter files in mapped host volumes
echo ""
echo "Verifying Saved LoRA Adapters..."
ADAPTER_FILE="adapters/adapters.safetensors"
if [ -f "$ADAPTER_FILE" ]; then
    FILE_SIZE=$(wc -c <"$ADAPTER_FILE" | tr -d ' ')
    FILE_SIZE_MB=$(( FILE_SIZE / 1024 / 1024 ))
    echo "✔ Found Adapter File:  $ADAPTER_FILE"
    echo "  - Path:              $(pwd)/$ADAPTER_FILE"
    echo "  - File Size:         $FILE_SIZE_MB MB ($FILE_SIZE bytes)"
else
    echo "⚠️  WARNING: Target adapter file adapters.safetensors not found."
    # Check alternate format extension
    ALT_ADAPTER="adapters/adapters.npz"
    if [ -f "$ALT_ADAPTER" ]; then
        FILE_SIZE=$(wc -c <"$ALT_ADAPTER" | tr -d ' ')
        FILE_SIZE_MB=$(( FILE_SIZE / 1024 / 1024 ))
        echo "✔ Found Adapter File:  $ALT_ADAPTER"
        echo "  - File Size:         $FILE_SIZE_MB MB"
    else
        echo "❌ ERROR: No LoRA adapter files compiled in adapters/."
        exit 1
    fi
fi
echo "========================================================="
echo "Phase 4 Completed. Ready to configure validation/evaluation in Phase 5."
exit 0
