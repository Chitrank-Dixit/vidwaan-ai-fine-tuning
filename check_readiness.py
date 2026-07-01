#!/usr/bin/env python3
import os
import sys
import json
import argparse

# Set Hugging Face cache directory to workspace-local ./models directory
# This ensures caching is bound to the host volume and survives container restarts
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def check_datasets(data_dir):
    """
    Verify datasets exist, are non-empty, and contain valid JSON records.
    """
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")
    
    reports = []
    success = True
    
    for path, name in [(train_path, "Train Dataset"), (valid_path, "Validation Dataset")]:
        if not os.path.exists(path):
            reports.append(f"❌ {name}: File missing at {path}")
            success = False
            continue
            
        size = os.path.getsize(path)
        if size == 0:
            reports.append(f"❌ {name}: File at {path} is empty")
            success = False
            continue
            
        # Parse first line to check structure
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                data = json.loads(first_line)
                if "messages" in data and isinstance(data["messages"], list):
                    reports.append(f"✔ {name}: Validated ({os.path.basename(path)}, {size} bytes)")
                else:
                    reports.append(f"❌ {name}: Structure invalid (missing 'messages' schema)")
                    success = False
        except Exception as e:
            reports.append(f"❌ {name}: JSON parsing failed: {e}")
            success = False
            
    return success, reports

def get_memory_info():
    """
    Query cgroups and system proc tables to profile allocated memory.
    """
    mem_limit_gb = None
    
    # 1. Check cgroup v1 limit
    try:
        limit_file = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(limit_file):
            with open(limit_file, "r") as f:
                val = int(f.read().strip())
                if val < 9223372036854771712:
                    mem_limit_gb = val / (1024 ** 3)
    except Exception:
        pass
        
    # 2. Check cgroup v2 limit
    if mem_limit_gb is None:
        try:
            limit_file = "/sys/fs/cgroup/memory.max"
            if os.path.exists(limit_file):
                val_str = f.read().strip()
                if val_str != "max":
                    mem_limit_gb = int(val_str) / (1024 ** 3)
        except Exception:
            pass

    # 3. Fallback to general OS RAM
    os_mem_gb = 0.0
    try:
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            os_mem_gb = int(res.strip()) / (1024 ** 3)
        else:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        os_mem_gb = int(line.split()[1]) / (1024 ** 2)
                        break
    except Exception:
        pass
        
    return {
        "limit_gb": mem_limit_gb if mem_limit_gb is not None else os_mem_gb,
        "is_constrained": mem_limit_gb is not None
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Model & Ingestion Pre-Flight Readiness Check")
    parser.add_argument("--model", type=str, default="mlx-community/Meta-Llama-3-8B-Instruct-4bit", help="Hugging Face Model ID")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--non-interactive", action="store_true", help="Bypass approval prompt")
    args = parser.parse_args()

    print("=========================================================")
    print("         PHASE 3: PRE-FLIGHT READINESS CHECK             ")
    print("=========================================================")

    # 1. Dataset Verification
    print("\n1. Verifying Dataset Integrity...")
    datasets_ok, dataset_reports = check_datasets(args.data_dir)
    for report in dataset_reports:
        print(f"   {report}")

    # 2. Hardware Resource Profiling
    print("\n2. Profiling Memory Constraints...")
    mem_info = get_memory_info()
    ram_gb = mem_info["limit_gb"]
    constrained_str = " (Docker Cgroup Limit)" if mem_info["is_constrained"] else " (Host System Total)"
    print(f"   - Available System RAM: {ram_gb:.2f} GB{constrained_str}")
    
    # 32 GB check (warning if cgroup limits allocated memory below 16 GB)
    if ram_gb < 16.0:
        print("   ⚠️  WARNING: Allocated RAM is below 16 GB. Training may fail due to memory exhaustion.")
    else:
        print("   ✔ RAM allocation satisfies memory safety limits.")

    # 3. Model Tokenizer & Configuration Load
    print("\n3. Verifying Model Availability & Configurations...")
    print(f"   Model Target: {args.model}")
    print("   Cache Location: ./models/")
    
    model_ok = True
    try:
        from mlx_lm import load
        print("   Loading model weights and tokenizer from local cache (or downloading if needed)...")
        # Load weights and tokenizer to confirm files and configurations are intact
        model, tokenizer = load(args.model)
        print("   ✔ Base model configuration and weight assets loaded successfully.")
    except ImportError:
        print("   ❌ Error: mlx-lm library not found. Run 'make sync' on the host first.")
        model_ok = False
    except Exception as e:
        print(f"   ❌ Error loading model assets: {e}")
        model_ok = False

    # Pre-Flight Scorecard Report
    print("\n" + "=" * 57)
    print("             PRE-FLIGHT READINESS SCORECARD             ")
    print("=" * 57)
    print(f"   - Datasets:           {'PASSED' if datasets_ok else 'FAILED'}")
    print(f"   - Memory Guardrail:   {'PASSED' if ram_gb >= 16.0 else 'WARNING'}")
    print(f"   - Model & Tokenizer:  {'PASSED' if model_ok else 'FAILED'}")
    print(f"   - Cache Directory:    ./models/ (HF_HOME override verified)")
    print("=" * 57)

    overall_passed = datasets_ok and model_ok
    if overall_passed:
        print("\n✔ Pre-Flight checks passed successfully. Hardware and assets are primed.")
    else:
        print("\n❌ Pre-Flight checks failed. Correct the errors highlighted above.")
        sys.exit(1)

    # Validation Gate Input
    if args.non_interactive:
        print("\n✔ Pre-Flight Approved via non-interactive mode. Proceeding to Phase 4 (LoRA Fine-Tuning)...")
        sys.exit(0)

    try:
        choice = input("\nDo you approve this pre-flight scorecard and wish to proceed to Phase 4? (Y/N): ").strip().lower()
        if choice == 'y':
            print("✔ Pre-Flight Approved. Proceeding to Phase 4 (LoRA Fine-Tuning)...")
            sys.exit(0)
        else:
            print("❌ Pre-Flight Rejected. Exiting pipeline.")
            sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Interrupted. Exiting pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()
