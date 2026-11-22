#!/usr/bin/env python3
"""Phase 3 Model & Ingestion Pre-Flight Readiness Check.

Verifies dataset integrity, profiles system RAM and NVIDIA CUDA GPU resources
(RTX 5060 Ti / 16GB VRAM), checks bfloat16 support, and audits the runtime environment.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure Hugging Face cache directory is workspace-local
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def check_datasets(data_dir: str) -> Tuple[bool, List[str]]:
    """Verify datasets exist, are non-empty, and contain valid JSON records."""
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


def get_memory_info() -> Dict[str, Any]:
    """Query cgroups and system proc tables to profile allocated memory."""
    mem_limit_gb = None

    # 1. Check cgroup v1 limit
    try:
        limit_file = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(limit_file):
            with open(limit_file, "r") as f:
                val = int(f.read().strip())
                if val < 9223372036854771712:
                    mem_limit_gb = val / (1024**3)
    except Exception:
        pass

    # 2. Check cgroup v2 limit
    if mem_limit_gb is None:
        try:
            limit_file = "/sys/fs/cgroup/memory.max"
            if os.path.exists(limit_file):
                with open(limit_file, "r") as f:
                    val_str = f.read().strip()
                    if val_str != "max":
                        mem_limit_gb = int(val_str) / (1024**3)
        except Exception:
            pass

    # 3. Fallback to general OS RAM
    os_mem_gb = 0.0
    try:
        if sys.platform == "darwin":
            import subprocess

            res = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            os_mem_gb = int(res.strip()) / (1024**3)
        else:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        os_mem_gb = int(line.split()[1]) / (1024**2)
                        break
    except Exception:
        pass

    return {
        "limit_gb": mem_limit_gb if mem_limit_gb is not None else os_mem_gb,
        "is_constrained": mem_limit_gb is not None,
    }


def check_gpu_environment() -> Dict[str, Any]:
    """Check NVIDIA GPU, CUDA capability, VRAM headroom, and bfloat16 support."""
    gpu_info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "N/A",
        "vram_gb": 0.0,
        "bf16_supported": False,
        "compute_capability": (0, 0),
    }

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info["vram_gb"] = round(props.total_memory / (1024**3), 2)
            gpu_info["bf16_supported"] = torch.cuda.is_bf16_supported()
            gpu_info["compute_capability"] = (props.major, props.minor)
    except Exception as e:
        gpu_info["error"] = str(e)

    return gpu_info


def main():
    default_model = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    try:
        import yaml

        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "model" in cfg:
                    default_model = cfg["model"]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="VidwaanAI Pre-Flight Readiness & Diagnostic Check"
    )
    parser.add_argument("--model", type=str, default=default_model, help="Target Model ID")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Bypass manual approval prompt"
    )
    args = parser.parse_args()

    print("=========================================================")
    print("         VIDWAAN-AI PRE-FLIGHT READINESS CHECK           ")
    print("=========================================================")

    # 1. Dataset Verification
    print("\n1. Verifying Dataset Integrity...")
    datasets_ok, dataset_reports = check_datasets(args.data_dir)
    for report in dataset_reports:
        print(f"   {report}")

    # 2. GPU & CUDA Acceleration Check
    print("\n2. Profiling GPU & CUDA Capabilities...")
    gpu = check_gpu_environment()
    if gpu["cuda_available"]:
        print(f"   ✔ CUDA Available: Yes ({gpu['device_count']} GPU detected)")
        print(f"   ✔ Device:         {gpu['device_name']}")
        print(f"   ✔ Total VRAM:     {gpu['vram_gb']} GB")
        print(f"   ✔ Compute Cap:    {gpu['compute_capability'][0]}.{gpu['compute_capability'][1]}")
        print(f"   ✔ bfloat16:       {'Supported' if gpu['bf16_supported'] else 'Not Supported'}")
        if gpu["vram_gb"] >= 15.0:
            print("   ✔ Target 16GB VRAM profile: READY for QLoRA 4-bit SFT.")
    else:
        print("   ℹ️ CUDA Not Detected on current process (CPU-only / host mode).")
        print("     Ensure execution runs inside WSL2 or Podman with NVIDIA CDI passthrough.")

    # 3. System RAM Profiling
    print("\n3. Profiling System Host RAM...")
    mem_info = get_memory_info()
    ram_gb = mem_info["limit_gb"]
    constrained_str = " (Cgroup Limit)" if mem_info["is_constrained"] else " (Total Available)"
    print(f"   RAM Capacity:     {ram_gb:.1f} GB{constrained_str}")

    # 4. Final Verdict
    print("\n" + "=" * 57)
    if not datasets_ok:
        print("❌ PRE-FLIGHT VERDICT: FAILED. Please resolve dataset issues.")
        sys.exit(1)
    else:
        print("✔ PRE-FLIGHT VERDICT: PASSED. Pipeline ready for execution.")
    print("=========================================================")


if __name__ == "__main__":
    main()
