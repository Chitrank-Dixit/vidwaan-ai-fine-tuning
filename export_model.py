#!/usr/bin/env python3
"""Cross-platform model fusion, weight merging, and multi-format export.

Supports merging LoRA adapters into base weights via Unsloth FastLanguageModel
or standard PEFT merge_and_unload. Exports 16-bit, 4-bit, and GGUF quantization formats.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import unsloth first before transformers/peft
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def export_model_weights(
    model_id: str,
    adapter_path: str = "./adapters",
    fused_path: str = "./fused_model",
    export_format: str = "16bit",
    quantization_method: Optional[str] = "q4_k_m",
):
    """Merge LoRA adapter into base model and save standalone weights."""
    print("=========================================================")
    print("        CROSS-PLATFORM MODEL FUSION & EXPORT             ")
    print("=========================================================")
    print(f"Base Model:       {model_id}")
    print(f"Adapter Source:   {adapter_path}")
    print(f"Fused Target:     {fused_path}")
    print(f"Export Format:    {export_format}")

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory '{adapter_path}' does not exist. Run fine-tuning training first."
        )

    out_path = Path(fused_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Attempt Unsloth merged export
    try:
        from unsloth import FastLanguageModel

        print("\nLoading model and adapter via Unsloth FastLanguageModel...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        if export_format == "gguf":
            print(f"\nExporting to GGUF ({quantization_method})...")
            model.save_pretrained_gguf(
                fused_path,
                tokenizer,
                quantization_method=quantization_method,
            )
        elif export_format == "4bit":
            print("\nExporting to 4-bit merged format...")
            model.save_pretrained_merged(fused_path, tokenizer, save_method="merged_4bit")
        else:
            print("\nExporting to 16-bit merged float format...")
            model.save_pretrained_merged(fused_path, tokenizer, save_method="merged_16bit")

        print(f"\n✔ Standalone merged model exported via Unsloth to: {fused_path}")
        print("=========================================================")
        return

    except Exception as e:
        print(
            f"ℹ️ Unsloth native export unavailable ({e}). Falling back to PEFT merge_and_unload..."
        )

    # PEFT merge_and_unload fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    )
    if device == "cpu":
        torch_dtype = torch.float32

    print(f"Processing on device: {device.upper()} ({torch_dtype})")
    print("1. Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",  # Merge on CPU for VRAM headroom safety
        trust_remote_code=True,
    )

    print("2. Ingesting LoRA adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch_dtype)

    print("3. Merging weights natively (unloading PEFT layers)...")
    merged_model = model.merge_and_unload()

    print(f"4. Saving fused model to {fused_path}...")
    merged_model.save_pretrained(fused_path)
    tokenizer.save_pretrained(fused_path)

    readme_path = out_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            f"# VidwaanAI Standalone Fused Scripture Model\n\n"
            f"Standalone fused weights combining base `{model_id}` with domain-specific LoRA adapters.\n"
        )

    print(f"\n✔ Standalone merged model exported successfully to: {fused_path}")
    print("=========================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base weights and export."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="16bit",
        choices=["16bit", "4bit", "gguf"],
        help="Export format (16bit, 4bit, or gguf)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="GGUF quantization method (e.g. q4_k_m, q8_0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fused_model",
        help="Output directory for merged weights",
    )
    args = parser.parse_args()

    model_id = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    adapter_path = "./adapters"

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            model_id = cfg.get("model", model_id)
            training_cfg = cfg.get("training", {})
            adapter_path = training_cfg.get("adapter_path", cfg.get("adapter_path", adapter_path))

    export_model_weights(
        model_id=model_id,
        adapter_path=adapter_path,
        fused_path=args.output_dir,
        export_format=args.format,
        quantization_method=args.quantization,
    )


if __name__ == "__main__":
    main()
