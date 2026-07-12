#!/usr/bin/env python3
import os
import sys

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local cache usage
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Standalone Fused Model Validation Engine")
    parser.add_argument("--fused-path", type=str, default="./fused_model", help="Path to fused model folder")
    args = parser.parse_args()

    print("=========================================================")
    print("         VERIFYING FUSED MODEL INTEGRITY                 ")
    print("=========================================================")
    
    # 1. Check folder existence
    if not os.path.exists(args.fused_path):
        print(f"❌ Error: Fused model directory not found at {args.fused_path}")
        print("          Run 'make fuse-model' or 'uv run python export_model.py' first.")
        sys.exit(1)
        
    print(f"✔ Found Fused Model directory: {args.fused_path}")

    # 2. Check required files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    # Qwen/HF models might shard the weights if needed, but for 1.5B it should save as model.safetensors.
    # Let's check for config.json and tokenizer.json and at least one .safetensors file.
    missing = []
    if not os.path.exists(os.path.join(args.fused_path, "config.json")):
        missing.append("config.json")
    if not os.path.exists(os.path.join(args.fused_path, "tokenizer.json")):
        missing.append("tokenizer.json")
        
    # Check for safetensors (can be model.safetensors or sharded files)
    has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(args.fused_path))
    if not has_safetensors:
        missing.append("*.safetensors")
        
    if missing:
        print(f"❌ Error: Fused model folder is incomplete. Missing files/patterns: {missing}")
        sys.exit(1)

    # 3. Load fused model weights & configs
    print("\nLoading merged weights configuration locally...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.fused_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.fused_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        if device_map is None:
            model = model.to(device)
        print("✔ Standalone model configuration and weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading merged weights: {e}")
        sys.exit(1)

    # 4. Generate test response to verify tensor matrices integrity
    print("\nRunning sample inference to verify tensor mapping integrity...")
    prompt = "Who freed Ahalya from Gautama's curse?"
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print("Generating response (max_tokens=64)...")
        inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print("\nGenerated Output:")
        print(response.strip())
        print("\n✔ Fused weight generation verification succeeded.")
    except Exception as e:
        print(f"❌ Error during generation validation: {e}")
        sys.exit(1)

    print("\n=========================================================")
    print("      FUSION INTEGRITY AND EXPORT CHECKS: PASSED         ")
    print("=========================================================")
    sys.exit(0)

if __name__ == "__main__":
    main()
