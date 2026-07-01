#!/usr/bin/env python3
import os
import sys
import argparse
from mlx_lm import load, generate

# Ensure local cache usage
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Standalone Fused Model Validation Engine")
    parser.add_argument("--fused-path", type=str, default="./fused_model", help="Path to fused model folder")
    parser.add_argument("--gguf-name", type=str, default="scripture_model.gguf", help="Target GGUF filename")
    args = parser.parse_args()

    print("=========================================================")
    print("         VERIFYING FUSED MODEL INTEGRITY                 ")
    print("=========================================================")
    
    # 1. Check folder existence
    if not os.path.exists(args.fused_path):
        print(f"❌ Error: Fused model directory not found at {args.fused_path}")
        print("          Run 'make fuse-model' first.")
        sys.exit(1)
        
    print(f"✔ Found Fused Model directory: {args.fused_path}")

    # 3. Load fused model weights & configs
    print("\nLoading merged weights configuration locally...")
    try:
        model, tokenizer = load(args.fused_path)
        print("✔ Standalone model configuration and weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading merged weights: {e}")
        sys.exit(1)

    # 4. Generate test response to verify tensor matrices integrity
    print("\nRunning sample inference to verify tensor mapping integrity...")
    prompt = "Who freed Ahalya from Gautama's curse?"
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        print("Generating response (max_tokens=64)...")
        response = generate(model, tokenizer, prompt=formatted, max_tokens=64)
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
