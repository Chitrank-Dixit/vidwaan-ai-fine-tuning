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

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure Hugging Face cache is local
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    print("=========================================================")
    print("        CROSS-PLATFORM MODEL FUSION & EXPORT             ")
    print("=========================================================")
    
    # Ingest configurations
    config_path = "./config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    model_id = config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = config.get("adapter_path", "./adapters")
    fused_path = "./fused_model"
    
    print(f"Base Model:      {model_id}")
    print(f"Adapters Source: {adapter_path}")
    print(f"Fused Target:    {fused_path}")
    
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Adapter path '{adapter_path}' does not exist. Run fine-tuning training first.")
        sys.exit(1)
        
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
        
    print(f"Processing on:   {device.upper()} ({torch_dtype})")
    
    print("\n1. Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cpu", # Merge on CPU for memory safety
        trust_remote_code=True
    )
    
    print("\n2. Ingesting LoRA adapter weights...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype
    )
    
    print("\n3. Merging weights natively (unloading PEFT layers)...")
    merged_model = model.merge_and_unload()
    
    print(f"\n4. Exporting standalone fused model to {fused_path}...")
    os.makedirs(fused_path, exist_ok=True)
    merged_model.save_pretrained(fused_path)
    tokenizer.save_pretrained(fused_path)
    
    # Generate a README.md file in the output directory
    readme_path = os.path.join(fused_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# Fused Scripture Model\n\nStandalone fused model combining base `{model_id}` with fine-tuned LoRA adapters.\n")
        
    print("\n✔ Standalone merged model exported successfully.")
    print("=========================================================")

if __name__ == "__main__":
    main()
