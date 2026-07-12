#!/usr/bin/env python3
import os
import sys
import getpass
from huggingface_hub import HfApi

def main():
    print("=========================================================")
    print("         HUGGING FACE MODEL PUBLISHING GATE              ")
    print("=========================================================")

    fused_dir = "./fused_model"
    if not os.path.exists(fused_dir):
        print(f"❌ Error: Standalone fused model folder not found at '{fused_dir}'.")
        print("          Run model weights fusion first using 'make fuse-model'.")
        sys.exit(1)

    # Verify that model files exist in fused_model
    required_files = ["model.safetensors", "config.json", "README.md"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(fused_dir, f))]
    if missing:
        print(f"❌ Error: Fused model folder is incomplete. Missing files: {missing}")
        sys.exit(1)

    # 1. Resolve HF Token
    # Check current directory and parents for env variables
    dotenv_paths = [".env", "../.env", "../vidwaan-ai-mvp/.env"]
    env_vars = {}
    for env_path in dotenv_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip().strip('"').strip("'")
                break
            except Exception:
                pass

    hf_token = os.environ.get("HF_TOKEN") or env_vars.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not found in environment or active .env files.")
        try:
            hf_token = getpass.getpass("Enter your Hugging Face write token (hidden input): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Cancelled.")
            sys.exit(1)

    if not hf_token:
        print("❌ Error: Hugging Face authentication token is required.")
        sys.exit(1)

    # 2. Get target repository ID
    try:
        repo_id = input("Enter target Hugging Face Repo ID (e.g. username/Llama-3-8B-Scripture): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Cancelled.")
        sys.exit(1)

    if not repo_id:
        print("❌ Error: Repository ID is required.")
        sys.exit(1)
        
    if "/" not in repo_id:
        print("❌ Error: Repository ID must contain your namespace/username (e.g., 'username/repo-name').")
        sys.exit(1)

    # 3. Create repo and upload folder
    api = HfApi()
    
    print(f"\n1. Authenticating and creating repository '{repo_id}' if it doesn't exist...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=hf_token
        )
        print("✔ Repository initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing repository: {e}")
        print("   Verify that your token has 'write' permissions and the repository name is valid.")
        sys.exit(1)

    print(f"\n2. Uploading './fused_model/' folder to Hugging Face Hub...")
    print("   (This uploads safetensors weights and model configs. Please wait...)")
    try:
        api.upload_folder(
            folder_path=fused_dir,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )
        print("\n" + "=" * 57)
        print("             HF UPLOAD COMPLETED SUCCESSFULLY           ")
        print("=========================================================")
        print(f"✔ Fused model folder uploaded successfully!")
        print(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
        print("=========================================================")
    except Exception as e:
        print(f"\n❌ Error uploading model weights folder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
