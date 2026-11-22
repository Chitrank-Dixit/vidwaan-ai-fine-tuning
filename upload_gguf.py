import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from huggingface_hub import HfApi


def upload():
    api = HfApi()
    env = {}
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")

    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    repo = env.get("HF_REPO_ID") or os.environ.get("HF_REPO_ID")

    print(f"Uploading GGUF models to Hugging Face repository: {repo}")
    files = ["vidwaan-scripture-qwen1.5b-Q4_K_M.gguf", "vidwaan-scripture-qwen1.5b-Q8_0.gguf"]
    for file_name in files:
        file_path = os.path.join("fused_model", file_name)
        if os.path.exists(file_path):
            print(f"Uploading {file_name}...")
            api.upload_file(
                path_or_fileobj=file_path, path_in_repo=file_name, repo_id=repo, token=token
            )
            print(f"✔ Uploaded {file_name}")


if __name__ == "__main__":
    upload()
