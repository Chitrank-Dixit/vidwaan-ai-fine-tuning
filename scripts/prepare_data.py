import json
import os

def main():
    print("Preparing mock scripture data...")
    os.makedirs("/app/data", exist_ok=True)
    
    train_data = [
        {"text": "And it came to pass that he went forth among the people..."},
        {"text": "Therefore, fear not, little flock; do good; let earth and hell combine against you..."}
    ]
    
    valid_data = [
        {"text": "Search these commandments, for they are true and faithful..."},
    ]
    
    # Check if we are running in the container or locally on host
    base_dir = "/app/data" if os.path.exists("/app") else "./data"
    os.makedirs(base_dir, exist_ok=True)
    
    train_path = os.path.join(base_dir, "train.jsonl")
    valid_path = os.path.join(base_dir, "valid.jsonl")
    
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(valid_path, "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Mock data files created successfully: {train_path}, {valid_path}")

if __name__ == "__main__":
    main()
