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

import json
import yaml
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

# Ensure Hugging Face cache is local
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class ScriptureDataset(Dataset):
    """
    Dataset to load conversation jsonl and tokenize with prompt masking.
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        self.examples = []
        if not os.path.exists(data_path):
            print(f"Dataset path {data_path} not found.")
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.examples.append(json.loads(line))
                except Exception as e:
                    print(f"Skipping invalid JSON line: {e}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        messages = item["messages"]
        
        # We perform prompt masking by splitting user/system prompt and assistant response
        user_part = messages[:-1]
        
        try:
            # Build prompt text using the template
            user_text = self.tokenizer.apply_chat_template(user_part, tokenize=False, add_generation_prompt=True)
            # Build full text containing the entire conversation
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            # Fallback chat formatting if template application fails
            user_text = ""
            for msg in user_part:
                user_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            user_text += "<|im_start|>assistant\n"
            full_text = user_text + messages[-1]["content"] + "<|im_end|>\n"

        user_tokens = self.tokenizer(user_text, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # Align where the assistant's response starts
        cutoff = len(user_tokens)
        if full_tokens[:cutoff] != user_tokens:
            # Find the overlap in case tokenizer added special tokens at boundary
            found = False
            for i in range(len(full_tokens) - cutoff + 1):
                if full_tokens[i:i+cutoff] == user_tokens:
                    cutoff = i + cutoff
                    found = True
                    break
            if not found:
                cutoff = len(user_tokens)
                
        labels = [-100] * cutoff + full_tokens[cutoff:]
        
        # Truncate if exceeds max length
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
            labels = labels[:self.max_length]
            
        attention_mask = [1] * len(full_tokens)
        
        return {
            "input_ids": torch.tensor(full_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_masks
    }

class LogLossCallback(TrainerCallback):
    """
    Callback to log training steps and loss to a file.
    """
    def __init__(self, log_path):
        self.log_path = log_path
        # Clear existing log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== Training Execution Log ===\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step_log = f"Step {state.global_step}: Loss = {logs['loss']:.4f}"
            if "learning_rate" in logs:
                step_log += f", Learning Rate = {logs['learning_rate']:.3e}"
            print(f"📝 {step_log}")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(step_log + "\n")

def main():
    print("=========================================================")
    print("         CROSS-PLATFORM LoRA TRAINING ENGINE             ")
    print("=========================================================")
    
    # 1. Load config
    config_path = "./config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    model_id = config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    data_dir = config.get("data", "./data")
    adapter_path = config.get("adapter_path", "./adapters")
    
    batch_size = config.get("batch_size", 1)
    grad_accumulation_steps = config.get("grad_accumulation_steps", 4)
    num_layers = config.get("num_layers", 8)
    iters = config.get("iters", 600)
    learning_rate = float(config.get("learning_rate", 1e-5))
    grad_checkpoint = config.get("grad_checkpoint", True)
    
    lora_params = config.get("lora_parameters", {})
    rank = lora_params.get("rank", 8)
    scale = lora_params.get("scale", 16.0)
    dropout = lora_params.get("dropout", 0.05)
    
    print(f"Base Model:      {model_id}")
    print(f"Data Directory:  {data_dir}")
    print(f"Adapter Output:  {adapter_path}")
    print(f"Iterations:      {iters}")
    print(f"Learning Rate:   {learning_rate}")
    print(f"Batch Size:      {batch_size}")
    
    # 2. Check Device Capabilities
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Device: {device.upper()}")
    
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
        print(f"CUDA Precision:  {torch_dtype}")
    else:
        torch_dtype = torch.float32
        device_map = None
        print("CUDA Precision:  float32 (CPU execution)")
        
    # 3. Load Tokenizer & Model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing if requested
    if grad_checkpoint:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        
    # 4. Set up PEFT LoRA Config
    print("\nConfiguring Low-Rank Adaptation (LoRA)...")
    total_layers = getattr(base_model.config, "num_hidden_layers", 28)
    
    # Limit adaptation to the last N layers to save memory, matching M1/macOS layers configuration
    if num_layers < total_layers:
        layers_to_transform = list(range(total_layers - num_layers, total_layers))
        print(f"Restricting LoRA to the last {num_layers} layers of the transformer: {layers_to_transform}")
    else:
        layers_to_transform = None
        print(f"Applying LoRA across all {total_layers} layers of the transformer.")
        
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(scale),
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers_to_transform
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Load Datasets
    print("\nPreparing train and validation datasets...")
    train_dataset = ScriptureDataset(os.path.join(data_dir, "train.jsonl"), tokenizer)
    valid_dataset = ScriptureDataset(os.path.join(data_dir, "valid.jsonl"), tokenizer)
    
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(valid_dataset)} validation samples.")
    
    # 6. Configure Trainer
    os.makedirs(data_dir, exist_ok=True)
    log_callback = LogLossCallback(os.path.join(data_dir, "training.log"))
    
    # Save checkpoint at the end of training
    training_args = TrainingArguments(
        output_dir="./tmp_checkpoints",
        max_steps=iters,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="steps" if len(valid_dataset) > 0 else "no",
        eval_steps=100 if len(valid_dataset) > 0 else None,
        fp16=(device == "cuda" and torch_dtype == torch.float16),
        bf16=(device == "cuda" and torch_dtype == torch.bfloat16),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if len(valid_dataset) > 0 else None,
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id),
        callbacks=[log_callback]
    )
    
    # 7. Execute Fine-Tuning
    print("\nStarting training loop...")
    trainer.train()
    
    # 8. Save final adapters
    print(f"\nSaving final LoRA adapters to {adapter_path}...")
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print("✔ Adapter files saved successfully.")
    
    # Clear temporary checkpoints directory
    import shutil
    if os.path.exists("./tmp_checkpoints"):
        shutil.rmtree("./tmp_checkpoints")
        
    print("=========================================================")
    print("             TRAINING RUN COMPLETE                       ")
    print("=========================================================")

if __name__ == "__main__":
    main()
