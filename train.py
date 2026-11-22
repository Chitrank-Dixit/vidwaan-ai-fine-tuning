#!/usr/bin/env python3
"""VidwaanAI Supervised Fine-Tuning (SFT) Engine powered by Unsloth QLoRA.

Implements memory-efficient 4-bit NF4 fine-tuning on NVIDIA RTX 5060 Ti (16GB VRAM)
using FastLanguageModel, PEFT LoRA across all 7 projection modules, and TRL SFTTrainer
with response-only loss masking. Supports resuming from checkpoints.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import unsloth first before transformers/trl/peft to ensure all Triton optimizations are patched
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer

from src.config_schema import TrainingPipelineConfig

# Configure Hugging Face Cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ScriptureDataset(TorchDataset):
    """Dataset to load conversation jsonl and tokenize with prompt masking."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.examples = []
        if not os.path.exists(data_path):
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.examples.append(json.loads(line))
                except Exception:
                    pass

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        messages = item["messages"]
        user_part = messages[:-1]

        try:
            user_text = self.tokenizer.apply_chat_template(
                user_part, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            user_text = ""
            for msg in user_part:
                user_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            user_text += "<|im_start|>assistant\n"
            full_text = user_text + messages[-1]["content"] + "<|im_end|>\n"

        user_tokens = self.tokenizer(user_text, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        cutoff = len(user_tokens)
        if full_tokens[:cutoff] != user_tokens:
            found = False
            for i in range(len(full_tokens) - cutoff + 1):
                if full_tokens[i : i + cutoff] == user_tokens:
                    cutoff = i + cutoff
                    found = True
                    break
            if not found:
                cutoff = len(user_tokens)

        labels = [-100] * cutoff + full_tokens[cutoff:]

        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[: self.max_length]
            labels = labels[: self.max_length]

        attention_mask = [1] * len(full_tokens)

        return {
            "input_ids": torch.tensor(full_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    """Collate batch items with padding for inputs and -100 for masked labels."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_masks,
    }


class LogLossCallback(TrainerCallback):
    """Callback to record training steps, loss, and learning rate to file."""

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Training Execution Log ===\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step_log = f"Step {state.global_step}: Loss = {logs['loss']:.4f}"
            if "learning_rate" in logs:
                step_log += f", Learning Rate = {logs['learning_rate']:.3e}"
            print(f"📝 {step_log}")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(step_log + "\n")


def load_conversations_to_dataset(jsonl_path: Path | str, tokenizer: Any) -> Optional[Dataset]:
    """Load JSONL dataset and format into continuous chat text using the model's chat template."""
    path = Path(jsonl_path)
    if not path.is_file():
        print(f"⚠️ Dataset file not found: {path}")
        return None

    texts: List[str] = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                messages = item.get("messages", [])
                if not messages:
                    continue

                try:
                    formatted_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    formatted_text = ""
                    for msg in messages:
                        formatted_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

                texts.append(formatted_text)
            except Exception:
                skipped += 1

    if not texts:
        print(f"⚠️ No valid dialogue samples could be loaded from {path}")
        return None

    print(f"Loaded {len(texts)} samples from {path.name} (skipped: {skipped})")
    return Dataset.from_dict({"text": texts})


def load_model_and_tokenizer(
    config: TrainingPipelineConfig,
) -> Tuple[Any, Any]:
    """Load base model and tokenizer with Unsloth fast kernels, falling back to PEFT if Unsloth is absent."""
    try:
        from unsloth import FastLanguageModel

        print(f"\n1. Loading {config.model} via Unsloth FastLanguageModel...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )

        print("\n2. Configuring Unsloth QLoRA PEFT model...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_parameters.rank,
            lora_alpha=int(config.lora_parameters.scale),
            lora_dropout=config.lora_parameters.dropout,
            target_modules=config.lora_parameters.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth" if config.training.grad_checkpoint else False,
            random_state=config.training.random_seed,
        )
        return model, tokenizer

    except ImportError:
        print(
            "\n⚠️ Unsloth not installed in current environment. Using Hugging Face PEFT fallback..."
        )
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if config.training.grad_checkpoint:
            base_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=config.lora_parameters.rank,
            lora_alpha=int(config.lora_parameters.scale),
            lora_dropout=config.lora_parameters.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.lora_parameters.target_modules,
        )
        model = get_peft_model(base_model, lora_config)
        return model, tokenizer


def execute_training(
    config: TrainingPipelineConfig,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete SFT training loop with response-only loss masking."""
    print("=========================================================")
    print("         VIDWAAN-AI UNSLOTH QLORA TRAINING ENGINE        ")
    print("=========================================================")
    print(f"Base Model:             {config.model}")
    print(f"Max Sequence Length:    {config.max_seq_length}")
    print(f"4-Bit Quantization:     {config.load_in_4bit}")
    print(f"Batch Size (Per Device):{config.training.batch_size}")
    print(f"Grad Accumulation:      {config.training.grad_accumulation_steps}")
    print(f"Learning Rate:          {config.training.learning_rate}")
    print(f"Adapter Output Path:    {config.training.adapter_path}")

    # Step 1: Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Step 2: Load Datasets
    data_dir = config.data_dir
    train_dataset = load_conversations_to_dataset(data_dir / "train.jsonl", tokenizer)
    valid_dataset = load_conversations_to_dataset(data_dir / "valid.jsonl", tokenizer)

    if train_dataset is None:
        raise ValueError(f"Training dataset at {data_dir / 'train.jsonl'} cannot be empty.")

    # Step 3: Setup Trainer Arguments
    output_dir = str(config.training.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_file = config.data_dir / "training.log"
    log_callback = LogLossCallback(log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16

    import inspect

    sft_params = inspect.signature(SFTTrainer.__init__).parameters

    try:
        from trl import SFTConfig

        training_args_cls = SFTConfig
    except ImportError:
        training_args_cls = TrainingArguments

    args_params = inspect.signature(training_args_cls.__init__).parameters
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.grad_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_steps": config.training.warmup_steps,
        "max_steps": config.training.max_steps if config.training.max_steps else -1,
        "num_train_epochs": config.training.num_train_epochs
        if not config.training.max_steps
        else 1.0,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "eval_steps": config.training.eval_steps if valid_dataset else None,
        "eval_strategy": "steps" if valid_dataset else "no",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": "adamw_8bit" if device == "cuda" else "adamw_torch",
        "seed": config.training.random_seed,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    # Pass dataset text formatting & sequence length to SFTConfig if supported
    if "dataset_text_field" in args_params:
        args_dict["dataset_text_field"] = "text"
    if "max_seq_length" in args_params:
        args_dict["max_seq_length"] = config.max_seq_length
    elif "max_length" in args_params:
        args_dict["max_length"] = config.max_seq_length
    if "packing" in args_params:
        args_dict["packing"] = False
    if "dataset_num_proc" in args_params:
        args_dict["dataset_num_proc"] = 1

    training_args = training_args_cls(**args_dict)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "args": training_args,
        "callbacks": [log_callback],
    }

    # Support modern TRL (processing_class) and legacy TRL (tokenizer)
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Pass dataset_text_field, max_seq_length, packing to SFTTrainer if supported directly
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = config.max_seq_length
    if "packing" in sft_params:
        trainer_kwargs["packing"] = False
    if "dataset_num_proc" in sft_params:
        trainer_kwargs["dataset_num_proc"] = 1

    trainer = SFTTrainer(**trainer_kwargs)

    # Step 4: Apply Response-Only Masking if supported
    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("✔ Applied response-only loss masking (user prompts ignored in backward pass).")
    except Exception as e:
        print(f"ℹ️ Skipping train_on_responses_only: {e}")

    # Step 5: Execute Training Loop
    print("\nStarting model training...")
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_result = trainer.train()

    # Step 6: Save Final LoRA Adapters
    adapter_path = Path(config.training.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final LoRA adapters to {adapter_path}...")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    metrics_path = adapter_path / "training_metrics.json"
    metrics = {
        "train_loss": getattr(train_result, "training_loss", None),
        "global_step": getattr(train_result, "global_step", None),
        "model_id": config.model,
        "max_seq_length": config.max_seq_length,
        "rank": config.lora_parameters.rank,
        "scale": config.lora_parameters.scale,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✔ Adapter files and metadata saved successfully.")
    print("=========================================================")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="VidwaanAI Scripture Fine-Tuning Pipeline (Unsloth QLoRA)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    args = parser.parse_args()

    config = TrainingPipelineConfig.from_yaml(args.config)
    execute_training(config, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
