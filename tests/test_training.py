import os
import sys
import json
import pytest
import torch
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train

def test_collate_fn():
    # Setup mock items in a batch
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([-100, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1])
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "labels": torch.tensor([-100, 5]),
            "attention_mask": torch.tensor([1, 1])
        }
    ]
    
    pad_token_id = 99
    collated = train.collate_fn(batch, pad_token_id)
    
    # Check shape
    assert collated["input_ids"].shape == (2, 3)
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    
    # Check padding values
    # Item 2 input_ids should be padded with 99
    assert collated["input_ids"][1, 2].item() == 99
    # Item 2 labels should be padded with -100
    assert collated["labels"][1, 2].item() == -100
    # Item 2 attention_mask should be padded with 0
    assert collated["attention_mask"][1, 2].item() == 0

def test_scripture_dataset_missing_file():
    # Missing file shouldn't crash, just report 0 len
    tokenizer = MagicMock()
    ds = train.ScriptureDataset("non_existent_file.jsonl", tokenizer)
    assert len(ds) == 0

def test_scripture_dataset_item_masking(tmp_path):
    # Create a small dataset line
    sample_data = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there."}
        ]
    }
    
    data_file = tmp_path / "train.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data) + "\n")
        
    # Setup mock tokenizer
    mock_tokenizer = MagicMock()
    
    # Mock template behavior
    # For user_text (messages[:-1])
    # For full_text (messages)
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        if add_generation_prompt:
            return "SYSTEM: You are helpful. USER: Hello. ASSISTANT:"
        else:
            return "SYSTEM: You are helpful. USER: Hello. ASSISTANT: Hi there."
            
    mock_tokenizer.apply_chat_template.side_effect = apply_chat_template
    
    # Mock tokenization outputs
    # Let prompt be 8 tokens
    # Let full text be 11 tokens (3 new assistant tokens)
    def tokenize_call(text, add_special_tokens=False):
        if text.startswith("SYSTEM: You are helpful. USER: Hello. ASSISTANT: Hi there."):
            return {"input_ids": [10, 11, 12, 13, 14, 15, 16, 17, 100, 101, 102]}
        elif text.startswith("SYSTEM: You are helpful. USER: Hello. ASSISTANT:"):
            return {"input_ids": [10, 11, 12, 13, 14, 15, 16, 17]}
        return {"input_ids": []}
        
    mock_tokenizer.side_effect = tokenize_call
    mock_tokenizer.pad_token_id = 99
    
    ds = train.ScriptureDataset(str(data_file), mock_tokenizer, max_length=50)
    assert len(ds) == 1
    
    item = ds[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    
    # The first 8 tokens (prompt) must be masked with -100
    # The last 3 tokens (assistant response) must be their original values [100, 101, 102]
    expected_labels = [-100] * 8 + [100, 101, 102]
    assert item["labels"].tolist() == expected_labels
    assert item["input_ids"].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 100, 101, 102]
    assert item["attention_mask"].tolist() == [1] * 11

def test_log_loss_callback(tmp_path):
    log_file = tmp_path / "training.log"
    callback = train.LogLossCallback(str(log_file))
    
    # Check log starts with header
    content = log_file.read_text(encoding="utf-8")
    assert "=== Training Execution Log ===" in content
    
    # Mock trainer state and args
    class StateMock:
        global_step = 10
    class ArgsMock:
        pass
        
    logs = {"loss": 0.5432, "learning_rate": 1e-5}
    callback.on_log(ArgsMock(), StateMock(), None, logs=logs)
    
    content = log_file.read_text(encoding="utf-8")
    assert "Step 10: Loss = 0.5432, Learning Rate = 1.000e-05" in content
