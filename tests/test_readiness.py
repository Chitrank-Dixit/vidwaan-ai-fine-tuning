import os
import sys
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_readiness

def test_check_datasets_missing(tmp_path):
    # Temp directory with no files
    success, reports = check_readiness.check_datasets(str(tmp_path))
    assert not success
    assert any("File missing" in r for r in reports)

def test_check_datasets_empty(tmp_path):
    # Temp directory with empty files
    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    train_file.touch()
    valid_file.touch()
    
    success, reports = check_readiness.check_datasets(str(tmp_path))
    assert not success
    assert any("File at" in r and "is empty" in r for r in reports)

def test_check_datasets_invalid_json(tmp_path):
    # Files contain corrupted JSON
    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    train_file.write_text("{invalid json", encoding="utf-8")
    valid_file.write_text('{"messages": []}', encoding="utf-8")
    
    success, reports = check_readiness.check_datasets(str(tmp_path))
    assert not success
    assert any("JSON parsing failed" in r for r in reports)

def test_check_datasets_invalid_schema(tmp_path):
    # Files contain valid JSON but wrong schema (missing 'messages')
    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    train_file.write_text('{"data": "some value"}', encoding="utf-8")
    valid_file.write_text('{"messages": []}', encoding="utf-8")
    
    success, reports = check_readiness.check_datasets(str(tmp_path))
    assert not success
    assert any("Structure invalid" in r for r in reports)

def test_check_datasets_valid(tmp_path):
    # Files contain correct format
    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    train_file.write_text('{"messages": [{"role": "user", "content": "hello"}]}', encoding="utf-8")
    valid_file.write_text('{"messages": [{"role": "user", "content": "hi"}]}', encoding="utf-8")
    
    success, reports = check_readiness.check_datasets(str(tmp_path))
    assert success
    assert all("Validated" in r for r in reports)

@patch("os.path.exists")
def test_get_memory_info_cgroup_v1(mock_exists):
    # Mock cgroup v1 configuration
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    
    m_open = mock_open(read_data="17179869184\n") # 16 GB
    with patch("builtins.open", m_open):
        mem_info = check_readiness.get_memory_info()
        assert mem_info["limit_gb"] == 16.0
        assert mem_info["is_constrained"]

@patch("sys.platform", "darwin")
@patch("subprocess.check_output")
def test_get_memory_info_darwin(mock_check_output):
    # Mock macOS sysctl check
    mock_check_output.return_value = b"34359738368\n" # 32 GB
    
    with patch("os.path.exists", return_value=False):
        mem_info = check_readiness.get_memory_info()
        assert mem_info["limit_gb"] == 32.0
        assert not mem_info["is_constrained"]

@patch("sys.platform", "linux")
def test_get_memory_info_linux_proc():
    # Mock /proc/meminfo check
    m_open = mock_open(read_data="MemTotal:       16384000 kB\n")
    
    with patch("os.path.exists", return_value=False), patch("builtins.open", m_open):
        mem_info = check_readiness.get_memory_info()
        # 16384000 kB -> ~15.625 GB
        assert 15.0 < mem_info["limit_gb"] < 16.0
        assert not mem_info["is_constrained"]
