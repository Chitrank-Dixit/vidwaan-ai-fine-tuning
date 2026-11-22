import os
import sys
from unittest.mock import mock_open, patch

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

    m_open = mock_open(read_data="17179869184\n")  # 16 GB
    with patch("builtins.open", m_open):
        mem_info = check_readiness.get_memory_info()
        assert mem_info["limit_gb"] == 16.0
        assert mem_info["is_constrained"]


@patch("sys.platform", "darwin")
@patch("subprocess.check_output")
def test_get_memory_info_darwin(mock_check_output):
    # Mock macOS sysctl check
    mock_check_output.return_value = b"34359738368\n"  # 32 GB

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


@patch("os.path.exists")
def test_get_memory_info_cgroup_v2_constrained(mock_exists):
    # Mock cgroup v2 configuration with numerical limit
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/memory.max"

    m_open = mock_open(read_data="17179869184\n")  # 16 GB
    with patch("builtins.open", m_open):
        mem_info = check_readiness.get_memory_info()
        assert mem_info["limit_gb"] == 16.0
        assert mem_info["is_constrained"]


@patch("os.path.exists")
def test_get_memory_info_cgroup_v2_unconstrained(mock_exists):
    # Mock cgroup v2 configuration with "max"
    mock_exists.side_effect = lambda path: path == "/sys/fs/cgroup/memory.max"

    m_open = mock_open(read_data="max\n")
    with patch("builtins.open", m_open), patch("sys.platform", "linux"):
        # When max, falls back to /proc/meminfo
        with patch("builtins.open", mock_open(read_data="MemTotal:       16384000 kB\n")):
            mem_info = check_readiness.get_memory_info()
            assert not mem_info["is_constrained"]
            assert 15.0 < mem_info["limit_gb"] < 16.0


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 5060 Ti")
@patch("torch.cuda.is_bf16_supported", return_value=True)
def test_check_gpu_environment_cuda_available(mock_bf16, mock_name, mock_count, mock_avail):
    from unittest.mock import MagicMock

    props = MagicMock()
    props.total_memory = 16 * 1024**3  # 16 GB
    props.major = 8
    props.minor = 9

    with patch("torch.cuda.get_device_properties", return_value=props):
        gpu = check_readiness.check_gpu_environment()
        assert gpu["cuda_available"] is True
        assert gpu["device_count"] == 1
        assert "RTX 5060 Ti" in gpu["device_name"]
        assert gpu["vram_gb"] == 16.0
        assert gpu["bf16_supported"] is True
        assert gpu["compute_capability"] == (8, 9)


@patch("torch.cuda.is_available", return_value=False)
def test_check_gpu_environment_cpu_only(mock_avail):
    gpu = check_readiness.check_gpu_environment()
    assert gpu["cuda_available"] is False
    assert gpu["device_count"] == 0
    assert gpu["vram_gb"] == 0.0
    assert gpu["bf16_supported"] is False


@patch("torch.cuda.is_available", side_effect=RuntimeError("CUDA driver mismatch"))
def test_check_gpu_environment_error_handling(mock_avail):
    gpu = check_readiness.check_gpu_environment()
    assert gpu["cuda_available"] is False
    assert "error" in gpu
    assert "CUDA driver mismatch" in gpu["error"]
