import os
import sys
import pytest
import shutil
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_local_dataset
import validate_local_data
import check_readiness
from scripts import create_test_pdf

@pytest.fixture
def backup_and_restore_data():
    # Files to backup
    files_to_backup = [
        "data/train.jsonl",
        "data/valid.jsonl",
        "data/corrupted.jsonl",
        "data/test_scripture.pdf"
    ]
    
    backups = {}
    for filepath in files_to_backup:
        if os.path.exists(filepath):
            backup_path = filepath + ".bak"
            shutil.copy(filepath, backup_path)
            backups[filepath] = backup_path
            
    yield
    
    # Restore backups and clean up
    for filepath in files_to_backup:
        # Delete generated file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        # Restore backup if it exists
        if filepath in backups:
            shutil.move(backups[filepath], filepath)

@patch("generate_local_dataset.call_local_lm_studio", return_value=[])
def test_pipeline_integration(mock_call_lm, backup_and_restore_data):
    # 1. Run create_test_pdf to generate a real scripture PDF in data/
    create_test_pdf.main()
    assert os.path.exists("data/test_scripture.pdf")
    
    # 2. Run generate_local_dataset.py using mock fallback
    gen_args = [
        "generate_local_dataset.py",
        "--ontology", "ontology.json" if os.path.exists("ontology.json") else "raw_entities.json",
        "--pdf-dir", "data",
        "--output-dir", "data",
        "--mock-fallback"
    ]
    with patch("sys.argv", gen_args):
        generate_local_dataset.main()
        
    assert os.path.exists("data/train.jsonl")
    assert os.path.exists("data/valid.jsonl")
    assert os.path.getsize("data/train.jsonl") > 0
    assert os.path.getsize("data/valid.jsonl") > 0
    
    # 3. Run validate_local_data.py
    val_args = [
        "validate_local_data.py",
        "--ontology", "ontology.json" if os.path.exists("ontology.json") else "raw_entities.json",
        "--pdf-dir", "data",
        "--mock-judge"
    ]
    
    with patch("sys.argv", val_args), patch("builtins.input", return_value="y"):
        try:
            validate_local_data.main()
        except SystemExit as e:
            assert e.code == 0
            
    # 4. Run check_readiness.py
    readiness_args = [
        "check_readiness.py",
        "--data-dir", "data",
        "--non-interactive"
    ]
    
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    with patch("sys.argv", readiness_args), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        
        try:
            check_readiness.main()
        except SystemExit as e:
            assert e.code == 0
