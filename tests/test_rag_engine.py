import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hybrid_rag_engine

def test_query_vector_db():
    # Test matching keyword "ahalya"
    res = hybrid_rag_engine.query_vector_db("Who was Ahalya?")
    assert "Gautama's wife Ahalya" in res
    
    # Test matching keyword "mithila"
    res = hybrid_rag_engine.query_vector_db("What happened in Mithila?")
    assert "Rama and Lakshmana went to Mithila" in res
    
    # Test fallback context
    res = hybrid_rag_engine.query_vector_db("Random question unrelated to anything")
    assert "Dasaratha's glorious son Rama" in res

def test_find_ontology_nodes(tmp_path):
    ontology_data = {
        "nodes": [
            {"id": "rama", "name": "Rama", "type": "Character", "attributes": {"description": "Dasaratha's son"}},
            {"id": "sita", "name": "Sita", "type": "Character", "attributes": {"description": "Janaka's daughter"}},
        ]
    }
    temp_file = tmp_path / "test_entities.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(ontology_data, f)
        
    res_str = hybrid_rag_engine.find_ontology_nodes("Rama was guide and Sita was there.", str(temp_file))
    res_data = json.loads(res_str)
    assert len(res_data) == 2
    assert {n["id"] for n in res_data} == {"rama", "sita"}

def test_find_ontology_nodes_not_found():
    res = hybrid_rag_engine.find_ontology_nodes("Rama was guide.", "non_existent_file.json")
    assert res == "Ontology database unavailable."

@patch("torch.no_grad")
def test_generate_rag_response(mock_no_grad, tmp_path):
    # Setup mock ontology
    ontology_data = {"nodes": [{"id": "rama", "name": "Rama", "type": "Character", "attributes": {"description": "Dasaratha's son"}}]}
    temp_file = tmp_path / "test_entities.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(ontology_data, f)
        
    # Setup mock model and tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
    
    # Mock return value of tokenizer to support .to(device) and key retrieval
    mock_inputs = MagicMock()
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 5) # mock input length of 5
    mock_inputs.__getitem__.side_effect = lambda key: mock_input_ids if key == "input_ids" else MagicMock()
    mock_inputs.to.return_value = mock_inputs
    
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "Mocked Assistant Response"
    
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = [[0, 1, 2, 3, 4, 5, 6, 7]] # inputs length is 5, remaining 3 new tokens
    
    response = hybrid_rag_engine.generate_rag_response(
        user_query="Tell me about Rama",
        model=mock_model,
        tokenizer=mock_tokenizer,
        ontology_path=str(temp_file),
        max_tokens=50
    )
    
    assert response == "Mocked Assistant Response"
    mock_tokenizer.assert_called_once()
    mock_model.generate.assert_called_once()
