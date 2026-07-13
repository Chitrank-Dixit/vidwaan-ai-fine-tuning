import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import validate_local_data

def test_estimate_tokens():
    assert validate_local_data.estimate_tokens("") == 1
    assert validate_local_data.estimate_tokens("hello") == 1
    assert validate_local_data.estimate_tokens("a" * 400) == 100

def test_find_matched_entities():
    ontology_nodes = [
        {"id": "vishvamitra", "name": "Viswamitra", "type": "Character", "attributes": {"description": "Sage guide"}},
        {"id": "mithila", "name": "Mithila", "type": "Location", "attributes": {"description": "Janaka's city"}},
    ]
    
    text = "The sage Viswamitra came to Ayodhya."
    matches = validate_local_data.find_matched_entities(text, ontology_nodes)
    assert len(matches) == 1
    assert matches[0]["id"] == "vishvamitra"
    assert matches[0]["name"] == "Viswamitra"

def test_find_best_context_chunk():
    ontology_nodes = [
        {"id": "ahalya", "name": "Ahalya", "type": "Character", "attributes": {"description": "turned to stone"}},
        {"id": "mithila", "name": "Mithila", "type": "Location", "attributes": {"description": "bow of Shiva"}},
    ]
    
    chunks = [
        "Rama and Lakshmana went to Mithila guided by the sage Viswamitra to attend Janaka's sacrifice.",
        "Ahalya was cursed to turn into a stone sculpture and perform penance for many years."
    ]
    
    # QA text matches Ahalya
    qa_text = "How was Ahalya freed from Gautama's curse?"
    best_chunk, matched_nodes = validate_local_data.find_best_context_chunk(qa_text, chunks, ontology_nodes)
    
    assert best_chunk == chunks[1]
    assert len(matched_nodes) == 1
    assert matched_nodes[0]["id"] == "ahalya"

@patch("urllib.request.urlopen")
def test_call_local_critic_success(mock_urlopen):
    # Mocking urllib.request.urlopen response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [
            {
                "message": {
                    "content": '{"faithfulness_score": 5, "ontological_score": 4, "tone_score": 5, "critical_review": "Excellent"}'
                }
            }
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    qa_pair = {"question": "Q?", "answer": "A."}
    result = validate_local_data.call_local_critic(
        "http://localhost:1234",
        "model-name",
        "Sample passage content",
        [{"id": "node1"}],
        qa_pair
    )
    
    assert result is not None
    assert result["faithfulness_score"] == 5
    assert result["ontological_score"] == 4
    assert result["tone_score"] == 5
    assert result["critical_review"] == "Excellent"

@patch("urllib.request.urlopen")
def test_call_local_critic_failure(mock_urlopen):
    mock_urlopen.side_effect = Exception("Connection Refused")
    qa_pair = {"question": "Q?", "answer": "A."}
    result = validate_local_data.call_local_critic(
        "http://localhost:1234",
        "model-name",
        "Sample passage content",
        [{"id": "node1"}],
        qa_pair
    )
    assert result is None
