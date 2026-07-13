import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_local_dataset
import generate_mlx_data

def test_load_ontology(tmp_path):
    # Test valid ontology loading
    ontology_data = {
        "nodes": [
            {"id": "entity_1", "name": "Entity One", "type": "Character", "attributes": {"description": "First entity"}},
            {"id": "entity_2", "name": "Entity Two", "type": "Location", "attributes": {"description": "Second entity"}}
        ]
    }
    temp_file = tmp_path / "test_ontology.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(ontology_data, f)
        
    nodes = generate_local_dataset.load_ontology(str(temp_file))
    assert len(nodes) == 2
    assert nodes[0]["id"] == "entity_1"
    assert nodes[1]["name"] == "Entity Two"

def test_load_ontology_not_found():
    with pytest.raises(FileNotFoundError):
        generate_local_dataset.load_ontology("non_existent_file.json")

def test_segment_text():
    text = "one two three four five six seven eight nine ten"
    # Chunk size of 3 words
    chunks = generate_local_dataset.segment_text(text, chunk_size=3)
    assert len(chunks) == 4
    assert chunks[0] == "one two three"
    assert chunks[1] == "four five six"
    assert chunks[2] == "seven eight nine"
    assert chunks[3] == "ten"

def test_find_semantic_intersections():
    ontology_nodes = [
        {"id": "rama", "name": "Rama", "type": "Character", "attributes": {"description": "Hero of Ramayana"}},
        {"id": "gautama_wife", "name": "Ahalya", "type": "Character", "attributes": {"description": "Wife of Gautama"}},
        {"id": "mithila", "name": "Mithila", "type": "Location", "attributes": {"description": "Ancient city"}},
        {"id": "non_match", "name": "NonMatch", "type": "Concept", "attributes": {"description": "Not in text"}}
    ]
    
    text_block = "Rama visited the hermitage of Ahalya, who was cursed. Later they traveled to Mithila."
    
    matches = generate_local_dataset.find_semantic_intersections(text_block, ontology_nodes)
    matched_ids = {m["id"] for m in matches}
    
    assert "rama" in matched_ids
    assert "gautama_wife" in matched_ids
    assert "mithila" in matched_ids
    assert "non_match" not in matched_ids
    
    # Check word boundaries: "here" should not match "he" if "he" is the ontology entity
    ontology_nodes_boundaries = [
        {"id": "he", "name": "he", "type": "Concept", "attributes": {"description": "pronoun"}}
    ]
    text_here = "They are here in the house."
    matches_boundaries = generate_local_dataset.find_semantic_intersections(text_here, ontology_nodes_boundaries)
    assert len(matches_boundaries) == 0

def test_sanitize_json_lines():
    raw_response = (
        "Here is the raw output you requested:\n"
        "```jsonl\n"
        '{"messages": [{"role": "user", "content": "Question 1?"}, {"role": "assistant", "content": "Answer 1."}]}\n'
        '{"messages": [{"role": "user", "content": "Question 2?"}, {"role": "assistant", "content": "Answer 2."}]}\n'
        "```\n"
        "Some trailing explanation."
    )
    records = generate_local_dataset.sanitize_json_lines(raw_response)
    assert len(records) == 2
    assert records[0]["messages"][0]["content"] == "Question 1?"
    assert records[1]["messages"][1]["content"] == "Answer 2."

def test_generate_mock_dataset_records():
    chunk = "Rama and Lakshmana walked in the forest with Sage Viswamitra."
    metadata = [
        {"id": "rama", "name": "Rama", "type": "Character", "description": "incarnation of Vishnu"},
        {"id": "lakshmana", "name": "Lakshmana", "type": "Character", "description": "brother of Rama"},
    ]
    records = generate_local_dataset.generate_mock_dataset_records(chunk, metadata)
    assert len(records) == 5
    assert "messages" in records[0]
    assert records[0]["messages"][0]["role"] == "user"
    assert records[0]["messages"][1]["role"] == "assistant"
