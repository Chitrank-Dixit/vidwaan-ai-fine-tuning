#!/usr/bin/env python3
"""
Convert exported Neo4j graph JSON into instruction-following JSONL dataset format.
Outputs train.jsonl and valid.jsonl compatible with VidwaanAI fine-tuning pipeline.
"""

import os
import sys
import json
import random

# Source & Target paths
SOURCE_JSON = os.getenv("SOURCE_JSON", "d:/Projects/vidwaan-ai-mvp/output/neo4j_dump_exported.json")
if not os.path.exists(SOURCE_JSON):
    # Fallback to local workspace check
    SOURCE_JSON = "output/neo4j_dump_exported.json"

DATA_DIR = os.getenv("DATA_DIR", "d:/Projects/vidwaan-ai-fine-tuning/data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VALID_PATH = os.path.join(DATA_DIR, "valid.jsonl")

def convert_neo4j_to_jsonl():
    print(f"Reading Neo4j export JSON from: {SOURCE_JSON}")
    if not os.path.exists(SOURCE_JSON):
        raise FileNotFoundError(f"Source JSON file not found at: {SOURCE_JSON}")

    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    print(f"Loaded {len(nodes)} nodes and {len(relationships)} relationships.")

    records = []
    node_id_to_name = {}

    print("Converting nodes to instruction pairs...")
    for node in nodes:
        props = node.get("properties", {})
        labels = node.get("labels", [])
        primary_label = labels[0] if labels else "Entity"
        
        name = props.get("name") or props.get("id") or ""
        scripture = props.get("scripture") or "Hindu scriptures"
        
        if props.get("id"):
            node_id_to_name[node["id"]] = (name, scripture)

        if primary_label == "Text":
            text_content = props.get("text", "").strip()
            translation = props.get("translation", "").strip()
            verse_id = props.get("verse_id") or name
            
            if not text_content:
                continue
                
            user_prompt = f"What does the text from {scripture} ({verse_id}) state?"
            assistant_resp = text_content
            if translation:
                assistant_resp += f"\n\nTranslation: {translation}"
                
            records.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_resp}
                ]
            })
        else:
            description = props.get("description", "").strip()
            if name and description:
                user_prompt = f"What is {name} ({primary_label}) according to {scripture}?"
                assistant_resp = f"{name} is a {primary_label} in {scripture}. {description}"
                records.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_resp}
                    ]
                })

    print("Converting relationships to instruction pairs...")
    for rel in relationships:
        source_id = rel.get("source")
        target_id = rel.get("target")
        rel_type = rel.get("type", "CONNECTED_TO")
        
        source_info = node_id_to_name.get(source_id)
        target_info = node_id_to_name.get(target_id)
        
        if source_info and target_info:
            s_name, s_scripture = source_info
            t_name, _ = target_info
            if s_name and t_name and s_name != t_name:
                user_prompt = f"How are '{s_name}' and '{t_name}' related in {s_scripture}?"
                assistant_resp = f"In {s_scripture}, '{s_name}' has the relationship [{rel_type}] with '{t_name}'."
                records.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_resp}
                    ]
                })

    print(f"Total instruction pairs generated: {len(records)}")

    # Shuffle records reproducible
    random.seed(42)
    random.shuffle(records)

    # 95% train, 5% validation split
    split_idx = int(len(records) * 0.95)
    train_records = records[:split_idx]
    valid_records = records[split_idx:]

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Writing {len(train_records)} records to {TRAIN_PATH}...")
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Writing {len(valid_records)} records to {VALID_PATH}...")
    with open(VALID_PATH, "w", encoding="utf-8") as f:
        for item in valid_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Conversion complete!")
    print(f"  - Train file size: {os.path.getsize(TRAIN_PATH) / (1024*1024):.2f} MB")
    print(f"  - Validation file size: {os.path.getsize(VALID_PATH) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_neo4j_to_jsonl()
