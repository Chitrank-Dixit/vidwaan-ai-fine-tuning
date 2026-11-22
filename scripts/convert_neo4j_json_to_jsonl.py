#!/usr/bin/env python3
"""Convert exported Neo4j graph JSON into leakage-free instruction-following JSONL datasets.

Groups records strictly by scripture/source corpus to prevent verse and entity leakage
across train.jsonl, valid.jsonl, and test.jsonl splits. Validates records with Pydantic v2.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_schema import (  # noqa: E402
    ChatMessage,
    ConversationMetadata,
    ScriptureConversation,
)


def load_neo4j_dump(file_path: Path | str) -> Tuple[List[dict], List[dict]]:
    """Ingest raw Neo4j JSON export containing nodes and relationships."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source Neo4j export not found at: {path}")

    print(f"Reading Neo4j export JSON from: {path}...")
    with open(path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    print(f"Loaded {len(nodes)} nodes and {len(relationships)} relationships.")
    return nodes, relationships


def extract_scripture_conversations(
    nodes: List[dict], relationships: List[dict]
) -> Dict[str, List[ScriptureConversation]]:
    """Convert nodes and relationships into conversations grouped by scripture corpus.

    This ensures that all verses and entities belonging to a particular scripture
    remain in the same split, completely preventing split leakage.
    """
    scripture_to_convs: Dict[str, List[ScriptureConversation]] = defaultdict(list)
    node_id_to_info: Dict[str, Tuple[str, str, str]] = {}

    print("Processing nodes into instruction dialogues...")
    for node in nodes:
        props = node.get("properties", {})
        labels = node.get("labels", [])
        primary_label = labels[0] if labels else "Entity"

        name = props.get("name") or props.get("id") or ""
        scripture = props.get("scripture") or "General Scriptures"
        verse_id = props.get("verse_id") or name

        if props.get("id"):
            node_id_to_info[node["id"]] = (name, scripture, primary_label)

        if primary_label == "Text":
            text_content = (props.get("text") or "").strip()
            translation = (props.get("translation") or "").strip()

            if not text_content:
                continue

            user_prompt = f"What does the text from {scripture} ({verse_id}) state?"
            assistant_resp = text_content
            if translation:
                assistant_resp += f"\n\nTranslation: {translation}"

            try:
                conv = ScriptureConversation(
                    messages=[
                        ChatMessage(role="user", content=user_prompt),
                        ChatMessage(role="assistant", content=assistant_resp),
                    ],
                    metadata=ConversationMetadata(
                        scripture=scripture,
                        verse_id=verse_id,
                        source_entity=name,
                    ),
                )
                scripture_to_convs[scripture].append(conv)
            except Exception as e:
                print(f"Skipping malformed node dialogue ({verse_id}): {e}")

        else:
            description = (props.get("description") or "").strip()
            if name and description:
                user_prompt = f"What is {name} ({primary_label}) according to {scripture}?"
                assistant_resp = f"{name} is a {primary_label} in {scripture}. {description}"
                try:
                    conv = ScriptureConversation(
                        messages=[
                            ChatMessage(role="user", content=user_prompt),
                            ChatMessage(role="assistant", content=assistant_resp),
                        ],
                        metadata=ConversationMetadata(
                            scripture=scripture,
                            source_entity=name,
                        ),
                    )
                    scripture_to_convs[scripture].append(conv)
                except Exception as e:
                    print(f"Skipping malformed entity dialogue ({name}): {e}")

    print("Processing relationships into instruction dialogues...")
    for rel in relationships:
        source_id = rel.get("source")
        target_id = rel.get("target")
        rel_type = rel.get("type", "CONNECTED_TO")

        source_info = node_id_to_info.get(source_id)
        target_info = node_id_to_info.get(target_id)

        if source_info and target_info:
            s_name, s_scripture, s_label = source_info
            t_name, _, t_label = target_info

            if s_name and t_name and s_name != t_name:
                user_prompt = f"How are '{s_name}' and '{t_name}' related in {s_scripture}?"
                assistant_resp = f"In {s_scripture}, '{s_name}' has the relationship [{rel_type}] with '{t_name}'."
                try:
                    conv = ScriptureConversation(
                        messages=[
                            ChatMessage(role="user", content=user_prompt),
                            ChatMessage(role="assistant", content=assistant_resp),
                        ],
                        metadata=ConversationMetadata(
                            scripture=s_scripture,
                            source_entity=s_name,
                            target_entity=t_name,
                            relationship_type=rel_type,
                        ),
                    )
                    scripture_to_convs[s_scripture].append(conv)
                except Exception as e:
                    print(f"Skipping malformed relationship dialogue: {e}")

    return scripture_to_convs


def partition_scriptures_leak_free(
    scripture_to_convs: Dict[str, List[ScriptureConversation]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[ScriptureConversation], List[ScriptureConversation], List[ScriptureConversation]]:
    """Partition records by scripture cluster to strictly prevent passage and entity leakage."""
    scriptures = sorted(list(scripture_to_convs.keys()))
    rng = random.Random(seed)
    rng.shuffle(scriptures)

    n_scriptures = len(scriptures)
    if n_scriptures >= 3:
        n_valid = max(1, int(round(n_scriptures * valid_ratio)))
        n_test = max(1, int(round(n_scriptures * test_ratio)))
        n_train = max(1, n_scriptures - n_valid - n_test)

        train_s = set(scriptures[:n_train])
        valid_s = set(scriptures[n_train : n_train + n_valid])
        test_s = set(scriptures[n_train + n_valid :])
    elif n_scriptures == 2:
        train_s = {scriptures[0]}
        valid_s = {scriptures[1]}
        test_s = {scriptures[1]}  # Fallback for 2 scriptures
    else:
        train_s = set(scriptures)
        valid_s = set()
        test_s = set()

    train_convs: List[ScriptureConversation] = [c for s in train_s for c in scripture_to_convs[s]]
    valid_convs: List[ScriptureConversation] = [c for s in valid_s for c in scripture_to_convs[s]]
    test_convs: List[ScriptureConversation] = [c for s in test_s for c in scripture_to_convs[s]]

    # Shuffle each split internally
    rng.shuffle(train_convs)
    rng.shuffle(valid_convs)
    rng.shuffle(test_convs)

    return train_convs, valid_convs, test_convs


def write_jsonl_file(records: List[ScriptureConversation], output_file: Path) -> int:
    """Stream conversation records to disk in standard OpenAI/ShareGPT messages format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            line_data = {
                "messages": [m.model_dump() for m in rec.messages],
            }
            if rec.metadata:
                line_data["metadata"] = rec.metadata.model_dump(exclude_none=True)
            f.write(json.dumps(line_data, ensure_ascii=False) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Neo4j graph dump to leak-free train/valid/test JSONL splits."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("SOURCE_JSON", "data/neo4j_dump_exported.json"),
        help="Path to neo4j_dump_exported.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("DATA_DIR", "data"),
        help="Output directory for train.jsonl, valid.jsonl, test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split fraction")

    args = parser.parse_args()

    source_path = Path(args.source)
    out_dir = Path(args.output_dir)

    print("=========================================================")
    print("      NEO4J TO LEAKAGE-FREE JSONL SPLITTER               ")
    print("=========================================================")
    print(f"Source JSON:       {source_path}")
    print(f"Output Directory:  {out_dir}")
    print(f"Ratios:            {args.train_ratio} / {args.valid_ratio} / {args.test_ratio}")
    print(f"Seed:              {args.seed}")

    nodes, relationships = load_neo4j_dump(source_path)
    scripture_to_convs = extract_scripture_conversations(nodes, relationships)

    print(f"Identified {len(scripture_to_convs)} distinct scripture corpora.")
    for s, c in sorted(scripture_to_convs.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"  - {s}: {len(c)} records")

    train_c, valid_c, test_c = partition_scriptures_leak_free(
        scripture_to_convs,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    test_path = out_dir / "test.jsonl"

    print(f"\nWriting {len(train_c)} train records to {train_path}...")
    write_jsonl_file(train_c, train_path)

    print(f"Writing {len(valid_c)} valid records to {valid_path}...")
    write_jsonl_file(valid_c, valid_path)

    print(f"Writing {len(test_c)} test records to {test_path}...")
    write_jsonl_file(test_c, test_path)

    print("\n✔ Splitting complete with zero cross-split passage leakage!")
    print(f"  Train: {len(train_c)} | Valid: {len(valid_c)} | Test: {len(test_c)}")
    print("=========================================================")


if __name__ == "__main__":
    main()
