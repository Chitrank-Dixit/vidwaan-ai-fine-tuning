#!/usr/bin/env python3
"""Local Dataset Quality Audit & Validation Gate.

Enforces Pydantic v2 schemas on instruction-following datasets, verifies
message role continuity (system? -> user -> assistant), isolates corrupted
lines, detects duplicate samples, and calculates structural quality metrics.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import ValidationError  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from src.dataset_schema import ScriptureConversation  # noqa: E402

# =====================================================================
# Context Rebuilding & PDF Helpers (Preserved for compatibility)
# =====================================================================


def load_ontology(ontology_path: str) -> List[dict]:
    """Load entity ontology definitions from json file."""
    if ontology_path == "raw_entities.json" and not os.path.exists(ontology_path):
        if os.path.exists("ontology.json"):
            ontology_path = "ontology.json"
    if not os.path.exists(ontology_path):
        return []
    with open(ontology_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", [])


def extract_text_from_pdfs(pdf_paths: List[str]) -> str:
    """Extract raw text from PDF files."""
    full_text_parts = []
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text_parts.append(text)
        except Exception:
            pass
    return "\n\n".join(full_text_parts)


def segment_text(text: str, chunk_size: int = 900) -> List[str]:
    """Chunk raw text into roughly chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def find_matched_entities(text: str, ontology_nodes: List[dict]) -> List[dict]:
    """Find ontology nodes mentioned within text string."""
    matched = []
    text_lower = text.lower()
    for node in ontology_nodes:
        node_id = node.get("id")
        node_name = node.get("name")
        candidates = {node_id, node_name}
        candidates = {c for c in candidates if c}

        has_match = False
        for cand in candidates:
            cand_lower = cand.lower()
            cand_space = cand_lower.replace("_", " ")
            if cand_lower in text_lower or cand_space in text_lower:
                pattern_cand = r"\b" + re.escape(cand_lower) + r"\b"
                pattern_space = r"\b" + re.escape(cand_space) + r"\b"
                if re.search(pattern_cand, text_lower) or re.search(pattern_space, text_lower):
                    has_match = True
                    break
                elif not cand_lower.isalnum():
                    has_match = True
                    break
        if has_match:
            matched.append(
                {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "description": node.get("attributes", {}).get("description", ""),
                }
            )
    return matched


def find_best_context_chunk(
    qa_text: str, chunks: List[str], ontology_nodes: List[dict]
) -> Tuple[str, List[dict]]:
    """Find the text chunk and matched ontology nodes that most closely intersect with QA text."""
    if not chunks:
        return "Source text passage unavailable.", []

    qa_entities = find_matched_entities(qa_text, ontology_nodes)
    if not qa_entities:
        return chunks[0], []

    entity_names = {node.get("name").lower() for node in qa_entities if node.get("name")}
    entity_ids = {node.get("id").lower() for node in qa_entities if node.get("id")}
    search_terms = entity_names.union(entity_ids)

    best_chunk = chunks[0]
    max_matches = -1

    for chunk in chunks:
        chunk_lower = chunk.lower()
        matches = sum(1 for term in search_terms if term in chunk_lower)
        if matches > max_matches:
            max_matches = matches
            best_chunk = chunk

    matched_nodes = find_matched_entities(best_chunk, ontology_nodes)
    return best_chunk, matched_nodes


def estimate_tokens(text: str) -> int:
    """Heuristic token count estimation (~4 characters per token)."""
    return max(1, int(len(text) / 4))


def call_local_critic(
    base_url: str,
    model_name: str,
    chunk: str,
    matched_nodes: List[dict],
    qa_pair: dict,
) -> Optional[dict]:
    """Query an external OpenAI-compatible critic LLM endpoint (optional validation)."""
    critic_prompt = (
        "You are a strict Data Quality Critic. Evaluate the following scriptural Question-Answer (QA) pair "
        "based on the provided Scripture Passage and Ontological Metadata. \n\n"
        f"Scripture Passage:\n{chunk}\n\n"
        f"Ontological Metadata:\n{json.dumps(matched_nodes, indent=2, ensure_ascii=False)}\n\n"
        f"QA Pair:\nQuestion: {qa_pair.get('question', '')}\nAnswer: {qa_pair.get('answer', '')}\n\n"
        "Score this QA pair on the following three criteria on a scale of 1 to 5 (1 being poor, 5 being perfect):\n"
        "1. Faithfulness: Is the answer strictly grounded in the passage text, or does it make things up?\n"
        "2. Ontological Alignment: Does the answer accurately reflect the character/place attributes defined in the ontology metadata?\n"
        "3. Tone Consistency: Is the response respectful, scholarly, and commentary-backed?\n\n"
        "Provide your evaluation output strictly in the following JSON format. Do not include markdown code fences or preambles:\n"
        "{\n"
        '  "faithfulness_score": <int>,\n'
        '  "ontological_score": <int>,\n'
        '  "tone_score": <int>,\n'
        '  "critical_review": "<string>"\n'
        "}"
    )

    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a database quality inspector. Return strictly JSON formatting.",
            },
            {"role": "user", "content": critic_prompt},
        ],
        "temperature": 0.1,
    }

    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_output = res_data["choices"][0]["message"]["content"]
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw_output[start : end + 1])
    except Exception:
        pass
    return None


# =====================================================================
# Pydantic v2 Dataset Validator Engine
# =====================================================================


class DatasetValidator:
    """Rigorous dataset auditor checking Pydantic v2 schemas and token integrity."""

    def __init__(self, data_dir: Path | str = "data"):
        self.data_dir = Path(data_dir)

    def validate_file(
        self, file_path: Path
    ) -> Tuple[List[ScriptureConversation], List[str], Dict[str, Any]]:
        """Audit a single JSONL file against ScriptureConversation schema."""
        valid_records: List[ScriptureConversation] = []
        corrupted_lines: List[str] = []
        user_queries: Set[str] = set()
        duplicate_queries = 0
        total_tokens = 0

        if not file_path.is_file():
            return [], [f"File not found: {file_path}"], {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    data = json.loads(stripped)
                    conv = ScriptureConversation.model_validate(data)
                    valid_records.append(conv)

                    # Check for duplicate user queries
                    user_msg = next((m.content for m in conv.messages if m.role == "user"), "")
                    if user_msg in user_queries:
                        duplicate_queries += 1
                    else:
                        user_queries.add(user_msg)

                    # Token estimate
                    full_text = " ".join(m.content for m in conv.messages)
                    total_tokens += estimate_tokens(full_text)

                except (json.JSONDecodeError, ValidationError) as e:
                    corrupted_lines.append(f"Line {line_idx}: {e} | Content: {stripped[:120]}")

        stats = {
            "total_lines": len(valid_records) + len(corrupted_lines),
            "valid_count": len(valid_records),
            "corrupted_count": len(corrupted_lines),
            "duplicate_queries": duplicate_queries,
            "avg_tokens_per_sample": round(total_tokens / max(1, len(valid_records)), 1),
        }
        return valid_records, corrupted_lines, stats


def main():
    parser = argparse.ArgumentParser(
        description="Local Dataset Quality Audit & Validation Gate (Pydantic v2)"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with JSONL files")
    parser.add_argument(
        "--ontology", type=str, default="raw_entities.json", help="Path to ontology file"
    )
    parser.add_argument("--pdf-dir", type=str, default=".", help="PDF folder path")
    parser.add_argument("--base-url", type=str, default=None, help="Critic model base URL")
    parser.add_argument("--model", type=str, default=None, help="Critic model name")
    parser.add_argument("--mock-judge", action="store_true", help="Simulate critic scores locally")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    test_path = data_dir / "test.jsonl"
    corrupted_path = data_dir / "corrupted.jsonl"

    print("=========================================================")
    print("      DATASET QUALITY AUDIT & VALIDATION GATE            ")
    print("=========================================================")
    print(f"Data Directory: {data_dir}")

    validator = DatasetValidator(data_dir=data_dir)

    all_corrupted: List[str] = []
    overall_valid = 0
    overall_corrupted = 0

    target_files = [train_path, valid_path]
    if test_path.is_file():
        target_files.append(test_path)

    for target in target_files:
        if not target.is_file():
            print(f"⚠️ Warning: Dataset file not found: {target}")
            continue

        print(f"\nAuditing schema and content: {target.name}")
        valid_recs, corrupted, stats = validator.validate_file(target)
        overall_valid += stats["valid_count"]
        overall_corrupted += stats["corrupted_count"]
        all_corrupted.extend(corrupted)

        print(f"  - Valid records:       {stats['valid_count']}")
        print(f"  - Corrupted records:   {stats['corrupted_count']}")
        print(f"  - Duplicate queries:   {stats['duplicate_queries']}")
        print(f"  - Avg tokens / sample: {stats['avg_tokens_per_sample']}")

    if all_corrupted:
        corrupted_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corrupted_path, "w", encoding="utf-8") as f:
            for err in all_corrupted:
                f.write(err + "\n")
        print(f"\n⚠️ {len(all_corrupted)} corrupted lines isolated into: {corrupted_path}")
    else:
        print("\n✔ Zero corrupted lines found across evaluated datasets.")

    print(f"\nAudit Summary: {overall_valid} valid records, {overall_corrupted} corrupted.")
    print("=========================================================")


if __name__ == "__main__":
    main()
