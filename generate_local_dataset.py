#!/usr/bin/env python3
import os
import re
import json
import glob
import random
import argparse
import urllib.request
from pypdf import PdfReader

def load_ontology(ontology_path):
    """
    Ingest Ontology: Load the ontology file (default raw_entities.json) into a local dictionary.
    Includes fallback to ontology.json if not found.
    """
    # Auto-fallback to ontology.json if default raw_entities.json is missing
    if ontology_path == "raw_entities.json" and not os.path.exists(ontology_path):
        if os.path.exists("ontology.json"):
            print("raw_entities.json not found. Falling back to ontology.json.")
            ontology_path = "ontology.json"
            
    print(f"Loading ontology from: {ontology_path}")
    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"Ontology file not found at {ontology_path}")
    
    with open(ontology_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    print(f"Successfully loaded {len(nodes)} ontology nodes.")
    return nodes

def extract_text_from_pdfs(pdf_paths):
    """
    Extracts text page-by-page from local PDFs.
    """
    full_text_parts = []
    for path in pdf_paths:
        print(f"Extracting text from PDF: {path}")
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text_parts.append(text)
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
    return "\n\n".join(full_text_parts)

def segment_text(text, chunk_size=900):
    """
    Segments raw text into sequential blocks of approximately 800-1000 words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
    print(f"Segmented text into {len(chunks)} blocks (approx. {chunk_size} words each).")
    return chunks

def find_semantic_intersections(text_block, ontology_nodes):
    """
    Semantic Intersection: Scan text block for references that match ontology node names.
    Gather matching entity nodes and their attributes -> description payloads.
    """
    matching_nodes = []
    text_lower = text_block.lower()
    
    for node in ontology_nodes:
        node_id = node.get("id")
        node_name = node.get("name")
        
        # Candidate names to search
        candidates = {node_id, node_name}
        candidates = {c for c in candidates if c}
        
        matched = False
        for cand in candidates:
            cand_lower = cand.lower()
            cand_space = cand_lower.replace("_", " ")
            
            # Fast substring search
            if cand_lower in text_lower or cand_space in text_lower:
                # Verify exact word boundaries to prevent matching subsets (e.g. matching "he" inside "here")
                pattern_cand = r'\b' + re.escape(cand_lower) + r'\b'
                pattern_space = r'\b' + re.escape(cand_space) + r'\b'
                
                if re.search(pattern_cand, text_lower) or re.search(pattern_space, text_lower):
                    matched = True
                    break
                # Fallback for non-alphanumeric entity keys (e.g. Bhrigu's_child)
                elif not cand_lower.isalnum():
                    matched = True
                    break
                    
        if matched:
            matching_nodes.append({
                "id": node.get("id"),
                "name": node.get("name"),
                "type": node.get("type"),
                "description": node.get("attributes", {}).get("description", "")
            })
            
    return matching_nodes

def sanitize_json_lines(response_text):
    """
    Validation wrapper: parses and strips accidental markdown styling (like backticks) from response stream.
    """
    records = []
    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Strip markdown fences if present
        if line.startswith("```"):
            continue
            
        try:
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1:
                json_str = line[start:end+1]
                data = json.loads(json_str)
                
                # Verify schema: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                if "messages" in data and isinstance(data["messages"], list) and len(data["messages"]) >= 2:
                    roles = {m.get("role") for m in data["messages"]}
                    if "user" in roles and "assistant" in roles:
                        records.append(data)
        except Exception:
            pass
            
    return records

def call_local_lm_studio(base_url, model_name, text_block, matching_metadata):
    """
    Standard OpenAI-compatible API client block targeting the local LM Studio instance.
    """
    system_prompt = (
        "You are an expert scriptural researcher. Generate 5 distinct, highly realistic user questions "
        "and highly accurate scripture-based answers from the perspective of an advanced assistant. "
        "Ground your reasoning strictly in the provided text block and its explicit ontological node metadata. "
        "\n\nOutput your response ONLY as valid JSON Lines (.jsonl). Every line must match this exact format:\n"
        '{"messages": [{"role": "user", "content": "QUESTION_HERE"}, {"role": "assistant", "content": "ANSWER_HERE"}]}'
        "\nDo not include any chat preamble, markdown code blocks (such as ```json), or trailing text. "
        "Output raw JSON lines directly."
    )

    user_content = f"""Scripture Text Passage:
{text_block}

Ontological Node Metadata:
{json.dumps(matching_metadata, indent=2, ensure_ascii=False)}"""

    # Normalize url endpoint
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    endpoint_url = f"{base_url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        # 60 second timeout for local inference
        with urllib.request.urlopen(req, timeout=90) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_output = res_data["choices"][0]["message"]["content"]
            print("\n--- RAW LM STUDIO RESPONSE ---")
            print(raw_output)
            print("------------------------------\n")
            return sanitize_json_lines(raw_output)
    except Exception as e:
        print(f"Connection to local LM Studio failed: {e}")
        return []

def generate_mock_dataset_records(chunk, matching_metadata):
    """
    Generates mock QA records locally to verify the pipeline structure if LM Studio is not active.
    """
    records = []
    words = chunk.split()
    first_few_words = " ".join(words[:15]) if len(words) > 15 else chunk
    
    # 1. Textual Question
    records.append({
        "messages": [
            {"role": "user", "content": f"What is introduced at the beginning of this passage: '{first_few_words}'?"},
            {"role": "assistant", "content": f"The scriptural passage opens by introducing: '{first_few_words}'."}
        ]
    })
    
    # 2. Relational Questions
    for node in matching_metadata[:4]:
        name = node.get("name") or node.get("id")
        node_type = node.get("type", "Concept")
        desc = node.get("description", "defined in the sacred lore.")
        
        records.append({
            "messages": [
                {"role": "user", "content": f"Who or what is '{name}' in this scriptural context?"},
                {"role": "assistant", "content": f"Based on the ontology, '{name}' is a {node_type} described as: {desc}"}
            ]
        })
        
    while len(records) < 5:
        records.append({
            "messages": [
                {"role": "user", "content": "What details can we draw from the surrounding context?"},
                {"role": "assistant", "content": "The scripture outlines historical, geographical, and relational connections among key figures."}
            ]
        })
        
    return records[:5]

def main():
    parser = argparse.ArgumentParser(description="Local Ontology-Driven QA Generation via LM Studio")
    parser.add_argument("--ontology", type=str, default="raw_entities.json", help="Path to raw_entities.json")
    parser.add_argument("--pdf-dir", type=str, default=".", help="Directory to search for PDFs")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--base-url", type=str, default=None, help="LM Studio base URL")
    parser.add_argument("--model", type=str, default=None, help="LM Studio target model")
    parser.add_argument("--chunk-size", type=int, default=900, help="Word block chunk size")
    parser.add_argument("--seed", type=int, default=42, help="Splitting seed")
    parser.add_argument("--mock-fallback", action="store_true", help="Use local mock QA if server connection fails")
    args = parser.parse_args()

    # Load environment settings manually
    dotenv_paths = [".env", "../.env", "../vidwaan-ai-mvp/.env"]
    env_vars = {}
    for env_path in dotenv_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip().strip('"').strip("'")
                print(f"Loaded environment variables from: {env_path}")
                break
            except Exception:
                pass

    # Resolve settings (priority: Arg -> Env -> Default)
    base_url = args.base_url or env_vars.get("LMSTUDIO_BASE_URL") or "http://localhost:1234"
    model_name = args.model or env_vars.get("LMSTUDIO_MODEL") or "mistralai/mistral-7b-instruct-v0.3"

    print(f"Configuring local LM Studio connection:")
    print(f"  Base URL:     {base_url}")
    print(f"  Target Model: {model_name}")

    # 1. Load ontology
    try:
        ontology_nodes = load_ontology(args.ontology)
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return

    # 2. Collect PDFs
    pdf_paths = glob.glob(os.path.join(args.pdf_dir, "**", "*.pdf"), recursive=True)
    if not pdf_paths:
        for alt_dir in ["data", "pdfs"]:
            pdf_paths.extend(glob.glob(os.path.join(alt_dir, "**", "*.pdf"), recursive=True))
            
    if not pdf_paths:
        print("Error: No scripture PDF files found. Place PDFs in current folder, './data/', or './pdfs/'.")
        return
        
    print(f"Found {len(pdf_paths)} scripture PDF file(s).")

    # 3. Extract text
    raw_text = extract_text_from_pdfs(pdf_paths)
    if not raw_text.strip():
        print("Error: Extracted scripture text is empty.")
        return
    print(f"Extracted {len(raw_text.split())} total words from scripture PDFs.")

    # 4. Chunk text
    blocks = segment_text(raw_text, chunk_size=args.chunk_size)

    # 5. Process blocks and request local model
    all_generated_lines = []
    print("\nStarting local QA generation loop...")
    for idx, block in enumerate(blocks):
        print(f"\nProcessing scripture block {idx + 1}/{len(blocks)}...")
        matching_metadata = find_semantic_intersections(block, ontology_nodes)
        print(f"Found {len(matching_metadata)} intersecting ontology entities in block.")
        
        # Run local LM Studio inference
        records = call_local_lm_studio(base_url, model_name, block, matching_metadata)
        
        # Fallback to local mock if connection failed or returned empty and fallback is enabled
        if not records and args.mock_fallback:
            print("Falling back to local mock QA generator for this block...")
            records = generate_mock_dataset_records(block, matching_metadata)
            
        print(f"Generated {len(records)} valid dataset lines.")
        all_generated_lines.extend(records)

    print(f"\nTotal validated dataset lines generated: {len(all_generated_lines)}")
    if not all_generated_lines:
        print("Error: No dataset lines generated. Ensure LM Studio server is running or pass --mock-fallback.")
        return

    # 6. Stratified Train/Valid Splitting (85% train / 15% valid)
    print("\nExecuting stratified shuffle and dataset split...")
    random.seed(args.seed)
    random.shuffle(all_generated_lines)

    split_index = int(len(all_generated_lines) * 0.85)
    train_set = all_generated_lines[:split_index]
    valid_set = all_generated_lines[split_index:]

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    valid_path = os.path.join(args.output_dir, "valid.jsonl")

    try:
        with open(train_path, "w", encoding="utf-8") as f:
            for rec in train_set:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(train_set)} records to: {train_path}")

        with open(valid_path, "w", encoding="utf-8") as f:
            for rec in valid_set:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(valid_set)} records to: {valid_path}")
        
        print("\nLocal dataset generation pipeline completed successfully!")
    except Exception as e:
        print(f"Error saving output datasets: {e}")

if __name__ == "__main__":
    main()
