#!/usr/bin/env python3
import os
import re
import json
import glob
import random
import argparse
from pypdf import PdfReader
from google import genai
from google.genai import types

def load_ontology(ontology_path):
    """
    Parse and map ontology definitions.
    Returns a list of node dictionaries.
    """
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
    Extracts text page-by-page from a list of PDF files.
    """
    full_text_parts = []
    for path in pdf_paths:
        print(f"Extracting text from PDF: {path}")
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text_parts.append(text)
            print(f"Successfully extracted {len(reader.pages)} pages from {path}")
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
    return "\n\n".join(full_text_parts)

def segment_text(text, chunk_size=900):
    """
    Segments raw text into clean, sequential chunks of approx chunk_size words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
    print(f"Segmented text into {len(chunks)} chunks (approx. {chunk_size} words each).")
    return chunks

def find_intersecting_metadata(text_chunk, ontology_nodes):
    """
    Scans a text chunk for keywords matching node id or name in the ontology.
    """
    matching_nodes = []
    text_lower = text_chunk.lower()
    
    for node in ontology_nodes:
        node_id = node.get("id")
        node_name = node.get("name")
        
        # Candidate keywords to check (ignoring None or empty values)
        candidates = {node_id, node_name}
        candidates = {c for c in candidates if c}
        
        matched = False
        for cand in candidates:
            cand_lower = cand.lower()
            # Handle underscores commonly found in ontology IDs
            cand_space = cand_lower.replace("_", " ")
            
            # Simple substring check first for speed
            if cand_lower in text_lower or cand_space in text_lower:
                # Use regex with word boundaries to avoid false positives (e.g. matching "he" inside "here")
                pattern_cand = r'\b' + re.escape(cand_lower) + r'\b'
                pattern_space = r'\b' + re.escape(cand_space) + r'\b'
                
                if re.search(pattern_cand, text_lower) or re.search(pattern_space, text_lower):
                    matched = True
                    break
                # Fallback for non-alphanumeric keys where word boundaries might fail
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

def sanitize_llm_json_lines(response_text):
    """
    Extracts and validates JSON lines from raw LLM output, handling markdown fences and formatting variations.
    """
    records = []
    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip markdown code fence markers if the model generates them
        if line.startswith("```"):
            continue
            
        try:
            # Find the starting and ending indices of the JSON object
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1:
                json_str = line[start:end+1]
                data = json.loads(json_str)
                
                # Verify that it matches the MLX conversational structure
                if "messages" in data and isinstance(data["messages"], list):
                    messages = data["messages"]
                    # We expect a pair: user prompt and assistant response
                    if len(messages) >= 2:
                        roles = {m.get("role") for m in messages}
                        if "user" in roles and "assistant" in roles:
                            records.append(data)
        except Exception as e:
            # Log parsing errors but do not crash the pipeline
            pass
            
    return records

def generate_records_for_chunk(client, chunk, matching_metadata):
    """
    Calls Gemini API to generate synthetic QA records grounded in the chunk and metadata.
    """
    prompt = f"""You are an expert scriptural scholar. You are creating fine-tuning training records.

Scripture Text Passage:
{chunk}

Validated Ontological Metadata:
{json.dumps(matching_metadata, indent=2, ensure_ascii=False)}

Generate 5 distinct, highly realistic user questions and perfectly accurate assistant answers. Mix the types of questions:
1. Factual/Textual: Grounded directly in the passage text.
2. Conceptual/Relational: Requiring understanding of how characters or places link together based on the provided ontology metadata.

Output your response STRICTLY as individual JSON Lines (.jsonl) following the exact MLX chat schema:
{{"messages": [{{"role": "user", "content": "QUESTION"}}, {{"role": "assistant", "content": "ANSWER"}}]}}
Do not include markdown code fences, trailing text, or preamble."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                system_instruction="You are a strict QA record generator. Your output must only contain valid JSON lines in the requested format."
            )
        )
        return sanitize_llm_json_lines(response.text or "")
    except Exception as e:
        print(f"Error during LLM API call: {e}")
        return []

def generate_mock_records_for_chunk(chunk, matching_metadata):
    """
    Generates mock QA records locally based on matched ontology metadata.
    """
    records = []
    
    # 1. Factual/Textual Question
    words = chunk.split()
    first_few_words = " ".join(words[:15]) if len(words) > 15 else chunk
    records.append({
        "messages": [
            {"role": "user", "content": f"Based on the text passage, what is mentioned near the start: '{first_few_words}'?"},
            {"role": "assistant", "content": f"The passage opens by discussing or describing: '{first_few_words}'."}
        ]
    })
    
    # 2. Conceptual/Relational Questions based on matched ontology nodes
    for idx, node in enumerate(matching_metadata[:4]): # limit to 4 to make exactly 5 records max
        name = node.get("name") or node.get("id")
        node_type = node.get("type", "concept")
        description = node.get("description", "Not described.")
        
        records.append({
            "messages": [
                {"role": "user", "content": f"How is the {node_type} '{name}' defined in the scripture ontology?"},
                {"role": "assistant", "content": f"In the scriptural ontology, '{name}' is a {node_type} described as: {description}"}
            ]
        })
        
    # Ensure we have at least 5 records
    while len(records) < 5:
        records.append({
            "messages": [
                {"role": "user", "content": "What is the significance of the scripture passage presented in this chunk?"},
                {"role": "assistant", "content": "The scripture passage provides vital narrative and context regarding ancient sages, characters, and events detailed in the ontology."}
            ]
        })
        
    return records[:5]

def main():
    parser = argparse.ArgumentParser(description="Ontology-Driven Scripture PDF Dataset Pipeline")
    parser.add_argument("--ontology", type=str, default="ontology.json", help="Path to ontology JSON file")
    parser.add_argument("--pdf-dir", type=str, default=".", help="Directory to search for source PDF files")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save train.jsonl and valid.jsonl")
    parser.add_argument("--chunk-size", type=int, default=900, help="Approximate word chunk size for text segmentation")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for deterministic dataset shuffling and splitting")
    parser.add_argument("--mock-llm", action="store_true", help="Simulate LLM QA generation without calling the API")
    args = parser.parse_args()

    # Load .env manually if present in current, parent, or sibling directories
    dotenv_paths = [".env", "../.env", "../vidwaan-ai-mvp/.env"]
    for env_path in dotenv_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            os.environ[key.strip()] = val.strip().strip('"').strip("'")
                print(f"Loaded environment variables from: {env_path}")
                break
            except Exception as e:
                print(f"Warning: Failed to read env file {env_path}: {e}")

    # Verify Gemini API configuration
    api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    if not api_key_set and not args.mock_llm:
        print("WARNING: GEMINI_API_KEY environment variable is not set. API calls will fail unless credential helper is configured.")

    # 1. Load ontology definitions
    try:
        ontology_nodes = load_ontology(args.ontology)
    except Exception as e:
        print(f"Critical error loading ontology: {e}")
        return

    # 2. Collect PDF source files
    pdf_pattern = os.path.join(args.pdf_dir, "**", "*.pdf")
    pdf_paths = glob.glob(pdf_pattern, recursive=True)
    
    # Check alternate common directories if none found in root
    if not pdf_paths:
        for alt_dir in ["data", "pdfs"]:
            pdf_paths.extend(glob.glob(os.path.join(alt_dir, "**", "*.pdf"), recursive=True))
            
    if not pdf_paths:
        print("Error: No PDF files found to parse. Please place your scripture PDFs in the repository directory, './data/', or './pdfs/'.")
        return
        
    print(f"Found {len(pdf_paths)} PDF file(s) to process.")

    # 3. Extract text page-by-page
    raw_text = extract_text_from_pdfs(pdf_paths)
    if not raw_text.strip():
        print("Error: Extracted text is empty. Verify that your PDF files contain selectable text elements.")
        return
    print(f"Total words extracted from PDF(s): {len(raw_text.split())}")

    # 4. Segment text into chunks
    chunks = segment_text(raw_text, chunk_size=args.chunk_size)

    # 5. Initialize Google GenAI client (only if not mocking and key is present)
    client = None
    if not args.mock_llm and api_key_set:
        client = genai.Client()
    
    all_generated_records = []

    # 6. Loop chunks, intersect metadata, and generate QA
    print("\nStarting synthetic dataset generation...")
    for idx, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {idx + 1}/{len(chunks)}...")
        matching_metadata = find_intersecting_metadata(chunk, ontology_nodes)
        print(f"Found {len(matching_metadata)} intersecting ontology entities.")
        
        # Request QA generation from Gemini 2.0 or simulate it
        if args.mock_llm or not api_key_set:
            if not api_key_set and not args.mock_llm:
                print("No API key found. Falling back to mock LLM generation to verify pipeline...")
            records = generate_mock_records_for_chunk(chunk, matching_metadata)
        else:
            records = generate_records_for_chunk(client, chunk, matching_metadata)
            
        print(f"Generated {len(records)} valid dataset records for this chunk.")
        all_generated_records.extend(records)

    print(f"\nTotal conversational dataset records generated: {len(all_generated_records)}")
    if not all_generated_records:
        print("Error: No valid records were successfully generated by the pipeline.")
        return

    # 7. Shuffle and split datasets (85% train, 15% validation)
    print("\nSplitting dataset into train.jsonl and valid.jsonl...")
    random.seed(args.seed)
    random.shuffle(all_generated_records)

    split_index = int(len(all_generated_records) * 0.85)
    train_set = all_generated_records[:split_index]
    valid_set = all_generated_records[split_index:]

    # Write output files
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    valid_path = os.path.join(args.output_dir, "valid.jsonl")

    try:
        with open(train_path, "w", encoding="utf-8") as f:
            for record in train_set:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(train_set)} records to: {train_path}")

        with open(valid_path, "w", encoding="utf-8") as f:
            for record in valid_set:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(valid_set)} records to: {valid_path}")
        
        print("\nDataset pipeline execution completed successfully!")
    except Exception as e:
        print(f"Error saving output dataset files: {e}")

if __name__ == "__main__":
    main()
