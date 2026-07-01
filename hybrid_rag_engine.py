#!/usr/bin/env python3
import os
import sys
import json
import re
import argparse
from mlx_lm import load, generate

# Ensure local cache usage
os.environ["HF_HOME"] = "./models"

def query_vector_db(user_query):
    """
    Mock retrieval function simulating a Vector DB lookup.
    Returns scriptural passages relevant to the user query.
    """
    # Sample Mock Database returns
    db_mocks = {
        "ahalya": (
            "Gautama's wife Ahalya was cursed to turn into a stone sculpture and perform penance for many years. "
            "She was promised redemption when Rama, the son of Dasaratha, set foot in her hermitage. "
            "Upon his arrival, she was freed from the curse, and her purity was restored."
        ),
        "mithila": (
            "Rama and Lakshmana went to Mithila guided by the sage Viswamitra to attend Janaka's sacrifice. "
            "King Janaka announced that whoever could bend the celestial bow of Shiva would marry his daughter Sita. "
            "Rama lifted, bent, and broke the mighty bow, winning Sita as his bride."
        ),
        "vishvamitra": (
            "Viswamitra, a warrior-king who attained the status of a Brahmarshi through severe austerities, "
            "came to Ayodhya to request Rama's help in guarding his sacrifices against demons like Maricha and Subahu. "
            "He served as a guide and preceptor to Rama and Lakshmana during their early travels."
        )
    }

    # Match query keywords to return relevant scripture context
    query_lower = user_query.lower()
    retrieved = []
    for key, text in db_mocks.items():
        if key in query_lower:
            retrieved.append(text)
            
    if not retrieved:
        # Default fallback context snippet if no keywords match
        retrieved.append(
            "Dasaratha's glorious son Rama, along with his brother Lakshmana, traveled through the sacred "
            "forests, protecting hermits, destroying demonic obstacles, and upholding righteousness (Dharma)."
        )
        
    return "\n\n".join(retrieved)

def find_ontology_nodes(text, ontology_path="raw_entities.json"):
    """
    Search ontology files (raw_entities.json or ontology.json) for keywords appearing in text,
    returning structured node information.
    """
    if ontology_path == "raw_entities.json" and not os.path.exists(ontology_path):
        if os.path.exists("ontology.json"):
            ontology_path = "ontology.json"
            
    if not os.path.exists(ontology_path):
        return "Ontology database unavailable."

    try:
        with open(ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            nodes = data.get("nodes", [])
    except Exception as e:
        return f"Error loading ontology: {e}"

    matched_nodes = []
    text_lower = text.lower()
    
    for node in nodes:
        node_id = node.get("id")
        node_name = node.get("name")
        candidates = {node_id, node_name}
        candidates = {c for c in candidates if c}
        
        has_match = False
        for cand in candidates:
            cand_lower = cand.lower()
            if cand_lower in text_lower:
                pattern = r'\b' + re.escape(cand_lower) + r'\b'
                if re.search(pattern, text_lower) or not cand_lower.isalnum():
                    has_match = True
                    break
        if has_match:
            matched_nodes.append({
                "id": node.get("id"),
                "name": node.get("name"),
                "type": node.get("type"),
                "description": node.get("attributes", {}).get("description", "")
            })
            # Limit count to keep prompt compact and prevent OOM
            if len(matched_nodes) >= 6:
                break
                
    if not matched_nodes:
        return "No explicit ontological records found in the text context."
        
    return json.dumps(matched_nodes, indent=2, ensure_ascii=False)

def generate_rag_response(user_query, model, tokenizer, ontology_path="raw_entities.json", max_tokens=256):
    """
    Performs the full Hybrid RAG pipeline step: retrieves vector text, intersects ontology,
    constructs the system context prompt, and runs adapted MLX-LM inference.
    """
    # 1. Retrieve Vector DB Scripture Snippets
    retrieved_chunks = query_vector_db(user_query)
    
    # 2. Extract matching nodes from Ontology
    matched_ontology = find_ontology_nodes(user_query + " " + retrieved_chunks, ontology_path)
    
    # 3. Construct System Prompt Context
    system_prompt = (
        "You are an advanced scriptural assistant. You must answer the user's inquiry using "
        "the precise philosophical definitions and tone learned during training. "
        "Ground your answer strictly in the factual context provided below.\n\n"
        f"Verified Scripture Context:\n{retrieved_chunks}\n\n"
        f"Ontological Metadata:\n{matched_ontology}"
    )

    # 4. Apply Chat Templates
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        # Fallback formatting
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    # 5. Run local inference
    print("\n--- [System context generated] ---")
    print(f"Verified Scripture Context size: {len(retrieved_chunks)} characters")
    print(f"Ontology Matches found: {matched_ontology != 'No explicit ontological records found in the text context.'}")
    print("----------------------------------\n")
    
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens)
    return response

def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Architecture Integration Module")
    parser.add_argument("--model", type=str, default=None, help="Base model path or Hugging Face ID")
    parser.add_argument("--adapter-path", type=str, default="./adapters", help="LoRA adapters path")
    parser.add_argument("--ontology", type=str, default="raw_entities.json", help="Ontology path")
    parser.add_argument("--query", type=str, default="Tell me how Ahalya was freed from Gautama's curse.", help="User inquiry")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens")
    args = parser.parse_args()

    # Determine base model path
    model_id = args.model
    if not model_id:
        local_path = "./models/Meta-Llama-3-8B-Instruct-4bit"
        if os.path.exists(local_path):
            model_id = local_path
        else:
            model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

    print("=========================================================")
    print("      HYBRID RAG INTEGRATION PIPELINE DEMONSTRATION      ")
    print("=========================================================")
    print(f"User Query:  {args.query}")
    print(f"Model ID:    {model_id}")
    print(f"Adapters:    {args.adapter_path}")
    print("\nLoading model weights into memory...")
    
    try:
        model, tokenizer = load(model_id, adapter_path=args.adapter_path)
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        sys.exit(1)

    print("Executing Hybrid RAG pipeline step...")
    answer = generate_rag_response(
        user_query=args.query,
        model=model,
        tokenizer=tokenizer,
        ontology_path=args.ontology,
        max_tokens=args.max_tokens
    )
    
    print("\n" + "=" * 60)
    print("             HYBRID RAG ASSISTANT ANSWER                  ")
    print("=" * 60)
    print(answer.strip())
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
