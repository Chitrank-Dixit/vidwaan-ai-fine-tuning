#!/usr/bin/env python3
import os
import sys

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
    constructs the system context prompt, and runs adapted Hugging Face inference.
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
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback formatting
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    # 5. Run local inference
    print("\n--- [System context generated] ---")
    print(f"Verified Scripture Context size: {len(retrieved_chunks)} characters")
    print(f"Ontology Matches found: {matched_ontology != 'No explicit ontological records found in the text context.'}")
    print("----------------------------------\n")
    
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response

def load_model_and_tokenizer(model_id, adapter_path=None):
    """
    Utility loader.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if adapter_path and os.path.exists(adapter_path):
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch_dtype)
        else:
            print(f"⚠️ Warning: adapter_config.json not found in {adapter_path}. Running base model only.")
            model = base_model
    else:
        model = base_model
        
    if device_map is None:
        model = model.to(device)
        
    return model, tokenizer

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
        try:
            import yaml
            if os.path.exists("config.yaml"):
                with open("config.yaml", "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg and "model" in cfg:
                        model_id = cfg["model"]
        except Exception:
            pass
        if not model_id:
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print("=========================================================")
    print("      HYBRID RAG INTEGRATION PIPELINE DEMONSTRATION      ")
    print("=========================================================")
    print(f"User Query:  {args.query}")
    print(f"Model ID:    {model_id}")
    print(f"Adapters:    {args.adapter_path}")
    print("\nLoading model weights into memory...")
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, adapter_path=args.adapter_path)
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
