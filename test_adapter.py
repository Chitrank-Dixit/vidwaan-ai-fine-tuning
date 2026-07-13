#!/usr/bin/env python3
import os
import gc
import sys

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

# Ensure cache bindings remain local
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def format_prompt(tokenizer, user_message):
    """
    Wrap user prompt into the correct chat template.
    """
    messages = [{"role": "user", "content": user_message}]
    try:
        # Resolves correct tokenizer format
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback if model has no chat template
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

def stream_generate(model, tokenizer, prompt, max_tokens=512):
    """
    Generate tokens and yield text chunks in real-time.
    """
    # Tokenize prompt and send to device
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Setup the streaming queue
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Run the generation inside a separate thread to let iterator yield tokens concurrently
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text
        
    thread.join()

def load_model_and_tokenizer(model_id, adapter_path=None):
    """
    Load base model or adapted model on available device (GPU or CPU).
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
        # Verify adapter is valid before loading
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

def run_comparative_test(model_id, adapter_path, test_prompt, max_tokens=256):
    """
    Run comparative evaluations sequentially to prevent memory pressure.
    """
    print("\n" + "=" * 60)
    print("           INITIATING COMPARATIVE DIAGNOSTIC             ")
    print("=" * 60)
    print(f"Prompt: {test_prompt}\n")

    # 1. Base Model Run
    print("Loading base model weights (no adapters)...")
    try:
        model_base, tokenizer_base = load_model_and_tokenizer(model_id)
        formatted_prompt = format_prompt(tokenizer_base, test_prompt)
        
        print("\n--- Generating Base Model Response ---")
        base_response = ""
        for text_chunk in stream_generate(model_base, tokenizer_base, prompt=formatted_prompt, max_tokens=max_tokens):
            print(text_chunk, end="", flush=True)
            base_response += text_chunk
        print("\n--------------------------------------")
        
        # Explicit garbage collection to release memory
        del model_base
        del tokenizer_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        base_response = "Base model execution failed."

    # 2. Fine-Tuned Model Run
    print("\nLoading fine-tuned model (base weights + LoRA adapters)...")
    try:
        model_tuned, tokenizer_tuned = load_model_and_tokenizer(model_id, adapter_path=adapter_path)
        formatted_prompt = format_prompt(tokenizer_tuned, test_prompt)
        
        print("\n--- Generating Fine-Tuned Adapter Response ---")
        tuned_response = ""
        for text_chunk in stream_generate(model_tuned, tokenizer_tuned, prompt=formatted_prompt, max_tokens=max_tokens):
            print(text_chunk, end="", flush=True)
            tuned_response += text_chunk
        print("\n----------------------------------------------")
        
        del model_tuned
        del tokenizer_tuned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        tuned_response = "Fine-tuned model execution failed."

    # 3. Final Side-by-Side Review
    print("\n" + "=" * 60)
    print("             COMPARATIVE EVALUATION SCORECARD             ")
    print("=" * 60)
    print("\n[BASE MODEL RESPONSE (UN-TUNED)]:")
    print(base_response.strip())
    print("\n" + "-" * 60)
    print("\n[FINE-TUNED ADAPTER RESPONSE (PHILOSOPHICAL TONE)]:")
    print(tuned_response.strip())
    print("=" * 60 + "\n")

def interactive_loop(model, tokenizer, max_tokens=512):
    """
    Run stream text generation character-by-character inside an interactive prompt loop.
    """
    print("\nChat session initialized. Type '/exit' or '/quit' to terminate.")
    print("Type '/compare <prompt>' to run a comparative evaluation of base vs tuned models.\n")
    
    while True:
        try:
            user_input = input("User 👤: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "/quit"]:
                print("Closing session. Goodbye!")
                break
                
            print("Assistant 🤖: ", end="", flush=True)
            formatted_prompt = format_prompt(tokenizer, user_input)
            
            # Stream response character by character
            for text_chunk in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens):
                print(text_chunk, end="", flush=True)
            print("\n")
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter Interactive Testing CLI Engine")
    parser.add_argument("--model", type=str, default=None, help="Path to base model or Hugging Face ID")
    parser.add_argument("--adapter-path", type=str, default="./adapters", help="Path to saved LoRA adapters")
    parser.add_argument("--compare", action="store_true", help="Run diagnostic comparison and exit")
    parser.add_argument("--prompt", type=str, default="Who freed Ahalya from her long curse?", help="Prompt for comparison")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
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

    if args.compare:
        run_comparative_test(model_id, args.adapter_path, args.prompt, max_tokens=args.max_tokens)
        sys.exit(0)

    # Standard Interactive Mode
    print(f"Loading Base model:    {model_id}")
    print(f"Loading LoRA Adapters: {args.adapter_path}")
    print("Loading model weights into memory...")
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, adapter_path=args.adapter_path)
        print("✔ Model and adapters successfully initialized.")
    except Exception as e:
        print(f"❌ Error loading model/adapters: {e}")
        sys.exit(1)

    interactive_loop(model, tokenizer, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
