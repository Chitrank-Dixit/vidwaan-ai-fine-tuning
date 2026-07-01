#!/usr/bin/env python3
import os
import gc
import sys
import argparse
from mlx_lm import load, stream_generate

# Ensure cache bindings remain local
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def format_prompt(tokenizer, user_message):
    """
    Wrap user prompt into the correct Llama-3 / Instruct chat template.
    """
    messages = [{"role": "user", "content": user_message}]
    try:
        # Resolves correct tokenizer format
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        # Fallback if model has no chat template
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

def run_comparative_test(model_id, adapter_path, test_prompt, max_tokens=256):
    """
    Run comparative evaluations sequentially to prevent unified memory pressure.
    """
    print("\n" + "=" * 60)
    print("           INITIATING COMPARATIVE DIAGNOSTIC             ")
    print("=" * 60)
    print(f"Prompt: {test_prompt}\n")

    # 1. Base Model Run
    print("Loading base model weights (no adapters)...")
    try:
        model_base, tokenizer_base = load(model_id)
        formatted_prompt = format_prompt(tokenizer_base, test_prompt)
        
        print("\n--- Generating Base Model Response ---")
        base_response = ""
        for response in stream_generate(model_base, tokenizer_base, prompt=formatted_prompt, max_tokens=max_tokens):
            text_chunk = getattr(response, "text", response) if not isinstance(response, str) else response
            print(text_chunk, end="", flush=True)
            base_response += text_chunk
        print("\n--------------------------------------")
        
        # Explicit garbage collection to release unified memory
        del model_base
        del tokenizer_base
        gc.collect()
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        base_response = "Base model execution failed."

    # 2. Fine-Tuned Model Run
    print("\nLoading fine-tuned model (base weights + LoRA adapters)...")
    try:
        model_tuned, tokenizer_tuned = load(model_id, adapter_path=adapter_path)
        formatted_prompt = format_prompt(tokenizer_tuned, test_prompt)
        
        print("\n--- Generating Fine-Tuned Adapter Response ---")
        tuned_response = ""
        for response in stream_generate(model_tuned, tokenizer_tuned, prompt=formatted_prompt, max_tokens=max_tokens):
            text_chunk = getattr(response, "text", response) if not isinstance(response, str) else response
            print(text_chunk, end="", flush=True)
            tuned_response += text_chunk
        print("\n----------------------------------------------")
        
        del model_tuned
        del tokenizer_tuned
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
            for response in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens):
                text_chunk = getattr(response, "text", response) if not isinstance(response, str) else response
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
        local_path = "./models/Meta-Llama-3-8B-Instruct-4bit"
        if os.path.exists(local_path):
            model_id = local_path
        else:
            model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

    if args.compare:
        run_comparative_test(model_id, args.adapter_path, args.prompt, max_tokens=args.max_tokens)
        sys.exit(0)

    # Standard Interactive Mode
    print(f"Loading Base model:    {model_id}")
    print(f"Loading LoRA Adapters: {args.adapter_path}")
    print("Loading model weights into unified memory space...")
    try:
        model, tokenizer = load(model_id, adapter_path=args.adapter_path)
        print("✔ Model and adapters successfully initialized.")
    except Exception as e:
        print(f"❌ Error loading model/adapters: {e}")
        sys.exit(1)

    interactive_loop(model, tokenizer, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
