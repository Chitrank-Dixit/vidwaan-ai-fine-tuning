#!/usr/bin/env python3
"""VidwaanAI Character-Streaming Inference CLI & Comparative Evaluator.

Supports fast generation using Unsloth inference kernels (when available) or
standard PEFT adapters, non-blocking token streaming via TextIteratorStreamer,
and side-by-side base model vs. adapter comparisons.
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from threading import Thread
from typing import Any, Generator, Optional, Tuple

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import unsloth first before transformers/peft to apply fast inference kernels
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def format_prompt(tokenizer, user_message: str) -> str:
    """Wrap user prompt into the correct chat template."""
    messages = [{"role": "user", "content": user_message}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


def stream_generate(
    model, tokenizer, prompt: str, max_tokens: int = 512
) -> Generator[str, None, None]:
    """Generate tokens and yield text chunks in real-time using a background thread."""
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()


def load_model_and_tokenizer(model_id: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
    """Load model and tokenizer with Unsloth inference acceleration when available."""
    # Attempt Unsloth FastLanguageModel loading
    try:
        from unsloth import FastLanguageModel

        if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"Loading Unsloth adapter from {adapter_path}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
        else:
            print(f"Loading Unsloth base model: {model_id}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=2048,
                load_in_4bit=True,
            )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        print(f"ℹ️ Falling back to Hugging Face / PEFT: {e}")

    # Standard Hugging Face / PEFT fallback
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    )
    if device == "cpu":
        torch_dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch_dtype)
    else:
        model = base_model

    return model, tokenizer


def run_comparative_test(model_id: str, adapter_path: str, test_prompt: str, max_tokens: int = 256):
    """Run sequential side-by-side evaluation between base model and fine-tuned adapter."""
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
        for text_chunk in stream_generate(
            model_base, tokenizer_base, prompt=formatted_prompt, max_tokens=max_tokens
        ):
            print(text_chunk, end="", flush=True)
            base_response += text_chunk
        print("\n--------------------------------------")

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
        for text_chunk in stream_generate(
            model_tuned, tokenizer_tuned, prompt=formatted_prompt, max_tokens=max_tokens
        ):
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

    # 3. Final Review
    print("\n" + "=" * 60)
    print("             COMPARATIVE EVALUATION SCORECARD             ")
    print("=" * 60)
    print("\n[BASE MODEL RESPONSE (UN-TUNED)]:")
    print(base_response.strip())
    print("\n" + "-" * 60)
    print("\n[FINE-TUNED ADAPTER RESPONSE (GROUNDED SCRIPTURE TONE)]:")
    print(tuned_response.strip())
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VidwaanAI Character-Streaming Inference Interface"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        help="Base model ID",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./adapters",
        help="Path to trained LoRA adapter weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to evaluate non-interactively",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run side-by-side comparative analysis",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generation tokens",
    )
    args = parser.parse_args()

    if args.compare:
        test_prompt = (
            args.prompt
            or "What does the text from Bhagavad Gita state regarding the nature of Atman?"
        )
        run_comparative_test(args.model, args.adapter, test_prompt, max_tokens=args.max_tokens)
        return

    if args.prompt:
        model, tokenizer = load_model_and_tokenizer(args.model, adapter_path=args.adapter)
        formatted_prompt = format_prompt(tokenizer, args.prompt)
        print(f"\nPrompt: {args.prompt}\nResponse: ", end="")
        for chunk in stream_generate(
            model, tokenizer, formatted_prompt, max_tokens=args.max_tokens
        ):
            print(chunk, end="", flush=True)
        print("\n")
        return

    print("Please provide --prompt or --compare to run inference.")


if __name__ == "__main__":
    main()
