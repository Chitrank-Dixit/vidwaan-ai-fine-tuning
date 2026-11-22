# VidwaanAI Scripture Fine-Tuning Pipeline (Unsloth QLoRA)

An end-to-end, highly optimized local fine-tuning, validation, and evaluation pipeline designed to adapt Large Language Models on Indian scripture commentaries (Vedas, Puranas, Bhagavad Gita, Ramayana, Mahabharata).

This pipeline utilizes **Unsloth with QLoRA (4-bit NF4 quantization)**, **Hugging Face Transformers**, **PEFT**, and **TRL SFTTrainer** with instruction-prompt loss masking, containerized via **Podman Compose** with NVIDIA CDI passthrough and optimized for a single **NVIDIA GeForce RTX 5060 Ti (16GB VRAM, CUDA 12.x/13.x)**.

---

## 🏛️ System Architecture & Responsibilities

VidwaanAI cleanly decouples fine-tuning and retrieval:
- **Fine-Tuning Responsibility**: Optimize instruction following, domain answer structure, scripture-related reasoning patterns, and multilingual (Sanskrit/Hindi/English) token comprehension. The model is trained to reject speculative queries when evidence is absent (hallucination resistance).
- **GraphRAG Retrieval Responsibility**: Retrieve authoritative scripture passages, exact Sanskrit/Hindi verses, and verified Neo4j entity relationships. The fine-tuned model does not replace GraphRAG or attempt to memorize the entire Neo4j database into model weights.

```
       Scripture PDFs / Neo4j Graph Export (data/neo4j_dump_exported.json)
                                     │
                                     ▼
                scripts/convert_neo4j_json_to_jsonl.py
               (Grouped by scripture / entity cluster)
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
    data/train.jsonl (80%)     data/valid.jsonl (10%)     data/test.jsonl (10%)
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     │
                                     ▼
                          validate_local_data.py
                        (Pydantic v2 Schema Gate)
                                     │
                                     ▼
                                  train.py
                     ┌───────────────────────────────┐
                     │ Unsloth FastLanguageModel     │
                     │ 4-bit NF4 Quantization        │
                     │ All 7 Projection Modules      │
                     │ TRL SFTTrainer (Response-Only)│
                     │ RTX 5060 Ti 16GB VRAM (bf16)  │
                     └───────────────────────────────┘
                                     │
                                     ▼
                          adapters/ (LoRA Weights)
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
            export_model.py                     evaluate.py
        (16-bit / 4-bit / GGUF)           (6-Category Scripture Benchmark
                                           Loss, Perplexity, Refusal Score)
```

---

## 💻 Hardware & Execution Environment

- **Target GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM).
- **Compute Architecture**: Blackwell / Ada / Ampere with native `bfloat16` and 4-bit tensor cores.
- **Operating Systems**: Windows 11 with WSL2 Ubuntu / Native Linux.
- **Containerization**: Podman and Podman Compose with Container Device Interface (CDI) (`nvidia.com/gpu=all`).
- **Python Package Manager**: Astral `uv` (fast, deterministic environment management via `pyproject.toml` and `uv.lock`).

---

## 📁 Workspace Map

### Core Components
- **`config.yaml`**: Single source of truth for hyperparameters (model ID, sequence length 2048, 4-bit NF4, LoRA rank 16 / alpha 16 / dropout 0.0, batch size 2, accumulation 4, learning rate 2e-4).
- **`src/config_schema.py`**: Pydantic v2 configuration schema enforcing strict type safety and parameter boundaries.
- **`src/dataset_schema.py`**: Pydantic v2 domain schemas for `ChatMessage`, `ScriptureConversation`, Neo4j graph dump parsing, and benchmark test cases.
- **`Containerfile`** / **`compose.yaml`**: Rootless Podman Compose container stack based on CUDA 12.4 runtime.
- **`pyproject.toml`**: Dependency manifests locking PyTorch, Unsloth, Transformers, PEFT, TRL, BitsAndBytes, and Ruff.

### Executables
1. **`scripts/convert_neo4j_json_to_jsonl.py`**: Extracts graph entities and verses from `data/neo4j_dump_exported.json`, clustering by scripture to guarantee **zero cross-split passage leakage** across `train.jsonl`, `valid.jsonl`, and `test.jsonl`.
2. **`validate_local_data.py`**: The Pydantic v2 validation gate. Verifies role sequences (`system`? -> `user` -> `assistant`), non-empty content, isolates corrupted samples into `data/corrupted.jsonl`, and calculates token distribution.
3. **`check_readiness.py`**: Pre-flight diagnostic inspecting CUDA availability, 16GB VRAM headroom, compute capability, bfloat16 support, and dataset file health.
4. **`train.py`**: Supervised Fine-Tuning engine utilizing Unsloth's `FastLanguageModel`, 4-bit NF4 quantization, and TRL's `SFTTrainer` with `train_on_responses_only` loss masking.
5. **`evaluate.py`**: Evaluates adapters against the 6-category scripture benchmark (`data/eval_benchmark.jsonl`), calculating validation loss, perplexity, and Category 5 hallucination refusal accuracy.
6. **`test_adapter.py`**: Non-blocking character-streaming CLI inference interface and side-by-side base model vs. fine-tuned adapter evaluator.
7. **`export_model.py`**: Merges LoRA adapters into base weights and exports in 16-bit float, 4-bit merged, or GGUF quantization formats.

---

## ⚡ Quick Start Command Reference

All pipeline steps can be executed via `make` or directly via `uv`:

### 1. Environment Synchronization
```bash
# Sync local Python environment
make sync
# or
uv sync
```

### 2. Data Engineering & Leak-Free Splitting
```bash
# Ingest Neo4j export and generate leak-free train/valid/test splits
make split-data

# Audit dataset files against Pydantic v2 schema
make validate-data
```

### 3. Pre-Flight Diagnostic Check
```bash
# Run hardware and dataset integrity check
make check-readiness
```

### 4. Fine-Tuning (Unsloth QLoRA)
```bash
# Start SFT training loop
make train

# Resume training from an existing checkpoint
uv run python train.py --config config.yaml --resume ./tmp_checkpoints/checkpoint-25
```

### 5. 6-Category Benchmark Evaluation
```bash
# Run benchmark evaluation against trained adapter
make evaluate

# Run headless evaluation in mock/test mode
make evaluate-mock
```

### 6. Streaming Inference & Comparative Diagnostic
```bash
# Run side-by-side comparison (base model vs tuned adapter)
make test-compare

# Test single prompt streaming
uv run python test_adapter.py --prompt "What does the text from Bhagavad Gita state regarding the nature of Atman?"
```

### 7. Model Fusion & Multi-Format Export
```bash
# Export 16-bit merged weights
make fuse-model

# Export directly to GGUF (Q4_K_M)
uv run python export_model.py --format gguf --quantization q4_k_m
```

### 8. Testing & Linting
```bash
# Run complete test suite (41 tests)
make test

# Run Ruff linter and formatter
make lint
```

---

## 🐳 Podman Container Execution

To run inside a fully isolated Podman container with NVIDIA GPU passthrough:

### Prerequisites
1. Podman 5.0+ installed with NVIDIA Container Device Interface (CDI):
   ```bash
   podman info | grep -A 2 discoveredDevices
   # Should list: nvidia.com/gpu=all
   ```
2. Build and run via Podman Compose:
   ```bash
   # Build container image
   podman compose build

   # Run readiness check inside container
   podman compose run --rm app uv run python check_readiness.py

   # Execute fine-tuning inside container
   podman compose run --rm app uv run python train.py --config config.yaml
   ```

---

## 📊 Evaluation Categories (`data/eval_benchmark.jsonl`)

The evaluation suite tests 6 distinct scripture capabilities:
1. **Direct Factual Scripture Questions**: Direct queries testing knowledge of canonical verses.
2. **Questions Requiring Passage Context**: Queries evaluating comprehension of specific episodic scripture contexts.
3. **Questions Requiring Entity Relationships**: Queries evaluating relationships between sages, deities, kings, and locations.
4. **Theological Interpretations & Perspectives**: Queries comparing theological schools (Advaita vs. Vishishtadvaita).
5. **Insufficient Evidence & Hallucination Rejection**: Queries asking about modern unscriptural concepts where the model **must refuse** rather than hallucinate.
6. **Multilingual Sanskrit / Hindi Input Comprehension**: Queries in Devanagari Hindi and Sanskrit evaluating cross-lingual token alignment.

Evaluation results are saved as structured reports:
- `reports/evaluation_report.json`
- `reports/evaluation_report.md`

---

## 🛡️ Memory & Safety Guardrails (16GB RTX 5060 Ti)

1. **4-Bit NF4 Quantization**: Base weights loaded in 4-bit with double quantization, reducing base model memory to ~1.2 GB for a 1.5B parameter model.
2. **Gradient Checkpointing (`grad_checkpoint: true`)**: Activation tensors recomputed during backpropagation, reducing activation memory overhead by ~60%.
3. **Gradient Accumulation (`batch_size: 2`, `grad_accumulation_steps: 4`)**: Simulates an effective batch size of 8 while keeping peak memory under 6GB VRAM.
4. **Response-Only Masking (`train_on_responses_only`)**: User prompts are masked with `-100` label IDs so loss is only calculated on assistant answers.
