# SPEC.md: Upgrade VidwaanAI Fine-Tuning Pipeline to Unsloth QLoRA

## 1. System Overview & Objectives
VidwaanAI is a scripture-focused AI assistant for answering queries on Indian scriptures (Vedas, Puranas, Bhagavad Gita, Ramayana, Mahabharata).
This specification governs upgrading the fine-tuning pipeline to **Unsloth with QLoRA**, replacing the legacy Hugging Face Trainer / MLX pipeline while preserving dataset preparation, validation, evaluation, adapter export, and inference capabilities.

### Separation of Responsibilities
- **Fine-Tuning Role**: Optimize instruction following, domain answer structure, scripture-related reasoning patterns, and multilingual (Sanskrit/Hindi/English) token comprehension.
- **GraphRAG / Retrieval Role**: Retrieve authoritative scripture passages, exact verses, and Neo4j graph evidence.
- The model must never hallucinate verses or fabricate missing translations.

---

## 2. Technical Stack & Hardware Constraints
- **Hardware Target**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM), CUDA 12.x / 13.x.
- **Operating Environment**: Windows host with WSL2 Ubuntu / Podman container.
- **Frameworks**:
  - `unsloth` (`FastLanguageModel`, `is_bfloat16_supported`)
  - `torch` (CUDA-enabled)
  - `transformers`, `peft`, `trl` (`SFTTrainer`)
  - `bitsandbytes` (4-bit NF4 quantization)
  - `pydantic` (v2 for strict configuration & dataset validation)
  - `pytest` (headless test execution)
  - `ruff` (formatting and linting)
- **Containerization**: Podman Compose with `compose.yaml` and `Containerfile`, using volume mounts (`:Z` where appropriate) and NVIDIA CDI device passthrough (`--device nvidia.com/gpu=all`). Local Python venv fallback supported and documented.

---

## 3. Architecture & Data Flow
```text
Scripture PDFs / Neo4j Dump (data/neo4j_dump_exported.json)
      │
      ▼
Dataset Extraction & Conversion (scripts/convert_neo4j_json_to_jsonl.py)
      │ (Scripture-grouped, leakage-resistant split: Train / Valid / Test)
      ▼
Dataset Validation Gate (validate_local_data.py)
      │ (Pydantic v2 schema, role validation, deduplication, outlier filter)
      ▼
Unsloth QLoRA Training Engine (train.py)
      │ (FastLanguageModel, 4-bit QLoRA, SFTTrainer, response-only masking)
      ▼
Checkpoints & Adapters (adapters/adapter_model.safetensors)
      │
      ├───► Adapter Evaluation & Benchmark (test_adapter.py / evaluate.py)
      │       (6-category scripture benchmark, metrics, structured JSON/MD report)
      │
      └───► Adapter Export & Merge (export_model.py)
              (Standalone 16-bit / 4-bit weights, optional GGUF conversion)
```

---

## 4. Configuration Schema (Pydantic v2)
Every configuration option must be parsed and strictly validated using Pydantic v2:
- `model`: Base model ID (e.g. `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` or `Qwen/Qwen2.5-1.5B-Instruct`)
- `max_seq_length`: Maximum token sequence length (e.g. 2048)
- `load_in_4bit`: Boolean flag for 4-bit quantization (default `true`)
- `lora_parameters`:
  - `rank` (r): LoRA rank dimension (default 16)
  - `scale` (alpha): LoRA scaling factor (default 16)
  - `dropout`: LoRA dropout rate (default 0 for Unsloth optimization)
  - `target_modules`: List of projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- `training`:
  - `batch_size`: Per-device train batch size (default 2 for 16GB VRAM)
  - `grad_accumulation_steps`: Gradient accumulation steps (default 4)
  - `learning_rate`: Float learning rate (default 2e-4)
  - `max_steps` / `num_train_epochs`: Training duration controls
  - `warmup_steps`: Warmup step count (default 5)
  - `weight_decay`: Regularization parameter (default 0.01)
  - `grad_checkpoint`: Boolean flag (`true` / `"unsloth"`)
  - `random_seed`: Seed for reproducibility (default 3407)
  - `output_dir`: Output checkpoints directory (`./tmp_checkpoints`)
  - `adapter_path`: Destination for final LoRA adapters (`./adapters`)
  - `logging_steps`: Frequency of logging (default 5)
  - `save_steps`: Frequency of checkpointing
  - `eval_steps`: Frequency of validation loss evaluation

---

## 5. Dataset Validation & Leakage-Free Splitting
1. **Neo4j Conversion**:
   - Ingest `data/neo4j_dump_exported.json` without modifying source.
   - Group by scripture entity/source document to prevent verse leakage between train and valid/test splits.
   - Generate `train.jsonl`, `valid.jsonl`, and `test.jsonl`.
2. **Pydantic Validation**:
   - Validate `messages` list structure with roles `system`, `user`, `assistant`.
   - Disallow empty prompts or empty assistant responses.
   - Detect and flag duplicate or near-duplicate messages.
   - Preserve verse metadata and scripture provenance.

---

## 6. Training Pipeline (Unsloth + QLoRA)
1. Initialize `FastLanguageModel.from_pretrained` with 4-bit quantization and specified `max_seq_length`.
2. Apply `FastLanguageModel.get_peft_model` with target modules and gradient checkpointing.
3. Configure chat template with `get_chat_template` or model's native template.
4. Use `SFTTrainer` with `train_on_responses_only` to mask user instructions during loss computation.
5. Support resuming from checkpoints (`trainer.train(resume_from_checkpoint=...)`).
6. Save LoRA adapters (`model.save_pretrained`) and tokenizer (`tokenizer.save_pretrained`).
7. Log metrics: training loss, validation loss, peak VRAM, step execution time.

---

## 7. Evaluation & Inference Engine
1. **Scripture Evaluation Dataset (`data/eval_benchmark.jsonl`)**:
   - Category 1: Direct factual scripture questions.
   - Category 2: Questions requiring passage context.
   - Category 3: Questions requiring entity relationships.
   - Category 4: Questions involving theological interpretations.
   - Category 5: Questions where evidence is insufficient (hallucination rejection).
   - Category 6: Multilingual Sanskrit / Hindi input comprehension.
2. **Evaluation Suite**:
   - Compute validation loss, perplexity, reference accuracy, refusal accuracy on insufficient evidence.
   - Export structured JSON and Markdown evaluation reports.
3. **Inference**:
   - Verify `test_adapter.py` loads base model + adapter smoothly.
   - Support streaming generation via `TextIteratorStreamer`.

---

## 8. Verification & Containerization Matrix
- Headless Pytest suite verifying:
  - Configuration loading & Pydantic v2 schemas.
  - Dataset parsing & format verification.
  - Leakage prevention in dataset splitting.
  - Model loading parameters and LoRA config.
  - Mocked SFTTrainer lifecycle and callback logging.
  - Evaluation report generator.
- Podman Compose (`compose.yaml`) & Containerfile setup for CUDA execution.
