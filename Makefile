.PHONY: init sync build up down logs shell split-data validate-data check-readiness train evaluate evaluate-mock test-compare fuse-model clean help test

# Default target: display available commands
help:
	@echo "========================================================================"
	@echo "          VidwaanAI Scripture Fine-Tuning Pipeline (Unsloth QLoRA)      "
	@echo "========================================================================"
	@echo "Container Orchestration (Podman):"
	@echo "  make build             - Build Podman container image via compose.yaml"
	@echo "  make up                - Spin up the container stack in detached mode"
	@echo "  make down              - Bring down the container stack"
	@echo "  make logs              - View container execution logs"
	@echo "  make shell             - Attach an interactive bash shell to the container"
	@echo ""
	@echo "Data Engineering:"
	@echo "  make split-data        - Convert Neo4j dump to leak-free train/valid/test splits"
	@echo "  make validate-data     - Audit dataset schema and structure via Pydantic v2"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make check-readiness   - Inspect CUDA GPU, 16GB VRAM, and dataset integrity"
	@echo "  make train             - Execute Unsloth QLoRA 4-bit SFT training loop"
	@echo "  make evaluate          - Run 6-category scripture benchmark evaluation"
	@echo "  make evaluate-mock     - Run 6-category benchmark evaluation in mock/test mode"
	@echo "  make test-compare      - Run side-by-side comparison of base model vs adapter"
	@echo "  make fuse-model        - Merge adapters into standalone weights (16bit/4bit/GGUF)"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test              - Run Pytest test suite"
	@echo "  make lint              - Run Ruff linter and formatter checks"
	@echo "  make clean             - Reset temporary checkpoints, locks, and logs"
	@echo "========================================================================"

# Sync host dependencies
sync:
	uv sync

# Build Podman containers
build:
	podman compose build

# Start Podman containers
up:
	podman compose up -d

# Stop Podman containers
down:
	podman compose down

# Tail container logs
logs:
	podman compose logs -f

# Attach bash shell to running container
shell:
	podman compose exec -it app bash

# Data splitting
split-data:
	uv run python scripts/convert_neo4j_json_to_jsonl.py

# Data validation
validate-data:
	uv run python validate_local_data.py

# Pre-flight readiness
check-readiness:
	uv run python check_readiness.py

# Fine-tuning training
train:
	uv run python train.py --config config.yaml

# Benchmark evaluation
evaluate:
	uv run python evaluate.py

# Headless mock benchmark evaluation
evaluate-mock:
	uv run python evaluate.py --mock

# Side-by-side inference test
test-compare:
	uv run python test_adapter.py --compare

# Merge & Export
fuse-model:
	uv run python export_model.py --config config.yaml --format 16bit

# Testing
test:
	uv run pytest tests/ -v

# Linting
lint:
	uv run ruff check .
	uv run ruff format --check .

# Clean temporary states
clean:
	rm -rf tmp_checkpoints .pytest_cache
