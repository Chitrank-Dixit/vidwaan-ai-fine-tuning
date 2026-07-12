.PHONY: init sync build up down logs shell playbook-syntax run-playbook generate-data generate-data-mock test-pdf clean help

# Default target: display available commands
help:
	@echo "========================================================================"
	@echo "              Scripture Fine-Tuning Workspace CLI                       "
	@echo "========================================================================"
	@echo "Workspace Orchestration (Docker & Local):"
	@echo "  make init              - Run workspace setup script (health checks)"
	@echo "  make sync              - Synchronize host Python dependencies using uv"
	@echo "  make build             - Build/rebuild Docker Compose image"
	@echo "  make up                - Spin up the container stack in detached mode"
	@echo "  make down              - Bring down the container stack"
	@echo "  make logs              - View container execution logs"
	@echo "  make shell             - Attach an interactive bash shell to the container"
	@echo ""
	@echo "Dataset Generation Pipeline:"
	@echo "  make test-pdf          - Generate a scripture-like test PDF file locally"
	@echo "  make generate-data     - Parse PDFs and generate MLX conversational dataset (Gemini)"
	@echo "  make generate-data-mock- Run Gemini dataset generator in mock mode"
	@echo "  make generate-data-local- Parse PDFs and generate MLX conversational dataset (LM Studio)"
	@echo "  make generate-data-local-mock - Run LM Studio dataset generator in mock mode"
	@echo "  make validate-data     - Audit dataset quality using LM Studio critic"
	@echo "  make validate-data-mock- Audit dataset quality in mock mode (no connection needed)"
	@echo "  make check-readiness   - Ingest base model and run hardware pre-flight checks"
	@echo "  make train             - Execute local QLoRA fine-tuning training loop"
	@echo "  make test-interactive  - Boot interactive adapters test chat loop"
	@echo "  make test-compare      - Run comparative diagnostics (base vs tuned model)"
	@echo "  make test-rag          - Run Hybrid RAG pipeline validation step"
	@echo "  make fuse-model        - Merge adapters and convert model to GGUF format"
	@echo "  make verify-fusion     - Execute validation tests on the fused model"
	@echo "  make publish           - Upload fused model directory to Hugging Face Hub"
	@echo ""
	@echo "Workflow Automation (Ansible):"
	@echo "  make playbook-syntax   - Run syntax verification on the Ansible playbook"
	@echo "  make run-playbook      - Execute the Ansible orchestration workflow"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Reset locks, datasets, mock PDFs, and temporary states"
	@echo "========================================================================"

# Run workspace setup script
init:
	@chmod +x init_workspace.sh
	@./init_workspace.sh

# Sync host dependencies
sync:
	uv sync

# Build Docker containers
build:
	docker compose build

# Start Docker containers
up:
	docker compose up -d

# Stop Docker containers
down:
	docker compose down

# Tail container logs
logs:
	docker compose logs -f

# Attach bash shell to running container
shell:
	docker compose exec -it training bash

# Verify Ansible playbook syntax
playbook-syntax:
	uv run ansible-playbook -i ansible/inventory.ini ansible/playbook.yml --syntax-check

# Run Ansible workflow orchestration
run-playbook:
	uv run ansible-playbook -i ansible/inventory.ini ansible/playbook.yml

# Generate a scripture-like test PDF file locally
test-pdf:
	uv run python scripts/create_test_pdf.py

# Parse PDFs and generate MLX conversational dataset (Gemini)
generate-data:
	uv run python generate_mlx_data.py

# Run Gemini dataset generator in mock mode
generate-data-mock:
	uv run python generate_mlx_data.py --mock-llm

# Parse PDFs and generate MLX conversational dataset (LM Studio)
generate-data-local:
	uv run python generate_local_dataset.py

# Run LM Studio dataset generator in mock mode
generate-data-local-mock:
	uv run python generate_local_dataset.py --mock-fallback

# Run dataset quality validator (LM Studio)
validate-data:
	uv run python validate_local_data.py

# Run dataset quality validator in mock mode (no connection needed)
validate-data-mock:
	uv run python validate_local_data.py --mock-judge

# Run pre-flight readiness checks (ingests model and verifies constraints)
check-readiness:
	uv run python check_readiness.py

# Run local QLoRA fine-tuning training loop
train:
	./run_training.sh

# Run interactive adapter test loop (streams characters dynamically)
test-interactive:
	uv run python test_adapter.py

# Run comparative evaluation of base model vs fine-tuned adapter model
test-compare:
	uv run python test_adapter.py --compare --prompt "Who freed Ahalya from her long curse?"

# Execute Hybrid RAG integration pipeline test run
test-rag:
	uv run python hybrid_rag_engine.py

# Run model weights fusion and GGUF conversion
fuse-model:
	./export_model.sh

# Execute validation checks on the fused model artifacts
verify-fusion:
	uv run python verify_fused_output.py

# Publish standalone fused model to Hugging Face Hub
publish:
	uv run python publish_model.py

# Clean up temporary state and build artifacts
clean:
	@echo "Resetting workspace state..."
	rm -f data/.data_prepared.lock data/train.jsonl data/valid.jsonl data/test_scripture.pdf data/corrupted.jsonl data/training.log
	rm -rf adapters/*
	rm -rf fused_model/*
	@echo "Workspace state cleared."

