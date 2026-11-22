# Multi-stage Containerfile for VidwaanAI Unsloth QLoRA Training
# Base: NVIDIA CUDA 12.4.1 Runtime on Ubuntu 22.04

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/home/appuser/.venv/bin:$PATH"

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Astral uv natively from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create application user and set up permissions
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/.cache/huggingface /app && \
    chown -R appuser:appuser /home/appuser /app

WORKDIR /app
USER appuser

# Copy dependency specifications
COPY --chown=appuser:appuser pyproject.toml uv.lock* /app/

# Synchronize virtual environment inside container
RUN uv venv /home/appuser/.venv --python python3.11 && \
    uv sync --frozen --no-install-project

# Copy workspace source files
COPY --chown=appuser:appuser . /app

CMD ["bash"]
