# Stage 1: Build virtual environment
FROM python:3.11-slim AS builder

# Set env vars for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv natively from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency configuration (and uv.lock if available)
COPY pyproject.toml uv.lock* /app/

# Synchronize the dependencies inside the virtual environment /app/.venv.
# Docker BuildKit cache mount speeds up package installations during rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Stage 2: Final runtime image
FROM python:3.11-slim AS runner

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Create a non-root system user for security
RUN useradd -u 1000 -m appuser

WORKDIR /app

# Copy the pre-compiled virtual environment and uv binary from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /bin/uv /bin/uv

# Copy current project workspace to /app
COPY . /app

# Ensure correct file permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Open interactive bash terminal by default
CMD ["bash"]
