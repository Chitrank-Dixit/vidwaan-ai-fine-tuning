#!/bin/bash
set -euo pipefail

# ANSI color codes for rich styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}====================================================${NC}"
echo -e "${BLUE}${BOLD}   Scripture Fine-Tuning Workspace Initializer       ${NC}"
echo -e "${BLUE}${BOLD}====================================================${NC}"

# 1. Verify Docker is running
echo -e "\n${CYAN}[1/4] Verifying Docker environment...${NC}"
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running. Please start Docker Desktop on your Mac.${NC}"
    exit 1
fi
echo -e "${GREEN}✔ Docker is running.${NC}"

if ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}Error: docker compose is not installed or configured correctly.${NC}"
    exit 1
fi
echo -e "${GREEN}✔ Docker Compose is available.${NC}"

# 2. Create required directories
echo -e "\n${CYAN}[2/4] Ensuring workspace directories exist...${NC}"
mkdir -p data adapters
echo -e "${GREEN}✔ Directories './data' and './adapters' verified/created.${NC}"

# 3. Boot container stack
echo -e "\n${CYAN}[3/4] Building and launching Docker containers...${NC}"
# Enable Docker BuildKit for advanced caching mounts
export DOCKER_BUILDKIT=1
docker compose up --build -d

# 4. Verify uv sync inside container
echo -e "\n${CYAN}[4/4] Verifying container initialization and dependencies...${NC}"
echo -e "${YELLOW}Waiting for container to stabilize...${NC}"
sleep 3

# Check if container is running
CONTAINER_STATUS=$(docker compose ps --format json | grep -o '"State":"running"' || true)
if [ -z "$CONTAINER_STATUS" ]; then
    echo -e "${RED}Error: Container failed to start. Check 'docker compose logs'.${NC}"
    exit 1
fi

echo -e "Verifying uv sync status..."
if docker compose exec -T training uv pip list > /dev/null 2>&1; then
    echo -e "${GREEN}✔ Dependencies synchronized successfully inside virtualenv.${NC}"
else
    echo -e "${RED}Error: uv sync validation failed inside the container.${NC}"
    exit 1
fi

# 5. Generate System Readiness Report
echo -e "\n${BLUE}${BOLD}====================================================${NC}"
echo -e "${BLUE}${BOLD}             SYSTEM READINESS REPORT                ${NC}"
echo -e "${BLUE}${BOLD}====================================================${NC}"

# Host details
echo -e "${BOLD}1. Host Machine Details:${NC}"
HOST_MEM=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
if [ "$HOST_MEM" -ne "0" ]; then
    HOST_MEM_GB=$((HOST_MEM / 1024 / 1024 / 1024))
    echo -e "  - Host Memory: ${GREEN}${HOST_MEM_GB} GB RAM${NC}"
else
    echo -e "  - Host Memory: ${YELLOW}Unknown (macOS sysctl failed)${NC}"
fi

# Container memory limit details
echo -e "\n${BOLD}2. Docker Container Limits:${NC}"
CONTAINER_MEM_LIMIT=$(docker compose exec -T training cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "")
if [ -n "$CONTAINER_MEM_LIMIT" ] && [ "$CONTAINER_MEM_LIMIT" -lt 9223372036854771712 ] 2>/dev/null; then
    CONTAINER_MEM_GB=$((CONTAINER_MEM_LIMIT / 1024 / 1024 / 1024))
    echo -e "  - Allocated Container Memory: ${GREEN}${CONTAINER_MEM_GB} GB RAM${NC}"
else
    # Try cgroup v2 format
    CONTAINER_MEM_LIMIT_V2=$(docker compose exec -T training cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "")
    if [ -n "$CONTAINER_MEM_LIMIT_V2" ] && [ "$CONTAINER_MEM_LIMIT_V2" != "max" ]; then
        CONTAINER_MEM_GB=$((CONTAINER_MEM_LIMIT_V2 / 1024 / 1024 / 1024))
        echo -e "  - Allocated Container Memory: ${GREEN}${CONTAINER_MEM_GB} GB RAM${NC}"
    else
        echo -e "  - Allocated Container Memory: ${GREEN}Unlimited / Default Host Allocation${NC}"
    fi
fi

# Volume Attachment Verification
echo -e "\n${BOLD}3. Volume Attachment Verification:${NC}"
# Write a temporary test file from within the container
TEST_FILE="data/.volume_test"
if docker compose exec -T training touch /app/$TEST_FILE 2>/dev/null; then
    if [ -f "$TEST_FILE" ]; then
        echo -e "  - Volume Bind Mounts: ${GREEN}Success (Read/Write OK)${NC}"
        rm -f "$TEST_FILE"
    else
         echo -e "  - Volume Bind Mounts: ${RED}Failed (File not written to host)${NC}"
    fi
else
    echo -e "  - Volume Bind Mounts: ${RED}Failed (Write permission denied)${NC}"
fi

# Base Model Accessibility (Hugging Face check)
echo -e "\n${BOLD}4. External Connectivity & Model Access:${NC}"
if docker compose exec -T training curl -I -s --connect-timeout 5 https://huggingface.co >/dev/null; then
    echo -e "  - Hugging Face Access: ${GREEN}Online (huggingface.co is reachable)${NC}"
else
    echo -e "  - Hugging Face Access: ${RED}Offline / Timeout (Check internet settings)${NC}"
fi

# Check if HF_TOKEN is injected
HF_TOKEN_VAL=$(docker compose exec -T training sh -c 'echo $HF_TOKEN' 2>/dev/null || echo "")
if [ -n "$HF_TOKEN_VAL" ]; then
    echo -e "  - Hugging Face Token: ${GREEN}Injected (Secure credentials loaded)${NC}"
else
    echo -e "  - Hugging Face Token: ${YELLOW}Not Set (Access restricted to public models)${NC}"
fi

echo -e "\n${BLUE}${BOLD}====================================================${NC}"
echo -e "${GREEN}${BOLD}             WORKSPACE READY FOR USE                ${NC}"
echo -e "${BLUE}${BOLD}====================================================${NC}"
echo -e "To execute code inside the container:"
echo -e "  ${CYAN}docker compose exec -it training bash${NC}"
echo -e "To run fine-tuning on the host Mac (recommended for GPU/Metal MPS):"
echo -e "  ${CYAN}uv sync${NC}"
echo -e "  ${CYAN}uv run python <fine_tuning_script.py>${NC}"
echo -e "${BLUE}${BOLD}====================================================${NC}"
