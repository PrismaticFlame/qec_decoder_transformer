# FROM python:3.10-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     git \
#     build-essential \
#     tmux \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade pip

# # Copy requirements and install Python packages
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# CMD ["bash"]

# CUDA runtime only (no heavy dev toolkit) on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python and basic build tools (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

# Copy only requirements first so this layer is cached
COPY requirements_transformer.txt /tmp/requirements.txt

# Install CUDA-enabled PyTorch first
# (cu121 wheels; if this ever changes, check pytorch.org/get-started/locally)
RUN pip install --upgrade pip && \
    pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch

# Install the rest of your Python deps
RUN pip install -r /tmp/requirements.txt

# Default workdir for your mounted source
WORKDIR /workspace

# Default command: drop into a shell for dev
CMD ["bash"]
