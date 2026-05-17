#!/usr/bin/env bash
# =============================================================================
# VDA QAT Environment Setup
# Hardware : 2x A100-PCIE-40GB  |  CUDA driver 13.0
# Python   : 3.10.x
# PyTorch  : 2.6.0+cu126
# AIMET    : 2.30.0 + cu126
# =============================================================================

set -e

ENV_NAME="vda_qat"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
AIMET_ONNX_WHL="https://github.com/quic/aimet/releases/download/2.30.0/aimet_onnx-2.30.0+cu126-cp310-abi3-manylinux_2_34_x86_64.whl"
AIMET_TORCH_WHL="https://github.com/quic/aimet/releases/download/2.30.0/aimet_torch-2.30.0+cu126-py310-none-any.whl"
VDA_DIR="/media/rvcse22/CSERV/vdaproj/Video-Depth-Anything"

# ── Step 1: Initialize conda shell support ───────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh

# ── Step 2: Create environment if missing ────────────────────────────────────
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10.14
fi

# ── Step 3: Activate environment ─────────────────────────────────────────────
conda activate "${ENV_NAME}"

# ── Step 4: Upgrade pip tooling ──────────────────────────────────────────────
# python -m pip install --upgrade pip setuptools wheel

# ── Step 5: Install PyTorch CUDA 12.6 stack ──────────────────────────────────
pip install \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --index-url "${TORCH_INDEX}"

# ── Step 6: Install AIMET ────────────────────────────────────────────────────
pip install \
    "${AIMET_ONNX_WHL}" \
    "${AIMET_TORCH_WHL}"

# ── Step 7: Install project requirements ─────────────────────────────────────
pip install -r requirements.txt

# ── Step 8: Clone Video-Depth-Anything repo if missing ──────────────────────
if [ ! -d "${VDA_DIR}" ]; then
    git clone https://github.com/DepthAnything/Video-Depth-Anything.git "${VDA_DIR}"
fi

# ── Step 9: Install VDA editable package ─────────────────────────────────────
# pip install -e "${VDA_DIR}" --no-deps

# ── Step 10: Verify installation ─────────────────────────────────────────────
python - <<'EOF'
import torch
import torchvision
import aimet_torch

print(f"torch           : {torch.__version__}")
print(f"torchvision     : {torchvision.__version__}")
print(f"aimet_torch     : {aimet_torch.__version__}")

print(f"cuda available  : {torch.cuda.is_available()}")
print(f"gpu count       : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}           : {torch.cuda.get_device_name(i)}")
EOF

echo ""
echo "=================================================="
echo " Environment '${ENV_NAME}' ready."
echo " Activate with:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo " Run pipeline:"
echo "     python run_qat_pipeline.py"
echo "=================================================="