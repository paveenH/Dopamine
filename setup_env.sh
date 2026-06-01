#!/bin/bash
set -euo pipefail

# ==================== Dopamine Environment Setup ====================
# Creates or updates the conda environment used by this repository.
#
# Usage: bash setup_env.sh
#
# Note: Run this script in an interactive shell (not via sbatch)

ENV_NAME="dopamine"
PYTHON_VERSION="3.10"

echo "=================================================="
echo "Setting up Dopamine environment"
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "=================================================="

echo "[1/5] Loading conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[2/5] Creating or reusing conda environment..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists; reusing it."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# Install PyTorch with CUDA support
echo "[3/5] Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install required packages
echo "[4/5] Installing Python packages..."
python -m pip install --upgrade pip

# Core packages
python -m pip install accelerate datasets

# For Mistral3 models (requires latest transformers from main)
python -m pip install git+https://github.com/huggingface/transformers
python -m pip install "mistral-common>=1.8.6"

# Data processing & analysis
python -m pip install numpy pandas scikit-learn scipy

# Symbolic math: MATH answer equivalence checking (utils.is_correct_math).
# antlr4-python3-runtime is required for sympy.parsing.latex.parse_latex.
python -m pip install sympy "antlr4-python3-runtime==4.11"

# Visualization
python -m pip install matplotlib seaborn

# Hugging Face utilities + Phase 1b hidden-state pipeline
python -m pip install huggingface_hub tqdm h5py safetensors

echo "[5/5] Verifying installation..."
python - <<'PY'
import torch
import h5py
import safetensors
import transformers

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Transformers:", transformers.__version__)
PY

# Optional: for notebook support
# python -m pip install jupyter ipykernel

echo "=================================================="
echo "Environment setup complete!"
echo ""
echo "To activate: conda activate ${ENV_NAME}"
echo "=================================================="
