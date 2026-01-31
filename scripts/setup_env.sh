#!/bin/bash
# Run this interactively on the cluster login node

echo "=== R2P-GEN Environment Setup ==="

# Load modules
module load cuda/12.4
module load python/3.12

# Create venv
echo "Creating virtual environment..."
python -m venv ~/venvs/r2p-gen
source ~/venvs/r2p-gen/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (CUDA 12.4)
echo "Installing PyTorch..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install CLIP from git
echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Install flash-attn (Linux wheel)
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

# Install main requirements (but skip conflicting ones)
echo "Installing requirements..."
cd ~/R2P-GEN
pip install -r requirements.txt --ignore-installed torch torchvision torchaudio

# Upgrade transformers for Qwen2-VL
echo "Upgrading transformers for Qwen2-VL..."
pip install transformers>=4.45.0

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source ~/venvs/r2p-gen/bin/activate"