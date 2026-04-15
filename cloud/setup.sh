#!/bin/bash
# Cloud GPU setup for MoC experiments
# Tested on: RunPod RTX 4090 (CUDA 12.1, Ubuntu 22.04)

set -e

echo "=== MoC Cloud Setup ==="

# 1. Create and activate conda environment
conda create -n moc python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate moc

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install transformers>=4.50.0 datasets>=2.18.0 wandb numpy matplotlib tqdm

# 4. Log in to WandB (interactive)
echo ""
echo "Run: wandb login"
echo "Then: python run_experiment.py --phase 4 --max-samples 50000"
