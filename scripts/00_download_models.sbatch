#!/bin/bash
#SBATCH --job-name=r2p_download
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# === SETUP ===
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Load modules
module load cuda/12.4
module load python/3.12

# Activate venv
source ~/venvs/r2p-gen/bin/activate

# Go to project dir
cd ~/R2P-GEN

# Create logs dir
mkdir -p logs

# === RUN ===
python scripts/download_models.py

echo "Job finished at $(date)"