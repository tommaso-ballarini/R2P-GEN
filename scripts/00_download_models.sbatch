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
#SBATCH --gres=gpu:1

# === SETUP ===
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Pulisce tutto per sicurezza
module purge

# CARICA I MODULI CORRETTI (Quelli che abbiamo usato per installare)
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# Activate venv
source ~/venvs/r2p-gen/bin/activate

# Go to project dir
cd ~/R2P-GEN

# Create logs dir
mkdir -p logs

# === RUN ===
# Aggiungo -u a python per vedere i log in tempo reale
python -u scripts/download_models.py

echo "Job finished at $(date)"