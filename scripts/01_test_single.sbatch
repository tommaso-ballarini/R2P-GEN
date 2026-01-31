#!/bin/bash
#SBATCH --job-name=r2p_single
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
#SBATCH --nodelist=edu02

# === SETUP ===
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

# Load modules
module load cuda/12.4
module load python/3.12

# Activate venv
source ~/venvs/r2p-gen/bin/activate

# Set environment
export R2P_CLUSTER_MODE=true
export CUDA_VISIBLE_DEVICES=0

# Go to project dir
cd ~/R2P-GEN

# Create output dirs
mkdir -p logs output/single

# === FIND TEST IMAGE ===
TEST_IMAGE=$(find ~/data/perva-data/test/bag/alx -type f -name "*.jpg" -o -name "*.png" | head -1)
echo "Test image: $TEST_IMAGE"

# === RUN SINGLE IMAGE TEST ===
python full_loop.py \
    --mode single \
    --image "$TEST_IMAGE" \
    --output output/single \
    --no-refinement

echo "Job finished at $(date)"