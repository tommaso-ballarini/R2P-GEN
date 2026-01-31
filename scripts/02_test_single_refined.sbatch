#!/bin/bash
#SBATCH --job-name=r2p_refined
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/refined_%j.out
#SBATCH --error=logs/refined_%j.err
#SBATCH --nodelist=edu02

# === SETUP ===
echo "Job started at $(date)"
nvidia-smi

module load cuda/12.4
module load python/3.12
source ~/venvs/r2p-gen/bin/activate

export R2P_CLUSTER_MODE=true
export CUDA_VISIBLE_DEVICES=0

cd ~/R2P-GEN
mkdir -p logs output/refined

# === FIND TEST IMAGE ===
TEST_IMAGE=$(find ~/data/perva-data/test/bag/alx -type f -name "*.jpg" -o -name "*.png" | head -1)
echo "Test image: $TEST_IMAGE"

# === RUN WITH REFINEMENT + FINAL JUDGE ===
python full_loop.py \
    --mode single \
    --image "$TEST_IMAGE" \
    --output output/refined

echo "Job finished at $(date)"