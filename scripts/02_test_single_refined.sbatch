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

# Pulisce i moduli precedenti per evitare conflitti
module purge

# --- CORREZIONE MODULI ---
# Usa SEMPRE questi nomi specifici, altrimenti lo script non parte
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# Attiva l'ambiente virtuale
source ~/venvs/r2p-gen/bin/activate

export R2P_CLUSTER_MODE=true
export CUDA_VISIBLE_DEVICES=0

cd ~/R2P-GEN
mkdir -p logs output/refined

# === FIND TEST IMAGE ===
TEST_IMAGE=$(find ~/data/perva-data/test/bag/alx -type f -name "*.jpg" -o -name "*.png" | head -1)
echo "Test image: $TEST_IMAGE"

# === RUN WITH REFINEMENT + FINAL JUDGE ===
# Aggiunto -u per vedere l'output in tempo reale nel file di log
python -u full_loop.py \
    --mode single \
    --image "$TEST_IMAGE" \
    --output output/refined

echo "Job finished at $(date)"