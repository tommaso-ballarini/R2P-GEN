#!/bin/bash
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32      # Più CPU per alimentare 4 GPU velocemente
#SBATCH --gres=gpu:4            # PRENOTIAMO 4 GPU!
#SBATCH --job-name=R2P_Flux_Gen
#SBATCH --output=logs/flux_gen/%j.out
#SBATCH --error=logs/flux_gen/%j.err

# 1. Caricamento Moduli HPC (Leonardo)
module purge
module load profile/deeplrn
module load cuda/12.1 
module load cudnn

# 2. Setup Ambiente
# Sostituisci questo percorso con il path reale del tuo progetto
cd /percorso/del/tuo/progetto/R2P-GEN 
source $HOME/miniconda3/bin/activate FM_env

# 3. Variabili d'Ambiente 
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Percorsi
DATABASE_JSON="database/database_perva_train.json"
OUTPUT_DIR="output/generazioni_flux"
NUM_GPUS=4

echo "=========================================================="
echo "🚀 INIZIO GENERAZIONE PARALLELA SU $NUM_GPUS GPU"
echo "=========================================================="

# 4. Esecuzione Parallela dinamica
for ((i=0; i<$NUM_GPUS; i++)); do
    echo "Lancio Worker $i su GPU Fisica $i..."
    # Assegniamo una singola GPU fisica a ogni iterazione
    # Chiamiamo l'orchestratore flux_loop.py
    CUDA_VISIBLE_DEVICES=$i python flux_loop.py \
        --stage generate_only \
        --database $DATABASE_JSON \
        --output $OUTPUT_DIR \
        --num-shards $NUM_GPUS \
        --shard-index $i &
done

# Sincronizzazione: aspetta che tutti i processi in background finiscano
wait

echo "=========================================================="
echo "🎉 Generazione parallela completata su tutte le GPU!"
echo "=========================================================="