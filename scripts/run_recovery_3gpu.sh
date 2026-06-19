#!/bin/bash
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3       # <--- aggiornato da 2 a 3
#SBATCH --mem=128GB
#SBATCH --job-name=R2P_Refine
#SBATCH --output=logs/pipeline_40/refine_%j.out
#SBATCH --error=logs/pipeline_40/refine_%j.err

# ==========================================
# SETUP HPC E CONDA
# ==========================================
module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

# Esporta il path per far trovare config.py
export PYTHONPATH=$PWD:$PYTHONPATH

# ==========================================
# VARIABILI AMBIENTE (Offline & Paths)
# ==========================================
export HF_HOME="/leonardo_work/IscrC_MUSE/tballari/models_cache/.hf_cache"
export R2P_MODELS_BASE="/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DATABASE_JSON="database/database.json"
OUTPUT_DIR="/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_40"
FLUX_PORT=8766
FLUX_TEXT_PORT=8767

# Funzione per pulire i background process se il job viene killato
cleanup() {
    if [ -n "$FLUX_PID" ]; then
        echo "🛑 Chiudo server FLUX (PID=$FLUX_PID)"
        kill $FLUX_PID 2>/dev/null
        wait $FLUX_PID 2>/dev/null
    fi
    if [ -n "$FLUX_TEXT_PID" ]; then
        echo "🛑 Chiudo server FluxText (PID=$FLUX_TEXT_PID)"
        kill $FLUX_TEXT_PID 2>/dev/null
        wait $FLUX_TEXT_PID 2>/dev/null
    fi
}
trap cleanup EXIT

# ==========================================
# 1. AVVIO SERVER FLUX (Su GPU 1)
# ==========================================
echo "🚀 Avvio server FLUX in background sulla porta $FLUX_PORT (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python flux_server.py --port $FLUX_PORT > logs/pipeline_50/flux_server_${SLURM_JOB_ID}.log 2>&1 &
FLUX_PID=$!

echo "⏳ Attendo che FLUX sia pronto..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$FLUX_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ FLUX Server pronto!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ❌ FLUX Server non risponde dopo 10 minuti. Abort."
        exit 1
    fi
    sleep 10
done

# ==========================================
# 2. AVVIO FLUXTEXT SERVER (Su GPU 2)
# ==========================================
echo "✍️  Avvio FluxText server (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python flux_text_server.py --port $FLUX_TEXT_PORT \
    > logs/pipeline_50/flux_text_server_${SLURM_JOB_ID}.log 2>&1 &
FLUX_TEXT_PID=$!

# Health check FluxText (più lungo, carica 58GB)
echo "⏳ Attendo FluxText (può richiedere fino a 15 minuti)..."
for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:$FLUX_TEXT_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ FluxText Server pronto!"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "   ❌ FluxText non risponde. Abort."
        exit 1
    fi
    sleep 10
done

# ==========================================
# 3. AVVIO REFINE LOOP (Su GPU 0 con Qwen3)
# ==========================================
echo "⚖️ Avvio Pipeline Refine (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage refine \
    --database $DATABASE_JSON \
    --output $OUTPUT_DIR

# ==========================================
# [DEPRECATO] TEXT FIX STAGE — da reimplementare
# In flux_loop.py --stage text_fix era un placeholder vuoto.
# Se serve, reimplementare con la logica in archive/legacy/pipeline/text_fix.py
# ==========================================
echo "🏁 Job Completato!"