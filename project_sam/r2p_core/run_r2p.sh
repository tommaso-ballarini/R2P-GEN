#!/bin/bash
#SBATCH --job-name=r2p_test
#SBATCH --output=out_r2p_%j.txt
#SBATCH --error=err_r2p_%j.txt
#SBATCH --partition=edu-long         # Usiamo la stessa del tuo amico
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # Mettiamo 32G per sicurezza con SDXL (16G è al limite)
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00              # 40 minuti bastano per il primo test

# Nota: Ho rimosso #SBATCH --nodelist=edu01 per non farti aspettare in coda se quel nodo è pieno.

# --- 1. SETUP AMBIENTE (VENV) ---
echo "🚀 Job avviato su nodo: $HOSTNAME"

# Pulisci moduli precedenti
module purge

# CARICA LO STESSO PYTHON CHE HAI USATO PER CREARE IL VENV
# Copio quello del tuo amico, presumendo usiate lo stesso cluster/config
module load Python/3.11.3-GCCcore-12.3.0

# ATTIVA IL TUO VENV
# ATTENZIONE: Cambia 'venv' col nome della tua cartella se diverso (es. .venv o r2p_env)
source test_venv/bin/activate

# --- 2. VARIABILI AMBIENTE ---
# Fondamentale: Sposta la cache dei modelli nello spazio scratch
export HF_HOME="/scratch/user/$USER/.cache/huggingface"
export TORCH_HOME="/scratch/user/$USER/.cache/torch"
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# Abilitiamo ottimizzazioni CUDA (come il tuo amico)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NOTA SU FLASH ATTENTION:
# Il tuo amico aveva 'export FLASH_ATTN="False"'.
# Noi invece lo VOGLIAMO usare se possibile per MiniCPM.
# Quindi NON mettiamo quella riga.
export FLASH_ATTN="False"

# --- 3. DIAGNOSTICA ---
echo "=== GPU INFO ==="
nvidia-smi
echo "================"
echo "Python in uso: $(which python)"

# --- 4. ESECUZIONE ---
echo "Avvio Pipeline R2P..."

python full_loop.py

echo "✅ Finito!"