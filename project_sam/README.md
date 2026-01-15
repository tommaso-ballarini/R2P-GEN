# Guida Operativa Cluster - Progetto R2P-Gen

PREREQUISITI

- VPN attiva: GlobalProtect connesso


SETUP AMBIENTE (UNA SOLA VOLTA)

Collegati al cluster:

ssh tuo_user@indirizzo_cluster

clona la repo

muoviti dentro la cartella project

Crea l'ambiente dentro la cartella:

python -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt



--------------------------------------------------

SCRIPT SLURM (run_r2p.sh)

Crea il file:

nano run_r2p.sh

Copia il Contenuto COMPLETO:

#!/bin/bash
#SBATCH --job-name=r2p_gen  

#SBATCH --output=out_r2p_%j.txt

#SBATCH --error=err_r2p_%j.txt

#SBATCH --partition=edu-medium

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --gres=gpu:1

#SBATCH --time=01:00:00

module purge

module load Python/3.11.3-GCCcore-12.3.0

source venv/bin/activate


export HF_HOME="/scratch/user/$USER/.cache/huggingface"

export TORCH_HOME="/scratch/user/$USER/.cache/torch"

mkdir -p $HF_HOME

mkdir -p $TORCH_HOME

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export FLASH_ATTN="False"

python full_loop.py

--------------------------------------------------

fare ctrl o e ctrl x per salvare e uscire


LANCIARE UN JOB

sbatch run_r2p.sh

Controlla:

squeue --me



