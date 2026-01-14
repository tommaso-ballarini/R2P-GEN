# Guida Operativa Cluster - Progetto R2P-Gen

PREREQUISITI

- VPN attiva: GlobalProtect connesso
- Terminale:
  - Mac / Linux: Terminale
  - Windows: PowerShell o WSL
- Cartella progetto locale: R2P_Gen_Project con:

R2P_Gen_Project/
  data/                (immagini di test, es. 1.jpg)
  r2p_core/            (copiata dalla repo originale R2P)
  full_loop.py
  step1_*.py
  step2_*.py
  ...
  requirements.txt

--------------------------------------------------

CARICARE I FILE SUL CLUSTER

Usiamo scp. Carichiamo tutto in /scratch (NON /home).

Dal TUO PC:

scp -r R2P_Gen_Project tuo_user@indirizzo_cluster:/scratch/user/tuo_user/

--------------------------------------------------

SETUP AMBIENTE (UNA SOLA VOLTA)

Collegati al cluster:

ssh tuo_user@indirizzo_cluster

Vai nella cartella:

cd /scratch/user/tuo_user/R2P_Gen_Project

Crea l'ambiente:

module load Python/3.11.3-GCCcore-12.3.0
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Se non hai requirements.txt:

pip install torch torchvision torchaudio diffusers transformers accelerate peft pillow flash-attn --no-build-isolation

--------------------------------------------------

SCRIPT SLURM (run_r2p.sh)

Crea il file:

nano run_r2p.sh

Contenuto COMPLETO:

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



