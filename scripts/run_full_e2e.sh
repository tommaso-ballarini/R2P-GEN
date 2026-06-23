#!/bin/bash
#SBATCH --job-name=r2p_full_e2e
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_e2e/%j.out
#SBATCH --error=logs/full_e2e/%j.err

# ===========================================================================
# 1. Setup
# ===========================================================================
echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

mkdir -p logs/full_e2e

# ===========================================================================
# 2. Env var
# ===========================================================================
export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/data/perva-data
export R2P_MODELS_BASE=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_CLUSTER_MODE=true
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_FLUX_MODEL=/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# Directory dedicata a questo test — completamente indipendente da test_100
# e test_debug, nessun file condiviso.
OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_e2e
DATABASE=$OUTPUT_DIR/database_e2e.json

mkdir -p $OUTPUT_DIR

# ===========================================================================
# 3. Stage 0 — Build database (10 concetti)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 0 — BUILD DATABASE (debug_limit=10)"
echo "=========================================================="
START=$(date +%s)

python -u pipeline/build_database.py \
    --perva-data $R2P_PERVA_DATA \
    --debug \
    --debug-limit 10 \
    --output $DATABASE

END=$(date +%s)
echo "⏱️  Tempo build_database: $((END - START))s"

# Verifica che il database sia stato creato correttamente
if [ ! -f "$DATABASE" ]; then
    echo "❌ database_e2e.json NON trovato — pipeline interrotta."
    exit 1
fi

N_CONCEPTS=$(python3 -c "
import json
with open('$DATABASE') as f:
    d = json.load(f)
print(len(d.get('concept_dict', {})))
")
echo "✅ Database creato: $N_CONCEPTS concetti"

# ===========================================================================
# 4. Stage 1-4 — Full auto (generate → verify → refine → judge)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 1-4 — FULL AUTO PIPELINE"
echo "=========================================================="
START=$(date +%s)

python -u flux_loop.py \
    --stage full_auto \
    --database $DATABASE \
    --output $OUTPUT_DIR

END=$(date +%s)
echo "⏱️  Tempo full_auto: $((END - START))s  ($((( END - START ) / 60))m)"

# ===========================================================================
# 5. Verifica output completo
# ===========================================================================
echo ""
echo "--- Output check ---"

# Immagini generate
N_GEN=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_generated.png" | wc -l)
echo "Immagini *_generated.png:  $N_GEN"

# rejected_concepts.json
if [ -f "$OUTPUT_DIR/rejected_concepts.json" ]; then
    N_REJ=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/rejected_concepts.json'))))")
    echo "✅ rejected_concepts.json: $N_REJ concetti rifiutati al verify base"
else
    echo "⚠️  rejected_concepts.json non trovato (tutti passed al primo verify?)"
fi

# recovery_results.json
if [ -f "$OUTPUT_DIR/recovery_results.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/recovery_results.json') as f:
    r = json.load(f)
n_rec   = sum(1 for d in r.values() if d.get('status') == 'recovered')
n_grave = sum(1 for d in r.values() if d.get('status') == 'unrecoverable')
print(f'✅ recovery_results.json: {len(r)} nel refine → {n_rec} recovered, {n_grave} graveyard')
"
else
    echo "⚠️  recovery_results.json non trovato (nessun concetto è entrato nel refine?)"
fi

# final_judge_results.json
if [ -f "$OUTPUT_DIR/final_judge_results.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/final_judge_results.json') as f:
    r = json.load(f)
print(f'✅ final_judge_results.json: {len(r)} concetti valutati')
for cid, data in r.items():
    m = data.get('metrics', {})
    print(f'   {cid}: CLIP-I={m.get(\"clip_i\", 0):.3f} '
          f'| DINO-I={m.get(\"dino_i\", 0):.3f} '
          f'| TIFA={m.get(\"tifa_score\", 0):.1%}')
"
else
    echo "❌ final_judge_results.json NON trovato"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="