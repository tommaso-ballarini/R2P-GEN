#!/bin/bash
set -e

module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HOME="/leonardo_work/IscrC_MUSE/tballari/models_cache/.hf_cache"
export R2P_MODELS_BASE="/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR="/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_40"

cleanup() {
    if [ -n "$FLUX_TEXT_PID" ]; then
        echo "🛑 Chiudo FluxText server (PID=$FLUX_TEXT_PID)"
        kill $FLUX_TEXT_PID 2>/dev/null
        wait $FLUX_TEXT_PID 2>/dev/null
    fi
}
trap cleanup EXIT

echo "📦 Backup recovery_results.json..."
cp $OUTPUT_DIR/recovery_results.json $OUTPUT_DIR/recovery_results.json.bak

echo "🔬 Isolamento concept <kqc> per test..."
python3 -c "
import json
results = json.load(open('$OUTPUT_DIR/recovery_results.json'))
test_results = {'<kqc>': results['<kqc>']}
json.dump(test_results, open('$OUTPUT_DIR/recovery_results.json', 'w'), indent=2)
print('OK')
"

echo "✍️  Avvio FluxText server (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python flux_text_server.py --port 8767 > logs/flux_text_test.log 2>&1 &
FLUX_TEXT_PID=$!

echo "⏳ Attendo FluxText (può richiedere fino a 15 minuti)..."
for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:8767/health" > /dev/null 2>&1; then
        echo "   ✅ FluxText pronto!"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "   ❌ FluxText non risponde dopo 15 minuti. Abort."
        exit 1
    fi
    sleep 10
done

echo "🚀 Lancio stage text_fix..."
CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage text_fix \
    --database database/database.json \
    --output $OUTPUT_DIR

echo "📦 Ripristino recovery_results.json originale..."
cp $OUTPUT_DIR/recovery_results.json.bak $OUTPUT_DIR/recovery_results.json

echo "🏁 Test completato!"
