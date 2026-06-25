#!/bin/bash
# HaluEval-QA discrimination + RSN α steering, α = 0 / +4 / −4, layers 11–20.
# SIGNED-BIAS hallucination probe (headline = FNR/credulity & acceptance, NOT accuracy).
# Standalone — does not touch any existing task's loader/runner.
#
# Step 0 (once): build the balanced discrimination JSON (300 src rows -> 600 items).
#   python data_halueval.py --n 300 --out "${BASE_DIR}/benchmark/halueval_qa_disc.json"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/Dopamine/components"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"
MASK_TYPE="nmd"
PERCENTAGE=0.5

N_SAMPLES=300
# Full −8→+8 sweep on the SAME 600-item file (do NOT re-sample — keep the cells comparable).
# All 9 cells re-run at max_new_tokens=128 so raw generated text is comparable across α.
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

# Build dataset if missing
if [ ! -f "${BASE_DIR}/benchmark/halueval_qa_disc.json" ]; then
    echo "[setup] building HaluEval-QA discrimination JSON (${N_SAMPLES} src rows -> $((N_SAMPLES*2)) items)"
    python data_halueval.py --n "${N_SAMPLES}" \
        --out "${BASE_DIR}/benchmark/halueval_qa_disc.json"
fi

echo "[run] HaluEval-QA discrimination, α = ${CONFIGS}"
python get_answer_regenerate_halueval.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --max_new_tokens 128 \
    --save_all_raw \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    --ans_file "answer_halueval"
