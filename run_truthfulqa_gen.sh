#!/bin/bash
# TruthfulQA-Generation (open-ended) + RSN α steering, α = 0 / +4 / −4, layers 11–20.
# Judge-free over-generation readouts (length / hedge-rate / assertion density);
# truthful/informative judged offline by reading raw vs the carried reference lists.
# Standalone — does not touch any existing task's loader/runner.
#
# Step 0 (once): build the dataset JSON (100 sampled questions WITH reference lists).
#   python data_truthfulqa_gen.py --n 100 --out "${BASE_DIR}/benchmark/truthfulqa_gen.json"

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

N_SAMPLES=100
CONFIGS="0-11-20 4-11-20 neg4-11-20"

# Build dataset if missing
if [ ! -f "${BASE_DIR}/benchmark/truthfulqa_gen.json" ]; then
    echo "[setup] building TruthfulQA-Gen JSON (${N_SAMPLES} questions)"
    python data_truthfulqa_gen.py --n "${N_SAMPLES}" \
        --out "${BASE_DIR}/benchmark/truthfulqa_gen.json"
fi

echo "[run] TruthfulQA-Generation, α = ${CONFIGS}"
python get_answer_regenerate_truthfulqa_gen.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --max_new_tokens 128 \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    --ans_file "answer_truthfulqa_gen"
