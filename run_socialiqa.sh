#!/bin/bash
# SocialIQA (3-choice) generation + RSN α steering, α = 0 / +4 / −4, layers 11–20.
# Standalone pipeline — does not touch any existing task's loader/runner.
#
# Step 0 (once): build the dataset JSON from HF social_i_qa validation (1954 items).
#   python data_socialiqa.py --out "${BASE_DIR}/socialiqa/socialiqa.json"

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

CONFIGS="0-11-20 4-11-20 neg4-11-20"

# Build dataset if missing
if [ ! -f "${BASE_DIR}/socialiqa/socialiqa.json" ]; then
    echo "[setup] building SocialIQA JSON (validation, 1954 items)"
    python data_socialiqa.py --out "${BASE_DIR}/socialiqa/socialiqa.json"
fi

echo "[run] SocialIQA generation, α = ${CONFIGS}"
python get_answer_regenerate_socialiqa.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --max_new_tokens 8 \
    --save_all_raw \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    --ans_file "answer_socialiqa"
