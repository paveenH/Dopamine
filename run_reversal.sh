#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/RolePlaying/components"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
MASK_TYPE="nmd"
PERCENTAGE=0.5
CONFIGS="0-11-20 4-11-20 neg4-11-20"

python get_answer_reversal.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --num_runs 30 \
    --num_rounds 40 \
    --phase_switch 20 \
    --reward_prob 0.8 \
    --history_window 5 \
    --counterbalance \
    --ans_file "answer_reversal_v2" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}"
