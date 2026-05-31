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
# CONFIGS="0-11-20 4-11-20 neg4-11-20"
CONFIGS="neg4-11-20"

# All 30 ScienceWorld tasks
python get_answer_sciworld.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --task_nums 28 29 \
    --num_episodes 5 \
    --max_steps 50 \
    --max_new_tokens 32 \
    --ans_file "answer_sciworld" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}"
