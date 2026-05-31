#!/bin/bash
# MATH open-ended generation with RSN steering
# Model: Llama3-8B | Conditions: alpha=0 (baseline), alpha=+4, alpha=-4

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5

CONFIGS="0-11-20 4-11-20 neg4-11-20"

MATH_FILE="benchmark/math_test_sample.json"

WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "Start: $(date)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Configs: ${CONFIGS}"
echo "Task: MATH open-ended generation"
echo "=================================================="

cd ${WORK_DIR}

python get_answer_math.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "${MATH_FILE}" \
    --ans_file "answer_math" \
    --data "${DATA}" \
    --batch_size 8 \
    --max_new_tokens 1024 \
    --cot

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
