#!/bin/bash
# CRT (Cognitive Reflection Test) with RSN steering
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

CRT_FILE="benchmark/crt_questions.json"

WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "Start: $(date)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Configs: ${CONFIGS}"
echo "Task: CRT (Cognitive Reflection Test)"
echo "=================================================="

cd ${WORK_DIR}

python get_answer_crt.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --crt_file "${CRT_FILE}" \
    --ans_file "answer_crt" \
    --data "${DATA}"

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
