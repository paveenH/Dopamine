#!/bin/bash
# Effort-based Task Choice: GSM8K (easy) vs. MATH (hard)
# Models: Llama3-8B + Qwen3-8B
# Conditions: alpha=0/+4/-4 (best layers) + alpha=+1/-1 (all layers 1-33)

DATA="data1"
MASK_TYPE="nmd"
PERCENTAGE=0.5

GSM8K_FILE="benchmark/gsm8k_test_sample.json"
MATH_FILE="benchmark/math_test_sample.json"

WORK_DIR="/${DATA}/paveen/RolePlaying"

cd ${WORK_DIR}

# ==================================================
# Llama3-8B
# ==================================================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
# llama3: alpha=0/+4/-4 use 11-20; alpha=+1/-1 use full layers 1-33
CONFIGS="0-11-20 4-11-20 neg4-11-20 1-1-33 neg1-1-33"

echo "=================================================="
echo "Start Llama3: $(date)"
echo "Configs: ${CONFIGS}"
echo "=================================================="

python get_action_effort_choice.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --gsm8k_file "${GSM8K_FILE}" \
    --math_file "${MATH_FILE}" \
    --ans_file "answer_effort_choice" \
    --data "${DATA}"

echo "=================================================="
echo "Done Llama3: $(date)"
echo "=================================================="

# ==================================================
# Qwen3-8B
# ==================================================
MODEL_NAME="qwen3"
MODEL_DIR="Qwen/Qwen3-8B"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="qwen3"
# qwen3: alpha=0 uses 11-20; alpha=+4/-4 use 17-26; alpha=+1/-1 use full layers 1-33
CONFIGS="0-11-20 4-17-26 neg4-17-26 1-1-33 neg1-1-33"

echo "=================================================="
echo "Start Qwen3: $(date)"
echo "Configs: ${CONFIGS}"
echo "=================================================="

python get_action_effort_choice.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --gsm8k_file "${GSM8K_FILE}" \
    --math_file "${MATH_FILE}" \
    --ans_file "answer_effort_choice" \
    --data "${DATA}"

echo "=================================================="
echo "Done Qwen3: $(date)"
echo "=================================================="
