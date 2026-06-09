#!/bin/bash
# GSM8K Willingness Self-Evaluation (0-9) with Neuron Editing
# Model: Llama3-8B | Conditions: neutral (alpha=0), alpha=+4, alpha=-4

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5

# alpha=0 uses get_logits (no mask loaded); alpha=+4/-4 uses diff_mtx * alpha
# format: alpha-start-end  (0 means neutral, no steering)
# Full dose-response sweep -8 -> +8, layers 11-20 only (fast experiment)
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"
# Previous (best-layer ±4 + all-layer ±1):
# CONFIGS="0-11-20 4-11-20 neg4-11-20 1-1-33 neg1-1-33"

GSM8K_FILE="benchmark/gsm8k_test_sample.json"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "Start: $(date)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Configs: ${CONFIGS}"
echo "Task: GSM8K Willingness (0-9)"
echo "=================================================="

cd ${WORK_DIR}

python get_action_regenerate_gsm8k.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "${GSM8K_FILE}" \
    --ans_file "answer_action_gsm8k" \
    --data "${DATA}"

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
