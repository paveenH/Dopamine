#!/bin/bash

DATA="data1"
TRAIT_FILE="benchmark/trait_test.json"
BASE_DIR="/${DATA}/paveen/RolePlaying/components"

# --- Qwen3-8B (commented out) ---
# MODEL_NAME="qwen3"
# MODEL_DIR="Qwen/Qwen3-8B"
# MODEL_SIZE="8B"
# TYPE="non"
# HS_PREFIX="qwen3"
# MASK_TYPE="nmd"
# PERCENTAGE=0.5
# CONFIGS="0-17-26 4-17-26 neg4-17-26 1-1-33 neg1-1-33"
# python get_answer_trait.py \
#     --model "${MODEL_NAME}" \
#     --model_dir "${MODEL_DIR}" \
#     --hs "${HS_PREFIX}" \
#     --size "${MODEL_SIZE}" \
#     --type "${TYPE}" \
#     --percentage "${PERCENTAGE}" \
#     --configs ${CONFIGS} \
#     --mask_type "${MASK_TYPE}" \
#     --test_file "${TRAIT_FILE}" \
#     --ans_file "answer_trait" \
#     --data "${DATA}" \
#     --base_dir "${BASE_DIR}" \
#     --suite "vanilla" \
#     --roles "neutral"

# --- Llama3-8B-Instruct: full-layer configs (1-1-33, neg1-1-33) ---
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
MASK_TYPE="nmd"
PERCENTAGE=0.5
CONFIGS="6-11-20 neg6-11-20 10-11-20 neg10-11-20"

python get_answer_trait.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "${TRAIT_FILE}" \
    --ans_file "answer_trait" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    --suite "vanilla" \
    --roles "neutral"

# --- Llama3-8B-Base: mask and configs from IT model (hs=llama3, type=non) ---
# MODEL_NAME="llama3_base"
# MODEL_DIR="meta-llama/Llama-3.1-8B"
# MODEL_SIZE="8B"
# TYPE="non"
# HS_PREFIX="llama3"
# MASK_TYPE="nmd"
# PERCENTAGE=0.5
# CONFIGS="0-11-20 4-11-20 neg4-11-20 1-1-33 neg1-1-33"

python get_answer_trait.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "${TRAIT_FILE}" \
    --ans_file "answer_trait" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    --suite "vanilla" \
    --roles "neutral"
