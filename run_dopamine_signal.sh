#!/bin/bash
set -euo pipefail

# Phase 0: RSN dopamine signal tracking
# Records per-token RSN projection curves during original generation (no steering)
# Output: JSON with x_prefill, x_decode, ema_decode per sample

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5
LAYER_START=11
LAYER_END=20
EMA_ALPHA=0.95

MAX_NEW_TOKENS_GSM=512
MAX_NEW_TOKENS_MATH=1024
N_SAMPLES=300

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
MASK_PATH="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/${MASK_TYPE}_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"

echo "=================================================="
echo "Phase 0: Dopamine Signal Tracking"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Layers: ${LAYER_START}-${LAYER_END} | EMA alpha: ${EMA_ALPHA}"
echo "Start: $(date)"
echo "=================================================="

cd ${WORK_DIR}

echo ""
echo "[sanity] NMD mask indexing check"
python sanity_mask_indexing.py \
    --mask_path "${MASK_PATH}" \
    --expect_layer_start ${LAYER_START} \
    --expect_layer_end ${LAYER_END} \
    --skip_model

# ── GSM8K No-CoT ──
echo ""
echo "[1/4] GSM8K No-CoT"
python track_dopamine_signal.py \
    --task gsm8k \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" \
    --layer_start ${LAYER_START} \
    --layer_end ${LAYER_END} \
    --ema_alpha ${EMA_ALPHA} \
    --test_file "benchmark/gsm8k_test_sample.json" \
    --n_samples ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS_GSM} \
    --temperature 0.0 \
    --base_dir "${BASE_DIR}"

echo "[Done] GSM8K No-CoT"

# ── GSM8K CoT ──
echo ""
echo "[2/4] GSM8K CoT"
python track_dopamine_signal.py \
    --task gsm8k \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" \
    --layer_start ${LAYER_START} \
    --layer_end ${LAYER_END} \
    --ema_alpha ${EMA_ALPHA} \
    --test_file "benchmark/gsm8k_test_sample.json" \
    --n_samples ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS_GSM} \
    --temperature 0.0 \
    --base_dir "${BASE_DIR}" \
    --cot

echo "[Done] GSM8K CoT"

# ── MATH CoT ──
echo ""
echo "[3/4] MATH CoT"
python track_dopamine_signal.py \
    --task math \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" \
    --layer_start ${LAYER_START} \
    --layer_end ${LAYER_END} \
    --ema_alpha ${EMA_ALPHA} \
    --test_file "benchmark/math_test_sample.json" \
    --n_samples ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS_MATH} \
    --temperature 0.0 \
    --base_dir "${BASE_DIR}" \
    --cot

echo "[Done] MATH CoT"

# ── MATH No-CoT ──
echo ""
echo "[4/4] MATH No-CoT"
python track_dopamine_signal.py \
    --task math \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" \
    --layer_start ${LAYER_START} \
    --layer_end ${LAYER_END} \
    --ema_alpha ${EMA_ALPHA} \
    --test_file "benchmark/math_test_sample.json" \
    --n_samples ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS_MATH} \
    --temperature 0.0 \
    --base_dir "${BASE_DIR}"

echo "[Done] MATH No-CoT"

echo ""
echo "=================================================="
echo "All done: $(date)"
echo "Output → ${BASE_DIR}/${MODEL_NAME}/dopamine_signal/"
echo "=================================================="
