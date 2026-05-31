#!/bin/bash
# ==================== Refusal (−4) Supplement — Llama3-8B ====================
# Fills in the missing α=−4 neutral-role data for:
#   (1) MMLU-E       → get_answer_regenerate_logits.py     --E
#   (2) MMLU-Pro+E   → get_answer_regenerate_logits_mmlupro.py  --use_E
#
# Results saved to: components/llama3/answer_mdf_refusal/mdf_-4/
#
# Usage: bash run_refusal_neg4_llama3.sh

# ==================== Model Configuration ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
SUITE="default"
DATA="data1"
MASK_TYPE="nmd"
PERCENTAGE=0.5
ROLES="neutral"

# α=−4, layers 11–20
CONFIG="neg4-11-20"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
ANS_FILE_MMLU="answer_mdf_refusal_mmlu"
ANS_FILE_MMLUPRO="answer_mdf_refusal_mmlupro"

echo "=================================================="
echo "Refusal (−4) Supplement — Llama3-8B"
echo "Start time: $(date)"
echo "Config: ${CONFIG} | Roles: ${ROLES}"
echo "=================================================="

cd "${WORK_DIR}"

# ==================== 1. MMLU-E ====================
echo ""
echo "=========================================="
echo "[1] MMLU-E — neutral, α=−4"
echo "=========================================="

python get_answer_regenerate_logits.py \
    --data "${DATA}" \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIG} \
    --mask_type "${MASK_TYPE}" \
    --ans_file "${ANS_FILE_MMLU}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    --E

if [ $? -eq 0 ]; then
    echo "[✓ Done] MMLU-E neutral α=−4"
else
    echo "[✗ Failed] MMLU-E neutral α=−4"
    exit 1
fi

# ==================== 2. MMLU-Pro+E ====================
echo ""
echo "=========================================="
echo "[2] MMLU-Pro+E — neutral, α=−4"
echo "=========================================="

python get_answer_regenerate_logits_mmlupro.py \
    --data "${DATA}" \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIG} \
    --mask_type "${MASK_TYPE}" \
    --test_file "benchmark/mmlupro_test.json" \
    --ans_file "${ANS_FILE_MMLUPRO}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    --use_E

if [ $? -eq 0 ]; then
    echo "[✓ Done] MMLU-Pro+E neutral α=−4"
else
    echo "[✗ Failed] MMLU-Pro+E neutral α=−4"
    exit 1
fi

# ==================== Summary ====================
echo ""
echo "=================================================="
echo "All done at: $(date)"
echo "Results → ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE_MMLU}/mdf_-4/"
echo "          ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE_MMLUPRO}/mdf_-4/"
echo "=================================================="
