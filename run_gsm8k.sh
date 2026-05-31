#!/bin/bash
# ==================== GSM8K regenerate re-run (2026-05-31) ====================
# Single entry-point: get_answer_regenerate_gsm8k.py. Non-Cartesian matrix —
# multi-role only at alpha=0, steering only on neutral.
#
#   alpha=0  : No-CoT (expert, non_expert, primary_teacher, neutral) + CoT (neutral)
#   alpha=+4 : No-CoT (neutral)
#   alpha=-4 : No-CoT (neutral)
#
# Implemented as 3 calls (the script is roles x configs Cartesian, so we split):
#   [1] configs=0-11-20         roles=4 chars       (No-CoT)
#   [2] configs=0-11-20         roles=neutral  --cot (CoT)
#   [3] configs="4-11-20 neg4-11-20"  roles=neutral (No-CoT)
#
# Prompt: roles passed as full character strings. GSM8K has no E-option → NO
# "honest" framing. get_answer_regenerate_gsm8k.py routes neutral→neutral and
# any other role→neg ("Now you are {role}."), matching track_dopamine_signal.py.
# All 6 GSM8K templates now carry the "#### <number>" final-answer directive.
#
# Output isolation: --ans_file differs for CoT vs No-CoT (the .py output path
# does NOT encode cot), so the two alpha=0 calls don't overwrite each other.
#   No-CoT → answer_mdf_gsm8k        (mdf_0 / mdf_4 / mdf_-4)
#   CoT    → answer_mdf_gsm8k_cot    (mdf_0)
#
# Usage: bash run_gsm8k_regen.sh

# ==================== Model ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"

# ==================== Shared config ====================
SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# Full character strings (NOT short names) so prompts align with track.
ROLES_ALL="an expert,a non expert,a primary school teacher,neutral"
ROLES_NEUTRAL="neutral"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "GSM8K regenerate re-run | ${MODEL_NAME} (${MODEL_SIZE})"
echo "Start: $(date)"
echo "=================================================="
cd "${WORK_DIR}"

# ==================== [1] alpha=0, No-CoT, all roles ====================
echo ""
echo "[1/3] alpha=0 No-CoT — roles: ${ROLES_ALL}"
python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    0-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_mdf_gsm8k" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_ALL}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
[ $? -eq 0 ] && echo "[✓] step 1" || { echo "[✗] step 1"; exit 1; }

# ==================== [2] alpha=0, CoT, neutral ====================
echo ""
echo "[2/3] alpha=0 CoT — neutral"
python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    0-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_mdf_gsm8k_cot" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE} \
    --cot
[ $? -eq 0 ] && echo "[✓] step 2" || { echo "[✗] step 2"; exit 1; }

# ==================== [3] alpha=+4 / -4, No-CoT, neutral ====================
echo ""
echo "[3/3] alpha=+4/-4 No-CoT — neutral"
python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    4-11-20 neg4-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_mdf_gsm8k" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
[ $? -eq 0 ] && echo "[✓] step 3" || { echo "[✗] step 3"; exit 1; }

echo ""
echo "=================================================="
echo "All GSM8K regenerate runs finished: $(date)"
echo "=================================================="
