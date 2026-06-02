#!/bin/bash
# ==================== MATH role × α re-run ====================
# Entry-point: get_answer_regenerate_math.py. Non-Cartesian matrix (mirrors run_gsm8k.sh):
# multi-role only at alpha=0, steering only on neutral.
#
#   alpha=0  : No-CoT (neutral, expert, non_expert, math_expert) + CoT (neutral)
#   alpha=+4 : No-CoT (neutral)
#   alpha=-4 : No-CoT (neutral)
#
# Implemented as 3 calls (get_answer_regenerate_math.py is roles x configs Cartesian):
#   [1] configs=0-11-20         roles=4              (No-CoT)
#   [2] configs=0-11-20         roles=neutral --cot  (CoT)
#   [3] configs="4-11-20 neg4-11-20"  roles=neutral  (No-CoT)
#
# Prompt: roles passed as short names (neutral/expert/non_expert/math_expert);
# get_answer_regenerate_math.py routes neutral→neutral, others→neg ("Now you are {char}.")
# via utils.ROLE_TO_CHARACTER. NO "honest" framing (MATH has no E-option, same
# as GSM8K). All templates carry the "Provide your final answer in \boxed{}."
# directive (MATH analogue of GSM8K's #### directive).
#
# Output isolation: the .py output path does NOT encode cot, so the two alpha=0
# calls would collide. Use a separate --ans_file for CoT vs No-CoT:
#   No-CoT → answer_math        (mdf_0 / mdf_4 / mdf_neg4)
#   CoT    → answer_math_cot    (mdf_0)
#
# Usage: bash run_math_llama3.sh

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
BATCH_SIZE=8
N_SAMPLES=300
MATH_FILE="benchmark/math_test_sample.json"

# Short role names (routed to characters via utils.ROLE_TO_CHARACTER):
#   expert→"an expert", non_expert→"a non expert", math_expert→"a mathematician"
ROLES_ALL="neutral,expert,non_expert,math_expert"
ROLES_NEUTRAL="neutral"

ANS_NOCOT="answer_math"
ANS_COT="answer_math_cot"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "MATH role × α re-run | ${MODEL_NAME} (${MODEL_SIZE})"
echo "Start: $(date)"
echo "=================================================="
cd "${WORK_DIR}"

# ==================== [1] alpha=0, No-CoT, all roles ====================
echo ""
echo "[1/3] alpha=0 No-CoT — roles: ${ROLES_ALL}"
python get_answer_regenerate_math.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    0-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${MATH_FILE}" \
    --ans_file   "${ANS_NOCOT}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_ALL}" \
    --n_samples      ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
[ $? -eq 0 ] && echo "[✓] step 1" || { echo "[✗] step 1"; exit 1; }

# ==================== [2] alpha=0, CoT, neutral ====================
echo ""
echo "[2/3] alpha=0 CoT — neutral"
python get_answer_regenerate_math.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    0-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${MATH_FILE}" \
    --ans_file   "${ANS_COT}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --n_samples      ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE} \
    --cot
[ $? -eq 0 ] && echo "[✓] step 2" || { echo "[✗] step 2"; exit 1; }

# ==================== [3] alpha=+4 / -4, No-CoT, neutral ====================
echo ""
echo "[3/3] alpha=+4/-4 No-CoT — neutral"
python get_answer_regenerate_math.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    4-11-20 neg4-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${MATH_FILE}" \
    --ans_file   "${ANS_NOCOT}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --n_samples      ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
[ $? -eq 0 ] && echo "[✓] step 3" || { echo "[✗] step 3"; exit 1; }

echo ""
echo "=================================================="
echo "All MATH runs finished: $(date)"
echo "No-CoT → ${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}/"
echo "CoT    → ${BASE_DIR}/${MODEL_NAME}/${ANS_COT}/"
echo "=================================================="
