#!/bin/bash
# Verify: re-run GSM8K +4 / neutral / No-CoT through the EXACT run_gsm8k.sh path,
# but write to a SEPARATE dir (answer_verify_gsm8k) so the stored mdf_4 is NOT
# overwritten. Then byte-compare the two to confirm reproducibility.
#
# All args copied verbatim from run_gsm8k.sh (plain wording, bs=24, 768 tokens).
set -euo pipefail

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"
SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
cd "${WORK_DIR}"

echo "[verify] +4 neutral No-CoT plain → answer_verify_gsm8k/mdf_4"
python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    4-11-20 \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_verify_gsm8k" \
    --suite      "${SUITE}" \
    --fmt_wording plain \
    --base_dir   "${BASE_DIR}" \
    --roles      "neutral" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}

NEW="${BASE_DIR}/${MODEL_NAME}/answer_verify_gsm8k/mdf_4/gsm8k_8B_answers_20_11_20.json"
echo ""
echo "[verify] saved: ${NEW}"
echo "[verify] scp this to local and diff against gsm8k_eot/mdf_4/."
