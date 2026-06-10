#!/bin/bash
# ==================== GSM8K regenerate — Mistral cross-model (2026-06-09) =======
# Mistral-7B-Instruct-v0.3 GSM8K CoT-only dose-response, full -8 -> +8 sweep,
# neutral only, layers 14-22, plain wording. Cross-model extension of the Llama
# dose curve (run_gsm8k.sh stays Llama-only — do NOT edit that file).
#
# CoT-only because the No-CoT sweep collapsed to ~14-19% acc across ALL alpha
# (model forced to emit #### with no reasoning) — Mistral needs CoT on GSM8K.
# The No-CoT results already exist in answer_mdf_gsm8k/ (kept as contrast).
# CoT uses --ans_file answer_mdf_gsm8k_cot (the .py output path does NOT encode
# cot), so it never overwrites the No-CoT dirs.
#
# *** Same-machine rule: bf16 greedy is NOT byte-reproducible across GPUs, so
# this whole Mistral dose curve must run on ONE machine. Pick a box, keep it. ***
#
# Single entry-point: get_answer_regenerate_gsm8k.py. alpha=0 -> get_logits
# (no mask); other alpha -> diff_mtx = mask * alpha. NMD mask read from
# components/mask/mistral_non_logits/ (HS_PREFIX=mistral, layers 14-22).
#
# Prompt: neutral, GSM8K has no E-option -> NO "honest". plain wording =
# "Provide your final numeric answer after '####'." (main-line, history-aligned).
# ACC is reported offline by analyze_first_last_acc.py (first-####), NOT from the
# inline correct_* fields.
#
# Output: No-CoT -> components/mistral/answer_mdf_gsm8k/mdf_{alpha}/
#         CoT    -> components/mistral/answer_mdf_gsm8k_cot/mdf_{alpha}/
#         (one dir per alpha each).
#
# Usage:  bash run_gsm8k_mistral.sh

# ==================== Model ====================
MODEL_NAME="mistral"
MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SIZE="7B"
HS_PREFIX="mistral"
TYPE="non"

# ==================== Shared config ====================
SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

ROLES_NEUTRAL="neutral"

# Full dose-response sweep -8 -> +8, layers 14-22 (Mistral middle band).
CONFIGS="neg8-14-22 neg6-14-22 neg4-14-22 neg2-14-22 0-14-22 2-14-22 4-14-22 6-14-22 8-14-22"

# plain = main-line #### wording (matches the Llama main line).
WORDING="plain"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
cd "${WORK_DIR}"

echo "=================================================="
echo "GSM8K Mistral cross-model | ${MODEL_NAME} (${MODEL_SIZE})"
echo "CoT-only, neutral, layers 14-22, wording=${WORDING}"
echo "Configs: ${CONFIGS}"
echo "Start: $(date)"
echo "=================================================="

# ==================== CoT-only ====================
# CoT-only: No-CoT collapsed to ~14-19% acc (model forced to emit #### with no
# reasoning) — Mistral needs CoT to solve GSM8K. The No-CoT sweep already lives
# in answer_mdf_gsm8k/ (kept as contrast); CoT writes to answer_mdf_gsm8k_cot/.
echo ""
echo "CoT — neutral → answer_mdf_gsm8k_cot/mdf_{alpha}"
python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    ${CONFIGS} \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_mdf_gsm8k_cot" \
    --suite      "${SUITE}" \
    --fmt_wording "${WORDING}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE} \
    --cot
[ $? -eq 0 ] && echo "[✓] CoT dose sweep done" || { echo "[✗] CoT failed"; exit 1; }

echo ""
echo "=================================================="
echo "Mistral GSM8K CoT dose sweep finished: $(date)"
echo "ACC: run analyze_first_last_acc.py (first-####) offline in RoleAnswer/."
echo "=================================================="
