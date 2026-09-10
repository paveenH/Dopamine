#!/bin/bash
# ============== GSM8K Confidence-neuron steering — small-dose supplement ==============
# Adds FOUR small doses (alpha = -1, -0.5, +0.5, +1) to the existing 5-point
# Confidence-mask pilot (run_gsm8k_confidence.sh: -4/-2/0/+2/+4). Protocol is
# BYTE-IDENTICAL to that launcher — same model/prompt/eot/greedy/injection
# band/mask/generation params — ONLY the dose list differs. Does not modify
# run_gsm8k_confidence.sh, get_answer_regenerate_gsm8k.py, llms.py, template.py,
# or any existing output cell (mdf_-4/-2/0/2/4 are untouched and NOT rerun).
#
# Requires utils.parse_configs to accept fractional alpha tokens (e.g.
# "0.5-11-20", "neg0.5-11-20"). This was added as a COMPATIBLE extension:
# int parsing is tried first, so every existing integer config (all other
# experiments in this repo) returns the exact same value and mdf_{alpha}
# directory name as before; float is only reached when int() fails, and a
# whole-valued float (e.g. "1.0") is normalized back to int. Verified by
# regression test against every existing --configs style in the repo
# (integer single/multi-config, neg-prefixed, alpha=0) before this launcher
# was written — see utils.py::_parse_alpha_token.
#
# Doses: alpha = -1, -0.5, +0.5, +1 (new). Does NOT include -4/-2/0/+2/+4 —
# those cells already exist and must not be rerun or overwritten.
#
# Output: components/llama3/answer_mdf_gsm8k_confidence/mdf_{alpha}/
#   New cells: mdf_-1, mdf_-0.5, mdf_0.5, mdf_1
#   (same ANS_FILE tree as the existing 5-dose pilot — the four new cells sit
#   alongside mdf_-4/-2/0/2/4 in that tree; get_answer_regenerate_gsm8k.py's
#   own per-config directory (`mdf_{alpha}`) already fails closed against
#   silently overwriting a different alpha's data, since each alpha gets its
#   own subdirectory name.)
#
# Run all 4 new doses on ONE machine / ONE GPU model (bf16 greedy is not
# byte-reproducible across GPUs — see CLAUDE.md's cross-machine rule), and
# ideally the SAME machine/GPU model as the original 5-dose pilot for the
# fullest dose-response comparability (not required for correctness, since
# each alpha is independently valid, but recommended).
#
# Usage: bash run_gsm8k_confidence_small.sh
# ================================================================================

set -e

# ==================== Model ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"

# ==================== Shared config (byte-identical to run_gsm8k_confidence.sh) ====================
SUITE="default"
MASK_TYPE="confidence"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"
WORDING="plain"
ROLES_NEUTRAL="neutral"

LAYER_START=11
LAYER_END=20

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
ANS_FILE="answer_mdf_gsm8k_confidence"

cd "${WORK_DIR}"

echo "=================================================="
echo "GSM8K Confidence-neuron steering — SMALL-DOSE supplement | ${MODEL_NAME} (${MODEL_SIZE})"
echo "mask_type=${MASK_TYPE}  layers=${LAYER_START}-${LAYER_END}  role=neutral"
echo "doses (NEW ONLY): -1 -0.5 +0.5 +1"
echo "existing doses -4/-2/0/+2/+4 are NOT rerun"
echo "out: ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/"
echo "Start: $(date)"
echo "=================================================="

python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    neg1-${LAYER_START}-${LAYER_END} neg0.5-${LAYER_START}-${LAYER_END} \
                 0.5-${LAYER_START}-${LAYER_END} 1-${LAYER_START}-${LAYER_END} \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "${ANS_FILE}" \
    --suite      "${SUITE}" \
    --fmt_wording "${WORDING}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}

echo ""
echo "=================================================="
echo "Small-dose supplement finished: $(date)"
echo "=================================================="
