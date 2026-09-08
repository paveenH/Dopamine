#!/bin/bash
# ==================== GSM8K Confidence-neuron steering pilot ====================
# Replaces the RSN NMD mask with the Confidence mask and re-runs the SAME
# GSM8K protocol as the authoritative Llama3 eot re-run (run_gsm8k.sh, plain
# wording, MACHINE=182 branch) to check whether Confidence neurons also move
# GSM8K accuracy / commitment dynamics.
#
# Everything held identical to the authoritative RSN run:
#   - model (meta-llama/Llama-3.1-8B-Instruct), <|eot_id|> terminator (llms.py,
#     unconditional — not mask-dependent)
#   - prompt/template (select_templates_gsm8k, suite=default, fmt_wording=plain)
#   - No-CoT (neutral role only — this pilot does not touch CoT)
#   - greedy decoding (temperature=0.0)
#   - injection layers 11-20 (exclusive end, i.e. decoder layers 11-19),
#     prefill-only, tail=1 (script default — untouched)
#   - max_new_tokens=768, batch_size=24
#
# ONLY DIFFERENCE: --mask_type confidence instead of nmd, so the mask file
# resolved is components/mask/llama3_non_logits/confidence_0.5_11_20_8B.npy
# instead of nmd_0.5_11_20_8B.npy. No changes to get_answer_regenerate_gsm8k.py,
# llms.py, template.py, or the RSN mask/output tree.
#
# Doses: alpha = -4, -2, 0, +2, +4 (pilot). neutral role only (this experiment
# only asks about the neutral dose-response, so the alpha=0 baseline does not
# need the 4-role RSN sweep — see the launcher's own alpha=0 cell below).
#
# alpha=0 note: diff_mtx = mask * 0 is an all-zero matrix regardless of which
# mask file is loaded, so an alpha=0 run here is protocol-identical to the
# authoritative RSN alpha=0 baseline (same model/prompt/template/eot/greedy/
# batch). It is still re-run explicitly (not copied from the RSN mdf_0 JSON)
# so this experiment has its own complete, self-contained evidence chain in
# its own output tree, at the cost of one extra (cheap, alpha=0) generation
# pass.
#
# Output: components/llama3/answer_mdf_gsm8k_confidence/mdf_{alpha}/
#   (parallel to, and independent of, answer_mdf_gsm8k[_cot] — RSN tree is
#   untouched by this script.)
#
# Run all 5 doses on ONE machine / ONE GPU model (bf16 greedy is not
# byte-reproducible across GPUs — see CLAUDE.md's cross-machine rule).
#
# Usage: bash run_gsm8k_confidence.sh
# ================================================================================

set -e

# ==================== Model ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"

# ==================== Shared config (byte-identical to run_gsm8k.sh MACHINE=182, plain, No-CoT) ====================
SUITE="default"
MASK_TYPE="confidence"      # <-- the only steering-relevant change vs the RSN launcher
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
echo "GSM8K Confidence-neuron steering pilot | ${MODEL_NAME} (${MODEL_SIZE})"
echo "mask_type=${MASK_TYPE}  layers=${LAYER_START}-${LAYER_END}  role=neutral"
echo "doses: -4 -2 0 +2 +4"
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
    --configs    neg4-${LAYER_START}-${LAYER_END} neg2-${LAYER_START}-${LAYER_END} \
                 0-${LAYER_START}-${LAYER_END} \
                 2-${LAYER_START}-${LAYER_END} 4-${LAYER_START}-${LAYER_END} \
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
echo "Confidence pilot finished: $(date)"
echo "=================================================="
