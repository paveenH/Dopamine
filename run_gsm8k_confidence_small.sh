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
#
# OVERWRITE GUARD (load-bearing — get_answer_regenerate_gsm8k.py has NONE of
# its own): it writes every output file via plain `open(..., "w")`, so
# `os.makedirs(out_dir, exist_ok=True)` at line 149 is the ONLY thing standing
# between a rerun and silently clobbering an existing cell's JSON/CSV. This
# launcher therefore checks, BEFORE calling the script, that none of the four
# target mdf_* directories already contain that config's output file, and
# refuses to run if any do.
#
# Run all 4 new doses on ONE machine / ONE GPU model (bf16 greedy is not
# byte-reproducible across GPUs — see CLAUDE.md's cross-machine rule), and on
# the SAME machine/GPU model as the original 5-dose pilot — required for a
# strict dose-response comparison across all nine points, not merely
# recommended.
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

# ==================== Overwrite guard (fail BEFORE any generation) ====================
# get_answer_regenerate_gsm8k.py has no overwrite protection of its own — its
# output path is deterministic (SAVE_ROOT/mdf_{alpha}/gsm8k_{size}_answers_{TOP}_{st}_{en}.json),
# so refuse to launch if any of the four new-dose target files already exist.
TOP=$(python3 -c "print(max(1, int(${PERCENTAGE} / 100 * 4096)))" 2>/dev/null || python -c "print(max(1, int(${PERCENTAGE} / 100 * 4096)))")
SAVE_ROOT="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
NEW_ALPHAS=(neg1 neg0.5 0.5 1)
BLOCKED=0
for a in "${NEW_ALPHAS[@]}"; do
    # translate the config token (e.g. "neg0.5") to the mdf_{alpha} dirname
    # (e.g. "mdf_-0.5"), matching utils.parse_configs + f"mdf_{alpha}" exactly.
    if [[ "${a}" == neg* ]]; then
        DIRALPHA="-${a#neg}"
    else
        DIRALPHA="${a}"
    fi
    TARGET="${SAVE_ROOT}/mdf_${DIRALPHA}/gsm8k_${MODEL_SIZE}_answers_${TOP}_${LAYER_START}_${LAYER_END}.json"
    if [ -f "${TARGET}" ]; then
        echo "[REFUSE] existing output found, will not overwrite: ${TARGET}"
        BLOCKED=1
    fi
done
if [ "${BLOCKED}" -eq 1 ]; then
    echo "One or more target cells already exist. Aborting before any generation."
    echo "Delete the conflicting file(s) deliberately first if a rerun is truly intended."
    exit 1
fi
echo "[OK] no existing output at any of: mdf_{-1,-0.5,0.5,1} — safe to proceed."

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
