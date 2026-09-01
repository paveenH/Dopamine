#!/bin/bash
# ============ MATH fixed-workpoint transfer: Llama3-8B alpha=-6 ============
#
# PURPOSE. The GSM8K workpoint for Llama is alpha=-6. The stored MATH curve
# covers only -4/0/+4, so the workpoint itself has never been run on MATH.
# This adds exactly ONE cell (-6) so that the fixed-workpoint transfer test
# -- the same test P3 ran on GSM-Hard (docs/p3_amendment_05.json) -- can be
# run on MATH: alpha read from the frozen GSM8K record, NOT re-searched.
#
# THIS IS NOT A DOSE SEARCH AND NOT A NEW PREDICTION. It is a single
# pre-declared contrast, -6 vs the stored alpha=0, paired per question.
#
# HELD CONSTANT with the stored MATH cells (mdf_0 / mdf_4 / mdf_neg4 under
# llama3/math/math_eot/), verified against run_math.sh before writing:
#   same 300 problems (benchmark/math_test_sample.json, n_samples=300)
#   max_new_tokens=2048, temperature=0.0, batch_size=8, greedy
#   MATH budget 2048/bs=8 -- NOT GSM8K's 768/bs=24; reusing 768 truncates
#   MATH solutions and manufactures an extraction floor (see CLAUDE.md)
#   band 11-20 (L=9), mask nmd_0.5_11_20_8B.npy, prefill-only, tail_len=1
#   role neutral, No-CoT, \boxed{} directive
#   --ans_file answer_math, so -6 lands beside the existing cells as mdf_neg6
#
# ONE CARD. bf16 greedy is not byte-reproducible across GPUs, and this cell is
# compared per-question against a stored alpha=0. Pin CUDA_VISIBLE_DEVICES.
#
# NOTE the stored cells' physical GPU is unrecoverable (summary CSV carries no
# device field), so -6 vs 0 is a cross-run pairing. At temperature=0 the drift
# is small, but it is a real limitation and must be stated with the result.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_math_llama3_wp.sh > math_wp_llama.log 2>&1 &
#   cat math_wp_llama.log        # a wrong PY exits 127 before anything runs

set -u

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[x] refusing to run: CUDA_VISIBLE_DEVICES is unset."
    echo "    This cell is paired per question against a stored alpha=0 cell;"
    echo "    an unpinned or multi-card run mixes device differences into the"
    echo "    alpha effect irrecoverably."
    exit 1
fi
case "${CUDA_VISIBLE_DEVICES}" in
    *,*) echo "[x] refusing to run: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names more than one card."; exit 1 ;;
esac

PY="${PY:-python}"

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

ROLES_NEUTRAL="neutral"
ANS_NOCOT="answer_math"
CONFIG="neg6-11-20"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}/mdf_neg6/math_${MODEL_SIZE}_11_20.json"

echo "=================================================="
echo "MATH fixed-workpoint | ${MODEL_NAME} alpha=-6 | card ${CUDA_VISIBLE_DEVICES}"
echo "Start: $(date)"
echo "=================================================="
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

# interpreter names itself rather than failing obscurely under nohup
"${PY}" -c "import sys, numpy, torch; print('[py]', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)" \
    || { echo "[x] PY='${PY}' does not resolve or cannot import numpy/torch."; exit 1; }

if [ -e "${OUT}" ]; then
    echo "[x] refusing to overwrite an existing cell: ${OUT}"
    echo "    Delete it deliberately if this is a re-run."
    exit 1
fi

echo ""
echo "[1/1] alpha=-6 No-CoT — neutral"
"${PY}" get_answer_regenerate_math.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    "${CONFIG}" \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${MATH_FILE}" \
    --ans_file   "${ANS_NOCOT}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --n_samples      ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
[ $? -eq 0 ] && echo "[v] alpha=-6 done" || { echo "[x] alpha=-6 failed"; exit 1; }

echo ""
echo "=================================================="
echo "Finished: $(date)"
echo "Cell -> ${OUT}"
echo "Next (offline, from RoleAnswer/): python3.10 p3/math_workpoint.py"
echo "=================================================="
