#!/bin/bash
set -euo pipefail

# ==================== Role-Confidence Cross-Steering MMLU-E ====================
# RR / RC / CR / CC + matched-random controls (RRand/CRand) + alpha=0 baseline,
# on Llama3.1-8B-Instruct, MMLU-E, 57 tasks, confident/unconfident prompt roles.
#
# STANDALONE: does NOT touch run_hidden_mmlue_confidence_hs.sh,
# run_mmlue_qwen25.sh, get_answer_regenerate_logits.py, or any existing mask/
# mean/result. get_answer_cross_steering_mmlue.py is its own script.
#
# Steering vectors (RR/RC/CR/CC/RRand/CRand) must already be built and synced
# to the server via build_cross_steering_vectors.py (local, no GPU) BEFORE
# this launcher is run -- this script does not build them.
#
# PILOT DESIGN (per task instructions): a single symmetric nonzero dose
# alpha=+-1 first (the vectors are already norm-matched to RR at alpha=1;
# alpha=-1 is the mirror-direction dose, same magnitude). alpha=0 (baseline)
# is written ONCE, shared across all six conditions.
#
# Usage:
#   bash run_cross_steering_mmlue.sh baseline          # alpha=0 only, run ONCE
#   bash run_cross_steering_mmlue.sh RR                # alpha -1,0,1 (0 reused if present)
#   bash run_cross_steering_mmlue.sh RC
#   bash run_cross_steering_mmlue.sh CR
#   bash run_cross_steering_mmlue.sh CC
#   bash run_cross_steering_mmlue.sh RRand
#   bash run_cross_steering_mmlue.sh CRand
#
# Each condition writes to its OWN alpha_{a} subdirectory; the script itself
# is fail-closed against overwriting a non-empty output dir (no --allow_overwrite
# passed here, matching the "no cross-writes between conditions" requirement).

CONDITION="${1:?usage: bash run_cross_steering_mmlue.sh {baseline|RR|RC|CR|CC|RRand|CRand}}"

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

VECTOR_DIR="${BASE_DIR}/llama3_confidence/cross_steering_mmlue/vectors"
MMLU_DIR="${BASE_DIR}/mmlu"
OUT_ROOT="${BASE_DIR}/llama3_confidence/cross_steering_mmlue/results"

cd "${WORK_DIR}"

if [ "${CONDITION}" == "baseline" ]; then
    ALPHAS="0"
else
    ALPHAS="-1,0,1"
    # If the shared baseline (alpha=0) already exists, skip re-running it by
    # requesting only the nonzero doses -- alpha=0 is condition-independent
    # (all-zero diff matrix regardless of which vector alpha=0 would multiply).
    if [ -d "${OUT_ROOT}/baseline/alpha_0" ] && [ -n "$(ls -A "${OUT_ROOT}/baseline/alpha_0" 2>/dev/null)" ]; then
        echo "[info] baseline/alpha_0 already exists and is non-empty -- requesting only -1,1"
        ALPHAS="-1,1"
    fi
fi

echo "=================================================="
echo "Cross-steering MMLU-E | condition=${CONDITION} | alphas=${ALPHAS}"
echo "Vector dir: ${VECTOR_DIR}"
echo "MMLU dir  : ${MMLU_DIR}"
echo "Out root  : ${OUT_ROOT}"
echo "Start     : $(date)"
echo "=================================================="

python cross_steering_confidence/get_answer_cross_steering_mmlue.py \
    --model_dir "${MODEL_DIR}" \
    --size "${SIZE}" \
    --vector_dir "${VECTOR_DIR}" \
    --mmlu_dir "${MMLU_DIR}" \
    --out_root "${OUT_ROOT}" \
    --condition "${CONDITION}" \
    --alphas "${ALPHAS}"

echo ""
echo "[Done] condition=${CONDITION} — $(date)"
