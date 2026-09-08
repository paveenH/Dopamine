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
# FORMAL DOSE SET: alpha in {-4, -2, 0, +2, +4}. Baseline (alpha=0) is run
# ONCE via `bash run_cross_steering_mmlue.sh baseline` and shared across all
# six vector conditions. Each non-baseline condition runs the four nonzero
# doses -4,-2,2,4 in one invocation.
#
# NEGATIVE ALPHA LIST: passed to the Python script as --alphas="-4,-2,2,4"
# (the '=' form) so argparse does not mistake the leading '-' for a new flag.
#
# RECOMMENDED RUN ORDER (per task instructions): baseline -> RR -> CC -> RC
# -> CR -> RRand -> CRand. This launcher does not enforce the order itself
# (each condition is independent and resumable); run them in this sequence
# manually / via separate nohup invocations.
#
# RESUME: get_answer_cross_steering_mmlue.py checks completeness itself (57
# tasks present, correct sample counts, full confident/unconfident fields,
# matching run_meta). A COMPLETE cell is skipped; a PARTIAL cell has only its
# missing/incomplete tasks re-run; a cell whose stored metadata does not match
# the current invocation's config is a FATAL refusal (protocol mismatch), not
# a silent resume. So re-running this launcher on an already-complete
# condition is a safe no-op, and re-running it after an interruption completes
# only what is missing.
#
# Usage:
#   bash run_cross_steering_mmlue.sh baseline          # alpha=0 only
#   bash run_cross_steering_mmlue.sh RR                # alpha -4,-2,2,4
#   bash run_cross_steering_mmlue.sh CC
#   bash run_cross_steering_mmlue.sh RC
#   bash run_cross_steering_mmlue.sh CR
#   bash run_cross_steering_mmlue.sh RRand
#   bash run_cross_steering_mmlue.sh CRand

CONDITION="${1:?usage: bash run_cross_steering_mmlue.sh baseline-or-RR-or-RC-or-CR-or-CC-or-RRand-or-CRand}"

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
    ALPHAS="-4,-2,2,4"
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
    --alphas="${ALPHAS}"

echo ""
echo "[Done] condition=${CONDITION} — $(date)"
