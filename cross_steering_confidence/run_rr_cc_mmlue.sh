#!/bin/bash
set -euo pipefail

# ==================== Role self-steering (RR) / Confidence self-steering (CC) MMLU-E ====================
# SIMPLIFIED experiment: only RR (existing Role NMD mask) and CC (new
# Confidence NMD mask), each used AS-IS with no direction/support
# decomposition. NO RC, NO CR, NO random-support control, NO norm-matching
# in this stage.
#
# STANDALONE: does NOT touch run_hidden_mmlue_confidence_hs.sh,
# run_mmlue_qwen25.sh, get_answer_regenerate_logits.py, the existing Role
# mask, or any existing result.
#
# The Confidence mask (confidence_0.5_11_20_8B.npy) must already be built
# locally (build_confidence_nmd_mask_for_steering.py) and synced to
# ${MASK_DIR} on the server BEFORE this launcher is run.
#
# FORMAL DOSE SET: alpha in {-4, -2, 0, +2, +4}. Baseline (alpha=0) is run
# ONCE via `bash run_rr_cc_mmlue.sh baseline` and shared between RR and CC.
# Each non-baseline condition runs the four nonzero doses -4,-2,2,4.
#
# NEGATIVE ALPHA LIST: passed as --alphas="-4,-2,2,4" (the '=' form) so
# argparse does not mistake the leading '-' for a new flag.
#
# RESUME: get_answer_rr_cc_mmlue.py checks completeness itself (57 tasks
# present, correct sample counts, full confident/unconfident fields, matching
# run_meta). A COMPLETE cell is skipped; a PARTIAL cell has only its
# missing/incomplete tasks re-run; a metadata mismatch is a FATAL refusal.
#
# Usage:
#   bash run_rr_cc_mmlue.sh baseline   # alpha=0 only
#   bash run_rr_cc_mmlue.sh RR         # alpha -4,-2,2,4, using the Role mask
#   bash run_rr_cc_mmlue.sh CC         # alpha -4,-2,2,4, using the Confidence mask

CONDITION="${1:?usage: bash run_rr_cc_mmlue.sh {baseline|RR|CC}}"

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MASK_DIR="${BASE_DIR}/mask/llama3_non_logits"
MMLU_DIR="${BASE_DIR}/mmlu"
OUT_ROOT="${BASE_DIR}/llama3_confidence/rr_cc_mmlue/results"

cd "${WORK_DIR}"

if [ "${CONDITION}" == "baseline" ]; then
    ALPHAS="0"
else
    ALPHAS="-4,-2,2,4"
fi

echo "=================================================="
echo "RR/CC MMLU-E | condition=${CONDITION} | alphas=${ALPHAS}"
echo "Mask dir : ${MASK_DIR}"
echo "MMLU dir : ${MMLU_DIR}"
echo "Out root : ${OUT_ROOT}"
echo "Start    : $(date)"
echo "=================================================="

python cross_steering_confidence/get_answer_rr_cc_mmlue.py \
    --model_dir "${MODEL_DIR}" \
    --size "${SIZE}" \
    --mask_dir "${MASK_DIR}" \
    --mmlu_dir "${MMLU_DIR}" \
    --out_root "${OUT_ROOT}" \
    --condition "${CONDITION}" \
    --alphas="${ALPHAS}"

echo ""
echo "[Done] condition=${CONDITION} — $(date)"
