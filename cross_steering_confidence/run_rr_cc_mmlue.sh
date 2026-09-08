#!/bin/bash
set -euo pipefail

# ==================== Confidence self-steering (CC) MMLU-E ====================
# RSN-paper-style MMLU-E confidence-enhancement replication: baseline alpha=0
# vs POSITIVE-DOSE-ONLY steering alpha=+2,+4,+6.
#
# RR IS DELIBERATELY NOT RUN IN THIS SCRIPT (as of 2026-09-08). No verified,
# protocol-matching RR result exists to reuse -- the only locatable historical
# artifact (RoleAnswer/llama3/mmlue/logits_mdf_*) has ZERO E options in any
# rendered prompt, a null template field, invalid rates up to 71%, and no
# alpha=+6 cell at all, so it cannot be cited as "RR 0/2/4/6". Per explicit
# user decision (2026-09-08), this stage runs ONLY CC; RR is skipped rather
# than fabricated, and any RR column in the downstream analysis will be
# empty until RR is separately run or a verified matching result is
# supplied. Do NOT pass "RR" to this launcher -- it refuses.
#
# SIMPLIFIED experiment (still in force): CC uses the new Confidence NMD mask
# AS-IS, no direction/support decomposition, no norm-matching, no RC/CR, no
# random-support control.
#
# STANDALONE: does NOT touch run_hidden_mmlue_confidence_hs.sh,
# run_mmlue_qwen25.sh, get_answer_regenerate_logits.py, the existing Role
# mask, or any existing result.
#
# The Confidence mask (confidence_0.5_11_20_8B.npy) must already be built
# locally (build_confidence_nmd_mask_for_steering.py) and synced to
# ${MASK_DIR} on the server BEFORE this launcher is run.
#
# FORMAL DOSE SET (revised 2026-09-08): alpha in {0, +2, +4, +6}. NEGATIVE
# doses (-2, -4) are deliberately NOT run this round -- they are for
# verifying bidirectional control, not needed for this round's positive-
# effect RSN-paper-style comparison. Baseline (alpha=0) is run ONCE via
# `bash run_rr_cc_mmlue.sh baseline`.
#
# NEGATIVE ALPHA LIST (not used this round, kept for reference): would be
# passed as --alphas="-4,-2,2,4" (the '=' form) so argparse does not mistake
# a leading '-' for a new flag. This round's doses are all non-negative, so
# the '=' form is not strictly required, but is used anyway for consistency.
#
# RESUME: get_answer_rr_cc_mmlue.py checks completeness itself (57 tasks
# present, correct sample counts, full confident/unconfident fields, matching
# run_meta). A COMPLETE cell is skipped; a PARTIAL cell has only its
# missing/incomplete tasks re-run; a metadata mismatch is a FATAL refusal.
#
# Usage:
#   bash run_rr_cc_mmlue.sh baseline   # alpha=0 only
#   bash run_rr_cc_mmlue.sh CC         # alpha 2,4,6, using the Confidence mask

CONDITION="${1:?usage: bash run_rr_cc_mmlue.sh baseline-or-CC}"

if [ "${CONDITION}" == "RR" ]; then
    echo "[REFUSE] RR is not run by this launcher this round."
    echo "No verified, protocol-matching RR (alpha=0/2/4/6, MMLU-E) result exists to reuse"
    echo "(the only historical candidate has no E option, a null template, and no alpha=+6"
    echo "cell). Per explicit decision, RR is skipped rather than fabricated. If RR is"
    echo "needed later, run it as its own condition on this same script's underlying"
    echo "get_answer_rr_cc_mmlue.py, which still supports --condition RR."
    exit 1
fi

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
elif [ "${CONDITION}" == "CC" ]; then
    ALPHAS="2,4,6"
else
    echo "[REFUSE] Unknown condition: ${CONDITION} (expected baseline or CC)"
    exit 1
fi

echo "=================================================="
echo "CC MMLU-E | condition=${CONDITION} | alphas=${ALPHAS}"
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
