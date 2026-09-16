#!/bin/bash
set -euo pipefail

# ==================== RRSN -> MMLU-E reverse-transfer steering ====================
# Question: does RRSN (Reasoning RSN), reduced to its exact-NMD sparse support
# over this model's EXISTING MMLU-E-matched band and per-layer L2-norm-matched
# to this model's own MRSN exact-NMD support (the SAME norm-matched mask
# already deployed for the RRSN->GSM8K positive control), change MMLU-E
# accuracy / E-rate / conditional accuracy / role gap under steering?
#
# This is CROSS-TASK REVERSE transfer (MMLU-E did not participate in RRSN's
# construction) and must be read jointly with the RRSN->GSM8K positive
# control's result -- see get_answer_rrsn_mmlue.py's module docstring for the
# full interpretation boundary.
#
# FROZEN THIS ROUND (explicit user decision, do not widen without a new
# decision): alpha in {-4, 0, +4}, run on BOTH llama3 and qwen2.5. This is a
# NEW, independently frozen pilot grid -- it does NOT claim to be, and is not
# read from, any historical "formal" MRSN->MMLU-E grid (none was located with
# a verified matching protocol; run_mmlue_qwen25.sh, the only candidate with
# the right role/E-option wording, was NEVER actually run -- zero artifacts).
#
# Prompt: the ORIGINAL, UNMODIFIED get_answer_regenerate_logits.py /
# utils.construct_prompt() bare-string path, matching run_mmlue_qwen25.sh's
# exact configuration (suite=default, use_E=True,
# roles={task} expert / non {task} expert, use_chat=False). Both roles
# deliberately render "an honest {role}" -- NOT a bug, NOT patched, per
# explicit decision (see get_answer_rrsn_mmlue.py docstring).
#
# STANDALONE: does NOT touch run_mmlue_qwen25.sh, get_answer_regenerate_logits.py,
# cross_steering_confidence/ (RR/CC), the RRSN->GSM8K positive control tree,
# or any existing MMLU/MMLU-E/MRSN result. Writes to its own output tree only.
#
# The RRSN norm-matched mask must already be deployed at
# ${MASK_DIR}/rrsn_normmatched_0.5_<start>_<end>_<size>.npy on this machine
# (same file already used by the RRSN->GSM8K positive control -- see
# RoleHidden/build_rrsn_gsm8k_positive_control_mask.py and
# gsm8k_rrsn_positive_control/RUNBOOK.md for the local-build vs
# server-deployment distinction). This launcher does NOT rebuild it.
#
# RESUME: get_answer_rrsn_mmlue.py checks completeness itself (57 tasks
# present, correct sample counts, full expert/non-expert fields, matching
# run_meta). A COMPLETE cell is skipped; a PARTIAL cell has only its
# missing/incomplete tasks re-run; a metadata mismatch is a FATAL refusal,
# never a silent resume.
#
# alpha=0 is RE-RUN fresh in this experiment's own tree for BOTH models --
# never copied from any historical MRSN or RR/CC cell.
#
# *** SAME-MACHINE / SAME-GPU RULE *** bf16 greedy is not byte-reproducible
# across GPUs. All three alphas of ONE model must run on ONE card.
#
# ==================== Steps ====================
#   bash run_rrsn_mmlue.sh verify   <model>
#       renders and prints/saves one expert + one non-expert prompt. No model
#       load, no generation, no GPU.
#   bash run_rrsn_mmlue.sh full     <model>
#       runs the full frozen grid {-4,0,4} for that model, resumable.
#
# Output: components/{model}/rrsn_mmlue/alpha_<alpha>/
#   {task}_{size}_answers.json   (57 files)
#   run_meta_{size}.json

STEP="${1:?usage: bash run_rrsn_mmlue.sh {verify|full} {llama3|qwen2.5}}"
MODEL="${2:?usage: bash run_rrsn_mmlue.sh {verify|full} {llama3|qwen2.5}}"

if [[ "${MODEL}" != "llama3" && "${MODEL}" != "qwen2.5" ]]; then
  echo "[REFUSE] Unknown model: ${MODEL} (expected llama3 or qwen2.5)"
  exit 1
fi

if [[ "${MODEL}" == "llama3" ]]; then
  MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
  SIZE="8B"
else
  MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
  SIZE="7B"
fi

DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MASK_DIR="${BASE_DIR}/mask/${MODEL}_non_logits"
MMLU_DIR="${BASE_DIR}/mmlu"
OUT_ROOT="${BASE_DIR}/${MODEL}/rrsn_mmlue"
PY="${PY:-python}"

cd "${WORK_DIR}" || { echo "[✗] cannot cd ${WORK_DIR}"; exit 1; }

case "${STEP}" in
  verify)
    echo "=================================================="
    echo "RRSN -> MMLU-E | verify prompt rendering | ${MODEL}"
    echo "=================================================="
    ${PY} cross_steering_reasoning/get_answer_rrsn_mmlue.py \
        --model      "${MODEL}" \
        --model_dir  "${MODEL_DIR}" \
        --size       "${SIZE}" \
        --mask_dir   "${MASK_DIR}" \
        --mmlu_dir   "${MMLU_DIR}" \
        --out_root   "${OUT_ROOT}" \
        --verify_only
    ;;

  full)
    ALPHAS="-4,0,4"
    echo "=================================================="
    echo "RRSN -> MMLU-E | ${MODEL} | alphas=${ALPHAS}"
    echo "Mask dir : ${MASK_DIR}"
    echo "MMLU dir : ${MMLU_DIR}"
    echo "Out root : ${OUT_ROOT}"
    echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"
    echo "Start    : $(date)"
    echo "=================================================="
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
      echo "[warn] CUDA_VISIBLE_DEVICES is unset. Pin ONE card for all three alphas."
    fi
    ${PY} cross_steering_reasoning/get_answer_rrsn_mmlue.py \
        --model      "${MODEL}" \
        --model_dir  "${MODEL_DIR}" \
        --size       "${SIZE}" \
        --mask_dir   "${MASK_DIR}" \
        --mmlu_dir   "${MMLU_DIR}" \
        --out_root   "${OUT_ROOT}" \
        --alphas="${ALPHAS}"
    rc=$?
    echo ""
    if [ $rc -eq 0 ]; then
      echo "[✓] ${MODEL} finished: $(date)"
    else
      echo "[✗] ${MODEL} FAILED (rc=$rc): $(date)"
    fi
    exit $rc
    ;;

  *)
    echo "Usage: bash run_rrsn_mmlue.sh {verify|full} {llama3|qwen2.5}"
    echo ""
    echo "  verify  render + print/save one expert + one non-expert prompt, no GPU"
    echo "  full    the frozen {-4,0,4} grid for that model, resumable, one card"
    exit 1
    ;;
esac
