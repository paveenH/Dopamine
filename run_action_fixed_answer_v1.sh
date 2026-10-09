#!/bin/bash
# Fixed-Answer Reliability and Submission Pilot (protocol fixed-answer-v1).
# Llama-3.1-8B-Instruct, MMLU pooled, subject-specific Non-expert role, bare string, NMD mask 11-20,
# prefill last-token injection, alpha -4/0/+4, ONE process (all conditions same machine/GPU; device is recorded, not enforced).
#
# usage:  bash run_action_fixed_answer_v1.sh pilot-or-full
#   pilot  25 questions NOT in the formal 300 -> /tmp/fixed_answer_v1_pilot   (throwaway, separate dir)
#   full   formal 300 questions               -> ${WORK_DIR}/components/llama3/fixed_answer_v1
# Re-running resumes only provenance-identical cells; any mismatch aborts.
# Analysis is OFFLINE: rsync the out dir to RoleAnswer and run analyze_fixed_answer_v1.py there.
set -euo pipefail

MODE="${1:?usage: run_action_fixed_answer_v1.sh pilot-or-full}"
PY="${PY:-python}"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${WORK_DIR}/components"
MASK_DIR="${BASE_DIR}/mask/llama3_non_logits"
MMLU_DIR="${BASE_DIR}/mmlu"
CONFIGS="neg4-11-20 0-11-20 4-11-20"

case "${MODE}" in
  pilot) PMODE="pilot";  OUT_ROOT="${OUT_ROOT:-/tmp/fixed_answer_v1_pilot}" ;;
  full)  PMODE="formal"; OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/llama3/fixed_answer_v1}" ;;
  *) echo "MODE must be pilot or full, got '${MODE}'"; exit 2 ;;
esac

cd "${WORK_DIR}"
"${PY}" -c "import numpy, torch, transformers" || { echo "interpreter ${PY} lacks numpy/torch/transformers"; exit 3; }
[ -d "${MASK_DIR}" ] || { echo "missing mask dir ${MASK_DIR}"; exit 4; }
[ -d "${MMLU_DIR}" ] || { echo "missing MMLU dir ${MMLU_DIR}"; exit 5; }
mkdir -p "${OUT_ROOT}"

echo "== fixed-answer-v1 ${MODE} | out: ${OUT_ROOT} | $(date) =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname)"
"${PY}" get_action_fixed_answer_v1.py \
  --mode "${PMODE}" --model llama3 --model_dir "${MODEL_DIR}" --size 8B \
  --mask_dir "${MASK_DIR}" --mask_type nmd --percentage 0.5 \
  --configs ${CONFIGS} --mmlu_dir "${MMLU_DIR}" --out_root "${OUT_ROOT}"
echo "== done $(date) =="
echo "next: rsync -a server:${OUT_ROOT}/ <local RoleAnswer>/AdarResult/llama3/fixed_answer_v1/ ; python3.10 analyze_fixed_answer_v1.py --root <that dir>"
