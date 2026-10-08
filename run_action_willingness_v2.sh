#!/bin/bash
# Willingness Self-Evaluation v2 (protocol willingness-v2), eight tasks.
# Llama-3.1-8B-Instruct, neutral, no CoT, bare string, NMD mask 11-20, prefill last token.
# Doses: alpha = -4, 0, +4 (alpha=0 is RE-RUN under the new prompt).
#
# usage:  bash run_action_willingness_v2.sh smoke-or-full [task ...]
#   smoke  first 20 samples per task -> willingness_v2_smoke (separate dir, throwaway)
#   full   all samples               -> willingness_v2
# Every alpha of a task runs in ONE process on ONE machine/GPU set (paired design).
# Do not set --skip_missing silently: SKIP_MISSING=1 records + skips absent tasks.
# Analysis is OFFLINE: rsync ${OUT_ROOT} to RoleAnswer/ and run analyze_willingness_v2.py there.
set -euo pipefail

MODE="${1:?usage: run_action_willingness_v2.sh smoke-or-full [task ...]}"
shift || true
TASKS="${*:-mmlu mmlupro gpqa arlsat logiqa medqa truthfulqa gsm8k}"

PY="${PY:-python}"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${WORK_DIR}/components"
MASK_DIR="${BASE_DIR}/mask/llama3_non_logits"
CONFIGS="neg4-11-20 0-11-20 4-11-20"

# Original sample files.  mmlu is a DIR of 57 per-subject JSON; the others are single files.
# These are best guesses of the server layout -- the preflight prints n and digest per task;
# verify them against the old runs' counts before trusting the full run.
TASK_FILES=(
  "mmlu=${BASE_DIR}/mmlu"
  "mmlupro=${BASE_DIR}/benchmark/mmlupro_test.json"
  "gpqa=${BASE_DIR}/benchmark/gpqa_train.json"
  "arlsat=${BASE_DIR}/benchmark/arlsat_all.json"
  "logiqa=${BASE_DIR}/benchmark/logiqa_mrc.json"
  "medqa=${BASE_DIR}/benchmark/medqa_source_test.json"
  "truthfulqa=${BASE_DIR}/benchmark/truthfulqa_mc2_validation.json"
  "gsm8k=${BASE_DIR}/benchmark/gsm8k_test_sample.json"
)

case "${MODE}" in
  smoke) OUT_ROOT="${BASE_DIR}/llama3/willingness_v2_smoke"; EXTRA="--limit 20" ;;
  full)  OUT_ROOT="${BASE_DIR}/llama3/willingness_v2";       EXTRA="" ;;
  *) echo "MODE must be smoke or full, got '${MODE}'"; exit 2 ;;
esac
[ "${SKIP_MISSING:-0}" = "1" ] && EXTRA="${EXTRA} --skip_missing"

cd "${WORK_DIR}"
"${PY}" -c "import numpy, torch, transformers" || { echo "interpreter ${PY} lacks numpy/torch/transformers"; exit 3; }
[ -d "${MASK_DIR}" ] || { echo "missing mask dir ${MASK_DIR}"; exit 4; }
if [ "${MODE}" = "full" ] && [ -d "${OUT_ROOT}" ] && [ -n "$(ls -A "${OUT_ROOT}" 2>/dev/null)" ]; then
  echo "NOTE: ${OUT_ROOT} not empty; complete cells with matching provenance are skipped, mismatches abort."
fi

echo "== willingness-v2 ${MODE} | tasks: ${TASKS} | configs: ${CONFIGS} | $(date) =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname)"

"${PY}" get_action_willingness_v2.py \
  --model llama3 --model_dir "${MODEL_DIR}" --size 8B \
  --mask_dir "${MASK_DIR}" --mask_type nmd --percentage 0.5 \
  --configs ${CONFIGS} \
  --tasks ${TASKS} \
  --data_root "${BASE_DIR}" \
  --task_file "${TASK_FILES[@]}" \
  --out_root "${OUT_ROOT}" ${EXTRA}

echo "== done $(date) =="
echo "next: rsync -a server:${OUT_ROOT}/ <local RoleAnswer>/llama3/willingness_v2/ ; python3.10 analyze_willingness_v2.py"
