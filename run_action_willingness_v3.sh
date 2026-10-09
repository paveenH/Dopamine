#!/bin/bash
# Willingness Self-Evaluation v3 (protocol willingness-v3): 8 tasks x 300 fixed items x 9 doses.
# Llama-3.1-8B-Instruct, neutral, no CoT, bare string, NMD mask 11-20, prefill last-token injection.
#
# usage:  bash run_action_willingness_v3.sh MODE [task ...]
#   MODE = smoke   3 items per task from the FIXED 300-item list, all 9 doses, in this process,
#                  output /tmp/willingness_v3_smoke (throwaway, separate from the formal tree)
#          full    given tasks (default all 8) in THIS process = one worker
#          full3   3 background workers (GPUS="0 1 2"): GPU a = mmlupro medqa gsm8k,
#                  GPU b = mmlu logiqa truthfulqa, GPU c = gpqa arlsat; plus a watcher that
#                  writes the final summary.  A task's nine doses always stay on ONE worker/GPU.
#          status  per-worker exit codes, FATAL lines, cells done (of 72); exit 0 only if all done
# Formal results: ${WORK_DIR}/components/llama3/{task}/mdf_{alpha}/willingness_v3_{task}_8B_11_20.json
# Logs: ${WORK_DIR}/wv3_logs/.  Analysis is OFFLINE (rsync components/llama3 to RoleAnswer, then
# analyze_willingness_v3.py).
set -euo pipefail

MODE="${1:?usage: run_action_willingness_v3.sh smoke-or-full-or-full3-or-status [task ...]}"
shift || true
ALL_TASKS="mmlu mmlupro gpqa arlsat logiqa medqa truthfulqa gsm8k"
TASKS="${*:-${ALL_TASKS}}"

PY="${PY:-python}"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${WORK_DIR}/components"
MASK_DIR="${BASE_DIR}/mask/llama3_non_logits"
FORMAL_ROOT="${BASE_DIR}/llama3"
LOG_DIR="${WORK_DIR}/wv3_logs"
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
GROUPS_SPEC=("mmlupro medqa gsm8k" "mmlu logiqa truthfulqa" "gpqa arlsat")

# The exact files the v2 run used (digests are re-checked against the v2 metadata by the runner).
TASK_FILES=(
  "mmlu=${BASE_DIR}/mmlu"
  "mmlupro=${BASE_DIR}/benchmark/mmlupro_test.json"
  "gpqa=${BASE_DIR}/benchmark/gpqa_train.json"
  "arlsat=${BASE_DIR}/benchmark/arlsat_all.json"
  "logiqa=${BASE_DIR}/benchmark/logiqa_mrc.json"
  "medqa=${BASE_DIR}/benchmark/medqa_source_test.json"
  "truthfulqa=${BASE_DIR}/benchmark/truthfulqa_mc1_validation.json"
  "gsm8k=${BASE_DIR}/benchmark/gsm8k_test_sample.json"
)

count_cells() {  # count_cells task -> number of finished v3 cells of that task
  { ls "${FORMAL_ROOT}/$1"/mdf_*/willingness_v3_"$1"_8B_11_20.json 2>/dev/null || true; } | wc -l | tr -d ' '
}

do_status() {
  local ok=1 total=0 i=0
  echo "== willingness-v3 status $(date) =="
  for g in "${GROUPS_SPEC[@]}"; do
    local ef="${LOG_DIR}/worker_g${i}.exit" lf="${LOG_DIR}/worker_g${i}.log" st
    if [ -f "${ef}" ]; then st="exit=$(cat "${ef}")"; [ "$(cat "${ef}")" = "0" ] || ok=0
    elif [ -f "${lf}" ]; then st="RUNNING-or-killed (no exit file)"; ok=0
    else st="NOT STARTED"; ok=0; fi
    echo "worker g${i} [${g}]: ${st}"
    for t in ${g}; do
      local c; c="$(count_cells "${t}")"; total=$((total + c))
      echo "    ${t}: ${c}/9 cells"; [ "${c}" = "9" ] || ok=0
    done
    [ -f "${lf}" ] && grep -nE "FATAL|Traceback|Error|steering_fires" "${lf}" | tail -n 5 | sed 's/^/    ! /' || true
    i=$((i + 1))
  done
  echo "cells done: ${total}/72"
  [ "${ok}" = "1" ] && { echo "ALL WORKERS OK"; return 0; } || { echo "NOT COMPLETE / FAILED"; return 1; }
}

case "${MODE}" in
  status) do_status; exit $? ;;
  watch)  # internal: wait for the three exit files, then write the summary
    while true; do
      n=0; for j in 0 1 2; do [ -f "${LOG_DIR}/worker_g${j}.exit" ] && n=$((n + 1)); done
      [ "${n}" = "3" ] && break; sleep 60
    done
    do_status > "${LOG_DIR}/summary.txt" 2>&1 || true; cat "${LOG_DIR}/summary.txt"; exit 0 ;;
  full3)
    read -r G0 G1 G2 <<< "${GPUS:-0 1 2}"
    GPU_IDS=("${G0}" "${G1}" "${G2}")
    mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/worker_g*.exit "${LOG_DIR}/summary.txt"
    for i in 0 1 2; do
      tk="${GROUPS_SPEC[$i]}"
      LOGF="${LOG_DIR}/worker_g${i}.log" EXITF="${LOG_DIR}/worker_g${i}.exit" \
      CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" \
        nohup bash -c 'bash "$1" full ${@:2} > "$LOGF" 2>&1; echo $? > "$EXITF"' _ "${SELF}" ${tk} > /dev/null 2>&1 &
      echo "worker g${i}: GPU ${GPU_IDS[$i]} tasks [${tk}] pid $! -> ${LOG_DIR}/worker_g${i}.log"
    done
    nohup bash "${SELF}" watch > "${LOG_DIR}/watch.log" 2>&1 &
    echo "watcher pid $! ; check:  bash ${SELF} status"
    exit 0 ;;
  smoke) OUT_ROOT="/tmp/willingness_v3_smoke"; EXTRA="--limit 3" ;;
  full)  OUT_ROOT="${FORMAL_ROOT}"; EXTRA="" ;;
  *) echo "MODE must be smoke, full, full3 or status, got '${MODE}'"; exit 2 ;;
esac

cd "${WORK_DIR}"
"${PY}" -c "import numpy, torch, transformers" || { echo "interpreter ${PY} lacks numpy/torch/transformers"; exit 3; }
[ -d "${MASK_DIR}" ] || { echo "missing mask dir ${MASK_DIR}"; exit 4; }
for kv in "${TASK_FILES[@]}"; do   # existence check for the requested tasks only
  name="${kv%%=*}"; path="${kv#*=}"
  case " ${TASKS} " in *" ${name} "*) [ -e "${path}" ] || { echo "missing data for ${name}: ${path}"; exit 5; } ;; esac
done
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

echo "== willingness-v3 ${MODE} | tasks: ${TASKS} | out: ${OUT_ROOT} | $(date) =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname)"
"${PY}" get_action_willingness_v3.py \
  --model llama3 --model_dir "${MODEL_DIR}" --size 8B \
  --mask_dir "${MASK_DIR}" --mask_type nmd --percentage 0.5 \
  --configs ${CONFIGS} --tasks ${TASKS} \
  --data_root "${BASE_DIR}" --task_file "${TASK_FILES[@]}" \
  --out_root "${OUT_ROOT}" ${EXTRA}
echo "== done $(date) =="
