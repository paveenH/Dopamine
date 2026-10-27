#!/bin/bash
# Granite-3.3-8B Bare No-CoT transfer (granite33-bare-transfer-v1): GSM-Hard and MATH, alpha 0 -6 -4 +6, split over TWO identical GPUs (one model load per process).
# Usage: nohup env GPUS="0 1" PY=python bash run_granite33_bare_transfer.sh > granite33_bare_transfer.log 2>&1 &
# GPU1: GSM-Hard {0,-6,-4,6} then MATH {0};  GPU2: MATH {-6,-4,6}. Logs: granite33_bare_transfer.gpu<N>.log
# -> components/granite33/gsm_hard_bare_transfer_v1/mdf_<alpha>/  components/granite33/math_bare_transfer_v1/mdf_<alpha>/
# Cells of one task may come from different GPUs (cross-run; recorded in meta, only warned).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
GPUS="${GPUS:-0 1}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
set -- ${GPUS}
[ $# -eq 2 ] || { echo "[x] GPUS must list exactly two ids, got [${GPUS}]"; exit 1; }
G1=$1; G2=$2
echo "granite33-bare-transfer-v1 | GPUS=[${GPUS}] | host=$(hostname) | $(date)"
( export CUDA_VISIBLE_DEVICES=$G1
  "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task gsm_hard --alphas 0 -6 -4 6 &&
  "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task math --alphas 0 ) > "granite33_bare_transfer.gpu${G1}.log" 2>&1 &
p1=$!
( export CUDA_VISIBLE_DEVICES=$G2
  "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task math --alphas -6 -4 6 ) > "granite33_bare_transfer.gpu${G2}.log" 2>&1 &
p2=$!
rc=0
wait $p1 || rc=1
wait $p2 || rc=1
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED (see granite33_bare_transfer.gpu*.log)"
exit $rc
