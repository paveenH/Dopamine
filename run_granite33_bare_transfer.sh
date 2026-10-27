#!/bin/bash
# Granite-3.3-8B Bare No-CoT transfer (granite33-bare-transfer-v1): GSM-Hard and MATH, alpha 0 -6 -4 +6, split over FOUR identical GPUs.
# Usage: nohup env GPUS="0 1 2 3" PY=python bash run_granite33_bare_transfer.sh > granite33_bare_transfer.log 2>&1 &
# GPU i takes ONE alpha (order 0 -6 -4 6) for both tasks: GSM-Hard cell then MATH cell (one model load each). Logs: granite33_bare_transfer.gpu<N>.log
# -> components/granite33/gsm_hard_bare_transfer_v1/mdf_<alpha>/  components/granite33/math_bare_transfer_v1/mdf_<alpha>/
# Every contrast vs alpha 0 is therefore cross-GPU (recorded in meta, only warned).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
GPUS="${GPUS:-0 1 2 3}"
ALPHAS=(0 -6 -4 6)
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
set -- ${GPUS}
[ $# -eq 4 ] || { echo "[x] GPUS must list exactly four ids, got [${GPUS}]"; exit 1; }
echo "granite33-bare-transfer-v1 | GPUS=[${GPUS}] | host=$(hostname) | $(date)"
pids=()
i=0
for g in "$@"; do
  a=${ALPHAS[$i]}
  ( export CUDA_VISIBLE_DEVICES=$g
    "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task gsm_hard --alphas $a &&
    "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task math --alphas $a ) > "granite33_bare_transfer.gpu${g}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED (see granite33_bare_transfer.gpu*.log)"
exit $rc
