#!/bin/bash
# Granite-3.3-8B Bare No-CoT fixed-dose transfer (granite33-bare-transfer-v1): GSM-Hard {0,-6} and MATH {0,-6} in parallel on two GPUs, one model load each.
# Usage: nohup env GSM_HARD_GPU=0 MATH_GPU=1 PY=python bash run_granite33_bare_transfer.sh > granite33_bare_transfer.log 2>&1 &
# -> components/granite33/gsm_hard_bare_transfer_v1/mdf_{0,-6}/  components/granite33/math_bare_transfer_v1/mdf_{0,-6}/ ; per-task logs granite33_bare_transfer.<task>.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
GSM_HARD_GPU="${GSM_HARD_GPU:-0}"
MATH_GPU="${MATH_GPU:-1}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
echo "granite33-bare-transfer-v1 | gsm_hard GPU=${GSM_HARD_GPU} math GPU=${MATH_GPU} | host=$(hostname) | $(date)"
CUDA_VISIBLE_DEVICES="${GSM_HARD_GPU}" "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task gsm_hard > granite33_bare_transfer.gsm_hard.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES="${MATH_GPU}" "${PY}" get_answer_granite33_bare_transfer.py --base_dir "${BASE_DIR}" --task math > granite33_bare_transfer.math.log 2>&1 &
p2=$!
rc=0
wait $p1 || { echo "[x] gsm_hard FAILED"; rc=1; }
wait $p2 || { echo "[x] math FAILED"; rc=1; }
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED (see granite33_bare_transfer.*.log)"
exit $rc
