#!/bin/bash
# Qwen3-8B Bare No-CoT fixed-workpoint transfer (qwen3-bare-transfer-v1): GSM-Hard {0,+10} then MATH {0,+8,+10}, one GPU, one env.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 PY=python bash run_qwen3_bare_transfer.sh > qwen3_bare_transfer.log 2>&1 &
# -> components/qwen3/gsm_hard_bare_transfer_v1/mdf_{0,10}/  components/qwen3/math_bare_transfer_v1/mdf_{0,8,10}/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
echo "qwen3-bare-transfer-v1 | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
for T in gsm_hard math; do
  "${PY}" get_answer_qwen3_bare_transfer.py --base_dir "${BASE_DIR}" --task "${T}" || { echo "[x] FAILED task=${T}"; exit 1; }
done
echo "[ok] finished $(date)"
