#!/bin/bash
# GSM8K Base-checkpoint check (gsm8k-base-qwen3-granite-v1): one model per process, alpha 0 + instruct workpoint, one model load.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base_qwen3_granite.sh qwen3 > tmp/gsm8k_base_qwen3.log 2>&1 &
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${1:?usage: run_gsm8k_base_qwen3_granite.sh qwen3-or-granite33}"
case "${MODEL}" in qwen3|granite33) ;; *) echo "[x] model must be qwen3 or granite33, got ${MODEL}"; exit 1;; esac
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
echo "gsm8k-base-qwen3-granite-v1 ${MODEL} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
"${PY}" get_answer_gsm8k_base_qwen3_granite.py --base_dir "${BASE_DIR}" --model "${MODEL}"
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
