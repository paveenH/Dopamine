#!/bin/bash
# GSM8K Mistral Native Chat No-CoT nine-point sweep (gsm8k-mistral-chat-sweep-v1), one GPU, one model load.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_mistral_chat_sweep.sh > gsm8k_mistral_chat_sweep.log 2>&1 &
# alpha order 0 -8 -6 -4 -2 2 4 6 8 -> components/mistral/gsm8k_chat_sweep_v1/mdf_<alpha>/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
"${PY}" -c "import numpy, torch, transformers" || { echo "[x] interpreter '${PY}' lacks numpy/torch/transformers"; exit 1; }
echo "gsm8k-mistral-chat-sweep-v1 | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
"${PY}" get_answer_gsm8k_mistral_chat_sweep.py --base_dir "${BASE_DIR}"
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
