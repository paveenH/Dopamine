#!/bin/bash
# ==================== GSM8K Native Chat Pushy (gsm8k-chat-pushy-v1) ====================
# Usage: nohup env LLAMA_GPU=0 QWEN_GPU=1 bash run_gsm8k_chat_pushy_4cell.sh > gsm8k_chat_pushy_4cell.log 2>&1 &
# Two parallel processes, one GPU each, one model load each:
#   llama3 : alpha 0 then -6        qwen2.5 : alpha 0 then +8
# Protocol: docs/PREREG_GSM8K_CHAT_PUSHY_V1.md. Outputs under components/<model>/gsm8k_chat_pushy_v1/mdf_<alpha>/.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LLAMA_GPU="${LLAMA_GPU:?set LLAMA_GPU to a GPU index}"
QWEN_GPU="${QWEN_GPU:?set QWEN_GPU to a GPU index}"
[ "${LLAMA_GPU}" != "${QWEN_GPU}" ] || { echo "[x] LLAMA_GPU and QWEN_GPU must differ"; exit 1; }
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
"${PY}" -c "import numpy, torch, transformers" || { echo "[x] interpreter '${PY}' lacks numpy/torch/transformers"; exit 1; }

echo "gsm8k-chat-pushy-v1 | llama GPU ${LLAMA_GPU} | qwen GPU ${QWEN_GPU} | host=$(hostname) | $(date)"
CUDA_VISIBLE_DEVICES="${LLAMA_GPU}" "${PY}" get_answer_gsm8k_chat_pushy.py --model llama3 --base_dir "${BASE_DIR}" \
  > gsm8k_chat_pushy_llama3.log 2>&1 &
PID_L=$!
CUDA_VISIBLE_DEVICES="${QWEN_GPU}" "${PY}" get_answer_gsm8k_chat_pushy.py --model qwen2.5 --base_dir "${BASE_DIR}" \
  > gsm8k_chat_pushy_qwen25.log 2>&1 &
PID_Q=$!
wait "${PID_L}"; RC_L=$?
wait "${PID_Q}"; RC_Q=$?
echo "llama3 rc=${RC_L} (gsm8k_chat_pushy_llama3.log) | qwen2.5 rc=${RC_Q} (gsm8k_chat_pushy_qwen25.log) | $(date)"
[ "${RC_L}" -eq 0 ] && [ "${RC_Q}" -eq 0 ]
