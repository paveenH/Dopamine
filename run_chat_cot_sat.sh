#!/bin/bash
# ==================== Chat-CoT Saturation Check (chat-cot-sat-v1) ====================
# Usage: bash run_chat_cot_sat.sh MODEL TASK STEP     (no braces in this usage text on purpose)
#   MODEL: llama3 | qwen2.5        TASK: gsm8k | gsm_hard        STEP: check | run | status
# Cells per (MODEL,TASK): alpha 0 and the frozen workpoint (llama3 -6, qwen2.5 +8), ONE model load, Native Chat + CoT line,
# greedy, 768 tokens, bs=24, prefill-only tail=1. Protocol: docs/PREREG_CHAT_COT_SAT_V1.md.
#   check  : loads tokenizer+model, verifies prompts/tokens/mask/existing cells, writes NOTHING
#   run    : 300 items x 2 alphas into <task>_chat_cot_sat_v1 (verified cells are skipped, unverifiable ones abort)
#   status : which cells exist (no GPU)
# GPU placement is NOT constrained; host / CUDA_VISIBLE_DEVICES / sharding are recorded in each cell.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$1"; TASK="$2"; STEP="$3"
case "${MODEL}" in llama3|qwen2.5) ;; *) echo "[x] MODEL must be llama3 or qwen2.5 (got '${MODEL}')"; exit 1;; esac
case "${TASK}"  in gsm8k|gsm_hard) ;; *) echo "[x] TASK must be gsm8k or gsm_hard (got '${TASK}')"; exit 1;; esac
case "${STEP}"  in check|run|status) ;; *) echo "[x] STEP must be check, run or status (got '${STEP}')"; exit 1;; esac

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

if [ "${STEP}" = "status" ]; then
  for d in "${TASK}_chat_cot_sat_v1"; do
    echo "== ${BASE_DIR}/${MODEL}/${d}"; ls -1 "${BASE_DIR}/${MODEL}/${d}"/mdf_*/*.json 2>/dev/null || echo "   (no cells)"
  done
  exit 0
fi

"${PY}" -c "import numpy, torch, transformers" || { echo "[x] interpreter '${PY}' lacks numpy/torch/transformers"; exit 1; }
echo "chat-cot-sat-v1 | ${MODEL} | ${TASK} | ${STEP} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
"${PY}" get_answer_chat_cot_sat.py --model "${MODEL}" --task "${TASK}" --mode "${STEP}" --base_dir "${BASE_DIR}"
rc=$?
[ $rc -eq 0 ] && echo "[ok] ${STEP} finished $(date)" || echo "[x] ${STEP} FAILED rc=${rc}"
exit $rc
