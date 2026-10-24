#!/bin/bash
# GSM8K Qwen3-8B Bare No-CoT 2048-token budget control (gsm8k-qwen3-bare-budget2048-v1). Only max_new_tokens differs (2048 vs 768).
# Baseline only: nohup env CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_qwen3_bare_budget2048.sh > gsm8k_qwen3_budget2048_a0.log 2>&1 &
# Later add the steering cell: WORKPOINT=<alpha> (existing verified cells are skipped) -> components/qwen3/gsm8k_bare_budget2048_v1/mdf_<alpha>/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
echo "gsm8k-qwen3-bare-budget2048-v1 | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | WORKPOINT=${WORKPOINT:-(none)} | host=$(hostname) | $(date)"
EXTRA=""
[ -n "${WORKPOINT}" ] && EXTRA="--workpoint ${WORKPOINT}"
"${PY}" get_answer_gsm8k_qwen3_bare_budget2048.py --base_dir "${BASE_DIR}" ${EXTRA}
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
