#!/bin/bash
# GSM8K Qwen3-8B Bare No-CoT sweep continuation (alpha -8 8 10) (gsm8k-qwen3-bare-sweep-v1); builds the NMD mask from Qwen3 means first, one GPU, one model load.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_qwen3_bare_sweep.sh > gsm8k_qwen3_bare_sweep_cont3.log 2>&1 &
# alpha order -8 8 10 (earlier cells 0 -4 4 -6 6 untouched) -> components/qwen3/gsm8k_bare_sweep_v1/mdf_<alpha>/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
echo "gsm8k-qwen3-bare-sweep-v1 | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
"${PY}" get_answer_gsm8k_qwen3_bare_sweep.py --base_dir "${BASE_DIR}"
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
