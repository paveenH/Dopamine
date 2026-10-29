#!/bin/bash
# GSM8K Granite-3.3-8B-Instruct Bare No-CoT seven-point sweep (gsm8k-granite33-bare-sweep-v1), split over TWO identical GPUs (one process each, one model load each).
# Usage: nohup env GPUS="0 1" PY=python bash run_gsm8k_granite33_bare_sweep.sh > gsm8k_granite33_bare_sweep.log 2>&1 &
# Split: first GPU = alpha 0 -6 -8 -10, second GPU = alpha 6 8 10 (override with SPLIT_A / SPLIT_B, e.g. SPLIT_A="-4 -2" SPLIT_B="2 4"). Logs: gsm8k_granite33_bare_sweep.gpu<N>.log. Cells land in components/granite33/gsm8k_bare_sweep_v1/mdf_<alpha>/.
# Per-cell GPU/host/CUDA_VISIBLE_DEVICES are recorded in meta (cross-run provenance); the mask file is built identically by both processes (atomic write).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
GPUS="${GPUS:-0 1}"
SPLITS=("${SPLIT_A:-0 -6 -8 -10}" "${SPLIT_B:-6 8 10}")
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
set -- ${GPUS}
[ $# -eq 2 ] || { echo "[x] GPUS must list exactly two ids, got [${GPUS}]"; exit 1; }
echo "gsm8k-granite33-bare-sweep-v1 | GPUS=[${GPUS}] | host=$(hostname) | $(date)"
pids=()
i=0
for g in "$@"; do
  CUDA_VISIBLE_DEVICES=$g "${PY}" get_answer_gsm8k_granite33_bare_sweep.py --base_dir "${BASE_DIR}" --alphas ${SPLITS[$i]} > "gsm8k_granite33_bare_sweep.gpu${g}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED (see gsm8k_granite33_bare_sweep.gpu*.log)"
exit $rc
