#!/bin/bash
# ==================== GSM-Symbolic formal sweep ================================
# FULL OFFICIAL test split of all 3 configs (main/p1/p2) -- NOT a fixed
# 300-item sample. Measured sizes (Hub dataset-viewer, 2026-09): main ~1319,
# p1 ~5000, p2 ~2500 rows -> ~8819 items x 4 alphas = ~35,276 generations for
# ONE model, ~70,552 for both. This is a real cost, not a rounding difference
# from "300 x 8" -- budget GPU time accordingly before launching.
#
# 4-point alpha sweep, one model, ONE GPU, sequential -- so the whole
# per-model curve is byte-comparable (bf16 greedy is not reproducible across
# GPUs/cards).
#
# alpha grid (fixed, read from the frozen GSM8K record -- NOT re-searched):
#   llama3   : -6 -4 0 +4   (band 11-20)
#   qwen2.5  : -6  0 +6 +8  (band 16-22)
#
# BATCH_SIZE is ONE shared knob per model, applied uniformly to every
# config/alpha for that model (same convention as run_gsm_symbolic_preflight.sh).
# If a cell OOMs, lower it here (or via the env var) for the WHOLE model and
# re-run that model's FULL sweep -- never drop it only for the offending
# config/alpha, which would make cells within one model's curve incomparable.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_symbolic_formal.sh llama3  > gsms_llama.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_symbolic_formal.sh qwen2.5 > gsms_qwen.log  2>&1 &
#
# Only run this AFTER the preflight has been reviewed and approved.

set -e
MODEL="${1:?usage: run_gsm_symbolic_formal.sh {llama3|qwen2.5}}"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
BATCH_SIZE="${BATCH_SIZE:-24}"
cd "${WORK_DIR}/gsm_symbolic" || { echo "[✗] cannot cd ${WORK_DIR}/gsm_symbolic"; exit 1; }

case "${MODEL}" in
  llama3)
    MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
    SIZE="8B"; HS="llama3"
    LS=11; LE=20
    CONFIGS="neg6-${LS}-${LE} neg4-${LS}-${LE} 0-${LS}-${LE} 4-${LS}-${LE}"
    ;;
  qwen2.5)
    MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    SIZE="7B"; HS="qwen2.5"
    LS=16; LE=22
    CONFIGS="neg6-${LS}-${LE} 0-${LS}-${LE} 6-${LS}-${LE} 8-${LS}-${LE}"
    ;;
  *)
    echo "[✗] unknown model ${MODEL}"; exit 1
    ;;
esac

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "[warn] CUDA_VISIBLE_DEVICES is unset -- llms.py device_map=auto will"
  echo "       claim every visible card. Pin ONE card for the whole curve."
fi

echo "=================================================="
echo "GSM-Symbolic FORMAL sweep | ${MODEL} (${SIZE}) | layers ${LS}-${LE}"
echo "configs: ${CONFIGS}"
echo "batch_size: ${BATCH_SIZE}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset)}"
echo "Start: $(date)"
echo "=================================================="

for CFG in main p1 p2; do
  DATA_FILE="benchmark/gsm_symbolic/gsm_symbolic_${CFG}_test.json"
  echo ""
  echo "########## gsm_config=${CFG} (data: ${DATA_FILE}) ##########"
  ${PY} get_answer_gsm_symbolic.py \
      --model      "${MODEL}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS}" \
      --size       "${SIZE}" \
      --type       "non" \
      --percentage 0.5 \
      --mask_type  "nmd" \
      --configs    ${CONFIGS} \
      --gsm_config "${CFG}" \
      --data_file  "${DATA_FILE}" \
      --base_dir   "${BASE_DIR}" \
      --ans_root   "answer_gsm_symbolic" \
      --max_new_tokens 768 \
      --temperature    0.0 \
      --batch_size     "${BATCH_SIZE}"
  [ $? -eq 0 ] && echo "[✓] ${CFG} done" || { echo "[✗] ${CFG} failed"; exit 1; }
done

echo ""
echo "=================================================="
echo "Formal sweep finished: $(date)"
echo "Output root: ${BASE_DIR}/${MODEL}/answer_gsm_symbolic/{main,p1,p2}/mdf_<alpha>/"
echo "=================================================="
