#!/bin/bash
# ==================== GSM-Symbolic preflight (alpha=0 only) ===================
# 30-item preflight (10 per config: main/p1/p2), alpha=0, both models.
# Checks data loading, prompt, gold parsing, model answer parsing, injection
# site (fires must read 0 at alpha=0), no-answer/truncation. Stops here --
# the formal sweep is a SEPARATE, manually-invoked step after this is reviewed.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm_symbolic_preflight.sh llama3
#   CUDA_VISIBLE_DEVICES=1 bash run_gsm_symbolic_preflight.sh qwen2.5
#
# Both models read the SAME preflight file
# (components/benchmark/gsm_symbolic/gsm_symbolic_preflight_30.json), built by
# data_gsm_symbolic.py -- so "same samples for both models" is a data-file
# property, not something this launcher has to enforce.
#
# BATCH_SIZE matches the formal sweep's default (24) -- kept as ONE shared
# knob for this whole script, not per-config. If a config OOMs, lower
# BATCH_SIZE here (or via the env var) for the WHOLE model and re-run the
# FULL preflight (all 3 configs) again -- never drop it only for the
# offending config, which would leave the preflight testing a different
# batch size than the rest of the run and than the formal sweep.

set -e
MODEL="${1:?usage: run_gsm_symbolic_preflight.sh llama3-or-qwen2.5}"

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
    ;;
  qwen2.5)
    MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    SIZE="7B"; HS="qwen2.5"
    LS=16; LE=22
    ;;
  *)
    echo "[✗] unknown model ${MODEL}"; exit 1
    ;;
esac

DATA_FILE="benchmark/gsm_symbolic/gsm_symbolic_preflight_30.json"

echo "=================================================="
echo "GSM-Symbolic PREFLIGHT | ${MODEL} (${SIZE}) | layers ${LS}-${LE} | alpha=0"
echo "data: ${BASE_DIR}/${DATA_FILE}"
echo "batch_size: ${BATCH_SIZE} (same knob as the formal sweep)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset)}"
echo "Start: $(date)"
echo "=================================================="

for CFG in main p1 p2; do
  echo ""
  echo "---------- config=${CFG} ----------"
  ${PY} get_answer_gsm_symbolic.py \
      --model      "${MODEL}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS}" \
      --size       "${SIZE}" \
      --type       "non" \
      --percentage 0.5 \
      --mask_type  "nmd" \
      --configs    "0-${LS}-${LE}" \
      --gsm_config "${CFG}" \
      --data_file  "${DATA_FILE}" \
      --preflight \
      --base_dir   "${BASE_DIR}" \
      --ans_root   "answer_gsm_symbolic_preflight" \
      --max_new_tokens 768 \
      --temperature    0.0 \
      --batch_size     "${BATCH_SIZE}"
done

echo ""
echo "=================================================="
echo "Preflight finished: $(date)"
echo "Inspect: ${BASE_DIR}/${MODEL}/answer_gsm_symbolic_preflight/{main,p1,p2}/mdf_0/"
echo "=================================================="
