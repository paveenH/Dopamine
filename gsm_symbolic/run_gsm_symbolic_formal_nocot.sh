#!/bin/bash
# ================ GSM-Symbolic NO-COT formal sweep (4-point robustness) =====
# FORK of run_gsm_symbolic_formal.sh, NOT a flag on it -- the CoT launcher's
# tree is frozen; this writes to a completely separate --ans_root so the two
# result trees can never be confused or overwritten by each other.
#
# Reuses IDENTICAL data/sampling/model/mask/layer-band/generation config/
# scoring as the CoT formal sweep -- the ONLY difference is the prompt drops
# "Let's think step by step." (get_answer_gsm_symbolic_nocot.py calls
# select_templates_gsm8k(suite="default", cot=False, wording="plain") instead
# of cot=True). Same gsm_symbolic_{config}_sample.json (900 items total,
# 300/config, identical sample_id/order/gold to the CoT run) -- this script
# does NOT re-run the loader or re-sample anything.
#
# alpha grid -- IDENTICAL to the CoT sweep, read from the frozen GSM8K record,
# NOT re-searched:
#   llama3   : -6 -4 0 +4   (band 11-20)
#   qwen2.5  : -6  0 +6 +8  (band 16-22)
#
# Output tree: ${BASE_DIR}/${MODEL}/gsm_symbolic_nocot/{main,p1,p2}/mdf_<alpha>/
# -- deliberately NOT "gsm_symbolic" (the CoT tree's actual on-disk name) and
# NOT "answer_gsm_symbolic" (that script's own argparse default) -- both are
# hard-refused by get_answer_gsm_symbolic_nocot.py itself as a second guard.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_symbolic_formal_nocot.sh llama3  > gsms_nocot_llama.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_symbolic_formal_nocot.sh qwen2.5 > gsms_nocot_qwen.log  2>&1 &
#
# Requires: the CoT formal sweep's data_gsm_symbolic.py loader must have
# already been run once (gsm_symbolic_{main,p1,p2}_sample.json must exist) --
# this script does NOT regenerate or touch that file.

set -e
MODEL="${1:?usage: run_gsm_symbolic_formal_nocot.sh llama3-or-qwen2.5}"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ANS_ROOT="${ANS_ROOT:-gsm_symbolic_nocot}"
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
echo "GSM-Symbolic NO-COT FORMAL sweep | ${MODEL} (${SIZE}) | layers ${LS}-${LE}"
echo "configs: ${CONFIGS}"
echo "ans_root: ${ANS_ROOT}"
echo "batch_size: ${BATCH_SIZE}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset)}"
echo "Start: $(date)"
echo "=================================================="

for CFG in main p1 p2; do
  DATA_FILE="benchmark/gsm_symbolic/gsm_symbolic_${CFG}_sample.json"
  echo ""
  echo "########## gsm_config=${CFG} (data: ${DATA_FILE}) ##########"
  ${PY} get_answer_gsm_symbolic_nocot.py \
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
      --ans_root   "${ANS_ROOT}" \
      --max_new_tokens 768 \
      --temperature    0.0 \
      --batch_size     "${BATCH_SIZE}"
  [ $? -eq 0 ] && echo "[✓] ${CFG} done" || { echo "[✗] ${CFG} failed"; exit 1; }
done

echo ""
echo "=================================================="
echo "No-CoT formal sweep finished: $(date)"
echo "Output root: ${BASE_DIR}/${MODEL}/${ANS_ROOT}/{main,p1,p2}/mdf_<alpha>/"
echo "=================================================="
