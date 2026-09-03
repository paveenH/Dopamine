#!/usr/bin/env bash
# workpoint-stability supplement (wps-v0): GSM-Hard neighbour cells.
#
# docs/PREREG_WORKPOINT_STABILITY.md, frozen BEFORE these cells were generated.
#
# PURPOSE. Each cell adds a NEIGHBOUR to an already-reported dose so the
# reported point has data on BOTH sides. This is NOT a dose search and does NOT
# re-open P3 or the P3 CoT supplement -- both stay frozen exactly as they are.
# No frozen workpoint may be redefined by what comes out of these cells.
#
#   llama_cot_neg4 : Llama CoT  alpha=-4  -- does GSM8K's CoT shift -6 -> -4
#                                            also appear on GSM-Hard
#   qwen_cot_6     : Qwen  CoT  alpha=+6  -- compare +6/+8 under CoT
#   qwen_cot_10    : Qwen  CoT  alpha=+10 -- right neighbour of +8
#   qwen_nocot_10  : Qwen  No-CoT alpha=+10 -- +8 is at the tested boundary
#
# GOLD IS NEVER READ HERE. Generation and scoring stay separate scripts, as in
# P3: get_answer_gsm_hard_blind.py computes no accuracy, and the questions file
# must declare contains_labels=false (checked below).
#
# The blind generator SKIPS an existing cell rather than overwriting it, so
# pointing at the stored tree is safe -- the stored cells are left untouched.
#
# Budgets are inherited from the stored cells: 768 / bs=24 / greedy. The stored
# cells' physical GPU is unrecoverable (no device field in the summary), so each
# new-vs-stored contrast is a CROSS-RUN pairing regardless of device. The
# launcher does NOT constrain the device; it records what was used.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_gsm_hard.sh llama_cot_neg4 > wps_g2.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_gsm_hard.sh qwen_cot_6     > wps_g3.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_gsm_hard.sh qwen_cot_10    > wps_g4.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_gsm_hard.sh qwen_nocot_10  > wps_g5.log 2>&1 &
#   cat wps_g2.log     # immediately -- a wrong PY exits 127 before anything runs
set -euo pipefail

CELL="${1:-}"
case "${CELL}" in
    llama_cot_neg4|qwen_cot_6|qwen_cot_10|qwen_nocot_10) ;;
    *) echo "usage: $0 {llama_cot_neg4|qwen_cot_6|qwen_cot_10|qwen_nocot_10}" >&2; exit 2 ;;
esac

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Device selection is the caller's choice; this only RECORDS it. See the
# DEVICES note in the header.
DEVICES="${CUDA_VISIBLE_DEVICES:-<unset: all visible>}"
if [[ "${DEVICES}" == *,* ]]; then
  echo "[i] multi-card run (${DEVICES}): the model will be sharded, so this"
  echo "    cell's numerical path differs from a single-card stored cell."
  echo "    Record it with the result."
fi

TYPE="non"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
WORK_DIR="/data1/paveen/Dopamine"
QUESTIONS="${WORK_DIR}/components/benchmark/gsm_hard_p3_questions.json"

case "${CELL}" in
  llama_cot_neg4)
    MODEL_NAME="llama3"; MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_SIZE="8B"; HS_PREFIX="llama3"; LS=11; LE=20
    CONFIGS="neg4-11-20"; COT_FLAG="--cot"; SUB="gsm_hard_p3_cot"
    DESC="Llama GSM-Hard CoT alpha=-4 (neighbour of -6)"
    ;;
  qwen_cot_6)
    MODEL_NAME="qwen2.5"; MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SIZE="7B"; HS_PREFIX="qwen2.5"; LS=16; LE=22
    CONFIGS="6-16-22"; COT_FLAG="--cot"; SUB="gsm_hard_p3_cot"
    DESC="Qwen GSM-Hard CoT alpha=+6 (neighbour of +8)"
    ;;
  qwen_cot_10)
    MODEL_NAME="qwen2.5"; MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SIZE="7B"; HS_PREFIX="qwen2.5"; LS=16; LE=22
    CONFIGS="10-16-22"; COT_FLAG="--cot"; SUB="gsm_hard_p3_cot"
    DESC="Qwen GSM-Hard CoT alpha=+10 (right neighbour of +8)"
    ;;
  qwen_nocot_10)
    MODEL_NAME="qwen2.5"; MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SIZE="7B"; HS_PREFIX="qwen2.5"; LS=16; LE=22
    CONFIGS="10-16-22"; COT_FLAG=""; SUB="gsm_hard_p3"
    DESC="Qwen GSM-Hard No-CoT alpha=+10 (right neighbour of +8)"
    ;;
esac

MASK="${WORK_DIR}/components/mask/${HS_PREFIX}_${TYPE}_logits/nmd_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
OUT_DIR="${WORK_DIR}/components/${MODEL_NAME}/${SUB}"

[[ -f "${QUESTIONS}" ]] || { echo "ERROR: missing ${QUESTIONS}" >&2; exit 1; }
[[ -f "${MASK}" ]]      || { echo "ERROR: missing mask ${MASK}" >&2; exit 1; }

${PY} -c "import sys, numpy, torch; print('[py]', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)" \
    || { echo "ERROR: PY='${PY}' does not resolve or cannot import numpy/torch." >&2; exit 1; }

${PY} - <<PYCHK
import json, sys
d = json.load(open("${QUESTIONS}", encoding="utf-8"))
if d["meta"].get("contains_labels") is not False:
    sys.exit("ERROR: questions file does not declare contains_labels=false")
bad = [k for s in d["data"] for k in s
       if k.lower() in ("answer","gold","gold_answer","correct","target","accuracy")]
if bad:
    sys.exit(f"ERROR: label fields present: {sorted(set(bad))}")
print(f"label-free check OK ({len(d['data'])} questions, "
      f"digest {d['meta']['questions_sha256'][:16]})")
PYCHK

echo "=================================================="
echo "wps-v0 | ${DESC}"
echo "  band ${LS}-${LE} | cell ${CONFIGS} | devices ${DEVICES}"
echo "  out: ${OUT_DIR}"
echo "  (an existing cell is SKIPPED, not overwritten)"
echo "=================================================="

${PY} get_answer_gsm_hard_blind.py \
  --model "${MODEL_NAME}" --size "${MODEL_SIZE}" --model_dir "${MODEL_DIR}" \
  --questions "${QUESTIONS}" --mask_path "${MASK}" \
  --configs ${CONFIGS} --out_dir "${OUT_DIR}" \
  --max_new_tokens ${MAX_NEW_TOKENS} --temperature ${TEMPERATURE} \
  --batch_size ${BATCH_SIZE} --limit 0 ${COT_FLAG}

echo ""
echo "done. No accuracy computed."
echo "NEXT: download to the local RoleAnswer/ tree, then score with the frozen"
echo "GSM-Hard gold via the existing evaluator. first_acc is MAIN."
