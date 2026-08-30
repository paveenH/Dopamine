#!/usr/bin/env bash
# P3 SUPPLEMENT (p3-supp-v1): GSM-Hard CoT condition transfer -- qwen25.
#
# docs/PREREG_P3_SUPPLEMENT.md, frozen BEFORE these cells were generated.
#
# THIS DOES NOT RE-OPEN P3. The closed blind validation stands as frozen. What
# is prospective here is the CONDITION (CoT), not the data -- the same 300
# questions were already unsealed on 2026-08-30.
#
# TWO CELLS ONLY: CoT alpha=0 and CoT alpha=+8. The alpha is the workpoint
# already established on GSM8K and already used in P3; it is NOT re-searched,
# and no CoT dose curve is run. Four cells across both models answer whether CoT
# and a fixed workpoint are complementary, redundant or conflicting.
#
# ONLY the CoT instruction changes. --cot inserts exactly one line,
# "Let's think step by step."; the '####' directive and the 'Answer: ' anchor
# are untouched, so the injection site and the commitment features keep the
# meaning they had when the predictor was fitted.
#
# Usage:
#   bash run_gsm_hard_cot_qwen25.sh --preflight   # 5 samples, format only
#   bash run_gsm_hard_cot_qwen25.sh --full        # 300, both cells
set -euo pipefail

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# BOTH CELLS OF ONE MODEL SHARE A CARD. The primary metric is a per-question
# paired contrast between them, and bf16 greedy is not byte-reproducible across
# GPUs, so splitting them would mix the device difference into the CoT effect.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one card." >&2
  exit 1
fi

MODEL_NAME="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
MODEL_SIZE="7B"
HS_PREFIX="qwen2.5"
TYPE="non"
PERCENTAGE=0.5
LS=16
LE=22

MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24

WORK_DIR="/data1/paveen/Dopamine"
QUESTIONS="${WORK_DIR}/components/benchmark/gsm_hard_p3_questions.json"
MASK="${WORK_DIR}/components/mask/${HS_PREFIX}_${TYPE}_logits/nmd_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
OUT_DIR="${WORK_DIR}/components/${MODEL_NAME}/gsm_hard_p3_cot"

CONFIGS="0-16-22 8-16-22"

MODE="${1:---preflight}"
case "${MODE}" in
  --preflight) LIMIT=5;  OUT_DIR="${OUT_DIR}_preflight" ;;
  --full)      LIMIT=0 ;;
  *) echo "usage: $0 [--preflight|--full]" >&2; exit 2 ;;
esac

[[ -f "${QUESTIONS}" ]] || { echo "ERROR: missing ${QUESTIONS}" >&2; exit 1; }
[[ -f "${MASK}" ]]      || { echo "ERROR: missing mask ${MASK}" >&2; exit 1; }

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

echo "[qwen2.5 CoT] band ${LS}-${LE}  cells: ${CONFIGS}"
echo "  out: ${OUT_DIR}"

${PY} get_answer_gsm_hard_blind.py \
  --model "${MODEL_NAME}" --size "${MODEL_SIZE}" --model_dir "${MODEL_DIR}" \
  --questions "${QUESTIONS}" --mask_path "${MASK}" \
  --configs ${CONFIGS} --out_dir "${OUT_DIR}" \
  --max_new_tokens ${MAX_NEW_TOKENS} --temperature ${TEMPERATURE} \
  --batch_size ${BATCH_SIZE} --limit ${LIMIT} --cot

echo "done. No accuracy computed."
