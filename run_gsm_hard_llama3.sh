#!/usr/bin/env bash
# GSM-Hard generation for the P3 BLIND cross-task validation -- llama3.
#
# Protocol p3-v1 (docs/PREREG_P3.md, tag p3-prereg-v1).
#
# SEPARATE LAUNCHER PER MODEL, DELIBERATELY. The two models differ in mask,
# layer band, injection token and dose set; a shared launcher would invite
# exactly the cross-model parameter mixing this project has been bitten by.
#   llama3: band 11-20 (L=9), doses -8 / -6 / -4 / 0 / +4
#
# THIS SCRIPT COMPUTES NO ACCURACY. It drives get_answer_gsm_hard_blind.py,
# which has no correctness code path at all. The gold file stays SEALED until
# p3_predictions.json is frozen.
#
# ONE CURVE, ONE GPU: all five doses must run on the same card with the same
# batch settings -- bf16 greedy is not byte-reproducible across GPUs, and a
# split curve mixes the machine difference into the alpha effect.
#
# Usage:
#   bash run_gsm_hard_llama3.sh --preflight   # 5 samples, format only
#   bash run_gsm_hard_llama3.sh --full        # all 300, five doses
set -euo pipefail

PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ONE MODEL'S FIVE DOSES STAY ON ONE CARD. bf16 greedy is not byte-reproducible
# across GPUs (cuBLAS accumulation order flips logit ties; a measured re-run
# differed on 205/300 samples), so splitting one dose curve across cards would
# mix the device difference into the alpha effect. Different MODELS may run on
# different cards concurrently -- they are never compared per-question.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one card." >&2
  echo "  One model's five doses must share a card; two models may use two." >&2
  exit 1
fi

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"
PERCENTAGE=0.5
LS=11
LE=20

# Identical to the frozen GSM8K main line, so commitment features keep meaning.
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24

WORK_DIR="/data1/paveen/Dopamine"
QUESTIONS="${WORK_DIR}/components/benchmark/gsm_hard_p3_questions.json"
MASK="${WORK_DIR}/components/mask/${HS_PREFIX}_${TYPE}_logits/nmd_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
OUT_DIR="${WORK_DIR}/components/${MODEL_NAME}/gsm_hard_p3"

CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 0-11-20 4-11-20"

MODE="${1:---preflight}"
case "${MODE}" in
  --preflight) LIMIT=5;  OUT_DIR="${OUT_DIR}_preflight" ;;
  --full)      LIMIT=0 ;;
  *) echo "usage: $0 [--preflight|--full]" >&2; exit 2 ;;
esac

[[ -f "${QUESTIONS}" ]] || { echo "ERROR: missing ${QUESTIONS}" >&2; exit 1; }
[[ -f "${MASK}" ]]      || { echo "ERROR: missing mask ${MASK}" >&2; exit 1; }

# Fail closed: the questions file must declare itself label-free.
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

echo "[${MODEL_NAME}] band ${LS}-${LE}  doses: ${CONFIGS}"
echo "  out: ${OUT_DIR}"

${PY} get_answer_gsm_hard_blind.py \
  --model "${MODEL_NAME}" \
  --size "${MODEL_SIZE}" \
  --model_dir "${MODEL_DIR}" \
  --questions "${QUESTIONS}" \
  --mask_path "${MASK}" \
  --configs ${CONFIGS} \
  --out_dir "${OUT_DIR}" \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --temperature ${TEMPERATURE} \
  --batch_size ${BATCH_SIZE} \
  --limit ${LIMIT}

echo "done. No accuracy computed."
