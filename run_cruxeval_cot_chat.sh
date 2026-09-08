#!/usr/bin/env bash
# CRUXEval-O CoT chat-template interface-condition diagnostic, llama3 ONLY.
# Protocol `cruxeval-o-cot-chat-v1`.
#
#   bash run_cruxeval_cot_chat.sh PREFLIGHT    # FORMAT ONLY, 8 items
#   bash run_cruxeval_cot_chat.sh BASELINE     # alpha=0 (chat)
#   bash run_cruxeval_cot_chat.sh WORKPOINT    # alpha=-6, llama3's own frozen
#                                              # GSM8K workpoint -- the single
#                                              # planned comparison
#
# THE ONLY VARIABLE vs the bare CoT follow-up (run_cruxeval_cot.sh /
# get_answer_cruxeval_cot.py, protocol cot-transfer-followup-v0) is the
# prompt wrapper: the SAME 300 items / order / gold / CoT prompt body /
# FIRST-marker parser / mask / band / budget / batch size / greedy decoding
# are all inherited unchanged. This script cannot express any other alpha,
# any other model, or any other dose -- llama3 ONLY, {0, -6} ONLY.
#
# Output is isolated by TAG under the SAME cruxeval_cot_followup parent tree
# the bare CoT follow-up uses (not a different top-level directory):
#   components/llama3/cruxeval_cot_followup/formal_chat_v1_mdf_0/
#   components/llama3/cruxeval_cot_followup/formal_chat_v1_mdf_neg6/
# meta.protocol = "cruxeval-o-cot-chat-v1", never the bare follow-up's
# "cot-transfer-followup-v0" -- no existing file is ever touched or
# overwritten by this launcher.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_cruxeval_cot_chat.sh PREFLIGHT|BASELINE|WORKPOINT" >&2
  exit 1
fi
STEP="$1"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/llama3/cruxeval_cot_followup}"

"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

SIZE=8B
MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
# The FULL 32-row mask -- never pre-sliced to the 9-layer band. Injection
# range is controlled by the layer band (11-20) passed via --configs, exactly
# as every other launcher in this repo does it.
MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
BAND=11_20
A0=0-11-20
AWP=neg6-11-20

QFILE="${QFILE:-$BENCH/cruxeval_p4c_formal_blind.json}"
[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] blind questions not found: $QFILE" >&2
                       echo "        (same file the bare cruxeval-p4c-v0 / "
                       echo "         cot-transfer-followup-v0 cells used)" >&2
                       exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2
  echo "        This protocol uses the HF repo id, not a local model dir." >&2
  exit 1
fi

A0FILE="$OUT_ROOT/formal_chat_v1_mdf_0/cruxeval_o_cot_chat_${SIZE}_${BAND}.json"
need_baseline() {
  if [[ ! -f "$A0FILE" ]]; then
    echo "[FATAL] chat alpha=0 cell missing: $A0FILE" >&2
    echo "        Every contrast is against this cell's OWN chat alpha=0." >&2
    echo "        Run BASELINE first." >&2
    exit 1
  fi
}

case "$STEP" in
  PREFLIGHT)
    echo "[cot-chat] PREFLIGHT: format only -- chat template applied, marker /"
    echo "      literal parser / budget / steering_fires=0. Accuracy is NOT"
    echo "      computed and must not be sought."
    mkdir -p "$OUT_ROOT/_preflight_chat"
    cd "$WORK_DIR"
    exec "$PY" get_answer_cruxeval_cot_chat.py \
      --model_dir "$MODEL_DIR" --size "$SIZE" \
      --questions "$QFILE" --mask_path "$MASK" \
      --configs $A0 $AWP --out_dir "$OUT_ROOT/_preflight_chat" --preflight
    ;;
  BASELINE)  CONFIGS="$A0" ;;
  WORKPOINT) need_baseline; CONFIGS="$AWP" ;;
  *) echo "[FATAL] step must be PREFLIGHT, BASELINE or WORKPOINT" >&2
     exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "[cot-chat] CRUXEval-O CoT chat-template diagnostic, llama3 step=$STEP configs=$CONFIGS"
echo "[cot-chat] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[cot-chat] INTERFACE-CONDITION DIAGNOSTIC -- does not replace the bare"
echo "           cruxeval-p4c-v0 or cot-transfer-followup-v0 results"
echo "[cot-chat] start $(date)"

cd "$WORK_DIR"
"$PY" get_answer_cruxeval_cot_chat.py \
  --model_dir "$MODEL_DIR" --size "$SIZE" \
  --questions "$QFILE" --mask_path "$MASK" \
  --configs $CONFIGS --out_dir "$OUT_ROOT"

echo "[cot-chat] done $(date)"
echo "[cot-chat] Score once BOTH cells (alpha=0 and alpha=-6) exist:"
echo "     $PY eval_cruxeval.py --protocol cruxeval-o-cot-chat-v1 \\"
echo "       --generations \\"
echo "         $OUT_ROOT/formal_chat_v1_mdf_0/cruxeval_o_cot_chat_${SIZE}_${BAND}.json \\"
echo "         $OUT_ROOT/formal_chat_v1_mdf_neg6/cruxeval_o_cot_chat_${SIZE}_${BAND}.json \\"
echo "       --gold_file $BENCH/cruxeval_p4c_formal.json \\"
echo "       --out docs/cruxeval_cot_chat_v1_evaluation.json"
