#!/usr/bin/env bash
# CRUXEval-O explicit-CoT follow-up (cot-transfer-followup-v0).
#
#   bash run_cruxeval_cot.sh llama3  BASELINE     # alpha=0
#   bash run_cruxeval_cot.sh llama3  WORKPOINT    # frozen GSM8K alpha -- MAIN
#   bash run_cruxeval_cot.sh llama3  NEIGHBOUR    # local-stability diagnostic
#   bash run_cruxeval_cot.sh llama3  REVERSE      # direction-ordering diagnostic
#   bash run_cruxeval_cot.sh llama3  ALL          # all four, in order
#   bash run_cruxeval_cot.sh llama3  PREFLIGHT    # FORMAT ONLY, 8 items
#
# POST-HOC EXPLORATORY FOLLOW-UP -- does not replace the frozen No-CoT
# cruxeval-p4c-v0 result. Same doses, same items, same everything except the
# CoT cue.
#
#   llama3   0 | -6 workpoint | -4 neighbour | +4 reverse   band 11-20  L=9
#   qwen2.5  0 | +8 workpoint | +6 neighbour | -6 reverse   band 16-22  L=6
#
# CELLS ARE NOT REQUIRED TO SHARE A GPU. host and CUDA_VISIBLE_DEVICES are
# recorded as provenance; a cross-device contrast is reported as a CROSS-RUN
# pairing by item order, never hardware identity.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_cruxeval_cot.sh llama3|qwen2.5 BASELINE|WORKPOINT|NEIGHBOUR|REVERSE|ALL|PREFLIGHT" >&2
  exit 1
fi
MODEL="$1"; STEP="$2"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/$MODEL/cruxeval_cot_followup}"

"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

case "$MODEL" in
  llama3)
    SIZE=8B
    MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
    BAND=11_20
    A0=0-11-20 ; AWP=neg6-11-20 ; ANB=neg4-11-20 ; ARV=4-11-20 ;;
  qwen2.5)
    SIZE=7B
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
    BAND=16_22
    A0=0-16-22 ; AWP=8-16-22 ; ANB=6-16-22 ; ARV=neg6-16-22 ;;
  *) echo "[FATAL] unknown model '$MODEL' (llama3 | qwen2.5)" >&2; exit 1 ;;
esac

QFILE="${QFILE:-$BENCH/cruxeval_p4c_formal_blind.json}"
[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] blind questions not found: $QFILE" >&2
                       echo "        (same file the No-CoT cruxeval-p4c-v0 cells used)" >&2
                       exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

A0FILE="$OUT_ROOT/mdf_0/cruxeval_o_cot_${SIZE}_${BAND}.json"
need_baseline() {
  if [[ ! -f "$A0FILE" ]]; then
    echo "[FATAL] alpha=0 CoT cell missing: $A0FILE" >&2
    echo "        Every contrast is against this task's own CoT alpha=0." >&2
    echo "        Run BASELINE first." >&2
    exit 1
  fi
}

case "$STEP" in
  BASELINE)  CONFIGS="$A0" ;;
  WORKPOINT) need_baseline; CONFIGS="$AWP" ;;
  NEIGHBOUR) need_baseline; CONFIGS="$ANB" ;;
  REVERSE)   need_baseline; CONFIGS="$ARV" ;;
  ALL)       CONFIGS="$A0 $AWP $ANB $ARV" ;;
  PREFLIGHT)
    echo "[cot-followup] PREFLIGHT: format only -- marker / literal parser /"
    echo "      budget / steering_fires / injection-token equality."
    echo "      Accuracy is NOT computed and must not be sought."
    mkdir -p "$OUT_ROOT/_preflight"
    cd "$WORK_DIR"
    exec "$PY" get_answer_cruxeval_cot.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --questions "$QFILE" --mask_path "$MASK" \
      --configs $A0 $AWP --out_dir "$OUT_ROOT/_preflight" --preflight
    ;;
  *) echo "[FATAL] step must be BASELINE, WORKPOINT, NEIGHBOUR, REVERSE, ALL or PREFLIGHT" >&2
     exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "[cot-followup] CRUXEval-O model=$MODEL step=$STEP configs=$CONFIGS"
echo "[cot-followup] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[cot-followup] POST-HOC EXPLORATORY -- does not replace the frozen No-CoT cruxeval-p4c-v0 result"
echo "[cot-followup] start $(date)"

cd "$WORK_DIR"
"$PY" get_answer_cruxeval_cot.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
  --questions "$QFILE" --mask_path "$MASK" \
  --configs $CONFIGS --out_dir "$OUT_ROOT"

echo "[cot-followup] done $(date)"
echo "[cot-followup] Score with the shared eval script, once both models are done"
echo "     (NOTE: --nocot_generations needs all FOUR No-CoT cells per model --"
echo "     alpha=0, workpoint, neighbour, reverse -- since the DiD interaction"
echo "     term reads both alpha=0 and the workpoint dose for each model):"
echo "     $PY eval_cot_transfer_followup.py --task cruxeval \\"
echo "       --generations \\"
echo "         $BASE_DIR/llama3/cruxeval_cot_followup/mdf_{0,neg6,neg4,4}/cruxeval_o_cot_8B_11_20.json \\"
echo "         $BASE_DIR/qwen2.5/cruxeval_cot_followup/mdf_{0,8,6,neg6}/cruxeval_o_cot_7B_16_22.json \\"
echo "       --gold_file $BENCH/cruxeval_p4c_formal.json \\"
echo "       --nocot_generations \\"
echo "         $BASE_DIR/llama3/cruxeval/mdf_{0,neg6,neg4,4}/cruxeval_o_8B_11_20.json \\"
echo "         $BASE_DIR/qwen2.5/cruxeval/mdf_{0,8,6,neg6}/cruxeval_o_7B_16_22.json \\"
echo "       --nocot_evaluation docs/p4c_cruxeval_evaluation.json \\"
echo "       --out docs/cot_followup_cruxeval_evaluation.json"
