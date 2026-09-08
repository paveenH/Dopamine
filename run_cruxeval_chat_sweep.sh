#!/usr/bin/env bash
# CRUXEval-O chat-template four-point dose sweep, llama3 ONLY.
#
#   bash run_cruxeval_chat_sweep.sh cot    PREFLIGHT   # 8 items, alpha=0 only
#   bash run_cruxeval_chat_sweep.sh cot    ALL         # all 4 alpha, one process
#   bash run_cruxeval_chat_sweep.sh nocot  PREFLIGHT
#   bash run_cruxeval_chat_sweep.sh nocot  ALL
#
# EXPLORATORY four-point dose sweep under the HF chat template -- does NOT
# replace or redefine:
#   - the bare cruxeval-p4c-v0 fixed-workpoint transfer result (No-CoT)
#   - the bare cot-transfer-followup-v0 result (CoT)
#   - the SINGLE {0,-6} chat comparison cruxeval-o-cot-chat-v1 (m=1)
# All three stay on disk, untouched, under their own protocols/output paths.
#
# Doses: {-6,-4,0,+4}, llama3's own frozen four-point matrix, band 11-20.
# CoT and No-CoT write to DIFFERENT parent trees and DIFFERENT protocol
# strings; they are two independent generation runs, never mixed.
#
#   cot:    components/llama3/cruxeval_cot_followup/chat_sweep_v1_mdf_<a>/
#           meta.protocol = cruxeval-o-cot-chat-sweep-v1
#   nocot:  components/llama3/cruxeval/chat_sweep_v1_mdf_<a>/
#           meta.protocol = cruxeval-o-nocot-chat-sweep-v1
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_cruxeval_chat_sweep.sh cot|nocot PREFLIGHT|ALL" >&2
  exit 1
fi
MODE="$1"; STEP="$2"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"

"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

SIZE=8B
MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
QFILE="${QFILE:-$BENCH/cruxeval_p4c_formal_blind.json}"

case "$MODE" in
  cot)
    OUT_ROOT="${OUT_ROOT:-$BASE_DIR/llama3/cruxeval_cot_followup}"
    ;;
  nocot)
    OUT_ROOT="${OUT_ROOT:-$BASE_DIR/llama3/cruxeval}"
    ;;
  *) echo "[FATAL] mode must be cot or nocot" >&2; exit 1 ;;
esac

[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] blind questions not found: $QFILE" >&2; exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2
  echo "        This protocol uses the HF repo id, not a local model dir." >&2
  exit 1
fi

case "$STEP" in
  PREFLIGHT)
    echo "[chat-sweep] PREFLIGHT mode=$MODE: format only, alpha=0 ONLY."
    echo "      Accuracy is NOT computed and must not be sought."
    mkdir -p "$OUT_ROOT/_preflight_chat_sweep"
    cd "$WORK_DIR"
    exec "$PY" get_answer_cruxeval_chat_sweep.py \
      --mode "$MODE" --model_dir "$MODEL_DIR" --size "$SIZE" \
      --questions "$QFILE" --mask_path "$MASK" \
      --configs 0-11-20 --out_dir "$OUT_ROOT/_preflight_chat_sweep" --preflight
    ;;
  ALL)
    CONFIGS="0-11-20 neg6-11-20 neg4-11-20 4-11-20"
    ;;
  *) echo "[FATAL] step must be PREFLIGHT or ALL" >&2; exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "[chat-sweep] CRUXEval-O chat four-point sweep mode=$MODE step=$STEP configs=$CONFIGS"
echo "[chat-sweep] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[chat-sweep] EXPLORATORY -- does not replace the bare or the {0,-6} chat results"
echo "[chat-sweep] start $(date)"

cd "$WORK_DIR"
"$PY" get_answer_cruxeval_chat_sweep.py \
  --mode "$MODE" --model_dir "$MODEL_DIR" --size "$SIZE" \
  --questions "$QFILE" --mask_path "$MASK" \
  --configs $CONFIGS --out_dir "$OUT_ROOT"

echo "[chat-sweep] done $(date)"
if [[ "$MODE" == "cot" ]]; then
  STEM=cruxeval_o_cot_chat_sweep
else
  STEM=cruxeval_o_nocot_chat_sweep
fi
PROTOCOL_NAME="cruxeval-o-${MODE}-chat-sweep-v1"
echo "[chat-sweep] Score once all FOUR cells exist:"
echo "     $PY eval_cruxeval_chat_sweep.py --protocol $PROTOCOL_NAME \\"
echo "       --generations \\"
echo "         $OUT_ROOT/chat_sweep_v1_mdf_{0,neg6,neg4,4}/${STEM}_${SIZE}_11_20.json \\"
echo "       --gold_file $BENCH/cruxeval_p4c_formal.json \\"
echo "       --out docs/cruxeval_${MODE}_chat_sweep_v1_evaluation.json"
