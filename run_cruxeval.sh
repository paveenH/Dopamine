#!/usr/bin/env bash
# CRUXEval-O fixed-workpoint transfer (cruxeval-p4c-v0).
#
#   bash run_cruxeval.sh llama3  BASELINE     # alpha=0
#   bash run_cruxeval.sh llama3  WORKPOINT    # the frozen GSM8K alpha -- MAIN
#   bash run_cruxeval.sh llama3  NEIGHBOUR    # local-stability diagnostic
#   bash run_cruxeval.sh llama3  REVERSE      # direction-ordering diagnostic
#   bash run_cruxeval.sh llama3  ALL          # all four, in order, one process
#   bash run_cruxeval.sh llama3  PREFLIGHT    # FORMAT ONLY, 8 items, no accuracy
#
# THE LAUNCHER CANNOT EXPRESS ANY OTHER ALPHA. This protocol never searches
# doses on CRUXEval: alpha is read from the frozen GSM8K record (llama -6 /
# qwen +8) and the two diagnostics are frozen alongside it.
#
#   llama3   0 | -6 workpoint | -4 neighbour | +4 reverse   band 11-20  L=9
#   qwen2.5  0 | +8 workpoint | +6 neighbour | -6 reverse   band 16-22  L=6
#
# The DIAGNOSTIC cells are reported whether or not they are significant, sit
# OUTSIDE the Holm family (m=2 stays the two workpoint contrasts), and MUST NOT
# redefine the workpoint. WORKPOINT/NEIGHBOUR/REVERSE are refused until the
# alpha=0 cell exists, so no cell can be mistaken for a dose search.
#
# CELLS ARE NOT REQUIRED TO SHARE A GPU. host and CUDA_VISIBLE_DEVICES are
# recorded as provenance by the runner; because bf16 greedy may vary across
# hardware, a contrast between cells that ran on different devices is reported
# as a CROSS-RUN pairing. Pairing means alignment by the frozen item order,
# never hardware identity.
set -euo pipefail

# NOTE: no braces inside a ${1:?...} message -- the expansion ends at the FIRST
# '}', which once made MODEL literally 'llama3}'. bash -n does NOT catch that.
if [[ $# -lt 2 ]]; then
  echo "usage: run_cruxeval.sh llama3|qwen2.5 BASELINE|WORKPOINT|NEIGHBOUR|REVERSE|ALL|PREFLIGHT" >&2
  exit 1
fi
MODEL="$1"; STEP="$2"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/$MODEL/cruxeval}"

# a wrong PY exits 127 before anything runs and the nohup log looks empty
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

# cheap checks BEFORE 8B of weights load: a missing local model dir is
# otherwise handed to HF, which parses it as a repo id and raises an
# unrelated-looking HFValidationError.
QFILE="${QFILE:-$BENCH/cruxeval_p4c_formal_blind.json}"
[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] blind questions not found: $QFILE" >&2
                       echo "        run: $PY data_cruxeval.py --out_dir $BENCH" >&2
                       exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

A0FILE="$OUT_ROOT/mdf_0/cruxeval_o_${SIZE}_${BAND}.json"
need_baseline() {
  if [[ ! -f "$A0FILE" ]]; then
    echo "[FATAL] alpha=0 cell missing: $A0FILE" >&2
    echo "        Every contrast is against alpha=0. Run BASELINE first." >&2
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
    # FORMAT ONLY, and it cannot become anything else: --preflight forces
    # n=8 and writes to a separate _preflight tree, while the runner computes
    # no correctness field at all. Preflight accuracy must not be sought and
    # alpha must not be adjusted from it -- there is nothing to read.
    echo "[p4c] PREFLIGHT: format only -- marker / literal parser / budget /"
    echo "      steering_fires. Accuracy is NOT computed and must not be sought."
    mkdir -p "$OUT_ROOT/_preflight"
    cd "$WORK_DIR"
    exec "$PY" get_answer_cruxeval.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --questions "$QFILE" --mask_path "$MASK" \
      --configs $A0 $AWP --out_dir "$OUT_ROOT/_preflight" --preflight
    ;;
  *) echo "[FATAL] step must be BASELINE, WORKPOINT, NEIGHBOUR, REVERSE, ALL or PREFLIGHT" >&2
     exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "[p4c] model=$MODEL step=$STEP configs=$CONFIGS"
echo "[p4c] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[p4c] start $(date)"

cd "$WORK_DIR"
"$PY" get_answer_cruxeval.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
  --questions "$QFILE" --mask_path "$MASK" \
  --configs $CONFIGS --out_dir "$OUT_ROOT"

echo "[p4c] done $(date)"
echo "[p4c] Score with the ONLY script that reads gold, once both models are done:"
echo "     $PY eval_cruxeval.py \\"
echo "       --generations $BASE_DIR/llama3/cruxeval/mdf_{0,neg6,neg4,4}/cruxeval_o_8B_11_20.json \\"
echo "                     $BASE_DIR/qwen2.5/cruxeval/mdf_{0,8,6,neg6}/cruxeval_o_7B_16_22.json \\"
echo "       --gold_file $BENCH/cruxeval_p4c_formal.json \\"
echo "       --out docs/p4c_cruxeval_evaluation.json"
