#!/usr/bin/env bash
# BBH numeric fixed-workpoint transfer (bbh-p4b-v0).
#
#   bash run_bbh_numeric.sh llama3  object_counting STAGE0
#   bash run_bbh_numeric.sh qwen2.5 object_counting STAGE0
#   bash run_bbh_numeric.sh llama3  object_counting WORKPOINT   # only after PASS
#   bash run_bbh_numeric.sh llama3  object_counting REVERSE     # diagnostic only
#
# STAGE0 runs alpha=0 ONLY, on all 250 items. WORKPOINT runs that model's
# frozen GSM8K alpha (llama -6 / qwen +8) and is refused unless the alpha=0
# cell already exists -- the gate is judged on alpha=0, so running the
# workpoint first would make the gate decision unfalsifiable.
#
# REVERSE (p4b-amend-01) runs ONE opposite-signed dose per model as a
# DIRECTION-ORDERING DIAGNOSTIC: llama +4, qwen -6. It exists only to see
# whether the ordering continues (llama -6 > 0 > +4; qwen +8 > 0 > -6).
# It is NOT part of the primary test, it is NOT in the Holm family, and it
# MUST NOT redefine the workpoint -- that stays read from the frozen GSM8K
# record. It is refused until BOTH the alpha=0 and the workpoint cell exist,
# so it can never be mistaken for a dose search that produced the workpoint.
#
# Apart from those two frozen reverse doses the launcher CANNOT express any
# other alpha: this protocol never searches doses.
#
# ONE MODEL PER CARD, and a model's THREE cells must share it: they are paired
# per-item contrasts and bf16 greedy is not byte-reproducible across GPUs. The
# two MODELS may run on two cards -- they are never compared per item.
set -euo pipefail

# NOTE: no braces in a ${1:?...} message -- the expansion ends at the FIRST '}',
# which once made MODEL literally 'llama3}'. bash -n does NOT catch that.
if [[ $# -lt 3 ]]; then
  echo "usage: run_bbh_numeric.sh llama3|qwen2.5 <task> STAGE0|WORKPOINT|REVERSE" >&2
  echo "  task: object_counting | multistep_arithmetic_two" >&2
  exit 1
fi
MODEL="$1"; TASK="$2"; STEP="$3"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/$MODEL/bbh/$TASK}"

case "$TASK" in
  object_counting|multistep_arithmetic_two) ;;
  *) echo "[FATAL] task must be object_counting or multistep_arithmetic_two" >&2; exit 1 ;;
esac

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card." >&2
  echo "        This cell is paired per item against the alpha=0 cell; an" >&2
  echo "        unpinned run mixes device differences into the alpha effect." >&2
  exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "[FATAL] one card only; got '$CUDA_VISIBLE_DEVICES'." >&2; exit 1
fi

# a wrong PY exits 127 before anything runs and the nohup log looks empty
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

case "$MODEL" in
  llama3)
    SIZE=8B
    MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
    A0=0-11-20 ; AWP=neg6-11-20 ; WPTAG=mdf_neg6 ; BAND=11_20
    AREV=4-11-20 ; REVTAG=mdf_4 ;;
  qwen2.5)
    SIZE=7B
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
    A0=0-16-22 ; AWP=8-16-22 ; WPTAG=mdf_8 ; BAND=16_22
    AREV=neg6-16-22 ; REVTAG=mdf_neg6 ;;
  *) echo "[FATAL] unknown model '$MODEL'" >&2; exit 1 ;;
esac

# cheap checks BEFORE 8B of weights load: a missing local model dir is
# otherwise handed to HF, which parses it as a repo id and raises an
# unrelated-looking HFValidationError.
QFILE="${QFILE:-$BENCH/bbh_p4b_${TASK}_blind.json}"
[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] blind questions not found: $QFILE" >&2
                       echo "        run: $PY data_bbh_numeric.py --task $TASK --out_dir $BENCH" >&2
                       exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

case "$STEP" in
  STAGE0)    CONFIGS="$A0" ;;
  WORKPOINT)
    A0DIR="$OUT_ROOT/mdf_0/bbh_${TASK}_${SIZE}_${BAND}.json"
    if [[ ! -f "$A0DIR" ]]; then
      echo "[FATAL] alpha=0 cell missing: $A0DIR" >&2
      echo "        The stage-0 gate is judged on alpha=0. Run STAGE0 and" >&2
      echo "        score it first; if it FAILS the gate, do not run this." >&2
      exit 1
    fi
    CONFIGS="$AWP" ;;
  REVERSE)
    # Refused until alpha=0 AND the workpoint cell exist. The reverse dose is a
    # direction diagnostic on an already-completed transfer test; running it
    # earlier would make it look like the dose search this protocol forbids.
    A0DIR="$OUT_ROOT/mdf_0/bbh_${TASK}_${SIZE}_${BAND}.json"
    WPDIR="$OUT_ROOT/$WPTAG/bbh_${TASK}_${SIZE}_${BAND}.json"
    if [[ ! -f "$A0DIR" ]]; then
      echo "[FATAL] alpha=0 cell missing: $A0DIR" >&2; exit 1
    fi
    if [[ ! -f "$WPDIR" ]]; then
      echo "[FATAL] workpoint cell missing: $WPDIR" >&2
      echo "        REVERSE is a diagnostic on a COMPLETED transfer test." >&2
      echo "        Run WORKPOINT first; the workpoint is read from the" >&2
      echo "        frozen GSM8K record and is never chosen from BBH." >&2
      exit 1
    fi
    CONFIGS="$AREV" ;;
  *) echo "[FATAL] step must be STAGE0, WORKPOINT or REVERSE" >&2; exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "[p4b] model=$MODEL task=$TASK step=$STEP configs=$CONFIGS card=$CUDA_VISIBLE_DEVICES"
echo "[p4b] start $(date)"

cd "$WORK_DIR"
"$PY" get_answer_bbh_numeric.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
  --task "$TASK" --questions "$QFILE" --mask_path "$MASK" \
  --configs $CONFIGS --out_dir "$OUT_ROOT"

echo "[p4b] done $(date)"
if [[ "$STEP" == "STAGE0" ]]; then
  echo "[p4b] Next -- score the gate (this is the ONLY script that reads gold):"
  echo "     $PY eval_bbh_numeric.py \\"
  echo "       --generations $OUT_ROOT/mdf_0/bbh_${TASK}_${SIZE}_${BAND}.json \\"
  echo "       --gold_file $BENCH/bbh_p4b_${TASK}.json \\"
  echo "       --out docs/bbh_p4b_${TASK}_stage0_${MODEL}.json"
  echo "     Run WORKPOINT only if that prints PASS."
fi
