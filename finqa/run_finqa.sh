#!/usr/bin/env bash
# FinQA task-specific steering exploration (finqa-v0). NOT a GSM8K
# fixed-workpoint transfer test -- alpha is swept fresh on this task.
#
#   bash run_finqa.sh llama3  PREFLIGHT
#   bash run_finqa.sh qwen2.5 PREFLIGHT
#   bash run_finqa.sh llama3  FORMAL      # only after preflight is reviewed
#   bash run_finqa.sh qwen2.5 FORMAL
#
# PREFLIGHT runs alpha=0 ONLY, restricted to the 30-item frozen subset of the
# 300-item sample (data_finqa.py's `preflight_indices`), max_new_tokens=512.
# FORMAL runs all four frozen alphas for that model on the full 300 items.
#
# Doses (read from the task instructions, not searched here):
#   llama3   band 11-20   alpha in {-6,-4,0,4}
#   qwen2.5  band 16-22   alpha in {-6,0,6,8}
#
# ONE MODEL PER CARD, and a model's cells must share it: they are paired
# per-item contrasts and bf16 greedy is not byte-reproducible across GPUs.
# The two MODELS may run on two cards.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_finqa.sh llama3|qwen2.5 PREFLIGHT|FORMAL" >&2
  exit 1
fi
MODEL="$1"; STEP="$2"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/$MODEL/finqa}"
FINQA_DIR="${FINQA_DIR:-$WORK_DIR/finqa}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card." >&2
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
    CONFIGS_FORMAL="0-11-20 neg6-11-20 neg4-11-20 4-11-20" ;;
  qwen2.5)
    SIZE=7B
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
    CONFIGS_FORMAL="0-16-22 neg6-16-22 6-16-22 8-16-22" ;;
  *) echo "[FATAL] unknown model '$MODEL'" >&2; exit 1 ;;
esac

QFILE="${QFILE:-$BENCH/finqa_formal.json}"
[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] finqa_formal.json not found: $QFILE" >&2
                       echo "        run: $PY $FINQA_DIR/data_finqa.py --out_dir $BENCH" >&2
                       exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

mkdir -p "$OUT_ROOT"
cd "$WORK_DIR"

case "$STEP" in
  PREFLIGHT)
    PF_IDS=$("$PY" - "$QFILE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ".join(str(i) for i in d["meta"]["preflight_indices"]))
PYEOF
)
    if [[ "$MODEL" == llama3 ]]; then CONFIG_A0="0-11-20"; BAND="11_20"; else CONFIG_A0="0-16-22"; BAND="16_22"; fi
    echo "[finqa] model=$MODEL step=PREFLIGHT card=$CUDA_VISIBLE_DEVICES n=30"
    "$PY" "$FINQA_DIR/get_answer_finqa.py" \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --questions "$QFILE" --mask_path "$MASK" \
      --configs "$CONFIG_A0" \
      --out_dir "$OUT_ROOT/preflight" \
      --max_new_tokens 512 --batch_size 8 \
      --indices $PF_IDS
    echo "[finqa] done $(date)"
    echo "[finqa] Next -- inspect the preflight (this is the ONLY script that reads gold):"
    echo "     $PY $FINQA_DIR/eval_finqa.py --preflight \\"
    echo "       --generations $OUT_ROOT/preflight/mdf_0/finqa_${SIZE}_${BAND}.json \\"
    echo "       --gold_file $QFILE --out docs/finqa_preflight_${MODEL}.json"
    ;;
  FORMAL)
    echo "[finqa] model=$MODEL step=FORMAL card=$CUDA_VISIBLE_DEVICES configs=$CONFIGS_FORMAL"
    echo "[finqa] start $(date)"
    "$PY" "$FINQA_DIR/get_answer_finqa.py" \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --questions "$QFILE" --mask_path "$MASK" \
      --configs $CONFIGS_FORMAL \
      --out_dir "$OUT_ROOT/formal" \
      --max_new_tokens 512 --batch_size 8
    echo "[finqa] done $(date)"
    ;;
  *) echo "[FATAL] step must be PREFLIGHT or FORMAL" >&2; exit 1 ;;
esac
