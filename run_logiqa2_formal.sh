#!/usr/bin/env bash
# LogiQA 2.0 P4 FORMAL RUN (logiqa2-p4-v0, amendments 02/03/04/05/06).
#
# 300 items x 2 cells for ONE model at max_new_tokens=1024 (stage-1 frozen).
# Generates only -- accuracy is computed afterwards by eval_logiqa2.py.
#
#   bash run_logiqa2_formal.sh llama3
#   bash run_logiqa2_formal.sh qwen2.5
#
# Both cells of one model stay on ONE card: bf16 greedy is not byte-reproducible
# across GPUs and the two cells are a paired contrast. The two MODELS may run on
# two cards -- they are never compared per-question.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_logiqa2_formal.sh llama3|qwen2.5" >&2; exit 1
fi
MODEL="$1"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/logiqa2/formal}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card." >&2; exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "[FATAL] both cells of a model must share ONE card; got '$CUDA_VISIBLE_DEVICES'." >&2; exit 1
fi

"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the interpreter" >&2
  echo "        is 'python', not 'python3.10'." >&2; exit 1; }

case "$MODEL" in
  llama3)
    SIZE=8B
    MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
    CONFIGS=(0-11-20 neg6-11-20) ;;
  qwen2.5)
    SIZE=7B
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
    CONFIGS=(0-16-22 8-16-22) ;;
  *) echo "[FATAL] unknown model '$MODEL'" >&2; exit 1 ;;
esac

FORMAL_FILE="${FORMAL_FILE:-$BENCH/logiqa2_p4_formal.json}"
for f in "$MASK" "$FORMAL_FILE"; do
  if [[ ! -f "$f" ]]; then echo "[FATAL] not found: $f" >&2; exit 1; fi
done
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

mkdir -p "$OUT_DIR"
echo "[p4] FORMAL model=$MODEL configs=${CONFIGS[*]} card=$CUDA_VISIBLE_DEVICES"
echo "[p4] budget 1024 (stage-1 frozen), 300 items, ~2 cells"

"$PY" get_answer_logiqa2.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" --mask "$MASK" \
  --configs "${CONFIGS[@]}" \
  --formal_file "$FORMAL_FILE" \
  --out "$OUT_DIR/formal_${MODEL}.json"

echo "[p4] done. After BOTH models finish, score once:"
echo "     $PY eval_logiqa2.py \\"
echo "       --generations $OUT_DIR/formal_llama3.json $OUT_DIR/formal_qwen2.5.json \\"
echo "       --formal_file $FORMAL_FILE --out docs/p4_logiqa2_evaluation.json"
