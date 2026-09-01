#!/usr/bin/env bash
# LogiQA 2.0 P4 format-only preflight (logiqa2-p4-v0 + p4-amend-01).
#
# Runs 20 no-gold items through both cells of ONE model at max_new_tokens=512.
# Computes NO accuracy. The only downstream decision is the frozen 512/1024
# budget rule, applied across all four cells by decide_p4_budget.py.
#
#   bash run_logiqa2_preflight.sh llama3
#   bash run_logiqa2_preflight.sh qwen2.5
#
# Both cells of one model stay on ONE card: bf16 greedy is not byte-reproducible
# across GPUs, and these two cells are a paired contrast.
set -euo pipefail

# NOTE: do NOT put braces in a ${1:?...} message -- the parameter expansion
# ends at the FIRST '}', so "{llama3|qwen2.5}" made MODEL literally 'llama3}'.
if [[ $# -lt 1 ]]; then
  echo "usage: run_logiqa2_preflight.sh llama3|qwen2.5" >&2; exit 1
fi
MODEL="$1"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BENCH="${BENCH:-$WORK_DIR/components/benchmark}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/components/logiqa2/preflight}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card." >&2; exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "[FATAL] both cells of a model must share ONE card; got '$CUDA_VISIBLE_DEVICES'." >&2; exit 1
fi

# a wrong PY exits 127 before anything runs and the nohup log looks empty
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the interpreter" >&2
  echo "        is 'python', not 'python3.10'." >&2; exit 1; }

# MODEL_DIR is an HF REPO ID, matching every other launcher in this repo
# (run_gsm8k_qwen25.sh, run_bandit_reference.sh, ...). Weights come from the
# local HF cache; there is no filesystem model tree. MASK_DIR follows the same
# convention as those launchers: ${BASE_DIR}/mask/${MODEL}_non_logits.
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"

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

# Fail on a wrong path BEFORE loading 8B of weights. A missing local model dir
# is otherwise handed to HF, which tries to parse it as a repo id and raises an
# unrelated-looking HFValidationError.
if [[ ! -f "$MASK" ]]; then
  echo "[FATAL] mask not found: $MASK" >&2
  echo "        override with MASK=/path/to/nmd.npy" >&2; exit 1
fi
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a filesystem path but does not exist:" >&2
  echo "        $MODEL_DIR" >&2
  echo "        Use an HF repo id (the repo default) or a real directory." >&2
  exit 1
fi
PREFLIGHT_FILE="${PREFLIGHT_FILE:-$BENCH/logiqa2_p4_preflight.json}"
if [[ ! -f "$PREFLIGHT_FILE" ]]; then
  echo "[FATAL] preflight file not found: $PREFLIGHT_FILE" >&2
  echo "        run: $PY data_logiqa2.py --out_dir $BENCH" >&2; exit 1
fi

mkdir -p "$OUT_DIR"
echo "[p4] model=$MODEL configs=${CONFIGS[*]} card=$CUDA_VISIBLE_DEVICES"

"$PY" get_answer_logiqa2_preflight.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" --mask "$MASK" \
  --configs "${CONFIGS[@]}" \
  --preflight_file "$PREFLIGHT_FILE" \
  --out "$OUT_DIR/preflight_${MODEL}.json"

echo "[p4] done. After BOTH models finish:"
echo "     $PY decide_p4_budget.py --preflight $OUT_DIR/preflight_llama3.json $OUT_DIR/preflight_qwen2.5.json"
