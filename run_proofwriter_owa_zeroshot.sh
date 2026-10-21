#!/usr/bin/env bash
# ProofWriter-OWA CHAT ZERO-SHOT (protocol proofwriter-owa-zs-chat-v1).
# Independent of the closed bare line and of the 1-shot chat line; writes only
# under <components>/<model>/proofwriter_owa_zs/ and proofwriter_owa/results/zs_chat_v1/.
#
# Stages (MODEL = llama3 or qwen2.5; CONDITION = nocot (default) or cot):
#   check      MODEL [CONDITION]   tokenizer only, no GPU: prompt + injection-site report
#   base       MODEL [CONDITION]   generate the alpha=0 cell (needs a GPU)
#   workpoint  MODEL [CONDITION]   llama3: alpha -6 AND +6;  qwen2.5: alpha +8 AND -6 (one process each, alpha=0 excluded)
#   eval       [CONDITION]         score all cells (reads gold)
# There is no format gate. The CoT condition is a later stage and is never started by another stage.
set -euo pipefail

STAGE="${1:?usage: run_proofwriter_owa_zeroshot.sh stage model condition}"
if [[ "$STAGE" == "eval" ]]; then MODEL=""; CONDITION="${2:-nocot}"
else MODEL="${2:?model is required (llama3 or qwen2.5)}"; CONDITION="${3:-nocot}"; fi

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
PW="$WORK_DIR/proofwriter_owa"
RES="$PW/results/zs_chat_v1"
BLIND="$BENCH/manifest_blind.json"; GOLD="$BENCH/manifest_gold.json"
case "$CONDITION" in nocot|cot) ;; *) echo "[FATAL] condition must be nocot or cot" >&2; exit 1;; esac

cell_path() {  # model alpha_tag size band
  echo "$BASE_DIR/$1/proofwriter_owa_zs/${CONDITION}_mdf_$2/proofwriter_owa_zs_$3_$4.json"; }
model_vars() {
  case "$1" in
    llama3)  SIZE=8B; BAND=11_20; WP=neg6; MASK="$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy";;
    qwen2.5) SIZE=7B; BAND=16_22; WP=8;    MASK="$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy";;
    *) echo "[FATAL] unknown model '$1' (llama3 or qwen2.5)" >&2; exit 1;;
  esac; }
need_py() { "$PY" -c "import numpy, torch, transformers" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch/transformers (server interpreter is 'python')" >&2; exit 1; }; }
need_gpu_var() { [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "[FATAL] set CUDA_VISIBLE_DEVICES (recorded as provenance; both cells of a model should use the same host/GPU model)" >&2; exit 1; }; }

[[ -z "$MODEL" ]] || model_vars "$MODEL"   # reject an unknown model before anything else
cd "$WORK_DIR"
case "$STAGE" in
  check)
    model_vars "$MODEL"
    "$PY" -c "import transformers" >/dev/null 2>&1 || { echo "[FATAL] transformers missing" >&2; exit 1; }
    mkdir -p "$RES"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa_zeroshot.py --mode check \
      --model "$MODEL" --condition "$CONDITION" --cell a0 --manifest "$BLIND" \
      --report "$RES/check_${CONDITION}_${MODEL}.json";;
  base)
    model_vars "$MODEL"; need_py; need_gpu_var
    "$PY" proofwriter_owa/get_answer_proofwriter_owa_zeroshot.py --mode run \
      --model "$MODEL" --condition "$CONDITION" --cell a0 --manifest "$BLIND" \
      --mask_path "$MASK" --out_root "$BASE_DIR";;
  workpoint)
    model_vars "$MODEL"; need_py; need_gpu_var
    CELLS="wp rev"   # llama3: -6 and +6; qwen2.5: +8 and -6
    "$PY" proofwriter_owa/get_answer_proofwriter_owa_zeroshot.py --mode run \
      --model "$MODEL" --condition "$CONDITION" --cell $CELLS --manifest "$BLIND" \
      --mask_path "$MASK" --out_root "$BASE_DIR";;
  eval)
    model_vars llama3; L0="$(cell_path llama3 0 "$SIZE" "$BAND")"; L1="$(cell_path llama3 "$WP" "$SIZE" "$BAND")"
    model_vars qwen2.5; Q0="$(cell_path qwen2.5 0 "$SIZE" "$BAND")"; Q1="$(cell_path qwen2.5 "$WP" "$SIZE" "$BAND")"
    Q2="$(cell_path qwen2.5 neg6 "$SIZE" "$BAND")"; L2="$(cell_path llama3 6 8B 11_20)"
    mkdir -p "$RES"
    "$PY" proofwriter_owa/eval_proofwriter_owa_zeroshot.py \
      --condition "$CONDITION" --manifest_blind "$BLIND" --gold "$GOLD" \
      --cells "$L0" "$L1" "$L2" "$Q0" "$Q1" "$Q2" \
      --out "$RES/eval_${CONDITION}.json";;
  *) echo "[FATAL] unknown stage '$STAGE'" >&2; exit 1;;
esac
