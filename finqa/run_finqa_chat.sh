#!/usr/bin/env bash
# FinQA Chat 0-shot CoT (finqa-chat-0shot-cot-v1). Supplementary Native Chat run on the CLOSED finqa-v0 line;
# fixed workpoints only (llama3 {0,-6}, qwen2.5 {0,+8}); NOT a dose search, NOT a blind test.
#
#   CUDA_VISIBLE_DEVICES=0 bash finqa/run_finqa_chat.sh llama3  run   # 300 items, both alphas, ONE model load
#   CUDA_VISIBLE_DEVICES=1 bash finqa/run_finqa_chat.sh qwen2.5 run   # own card; the two may run in parallel
#   bash finqa/run_finqa_chat.sh both eval   # CPU only, needs all four formal cells under $BASE_DIR
#
# Outputs: $BASE_DIR/<model>/finqa/chat_0shot_cot_v1/formal/mdf_<alpha>/finqa_chat_<size>_<ls>_<le>.json
# Evaluation: docs/finqa_chat_0shot_cot_v1_evaluation.json (never overwritten).
set -euo pipefail

MODEL="${1:?usage: run_finqa_chat.sh llama3-or-qwen2.5-or-both run-or-eval}"
STEP="${2:?usage: run_finqa_chat.sh llama3-or-qwen2.5-or-both run-or-eval}"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
QFILE="${QFILE:-$BASE_DIR/benchmark/finqa_formal.json}"
cd "$WORK_DIR"

case "$STEP" in
  run)
    case "$MODEL" in llama3|qwen2.5) ;; *) echo "[FATAL] model must be llama3 or qwen2.5 for run" >&2; exit 1 ;; esac
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
      echo "[FATAL] set CUDA_VISIBLE_DEVICES to exactly one card (both alphas of a model share it)." >&2; exit 1
    fi
    "$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
      echo "[FATAL] '$PY' cannot import numpy/torch (server interpreter is 'python')." >&2; exit 1; }
    [[ -f "$QFILE" ]] || { echo "[FATAL] $QFILE not found" >&2; exit 1; }
    echo "[finqa-chat] model=$MODEL step=$STEP card=$CUDA_VISIBLE_DEVICES $(date)"
    "$PY" finqa/get_answer_finqa_chat.py --model "$MODEL" --base_dir "$BASE_DIR" --questions "$QFILE"
    echo "[finqa-chat] done $(date)"
    ;;
  eval)
    ROOT_OUT="finqa/chat_0shot_cot_v1/formal"
    LL="$BASE_DIR/llama3/$ROOT_OUT"; QW="$BASE_DIR/qwen2.5/$ROOT_OUT"
    CELLS=("$LL/mdf_0/finqa_chat_8B_11_20.json" "$LL/mdf_-6/finqa_chat_8B_11_20.json"
           "$QW/mdf_0/finqa_chat_7B_16_22.json" "$QW/mdf_8/finqa_chat_7B_16_22.json")
    for c in "${CELLS[@]}"; do [[ -f "$c" ]] || { echo "[FATAL] missing formal cell $c" >&2; exit 1; }; done
    EXTRA=()
    [[ -n "${DETECTOR_DIR:-}" ]] && EXTRA+=(--detector_dir "$DETECTOR_DIR")
    [[ -f docs/finqa_formal_evaluation.json ]] && EXTRA+=(--bare_eval docs/finqa_formal_evaluation.json)
    "$PY" finqa/eval_finqa_chat.py --cells "${CELLS[@]}" --gold_file "$QFILE" \
      --out docs/finqa_chat_0shot_cot_v1_evaluation.json "${EXTRA[@]}"
    ;;
  *) echo "[FATAL] step must be run or eval" >&2; exit 1 ;;
esac
