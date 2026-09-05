#!/usr/bin/env bash
# ProofWriter OWA v2 prompt -- alpha=0, N=30 feasibility preflight ONLY.
#
# Separate, standalone launcher (does NOT touch run_proofwriter_owa.sh's
# frozen v0/v1 stages): human decision, 2026-09-05, resuming the SUSPENDED
# ProofWriter-OWA line (CLAUDE.md row) with a v2 prompt revision (fixed
# 1-shot Unknown train-split exemplar + "#### <Label>" marker instead of
# v1's "Answer: <Label>"). This is STILL A FEASIBILITY PROBE, not a steering
# sweep -- alpha=0 only, both models, same 30-item preflight subset already
# used for v0/v1 so results are directly comparable.
#
# Usage:
#   bash run_proofwriter_owa_v2_preflight.sh generate   # runs both models
#   bash run_proofwriter_owa_v2_preflight.sh eval        # scores both (reads gold)
#
# 'generate' launches BOTH models in the background (one physical GPU each,
# via CUDA_VISIBLE_DEVICES set per-command below) and returns immediately;
# check the two log files to see progress/completion. 'eval' must be run
# AFTER both generate jobs have finished and requires no GPU.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_proofwriter_owa_v2_preflight.sh generate|eval" >&2
  exit 1
fi
STAGE="$1"

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
PW_DIR="$WORK_DIR/proofwriter_owa"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
mkdir -p "$LOG_DIR"

LLAMA_GPU="${LLAMA_GPU:-2}"
QWEN_GPU="${QWEN_GPU:-1}"

# a wrong PY exits 127 before anything runs and the nohup log looks empty
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

EXEMPLAR_FILE="$PW_DIR/exemplar_unknown_v2.json"
[[ -f "$EXEMPLAR_FILE" ]] || {
  echo "[FATAL] $EXEMPLAR_FILE not found." >&2; exit 1; }

LLAMA_PREFLIGHT="$BENCH/preflight_blind_llama3.json"
QWEN_PREFLIGHT="$BENCH/preflight_blind_qwen25.json"
LLAMA_GOLD="$BENCH/preflight_gold_llama3.json"
QWEN_GOLD="$BENCH/preflight_gold_qwen25.json"
for f in "$LLAMA_PREFLIGHT" "$QWEN_PREFLIGHT" "$LLAMA_GOLD" "$QWEN_GOLD"; do
  [[ -f "$f" ]] || {
    echo "[FATAL] $f not found; run the v0/v1 'preflight' stage of" >&2
    echo "        run_proofwriter_owa.sh first (it builds this 30-item" >&2
    echo "        subset from the frozen 300-item manifest) -- v2 reuses" >&2
    echo "        the SAME subset so results are directly comparable." >&2
    exit 1; }
done

LLAMA_OUT_DIR="$OUT_ROOT/llama3/proofwriter_owa"
QWEN_OUT_DIR="$OUT_ROOT/qwen2.5/proofwriter_owa"
LLAMA_GEN="$LLAMA_OUT_DIR/preflight_v2_mdf_0/proofwriter_owa_8B_11_20.json"
QWEN_GEN="$QWEN_OUT_DIR/preflight_v2_mdf_0/proofwriter_owa_7B_16_22.json"

RESULTS_DIR="$PW_DIR/results"
mkdir -p "$RESULTS_DIR"
LLAMA_EVAL_OUT="$RESULTS_DIR/preflight_check_llama3_v2.json"
QWEN_EVAL_OUT="$RESULTS_DIR/preflight_check_qwen25_v2.json"

case "$STAGE" in
  generate)
    LLAMA_LOG="$LOG_DIR/proofwriter_owa_v2_preflight_llama3.log"
    QWEN_LOG="$LOG_DIR/proofwriter_owa_v2_preflight_qwen25.log"

    echo "[proofwriter-owa v2] launching llama3 preflight in background"
    echo "  GPU=$LLAMA_GPU  log=$LLAMA_LOG"
    cd "$WORK_DIR"
    CUDA_VISIBLE_DEVICES="$LLAMA_GPU" nohup "$PY" \
      proofwriter_owa/get_answer_proofwriter_owa.py \
      --model llama3 --size 8B \
      --model_dir meta-llama/Llama-3.1-8B-Instruct \
      --manifest "$LLAMA_PREFLIGHT" \
      --mask_path "$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy" \
      --configs 0-11-20 \
      --out_dir "$LLAMA_OUT_DIR" \
      --n_shot 1 \
      --exemplar_file "$EXEMPLAR_FILE" \
      --tag preflight_v2 \
      > "$LLAMA_LOG" 2>&1 &
    LLAMA_PID=$!
    echo "  pid=$LLAMA_PID"

    echo "[proofwriter-owa v2] launching qwen2.5 preflight in background"
    echo "  GPU=$QWEN_GPU  log=$QWEN_LOG"
    CUDA_VISIBLE_DEVICES="$QWEN_GPU" nohup "$PY" \
      proofwriter_owa/get_answer_proofwriter_owa.py \
      --model qwen2.5 --size 7B \
      --model_dir Qwen/Qwen2.5-7B-Instruct \
      --manifest "$QWEN_PREFLIGHT" \
      --mask_path "$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy" \
      --configs 0-16-22 \
      --out_dir "$QWEN_OUT_DIR" \
      --n_shot 1 \
      --exemplar_file "$EXEMPLAR_FILE" \
      --tag preflight_v2 \
      > "$QWEN_LOG" 2>&1 &
    QWEN_PID=$!
    echo "  pid=$QWEN_PID"

    echo
    echo "[proofwriter-owa v2] both jobs launched. Check logs IMMEDIATELY"
    echo "  (a wrong PY / bad import exits fast and the log looks empty"
    echo "  otherwise looks like it's just quiet):"
    echo "    tail -f $LLAMA_LOG"
    echo "    tail -f $QWEN_LOG"
    echo
    echo "[proofwriter-owa v2] when both finish, expected outputs:"
    echo "    $LLAMA_GEN"
    echo "    $QWEN_GEN"
    echo "  steering_fires must read 0 for both (alpha=0)."
    echo
    echo "[proofwriter-owa v2] then run:"
    echo "    bash run_proofwriter_owa_v2_preflight.sh eval"
    ;;

  eval)
    for f in "$LLAMA_GEN" "$QWEN_GEN"; do
      [[ -f "$f" ]] || {
        echo "[FATAL] $f not found; the 'generate' stage has not finished" >&2
        echo "        (or has not been run) for this model yet." >&2
        exit 1; }
    done
    cd "$WORK_DIR"
    echo "[proofwriter-owa v2] scoring llama3 (reads gold)"
    "$PY" proofwriter_owa/eval_proofwriter_owa.py \
      --gold "$LLAMA_GOLD" --generations "$LLAMA_GEN" \
      --out "$LLAMA_EVAL_OUT" --allow_partial_alphas
    echo
    echo "[proofwriter-owa v2] scoring qwen2.5 (reads gold)"
    "$PY" proofwriter_owa/eval_proofwriter_owa.py \
      --gold "$QWEN_GOLD" --generations "$QWEN_GEN" \
      --out "$QWEN_EVAL_OUT" --allow_partial_alphas
    echo
    echo "[proofwriter-owa v2] wrote:"
    echo "    $LLAMA_EVAL_OUT"
    echo "    $QWEN_EVAL_OUT"
    echo "  Read results.<model>.cells.\"0\" for accuracy / parse_failure_rate"
    echo "  / loop_rate / truncation_rate. This is STILL a feasibility probe"
    echo "  -- report but do not draw any steering/workpoint conclusion."
    ;;

  *)
    echo "[FATAL] unknown stage '$STAGE' (generate|eval)" >&2
    exit 1
    ;;
esac
