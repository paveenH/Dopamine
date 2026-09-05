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
# GPU IS NOT PINNED BY THIS SCRIPT (matching zebralogic/run_zebralogic.sh's
# convention) -- pass CUDA_VISIBLE_DEVICES on each invocation, one model per
# call, e.g.:
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_proofwriter_owa_v2_preflight.sh \
#     generate llama3 > proofwriter_owa_v2_preflight_llama3.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=3 nohup bash run_proofwriter_owa_v2_preflight.sh \
#     generate qwen2.5 > proofwriter_owa_v2_preflight_qwen25.log 2>&1 &
#
#   python run_proofwriter_owa_v2_preflight.sh eval   # after BOTH finish, no GPU needed
#
# (the eval stage also works via `bash run_proofwriter_owa_v2_preflight.sh eval`)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_proofwriter_owa_v2_preflight.sh generate <llama3|qwen2.5>" >&2
  echo "       run_proofwriter_owa_v2_preflight.sh eval" >&2
  exit 1
fi
STAGE="$1"
MODEL="${2:-}"

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
PW_DIR="$WORK_DIR/proofwriter_owa"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR}"

# a wrong PY exits 127 before anything runs and the nohup log looks empty
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

EXEMPLAR_FILE="$PW_DIR/exemplar_unknown_v2.json"
[[ -f "$EXEMPLAR_FILE" ]] || {
  echo "[FATAL] $EXEMPLAR_FILE not found." >&2; exit 1; }

# Filenames must match run_proofwriter_owa.sh's own convention EXACTLY:
# PREFLIGHT_FILE="$BENCH/preflight_blind_${MODEL}.json" where $MODEL is the
# literal string passed on that script's command line -- "qwen2.5", not
# "qwen25". A mismatched filename here is a naming bug in THIS launcher, not
# evidence the v0/v1 preflight subset was never built.
LLAMA_PREFLIGHT="$BENCH/preflight_blind_llama3.json"
QWEN_PREFLIGHT="$BENCH/preflight_blind_qwen2.5.json"
LLAMA_GOLD="$BENCH/preflight_gold_llama3.json"
QWEN_GOLD="$BENCH/preflight_gold_qwen2.5.json"

LLAMA_OUT_DIR="$OUT_ROOT/llama3/proofwriter_owa"
QWEN_OUT_DIR="$OUT_ROOT/qwen2.5/proofwriter_owa"
LLAMA_GEN="$LLAMA_OUT_DIR/preflight_v2_mdf_0/proofwriter_owa_8B_11_20.json"
QWEN_GEN="$QWEN_OUT_DIR/preflight_v2_mdf_0/proofwriter_owa_7B_16_22.json"

RESULTS_DIR="$PW_DIR/results"
mkdir -p "$RESULTS_DIR"
LLAMA_EVAL_OUT="$RESULTS_DIR/preflight_check_llama3_v2.json"
QWEN_EVAL_OUT="$RESULTS_DIR/preflight_check_qwen25_v2.json"

require_preflight_files() {
  # $1 = model tag, $2 = blind file, $3 = gold file
  [[ -f "$2" ]] || {
    echo "[FATAL] $2 not found." >&2
    echo "        If the v0/v1 'preflight' stage of run_proofwriter_owa.sh" >&2
    echo "        was already run for $1, check for a filename mismatch" >&2
    echo "        (this launcher expects EXACTLY 'preflight_blind_$1.json'" >&2
    echo "        under $BENCH) before assuming the subset was never built." >&2
    exit 1; }
  [[ -f "$3" ]] || {
    echo "[FATAL] $3 not found (see the note above)." >&2; exit 1; }
}

case "$STAGE" in
  generate)
    if [[ -z "$MODEL" ]]; then
      echo "[FATAL] 'generate' requires a model argument: llama3 or qwen2.5" >&2
      exit 1
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card" >&2
      echo "        (this script does not pick a GPU for you)." >&2
      exit 1
    fi
    case "$MODEL" in
      llama3)
        require_preflight_files llama3 "$LLAMA_PREFLIGHT" "$LLAMA_GOLD"
        echo "[proofwriter-owa v2] llama3 preflight, alpha=0, N=30"
        cd "$WORK_DIR"
        "$PY" proofwriter_owa/get_answer_proofwriter_owa.py \
          --model llama3 --size 8B \
          --model_dir meta-llama/Llama-3.1-8B-Instruct \
          --manifest "$LLAMA_PREFLIGHT" \
          --mask_path "$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy" \
          --configs 0-11-20 \
          --out_dir "$LLAMA_OUT_DIR" \
          --n_shot 1 \
          --exemplar_file "$EXEMPLAR_FILE" \
          --tag preflight_v2
        echo "[proofwriter-owa v2] wrote $LLAMA_GEN"
        echo "  steering_fires must read 0 (alpha=0; checked by the"
        echo "  generator itself -- a mismatch already stopped the run)."
        ;;
      qwen2.5)
        require_preflight_files qwen2.5 "$QWEN_PREFLIGHT" "$QWEN_GOLD"
        echo "[proofwriter-owa v2] qwen2.5 preflight, alpha=0, N=30"
        cd "$WORK_DIR"
        "$PY" proofwriter_owa/get_answer_proofwriter_owa.py \
          --model qwen2.5 --size 7B \
          --model_dir Qwen/Qwen2.5-7B-Instruct \
          --manifest "$QWEN_PREFLIGHT" \
          --mask_path "$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy" \
          --configs 0-16-22 \
          --out_dir "$QWEN_OUT_DIR" \
          --n_shot 1 \
          --exemplar_file "$EXEMPLAR_FILE" \
          --tag preflight_v2
        echo "[proofwriter-owa v2] wrote $QWEN_GEN"
        echo "  steering_fires must read 0 (alpha=0; checked by the"
        echo "  generator itself -- a mismatch already stopped the run)."
        ;;
      *)
        echo "[FATAL] unknown model '$MODEL' (llama3 | qwen2.5)" >&2
        exit 1
        ;;
    esac
    echo
    echo "[proofwriter-owa v2] when BOTH models finish, run:"
    echo "    bash run_proofwriter_owa_v2_preflight.sh eval"
    ;;

  eval)
    for f in "$LLAMA_GEN" "$QWEN_GEN"; do
      [[ -f "$f" ]] || {
        echo "[FATAL] $f not found; run 'generate' for both models first" >&2
        echo "        (and wait for both background jobs to finish)." >&2
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
