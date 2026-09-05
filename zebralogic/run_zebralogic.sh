#!/usr/bin/env bash
# ZebraLogic-Easy four-point workpoint exploration (zebralogic-easy-v0).
# Protocol: docs/PREREG_ZEBRALOGIC_EASY.md
#
#   bash zebralogic/run_zebralogic.sh CHECK_ENV
#   bash zebralogic/run_zebralogic.sh CHECK_ACCESS
#   bash zebralogic/run_zebralogic.sh CHECK_DISTRIBUTION
#   CUDA_VISIBLE_DEVICES=0 bash zebralogic/run_zebralogic.sh PREFLIGHT   llama3
#   CUDA_VISIBLE_DEVICES=0 bash zebralogic/run_zebralogic.sh PREFLIGHT   qwen2.5
#   CUDA_VISIBLE_DEVICES=0 bash zebralogic/run_zebralogic.sh CANARY      llama3  card0
#   CUDA_VISIBLE_DEVICES=1 bash zebralogic/run_zebralogic.sh CANARY      llama3  card1
#   CUDA_VISIBLE_DEVICES=0 bash zebralogic/run_zebralogic.sh FORMAL_LLAMA
#   CUDA_VISIBLE_DEVICES=1 bash zebralogic/run_zebralogic.sh FORMAL_QWEN
#                          bash zebralogic/run_zebralogic.sh ANALYZE     llama3
#                          bash zebralogic/run_zebralogic.sh ANALYZE     qwen2.5
#
# SIX STAGES, run in this order:
#   1. CHECK_ENV           interpreter/deps, mask files, blind items file,
#                           model paths -- no GPU/data download needed beyond
#                           what is already local.
#   2. CHECK_ACCESS         confirms HF_TOKEN + allenai/ZebraLogicBench-private
#                           access BEFORE any GPU time is spent generating
#                           something that cannot later be scored.
#   3. CHECK_DISTRIBUTION   prints the official easy/hard/small/medium/large/xl
#                           size distribution (zebra_difficulty.py's own
#                           categories, read from data_zebralogic.py) next to
#                           this protocol's frozen 7-size list, THEN STOPS.
#                           The formal stages below refuse to run until a
#                           human has confirmed by exporting
#                           ZEBRALOGIC_DISTRIBUTION_CONFIRMED=1 -- this holds
#                           even though the two lists already match exactly
#                           (see docs/PREREG_ZEBRALOGIC_EASY.md section 8b);
#                           the instruction was to show and wait, not to skip
#                           the wait because they match.
#   4. PREFLIGHT <model>    5-item format/plumbing check, alpha=0 (+ one smoke
#                           alpha for the firing-count assertion). Writes to
#                           an isolated _preflight/ tree; does not gate on
#                           accuracy.
#   5. CANARY <model> <tag> alpha=0, 8-item deterministic subset, tagged by
#                           physical GPU. Run once per card in use BEFORE
#                           pooling that model's formal cells across cards.
#   6. FORMAL_LLAMA /       the full 280-item, four-alpha sweep for one model.
#      FORMAL_QWEN          Both refuse to run before CHECK_DISTRIBUTION has
#                           been explicitly confirmed (stage 3).
#      ANALYZE <model>      scores a model's formal cells (reads private gold
#                           -- the only stage besides CANARY/PREFLIGHT that
#                           does).
#
# ONE MODEL PER CARD for FORMAL/CANARY/PREFLIGHT: a model's cells are a
# paired per-item contrast and bf16 greedy is not byte-reproducible across
# GPUs. The two MODELS may use two different cards.
set -euo pipefail

STEP="${1:-}"
if [[ -z "$STEP" ]]; then
  echo "usage: run_zebralogic.sh CHECK_ENV|CHECK_ACCESS|CHECK_DISTRIBUTION|" >&2
  echo "                          PREFLIGHT|CANARY|FORMAL_LLAMA|FORMAL_QWEN|ANALYZE" >&2
  echo "                          [llama3|qwen2.5] [device_tag]" >&2
  exit 1
fi

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
ZDIR="$WORK_DIR/zebralogic"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-$BASE_DIR}"
ITEMS="${ITEMS:-$BENCH/zebralogic_easy_blind.json}"

model_config() {
  case "$1" in
    llama3)
      SIZE=8B
      MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
      MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
      CONFIGS="neg6-11-20 neg4-11-20 0-11-20 4-11-20"
      ZERO_CONFIG="0-11-20"
      SMOKE_CONFIG="8-11-20"   # non-zero, NOT one of the frozen doses -- used
                                # only for the preflight firing-count assertion
      OUT_ROOT="$OUT_ROOT_BASE/llama3/zebralogic_easy"
      ;;
    qwen2.5)
      SIZE=7B
      MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
      MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
      CONFIGS="neg6-16-22 0-16-22 6-16-22 8-16-22"
      ZERO_CONFIG="0-16-22"
      SMOKE_CONFIG="12-16-22"
      OUT_ROOT="$OUT_ROOT_BASE/qwen2.5/zebralogic_easy"
      ;;
    *) echo "[FATAL] unknown model '$1' (want llama3 or qwen2.5)" >&2; exit 1 ;;
  esac
}

cmd_check_env() {
  echo "[zebralogic] CHECK_ENV"
  "$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
    echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
    echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }
  "$PY" -c "import datasets" >/dev/null 2>&1 || {
    echo "[FATAL] '$PY' cannot import 'datasets' (needed for the loader/scorer)." >&2
    exit 1; }
  for m in llama3 qwen2.5; do
    # MASK/MODEL_DIR use `${VAR:-default}` inside model_config, which only
    # substitutes when the var is UNSET/empty -- once set on the first loop
    # iteration it silently persists into the second, reporting llama3's mask
    # path again while claiming to check qwen2.5. Force-unset before each
    # call so the env-var override still works for a single-model invocation
    # while this loop gets a fresh default each time.
    unset MASK MODEL_DIR
    model_config "$m"
    if [[ -f "$MASK" ]]; then
      echo "  OK   mask found: $MASK"
    else
      echo "  MISS mask not found: $MASK  (expected before FORMAL/CANARY/PREFLIGHT for $m)"
    fi
  done
  if [[ -f "$ITEMS" ]]; then
    echo "  OK   blind items file found: $ITEMS"
  else
    echo "  MISS blind items file not found: $ITEMS"
    echo "       run: $PY $ZDIR/data_zebralogic.py --out_dir $BENCH"
  fi
  echo "[zebralogic] CHECK_ENV done"
}

cmd_check_access() {
  echo "[zebralogic] CHECK_ACCESS -- verifying allenai/ZebraLogicBench-private"
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "[FATAL] neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN is set." >&2
    echo "        NOTE: the accepted names are HF_TOKEN or HUGGING_FACE_HUB_TOKEN" >&2
    echo "        (underscore between HUGGING and FACE). HUGGINGFACE_HUB_TOKEN" >&2
    echo "        (no underscore, one word 'HUGGINGFACE') is NOT the same variable" >&2
    echo "        and will not be picked up -- this has already caused one" >&2
    echo "        false 'not set' failure on this server. Use HF_TOKEN." >&2
    echo "        Real ZebraLogic gold requires:" >&2
    echo "          1. An HF account with access GRANTED on" >&2
    echo "             https://huggingface.co/datasets/allenai/ZebraLogicBench-private" >&2
    echo "             (click through the gated-repo access request while logged in)." >&2
    echo "          2. A valid token for that SAME account, exported here as" >&2
    echo "             HF_TOKEN (or run 'huggingface-cli login')." >&2
    exit 1
  fi
  "$PY" -c "
import sys
sys.path.insert(0, '$ZDIR')
from data_zebralogic import load_private_gold
g = load_private_gold(['lgp-test-2x2-1'])
print('  OK   private gold reachable; sample id lgp-test-2x2-1 has',
      len(g['lgp-test-2x2-1']['rows']), 'house(s)')
"
  echo "[zebralogic] CHECK_ACCESS passed"
}

cmd_check_distribution() {
  echo "[zebralogic] CHECK_DISTRIBUTION"
  "$PY" "$ZDIR/data_zebralogic.py" --check
  echo
  echo "=================================================================="
  echo " Frozen protocol size list (docs/PREREG_ZEBRALOGIC_EASY.md sec. 1):"
  echo "   2*2 2*3 2*4 2*5 2*6 3*2 3*3   (40 items each, 280 total)"
  echo "=================================================================="
  echo " This IS the official WildEval/ZeroEval 'easy_sizes' definition,"
  echo " verbatim (zebra_grid_eval.py). The two lists already match exactly."
  echo " Per instruction, the formal sweep still WAITS for an explicit human"
  echo " confirmation regardless -- export ZEBRALOGIC_DISTRIBUTION_CONFIRMED=1"
  echo " to unblock FORMAL_LLAMA / FORMAL_QWEN once you have reviewed the"
  echo " distribution printed above."
  echo "=================================================================="
}

require_distribution_confirmed() {
  if [[ "${ZEBRALOGIC_DISTRIBUTION_CONFIRMED:-0}" != "1" ]]; then
    echo "[FATAL] formal sweep blocked: run CHECK_DISTRIBUTION and review the" >&2
    echo "        printed size distribution, then re-invoke with" >&2
    echo "        ZEBRALOGIC_DISTRIBUTION_CONFIRMED=1 in the environment." >&2
    exit 1
  fi
}

require_card() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card." >&2
    echo "        This cell is paired per item against that model's own" >&2
    echo "        alpha=0 cell; an unpinned run mixes device differences" >&2
    echo "        into the alpha effect." >&2
    exit 1
  fi
  if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "[FATAL] one card only; got '$CUDA_VISIBLE_DEVICES'." >&2; exit 1
  fi
}

precheck_paths() {
  [[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
  [[ -f "$ITEMS" ]] || { echo "[FATAL] blind items not found: $ITEMS" >&2
                         echo "        run: $PY $ZDIR/data_zebralogic.py --out_dir $BENCH" >&2
                         exit 1; }
  if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
    echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
  fi
}

cmd_preflight() {
  local MODEL="$1"
  model_config "$MODEL"
  require_card
  precheck_paths
  local OUT="$OUT_ROOT/_preflight"
  mkdir -p "$OUT"
  echo "[zebralogic] PREFLIGHT model=$MODEL card=$CUDA_VISIBLE_DEVICES"
  # prereg S5: preflight is alpha=0 ONLY (the 5 fixed items), plus the
  # non-zero SMOKE_CONFIG purely for the steering_fires==L*N assertion --
  # NOT the full four-point frozen sweep. Passing $CONFIGS here (all four
  # frozen doses) would run three formal-dose cells at n=5, which the prereg
  # never asks for and which the preflight scorer (n=5) is not built to
  # analyze as a dose curve.
  cd "$WORK_DIR"
  # ZEBRA_MAX_NEW_TOKENS/ZEBRA_BATCH_SIZE default to the prereg's 2048/8;
  # override explicitly (never silently) to re-run preflight at the S3
  # 3072 escalation step, matching cmd_formal's own override convention.
  # max_new_tokens is still hard-enforced to {2048,3072} inside
  # get_answer_zebralogic.py regardless of what is passed here.
  "$PY" zebralogic/get_answer_zebralogic.py \
    --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
    --items "$ITEMS" --mask_path "$MASK" \
    --configs $ZERO_CONFIG $SMOKE_CONFIG \
    --out_dir "$OUT" --mode preflight \
    --max_new_tokens "${ZEBRA_MAX_NEW_TOKENS:-2048}" \
    --batch_size "${ZEBRA_BATCH_SIZE:-8}"
  echo "[zebralogic] PREFLIGHT generation done. Score it (requires private gold):"
  echo "  $PY zebralogic/eval_zebralogic.py --preflight_check \\"
  echo "      --preflight_file $OUT/mdf_0/zebralogic_easy_${SIZE}_*.json"
}

cmd_canary() {
  local MODEL="$1"; local TAG="$2"
  if [[ -z "$TAG" ]]; then
    echo "[FATAL] CANARY needs a device_tag, e.g.: CANARY llama3 card0" >&2; exit 1
  fi
  model_config "$MODEL"
  require_card
  precheck_paths
  local OUT="$OUT_ROOT/_canary/$TAG"
  mkdir -p "$OUT"
  echo "[zebralogic] CANARY model=$MODEL device_tag=$TAG card=$CUDA_VISIBLE_DEVICES"
  cd "$WORK_DIR"
  "$PY" zebralogic/get_answer_zebralogic.py \
    --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
    --items "$ITEMS" --mask_path "$MASK" \
    --configs $ZERO_CONFIG \
    --out_dir "$OUT" --mode canary --device_tag "$TAG" \
    --max_new_tokens 2048 --batch_size 8
  echo "[zebralogic] CANARY done. After running on every card in use, compare with:"
  echo "  $PY zebralogic/eval_zebralogic.py --canary_check --canary_files <file_card0> <file_card1> ..."
}

cmd_formal() {
  local MODEL="$1"
  model_config "$MODEL"
  require_distribution_confirmed
  require_card
  precheck_paths
  mkdir -p "$OUT_ROOT"
  echo "[zebralogic] FORMAL model=$MODEL configs=$CONFIGS card=$CUDA_VISIBLE_DEVICES"
  echo "[zebralogic] start $(date)"
  cd "$WORK_DIR"
  "$PY" zebralogic/get_answer_zebralogic.py \
    --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
    --items "$ITEMS" --mask_path "$MASK" \
    --configs $CONFIGS \
    --out_dir "$OUT_ROOT" --mode formal \
    --max_new_tokens "${ZEBRA_MAX_NEW_TOKENS:-2048}" --batch_size 8
  echo "[zebralogic] done $(date)"
  echo "[zebralogic] Next -- score (reads private gold, the ONLY step that does):"
  echo "  $PY zebralogic/eval_zebralogic.py --generations $OUT_ROOT/mdf_*/zebralogic_easy_${SIZE}_*.json \\"
  echo "      --out docs_local/zebralogic_easy_${MODEL}_result.json"
}

cmd_analyze() {
  local MODEL="$1"
  model_config "$MODEL"
  cd "$WORK_DIR"
  local OUT="docs_local/zebralogic_easy_${MODEL}_result.json"
  mkdir -p docs_local
  "$PY" zebralogic/eval_zebralogic.py \
    --generations "$OUT_ROOT"/mdf_*/zebralogic_easy_${SIZE}_*.json \
    --out "$OUT"
}

case "$STEP" in
  CHECK_ENV)          cmd_check_env ;;
  CHECK_ACCESS)        cmd_check_access ;;
  CHECK_DISTRIBUTION)  cmd_check_distribution ;;
  PREFLIGHT)           cmd_preflight "${2:?usage: PREFLIGHT llama3 or qwen2\.5}" ;;
  CANARY)              cmd_canary "${2:?usage: CANARY model device_tag}" "${3:-}" ;;
  FORMAL_LLAMA)        cmd_formal llama3 ;;
  FORMAL_QWEN)         cmd_formal qwen2.5 ;;
  ANALYZE)             cmd_analyze "${2:?usage: ANALYZE llama3 or qwen2\.5}" ;;
  *) echo "[FATAL] unknown step '$STEP'" >&2; exit 1 ;;
esac
