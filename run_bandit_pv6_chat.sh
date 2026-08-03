#!/bin/bash
# Track A reference-chat runner. Chat is a literature-interface comparator;
# it is never used for the competence gate or the main alpha sweep.
#
# Usage:
#   GPU=1 bash run_bandit_pv6_chat.sh --yes
#   GPU=1 bash run_bandit_pv6_chat.sh --yes --easy-only
#   GPU=1 bash run_bandit_pv6_chat.sh --yes --hard-only
#
# Default: Easy + Hard (40 episodes). Results are resumable.

set -uo pipefail

MODEL="${MODEL:-llama3}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
PY="${PY:-python3.10}"
LOGDIR="${LOGDIR:-./pv6_logs}"

CONFIRM=0
RUN_EASY=1
RUN_HARD=1
EASY_ONLY=0
HARD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --yes) CONFIRM=1 ;;
    --easy-only) EASY_ONLY=1; RUN_EASY=1; RUN_HARD=0 ;;
    --hard-only) HARD_ONLY=1; RUN_EASY=0; RUN_HARD=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done


if [ "$EASY_ONLY" -eq 1 ] && [ "$HARD_ONLY" -eq 1 ]; then
  echo "STOP: --easy-only and --hard-only are mutually exclusive."
  exit 2
fi

if [ "$CONFIRM" -ne 1 ]; then
  echo "STOP: formal N=20 runs require --yes (run smoke and inspect it first)."
  exit 2
fi
if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "STOP: choose a GPU, e.g. GPU=1 bash $0 --yes --easy-only"
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

run_checked () {
  local label="$1"; shift
  echo
  echo "========================================================================"
  echo "$label"
  echo "========================================================================"
  "$@" 2>&1 | tee "$LOGDIR/${label// /_}_${STAMP}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || {
    echo "STOP: $label failed"; exit 1;
  }
}

run_checked "manifest_check" "$PY" freeze_bandit_baseline.py --check
run_checked "gate_selftest" "$PY" evaluate_competence_gate.py --selftest

echo
echo "GPU=$CUDA_VISIBLE_DEVICES  MODEL=$MODEL  interface=chat"
if [ "$RUN_EASY" -eq 1 ]; then
  run_checked "A0_CHAT_EASY" env BASE_DIR="$BASE_DIR" PY="$PY" \
    bash run_bandit_reference.sh "$MODEL" A0_CHAT_EASY
fi
if [ "$RUN_HARD" -eq 1 ]; then
  run_checked "A0_CHAT_HARD" env BASE_DIR="$BASE_DIR" PY="$PY" \
    bash run_bandit_reference.sh "$MODEL" A0_CHAT_HARD
fi

echo
echo "chat comparator complete; no competence gate is run on chat."
echo "logs -> $LOGDIR"
