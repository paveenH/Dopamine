#!/bin/bash
# Track A reference-bare runner. Bare is the RSN-aligned interface and the
# only interface eligible for the competence gate.
#
# Usage:
#   GPU=0 bash run_bandit_pv6_bare.sh --yes
#   GPU=0 bash run_bandit_pv6_bare.sh --yes --easy-only
#   GPU=0 bash run_bandit_pv6_bare.sh --yes --hard-only
#
# Default: Easy + Hard (40 episodes). Results are resumable, so running
# --easy-only first and --hard-only later does not duplicate completed cells.

set -uo pipefail

MODEL="${MODEL:-llama3}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
# Default to `python`: the server conda env (roleplaying) names its
# interpreter that. `python3.10` is the LOCAL analysis-box convention and
# does not exist on the server, where it exits 127 before anything runs.
PY="${PY:-python}"
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
  echo "STOP: choose a GPU, e.g. GPU=0 bash $0 --yes --easy-only"
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

case "$MODEL" in
  llama3) OUT_ROOT="$BASE_DIR/llama3" ;;
  qwen25) OUT_ROOT="$BASE_DIR/qwen2.5" ;;
  *) echo "unknown MODEL '$MODEL' (want: llama3 | qwen25)"; exit 2 ;;
esac

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
echo "GPU=$CUDA_VISIBLE_DEVICES  MODEL=$MODEL  interface=bare"
if [ "$RUN_EASY" -eq 1 ]; then
  run_checked "A0_BARE_EASY" env BASE_DIR="$BASE_DIR" PY="$PY" \
    bash run_bandit_reference.sh "$MODEL" A0_BARE_EASY
fi
if [ "$RUN_HARD" -eq 1 ]; then
  run_checked "A0_BARE_HARD" env BASE_DIR="$BASE_DIR" PY="$PY" \
    bash run_bandit_reference.sh "$MODEL" A0_BARE_HARD
fi

GATE_ARGS=()
if [ "$RUN_EASY" -eq 1 ]; then
  GATE_ARGS+=(--result "$OUT_ROOT/pv6_easy_bare")
fi
if [ "$RUN_HARD" -eq 1 ]; then
  GATE_ARGS+=(--result "$OUT_ROOT/pv6_hard_bare")
fi
VERDICT_JSON="$LOGDIR/gate_bare_${STAMP}.json"
VERDICT_LOG="$LOGDIR/gate_bare_${STAMP}.log"

echo
echo "========================================================================"
echo "COMPETENCE GATE (bare only)"
echo "========================================================================"
"$PY" evaluate_competence_gate.py "${GATE_ARGS[@]}" \
  --json "$VERDICT_JSON" 2>&1 | tee "$VERDICT_LOG"
GATE_RC=${PIPESTATUS[0]}

if [ "$GATE_RC" -ne 0 ]; then
  [ -s "$VERDICT_JSON" ] || {
    echo "STOP: gate evaluator failed before writing a verdict"; exit 1;
  }
  "$PY" - "$VERDICT_JSON" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1]))
errors = [e for row in rows for e in row.get("errors", [])]
if errors:
    print("STOP: gate is not evaluable:")
    for error in errors:
        print("  -", error)
    raise SystemExit(1)
print("Gate completed: FAIL is a valid experimental outcome, not a launcher error.")
PYEOF
  [ $? -eq 0 ] || exit 1
else
  echo "Gate completed: PASS."
fi

echo
echo "bare results -> $OUT_ROOT/pv6_{easy,hard}_bare (selected environments only)"
echo "gate verdict -> $VERDICT_JSON"
echo "logs         -> $LOGDIR"
