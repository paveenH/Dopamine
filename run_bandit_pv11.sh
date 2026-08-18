#!/usr/bin/env bash
# PV11: Controlled Evidence-State Micro-Episodes.
#
#   bash run_bandit_pv11.sh llama3 CHECK   # no GPU, ~10s
#   bash run_bandit_pv11.sh llama3 SMOKE   # 12 states, both blocks
#   bash run_bandit_pv11.sh llama3 A0      # 160 states, alpha=0
#   bash run_bandit_pv11.sh llama3 GATE    # read M1/M2, then STOP
#
# The order is CHECK -> SMOKE -> A0 -> GATE and the launcher STOPS at GATE. There is
# deliberately no step that runs -4/+4:
#
#   M1 and M2 both pass -> implement and run the steered cells on the SAME
#                          bank, tapes and parser. That is a separate change.
#   either one fails    -> the protocol CLOSES. Do not modify the prompt to
#                          chase the gate, and do not run +-4.
#
# The second branch is the whole point of the file. PV10-A/B/C ended after four
# representation-layer interventions each raised recognition without moving
# acquisition; a fifth round of prompt edits aimed at a failing gate is exactly
# the loop this protocol was built to escape. A gate failure here is a finding
# about the interface, not a bug to fix.
#
# WHY ALPHA=0 CANNOT BE OVERRIDDEN HERE: run_bandit_pv11_episodes.py refuses a
# non-zero --alpha outright. This launcher exposes no alpha flag at all, so the
# two guards are independent -- editing one does not silently unlock the other.
#
# TWO-GPU LAYOUT: A0 is a single sequential cell, so there is nothing to
# parallelize. If you split it by block, give each job its own --ans_file AND
# its own CUDA_VISIBLE_DEVICES; llms.py loads with device_map="auto", which
# claims every VISIBLE device, so without it two jobs map across both cards and
# collide. Note the gate REQUIRES a complete block: a partial block is a hard
# error, not a subset.

set -euo pipefail

MODEL="${1:-llama3}"
STEP="${2:-CHECK}"

if [[ "$MODEL" != "llama3" ]]; then
  echo "PV11 is llama3-only: the 'Reason: ' anchor is token 220 and the bare"
  echo "candidate letters are 32-35 on the Llama-3.1 tokenizer. Neither has"
  echo "been audited on Qwen2.5. Adding a model means re-auditing both and"
  echo "re-freezing the state bank."
  exit 1
fi

PY="${PY:-python}"          # server conda env; python3.10 exits 127 there

MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
SIZE="${SIZE:-8B}"
HS="${HS:-llama3}"
TYPE="${TYPE:-non}"
MASK_TYPE="${MASK_TYPE:-nmd}"
PERCENTAGE="${PERCENTAGE:-0.5}"
LAYERS="${LAYERS:-11-20}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/${MODEL}/bandit/pv11}"
A0_DIR="${OUT_ROOT}/pv11_a0"
A0_FILE="${A0_DIR}/bandit_pv11_alpha0.json"
# Smoke output lives under the SAME tree as the formal cells, never /tmp:
# a smoke file in a system temp dir is deleted on reboot and is invisible to
# anyone reading the results tree, so a puzzling formal result cannot be
# checked against the smoke that preceded it. Its own subdirectory, and a
# name that says what it is, so it can never be mistaken for a gateable cell.
SMOKE_DIR="${OUT_ROOT}/smoke"

# Every check is offline and runs BEFORE any GPU time is spent. The bank check
# is first: a bank that does not reproduce from its builder makes everything
# downstream uninterpretable, and finding that out after a multi-hour run is
# the expensive way to learn it.
check () {
  echo "=== PV11 offline checks"
  ${PY} build_pv11_state_bank.py --check
  ${PY} test_pv11_state_bank.py
  ${PY} test_bandit_pv11.py
  ${PY} test_bandit_pv11_episode.py
  ${PY} analyze_bandit_pv11_gate.py --selftest
  ${PY} test_pv11_gate_end_to_end.py
  echo
  echo "NOTE: if test_bandit_pv11.py printed 'SKIP tokenizer unavailable',"
  echo "the token-220 anchor assertions did NOT run on this machine. Sync the"
  echo "HF cache and re-run CHECK before trusting the anchor."
}

# Smoke: the ONLY step before A0 where a real model sees a real prompt.
# Everything in CHECK is FakeVC or pure string work, so a prompt that the
# model cannot follow would pass every offline test and only surface an hour
# into A0. Deliberately covers BOTH blocks: M2 is the protocol's core question
# and a Commitment-only smoke leaves it untested.
smoke () {
  local block="$1" n="$2"
  local file="${SMOKE_DIR}/pv11_smoke_${block}.json"
  mkdir -p "${SMOKE_DIR}"
  echo
  echo "=== PV11 SMOKE  block=${block}  n=${n}  ->  ${file}"
  echo "    NOT gateable -- a partial run by construction."
  ${PY} run_bandit_pv11_episodes.py \
    --model_dir "${MODEL_DIR}" --size "${SIZE}" \
    --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
    --alpha 0 --block "${block}" --limit "${n}" \
    --base_dir "${BASE_DIR}" --ans_file "${file}"
}

case "${STEP}" in
  CHECK)
    check ;;

  # Read the generated TEXT, not only the summary line. What matters is
  # whether the model follows the two-line format, whether it drifts into
  # document/code completion, and whether the first action varies at all --
  # a first action that is constant across cells cannot move either gate rule.
  SMOKE)
    check
    smoke commitment "${SMOKE_N:-6}"
    smoke acquisition "${SMOKE_N:-6}"
    echo
    echo "Inspect the text before A0:"
    echo "  ${PY} inspect_pv11_smoke.py ${SMOKE_DIR}" ;;

  A0)
    check
    mkdir -p "${A0_DIR}"
    echo
    echo "=== PV11 alpha=0  160 states  ->  ${A0_FILE}"
    echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: ALL GPUs>}  pid=$$"
    ${PY} run_bandit_pv11_episodes.py \
      --model_dir "${MODEL_DIR}" --size "${SIZE}" \
      --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
      --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
      --alpha 0 \
      --base_dir "${BASE_DIR}" --ans_file "${A0_FILE}"
    echo
    echo "Done. Next: bash run_bandit_pv11.sh ${MODEL} GATE" ;;

  GATE)
    if [[ ! -f "${A0_FILE}" ]]; then
      echo "no alpha=0 result at ${A0_FILE}"
      echo "run: bash run_bandit_pv11.sh ${MODEL} A0"
      exit 1
    fi
    # The gate exits non-zero on FAIL, which under `set -e` would end the
    # script before the closing note prints. The verdict is the deliverable
    # either way, so capture it and report explicitly.
    set +e
    ${PY} analyze_bandit_pv11_gate.py --result "${A0_FILE}"
    verdict=$?
    set -e
    echo
    if [[ ${verdict} -eq 0 ]]; then
      echo "GATE PASSED. The -4/+4 cells may now be implemented and run on"
      echo "the SAME bank, tapes and parser. That is a separate change to"
      echo "the driver and this launcher; nothing here runs them."
    else
      echo "GATE FAILED. Per the frozen wording: manipulation-gate failure"
      echo "does not trigger prompt iteration. This controlled interface did"
      echo "not elicit the prerequisite behavior needed for an interpretable"
      echo "steering test; no steered cells are run."
    fi
    exit ${verdict} ;;

  *)
    echo "unknown step: ${STEP}"
    echo "expected: CHECK SMOKE A0 GATE"
    exit 1 ;;
esac
