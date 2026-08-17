#!/usr/bin/env bash
# PV10-A launcher: FIXED-BUDGET mechanism control for PV10-B.
#
#   bash run_bandit_pv10a.sh llama3 [CHECK|A0|AM4|AP4|PM4]
#
# TWO-GPU LAYOUT (one card each, run concurrently):
#   CUDA_VISIBLE_DEVICES=0 bash run_bandit_pv10a.sh llama3 A0
#   CUDA_VISIBLE_DEVICES=1 bash run_bandit_pv10a.sh llama3 PM4
#
# llms.py loads with device_map="auto", which claims every VISIBLE device, so
# CUDA_VISIBLE_DEVICES is what actually separates the two jobs -- without it
# both processes would map across both cards and collide.
#
# PV10-B's alpha=0 result confounds two things: the model committed after a
# median of 13 samples with some arm still at a single pull, so weak
# identification could be premature stopping OR a broken acquisition policy.
# PV10-A removes the stopping decision -- SAMPLE only until the budget is
# spent, then the FROZEN PV10-B terminal prompt asks for the commit.
#
# RUN alpha=0 FIRST AND READ IT BEFORE THE ALPHA CELLS:
#   * single-arm share still ~.6-.9  -> the defect is the ACQUISITION policy;
#     early stopping was a symptom, and +-4 on PV10-A adds little
#   * allocation/accuracy clearly improve -> PV10-B's defect is a too-low
#     subjective commitment threshold; THEN +-4 is worth running to see
#     whether RSN moves sampling, the threshold, or only the text
#
# PV10-A is a CONTROL, not a competence claim. Its accuracy is not comparable
# to a self-paced PV10-B number: the model did not choose when to stop.
#
# v1 (pv10a_a0/_am4/_ap4) is VOID: an alpha=0 / +-4 stop-path asymmetry plus a
# runtime/parser contract gap voided 58/60 episodes, and 0/20 alpha=0 episodes
# reached n=100. Fixed in pv10-strict-v2; v2 results go to pv10a_v2_*, which
# cannot resume into the v1 cells. See pv10_capability_amendment_02.json.
#
# INTERPRETATION ORDER IS UNCHANGED BY THE GPU LAYOUT. Running +-4 on a second
# card concurrently is a scheduling choice, not a licence to read them first:
# alpha=0 is what says whether the fixed-budget control WORKS at all. If A0
# does not reach n=100, the +-4 cells are void for the same reason v1 was, and
# no alpha contrast may be reported from them.

set -euo pipefail

MODEL="${1:-llama3}"
STEP="${2:-CHECK}"

if [[ "$MODEL" != "llama3" ]]; then
  echo "PV10 is llama3-only (token 220 anchor, 11-20 band)."
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
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/${MODEL}/bandit/pv10a}"

# The SAME frozen 20-seed bank as PV10-B, so A and B are seed-paired.
SEEDS="0 1 2 3 4 5 8 11 14 19 22 23 26 31 32 46 48 50 53 57"

run_cell () {
  local alpha="$1" dir="$2"
  echo "=== PV10-A  alpha=${alpha}  ->  ${dir}"
  echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: ALL GPUs>}  pid=$$"
  ${PY} run_bandit_pv10a_episodes.py \
    --model "${MODEL}" --model_dir "${MODEL_DIR}" --size "${SIZE}" \
    --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
    --alpha "${alpha}" --seeds ${SEEDS} \
    --base_dir "${BASE_DIR}" --ans_file "${dir}"
}

case "${STEP}" in
  CHECK) ${PY} evaluate_pv10_capability.py --check
         ${PY} test_bandit_pv10.py
         ${PY} test_bandit_pv10a.py
         ${PY} test_bandit_pv10_episode.py
         ${PY} test_pv10_gate_end_to_end.py
         ${PY} test_pv10_stop_parity.py ;;

  A0)    ${PY} test_bandit_pv10a.py
         ${PY} test_pv10_stop_parity.py
         run_cell 0 "${OUT_ROOT}/pv10a_v2_a0"
         echo
         echo "STOP. Read the alpha=0 allocation before running +-4:"
         echo "  single-arm share still high -> acquisition defect"
         echo "  allocation improved         -> stopping-threshold defect" ;;

  AM4)   run_cell -4 "${OUT_ROOT}/pv10a_v2_am4" ;;
  AP4)   run_cell  4 "${OUT_ROOT}/pv10a_v2_ap4" ;;

  # Both alpha cells SEQUENTIALLY on one card. They share a device, so they
  # must not run at the same time; A0 runs concurrently on the other card.
  PM4)   ${PY} test_pv10_stop_parity.py
         run_cell -4 "${OUT_ROOT}/pv10a_v2_am4"
         run_cell  4 "${OUT_ROOT}/pv10a_v2_ap4" ;;

  *) echo "unknown step: ${STEP}"
     echo "expected: CHECK A0 AM4 AP4 PM4"; exit 1 ;;
esac
