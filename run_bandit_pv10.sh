#!/usr/bin/env bash
# PV10-B launcher: self-paced BAI with subjective commitment.
#
#   bash run_bandit_pv10.sh llama3 [CHECK|A0|AM4|AP4|GATE|ALL]
#
# FROZEN EXECUTION ORDER -- do not reorder:
#
#   CHECK  verify the frozen bases reproduce           (no GPU)
#   A0     alpha = 0, the whole 20-seed bank
#   GATE   minimum capability check on the A0 cell
#   AM4    alpha = -4      \  only after GATE says the interface is executable
#   AP4    alpha = +4      /
#
# ALL runs CHECK -> A0 -> GATE and then STOPS. The alpha cells are deliberately
# not chained: the capability check is a human read, and running -4/+4 before it
# would make the check decorative.
#
# There is no separate smoke cell (a decision taken with the protocol). The
# first seed of every cell is attested inside run_pv10_episode, which raises on
# a steering-fire mismatch, so implementation errors surface on episode 1 rather
# than after the full budget is spent.
#
# PV10 IS LLAMA3-ONLY. The `Reason: ` anchor resolving to token 220 and the
# 11-20 layer band are Llama-3.1 facts. Qwen would need both re-audited and the
# order/seed bases re-frozen.

set -euo pipefail

MODEL="${1:-llama3}"
STEP="${2:-CHECK}"

if [[ "$MODEL" != "llama3" ]]; then
  echo "PV10 is llama3-only: the 'Reason: ' -> token 220 anchor and the 11-20"
  echo "band are Llama-3.1 tokenizer/layer facts, unaudited elsewhere."
  echo "Re-audit both and re-freeze the bases before adding a model."
  exit 1
fi

# The interpreter is named `python` in the server conda env (roleplaying);
# `python3.10` is the LOCAL analysis-box convention and exits 127 there,
# silently killing a nohup job before anything runs.
PY="${PY:-python}"

MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
SIZE="${SIZE:-8B}"
HS="${HS:-llama3}"
TYPE="${TYPE:-non}"
MASK_TYPE="${MASK_TYPE:-nmd}"
PERCENTAGE="${PERCENTAGE:-0.5}"
LAYERS="${LAYERS:-11-20}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/${MODEL}/bandit/pv10}"

# The frozen 20-seed bank. The driver REFUSES a subset: orders are assigned at
# the cell level, so a partial list re-derives different display/initial-pull
# orders and would break seed pairing across alpha cells.
SEEDS="0 1 2 3 4 5 8 11 14 19 22 23 26 31 32 46 48 50 53 57"

run_cell () {
  local alpha="$1" dir="$2"
  echo "=== PV10-B  alpha=${alpha}  ->  ${dir}"
  ${PY} run_bandit_pv10_episodes.py \
    --model "${MODEL}" --model_dir "${MODEL_DIR}" --size "${SIZE}" \
    --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
    --alpha "${alpha}" --seeds ${SEEDS} \
    --base_dir "${BASE_DIR}" --ans_file "${dir}"
}

check_bases () {
  echo "=== frozen bases"
  ${PY} evaluate_pv10_capability.py --check
  ${PY} pv10_env_prescreen.py --n_sim 10000 --check
  echo "=== protocol tests"
  ${PY} test_bandit_pv10.py
  ${PY} test_bandit_pv10_episode.py
  ${PY} evaluate_pv10_capability.py --selftest
}

case "${STEP}" in
  CHECK) check_bases ;;

  A0)    check_bases
         run_cell 0 "${OUT_ROOT}/pv10_a0" ;;

  GATE)  ${PY} evaluate_pv10_capability.py --result "${OUT_ROOT}/pv10_a0" ;;

  AM4)   run_cell -4 "${OUT_ROOT}/pv10_am4" ;;
  AP4)   run_cell  4 "${OUT_ROOT}/pv10_ap4" ;;

  ALL)   check_bases
         run_cell 0 "${OUT_ROOT}/pv10_a0"
         ${PY} evaluate_pv10_capability.py --result "${OUT_ROOT}/pv10_a0"
         echo
         echo "STOP. Read the capability verdict above before running the"
         echo "alpha cells:  bash run_bandit_pv10.sh llama3 AM4"
         echo "              bash run_bandit_pv10.sh llama3 AP4"
         ;;

  *) echo "unknown step: ${STEP}"
     echo "expected one of: CHECK A0 GATE AM4 AP4 ALL"
     exit 1 ;;
esac
