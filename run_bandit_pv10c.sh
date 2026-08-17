#!/usr/bin/env bash
# PV10-C: PV10-B + a competitor/falsification cue in the sampling clause.
#
#   bash run_bandit_pv10c.sh llama3 [CHECK|B0|C0]
#
# TWO-GPU LAYOUT (one card each, concurrently):
#   CUDA_VISIBLE_DEVICES=0 bash run_bandit_pv10c.sh llama3 B0
#   CUDA_VISIBLE_DEVICES=1 bash run_bandit_pv10c.sh llama3 C0
#
# llms.py loads with device_map="auto", which claims every VISIBLE device, so
# CUDA_VISIBLE_DEVICES is what actually separates the two jobs.
#
# WHY B-v2 A0 IS RE-RUN RATHER THAN REUSED: the stored PV10-B alpha=0 was
# generated under parser v1. Comparing C-A0 against it would differ in prompt
# AND parser version at once. The v2 boundary fix changes 1 round in 412 for
# that cell, and one changed action changes every later state, so it cannot be
# recovered by re-parsing -- it has to be regenerated. B-v1 stays citable
# against its own preserved v1 basis; it just cannot be C's baseline.
#
# The two cells share seeds, tapes, orders and parser v2. They differ ONLY in
# the Stage-1 sampling clause, which enters the resume key via
# STAGE1_INSTRUCTION_VERSION (p10 vs p10c) -- so they cannot resume into each
# other, and each writes its own directory.
#
# ALPHA=0 ONLY. Do not add +-4 here. The gate is whether the cue changes
# ACQUISITION: min_trials leaving 1, one-shot-zero arms being revisited,
# max_arm_share falling, true_top2 improving. If C-A0 still shows min_trials=1
# and no one-shot-zero revisits, the cue did not work and PV10-C stops.

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
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/${MODEL}/bandit/pv10c}"

# The SAME frozen 20-seed bank as PV10-A/B, so all cells are seed-paired.
SEEDS="0 1 2 3 4 5 8 11 14 19 22 23 26 31 32 46 48 50 53 57"

run_cell () {
  local variant="$1" dir="$2"
  echo "=== PV10 alpha=0  variant=${variant}  ->  ${dir}"
  echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: ALL GPUs>}  pid=$$"
  ${PY} run_bandit_pv10_episodes.py \
    --model "${MODEL}" --model_dir "${MODEL_DIR}" --size "${SIZE}" \
    --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
    --prompt_variant "${variant}" \
    --alpha 0 --seeds ${SEEDS} \
    --base_dir "${BASE_DIR}" --ans_file "${dir}"
}

case "${STEP}" in
  CHECK) ${PY} test_bandit_pv10c.py ;;

  # Baseline: PV10-B prompt under parser v2.
  B0)    ${PY} test_bandit_pv10c.py
         run_cell B "${OUT_ROOT}/pv10b_v2_a0" ;;

  # Intervention: + competitor cue.
  C0)    ${PY} test_bandit_pv10c.py
         run_cell C "${OUT_ROOT}/pv10c_a0" ;;

  *) echo "unknown step: ${STEP}"
     echo "expected: CHECK B0 C0"; exit 1 ;;
esac
