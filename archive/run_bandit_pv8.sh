#!/usr/bin/env bash
# pv8 Bandit launcher. pv8 = pv7 + a CHOICE HISTORY block in Stage 1.
#
# Stage 2 is byte-unchanged (the frozen S1 prompt, counts only, no history):
# putting the history there too would let a repeated label prime the candidate
# logits directly, and an arm change could no longer be attributed to the
# model's own reasoning.
#
# WHY THE ALPHA CELLS RUN WITHOUT A pv8 GATE FIRST
# ------------------------------------------------
# pv7's alpha=0 did not pass the frozen gate: SuffFail .250 vs Greedy .250, a
# tie, on the SAME five seeds Greedy fails (Jaccard 1.00). So this sweep is
# failure-mode characterization by default -- the words capability-effect,
# rescue and improvement are not available for it. A0_EASY is still run first
# and gated afterwards, because a pv8 pass would change that wording; but the
# alpha cells are not blocked on it, since the question here is whether alpha
# moves behaviour at all, not whether pv8 beats Greedy.
#
# WHY alpha=0 IS RE-RUN RATHER THAN INHERITED
# -------------------------------------------
# H1 changes the Stage-1 prompt. The stored pv7 alpha=0 trajectories were
# generated without the history block, so they are not this protocol's
# baseline. All three alphas run here on the same seed bank and the same
# reward tapes, one directory per alpha.
#
# Steps:
#   SMOKE      N=3  Easy alpha=0, --attest     protocol correctness only
#   A0_EASY    N=20 Easy alpha=0               the pv8 baseline cell
#   AM4_EASY   N=20 Easy rationale_alpha=-4
#   AP4_EASY   N=20 Easy rationale_alpha=+4
#   GATE       evaluate the stored A0_EASY cell (no GPU)
#
# The no-arg form runs SMOKE only. Each alpha cell is a separate multi-hour
# GPU process writing its own directory, so lumping them into one sequential
# run is never wanted -- launch them on separate cards.
#
# LLAMA3 ONLY: pv8 inherits pv7's token-id invariants (anchor 220, candidates
# 32-35), audited on the Llama-3.1 tokenizer only.
#
# Usage:  bash run_bandit_pv8.sh [llama3] [SMOKE|A0_EASY|AM4_EASY|AP4_EASY|GATE]

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="${DATA:-data1}"
BASE_DIR="${BASE_DIR:-/${DATA}/paveen/Dopamine/components}"
# `python`, NOT `python3.10`: the server conda env names its interpreter
# `python` and exits 127 before anything runs otherwise.
PY="${PY:-python}"

MODEL=${1:-llama3}
ONLY_STEP=${2:-SMOKE}

case "$MODEL" in
  llama3)
    MODEL_NAME="llama3"; MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_SIZE="8B";     HS_PREFIX="llama3";  LAYERS="11-20" ;;
  qwen25)
    echo "pv8 does not support qwen25: it inherits pv7's Llama-3.1" >&2
    echo "  token-id invariants (anchor 220, candidates 32-35)." >&2
    exit 1 ;;
  *) echo "unknown model '$MODEL' (pv8 supports: llama3)"; exit 1 ;;
esac

OUT_ROOT="${BASE_DIR}/${MODEL_NAME}/bandit/pv8"

SMOKE_SEEDS="6 12 13"
FORMAL_SEEDS="$($PY -c "import bandit_reference as br; \
print(' '.join(map(str, br.build_seed_bank(br.get_environment('easy')))))")"

run_cell () {
  local tag="$1" env="$2" seeds="$3" out="$4" ra="$5"; shift 5
  if [ "$ONLY_STEP" != "$tag" ]; then return; fi
  echo ""
  echo "######################################################################"
  echo "# STEP ${tag}: env=${env}  rationale_alpha=${ra}  action_alpha=0"
  echo "# out=${out}"
  echo "# seeds: ${seeds}"
  echo "######################################################################"
  "$PY" -u run_bandit_pv8_episodes.py \
      --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
      --type non --percentage 0.5 --mask_type nmd \
      --layers "${LAYERS}" \
      --reference_environment "${env}" \
      --seeds ${seeds} \
      --rationale_alpha "${ra}" --action_alpha 0 \
      --base_dir "${BASE_DIR}" \
      --ans_file "${out}" \
      "$@"
}

# ── SMOKE (N=3): protocol correctness ONLY ────────────────────────────────
# DO NOT read behaviour off this. Check afterwards:
#   - steering_fires == {rationale: 0, action: 0}   (alpha=0 registers no hook)
#   - invalid_rate == 0.0                            (structural; nonzero = bug)
#   - attestation.round_0.tokens_*.injection_token_id == 220, double_bos false
#   - the Stage 1 prompt CONTAINS "CHOICE HISTORY"
#   - the Stage 2 prompt does NOT contain it
#   - protocol == "pv8", resume_key carries pv8 + the history-block version
#   - wall-clock per episode -> the real N=20 budget
run_cell SMOKE easy "${SMOKE_SEEDS}" "${OUT_ROOT}/pv8_easy_bare_smoke" 0 --attest

# ── The three alpha cells, one directory each ─────────────────────────────
# Separate --ans_file per alpha is MANDATORY, not tidiness: the detail JSON
# name carries neither alpha nor scope, so two cells in one directory would
# overwrite each other.
run_cell A0_EASY  easy "${FORMAL_SEEDS}" "${OUT_ROOT}/pv8_easy_bare"      0
run_cell AM4_EASY easy "${FORMAL_SEEDS}" "${OUT_ROOT}/pv8_easy_bare_am4" -4
run_cell AP4_EASY easy "${FORMAL_SEEDS}" "${OUT_ROOT}/pv8_easy_bare_ap4"  4

# ── GATE: no GPU. Reads the stored A0_EASY cell. ──────────────────────────
if [ "$ONLY_STEP" = "GATE" ]; then
  echo ""
  echo "######################################################################"
  echo "# STEP GATE: frozen rules via the pv8 loader wrapper"
  echo "######################################################################"
  "$PY" -u evaluate_competence_gate_pv8.py \
      --result "${OUT_ROOT}/pv8_easy_bare" \
      --json "${OUT_ROOT}/pv8_gate_verdict.json"
  echo ""
  echo "A pass = competence under a STRUCTURED, PARSER-ASSISTED interface with"
  echo "Policy-following constrained action AND an explicit choice-history"
  echo "scaffold. NOT native free generation. pv7's alpha=0 did not pass, so"
  echo "unless this one does, the alpha cells are failure-mode"
  echo "characterization -- not capability effects."
fi
