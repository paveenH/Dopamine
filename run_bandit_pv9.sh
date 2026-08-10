#!/usr/bin/env bash
# PV9 Bandit launcher. PV9 = pv8 + four Stage-1 modifications + a `#` stop
# string, across TWO environments.
#
# THE FOUR STAGE-1 MODIFICATIONS
#   1. self-relevant reward framing ("Your score so far: N points.")
#   2. untried-arm exploration cue, on n=0 rows ONLY
#   3. generation control: 128 tokens, 50-word instruction, `#` stop string
#   4. explicit Bernoulli structure (fixed unknown p, differs across buttons)
#
# Stage 2 is BYTE-UNCHANGED (the frozen S1 prompt, counts only). The cue,
# score and history all stay in Stage 1: putting the cue in Stage 2 would
# raise that button's candidate logit directly, and an arm change could no
# longer be attributed to Stage 1's reasoning rather than to priming.
#
# TWO ENVIRONMENTS, DIFFERENT JOBS -- do not read them the same way
# -----------------------------------------------------------------
#   easy     .75/.25x3   competence-eligible. Carries discovery, one-shot-zero
#                        revisit, information investment, outcome.
#   neartie  .60/.55/.25/.25   NOT competence-eligible. Exists because easy
#                        structurally has no close-value states, so a
#                        precision effect has nowhere to show. At gap .05 with
#                        ~25 pulls/arm the empirical SE (~.10) is DOUBLE the
#                        gap, so its SuffFail measures the environment, not
#                        the policy. Its gate output is DIAGNOSTIC ONLY.
#
# WHAT NEARTIE ONLINE CAN AND CANNOT SHOW
# ---------------------------------------
# The three alphas diverge into different states within a few rounds, so this
# gives ECOLOGICAL evidence (does alpha change outcomes in a near-tie world),
# NOT matched-state precision evidence. The matched-state test is the frozen
# probe, deliberately deferred -- it is a later, cheaper, causal step and not
# a prerequisite for tonight.
#
# THE CUE IS A SCAFFOLD -- binding on how results may be worded
# ------------------------------------------------------------
# pv8 measured a ~3% EXPLORE floor, and H2 (information investment) is
# untestable at a floor, so the cue exists to lift the readout off it. It
# states a benefit direction, which puts it on the strategy side of the
# native/scaffold line. Any exploration observed here is SCAFFOLDED discovery.
# The cue also vanishes once an arm has one pull, so it CANNOT drive
# one-shot-zero revisits -- that failure mode stays untouched and measurable.
#
# WHY alpha=0 IS RE-RUN RATHER THAN INHERITED
# -------------------------------------------
# The Stage-1 prompt changed, so no stored pv7/pv8 alpha=0 cell is this
# protocol's baseline. All three alphas run here on the same seed bank and the
# same reward tapes, one directory per alpha.
#
# Steps:
#   SMOKE       N=3  Easy alpha=0, --attest      protocol correctness only
#   SMOKE_AM4   N=1  Easy alpha=-4, --attest     the STEERED path (see below)
#   A0_EASY / AM4_EASY / AP4_EASY        N=20 Easy,    ra = 0 / -4 / +4
#   A0_NT   / AM4_NT   / AP4_NT          N=20 NearTie, ra = 0 / -4 / +4
#   ALL         the six formal cells, sequentially (unattended overnight)
#   GATE        evaluate both stored alpha=0 cells (no GPU)
#
# The no-arg form runs SMOKE only. ALL is the overnight form; to split across
# two cards, run the three EASY steps on one and the three NT steps on the
# other.
#
# LLAMA3 ONLY: PV9 inherits pv7's token-id invariants (anchor 220, candidates
# 32-35), audited on the Llama-3.1 tokenizer only.
#
# Usage:
#   bash run_bandit_pv9.sh [llama3] [SMOKE|SMOKE_AM4|ALL|GATE|
#                                    A0_EASY|AM4_EASY|AP4_EASY|
#                                    A0_NT|AM4_NT|AP4_NT]
#
# Run SMOKE then SMOKE_AM4 before ALL. The two smokes cover DIFFERENT code:
# alpha=0 never registers a hook, so only the steered smoke exercises
# `regenerate`, the mask load and the 900-fire count. The formal cells do fail
# closed on a bad fire count, but by then A0_EASY may have burned hours.

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
    echo "PV9 does not support qwen25: it inherits pv7's Llama-3.1" >&2
    echo "  token-id invariants (anchor 220, candidates 32-35)." >&2
    exit 1 ;;
  *) echo "unknown model '$MODEL' (PV9 supports: llama3)"; exit 1 ;;
esac

OUT_ROOT="${BASE_DIR}/${MODEL_NAME}/bandit/pv9"

SMOKE_SEEDS="6 12 13"
# easy and neartie share a bank: make_arm_map depends only on (seed, k), and
# both are K=4. That is what makes the two environments seed-paired.
FORMAL_SEEDS="$($PY -c "import bandit_reference as br; \
print(' '.join(map(str, br.build_seed_bank(br.get_environment('easy')))))")"

# The frozen basis must reproduce BEFORE any GPU time is spent: if it does
# not, nothing produced tonight is citable.
"$PY" -u freeze_pv9_baseline.py --check >/dev/null || {
  echo "PV9 baseline manifest does not reproduce -- refusing to run." >&2
  exit 1; }

run_cell () {
  local tag="$1" env="$2" seeds="$3" out="$4" ra="$5"; shift 5
  if [ "$ONLY_STEP" != "$tag" ] && [ "$ONLY_STEP" != "ALL" ]; then return; fi
  if [ "$ONLY_STEP" = "ALL" ] && [ "$tag" = "SMOKE" ]; then return; fi
  echo ""
  echo "######################################################################"
  echo "# STEP ${tag}: env=${env}  rationale_alpha=${ra}  action_alpha=0"
  echo "# out=${out}"
  echo "# seeds: ${seeds}"
  echo "######################################################################"
  "$PY" -u run_bandit_pv9_episodes.py \
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
#   - the Stage 1 prompt CONTAINS "Your score so far", "CHOICE HISTORY" and
#     (when an arm is untried) the exploration cue
#   - the Stage 2 prompt contains NONE of those three
#   - stop_reason_counts: how many rounds ended clean vs continued
#   - protocol == "pv9"; resume_key carries pv9 + score/cue/stop versions
#   - wall-clock per episode -> the real 6 x N=20 budget
run_cell SMOKE easy "${SMOKE_SEEDS}" "${OUT_ROOT}/pv9_easy_bare_smoke" 0 --attest

# ── SMOKE_AM4 (N=1): the STEERED path, which alpha=0 cannot reach ─────────
# alpha=0 registers no hook anywhere, so SMOKE above never touches
# `regenerate`, the mask file, or the fire counter. Check afterwards:
#   - steering_fires == {rationale: 900, action: 0}  (L=9 x T=100; a 0 means
#     the hook never fired, 3200 means zero rows are being counted)
#   - the mask actually loaded from base_dir (the driver exits if it is absent)
#   - one episode's wall clock -> the six-cell budget
run_cell SMOKE_AM4 easy "6" "${OUT_ROOT}/pv9_easy_bare_smoke_am4" -4 --attest

# ── The six formal cells, one directory each ──────────────────────────────
# Separate --ans_file per cell is MANDATORY, not tidiness: the detail JSON
# name carries the environment but neither alpha nor scope, so two alphas in
# one directory would overwrite each other.
run_cell A0_EASY  easy    "${FORMAL_SEEDS}" "${OUT_ROOT}/pv9_easy_bare"      0
run_cell AM4_EASY easy    "${FORMAL_SEEDS}" "${OUT_ROOT}/pv9_easy_bare_am4" -4
run_cell AP4_EASY easy    "${FORMAL_SEEDS}" "${OUT_ROOT}/pv9_easy_bare_ap4"  4
run_cell A0_NT    neartie "${FORMAL_SEEDS}" "${OUT_ROOT}/pv9_nt_bare"        0
run_cell AM4_NT   neartie "${FORMAL_SEEDS}" "${OUT_ROOT}/pv9_nt_bare_am4"   -4
run_cell AP4_NT   neartie "${FORMAL_SEEDS}" "${OUT_ROOT}/pv9_nt_bare_ap4"    4

# ── GATE: no GPU. Reads the stored alpha=0 cells. ─────────────────────────
if [ "$ONLY_STEP" = "GATE" ]; then
  echo ""
  echo "######################################################################"
  echo "# STEP GATE: frozen rules via the PV9 loader wrapper"
  echo "######################################################################"
  "$PY" -u evaluate_competence_gate_pv9.py \
      --result "${OUT_ROOT}/pv9_easy_bare" \
      --json "${OUT_ROOT}/pv9_gate_verdict_easy.json"
  echo ""
  echo "# NearTie below is DIAGNOSTIC ONLY (competence_eligible=False)."
  "$PY" -u evaluate_competence_gate_pv9.py \
      --result "${OUT_ROOT}/pv9_nt_bare" \
      --json "${OUT_ROOT}/pv9_gate_diag_neartie.json"
  echo ""
  echo "Easy pass = competence under a STRUCTURED, PARSER-ASSISTED interface"
  echo "with Policy-following constrained action, a choice-history scaffold, a"
  echo "self-relevant score AND a scaffolded untried-arm cue. NOT native free"
  echo "generation, and NOT autonomous exploration. NearTie is never a"
  echo "competence verdict."
fi
