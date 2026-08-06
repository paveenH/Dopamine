#!/usr/bin/env bash
# pv7 Bandit launcher. Phase-gated: alpha comes LAST, and only after the
# alpha=0 Easy-bare competence gate passes on pv7's OWN interface.
#
# pv7 does not inherit pv6's gate verdict. pv6's Easy-bare pass was measured on
# a different interface (truncated rationale, self-contradicting Stage 2,
# drifting option display, uncontrolled label prior). pv7 fixes all four, so it
# needs its own alpha=0 anchor before any alpha claim.
#
# Interfaces frozen before any trajectory ran, selected on the frozen state
# bank by validity / grounding / completion / cost -- never by reward:
#   Stage 1 = P1b   (policy parse rate 60.7% -> 100%)
#   Stage 2 = S1    (non-A policy overridden by A 54.3% -> 1.9%, McNemar
#                    n=428, discordant 178/0, p<1e-4; rotation-invariant)
#
# Steps:
#   SMOKE      N=3  Easy alpha=0, --attest      protocol correctness only
#   A0_EASY    N=20 Easy alpha=0                the competence gate cell
#   GATE       evaluate the stored A0_EASY cell (no GPU)
#   A0_HARD    N=20 Hard alpha=0                optional second anchor
#
# The no-arg form runs SMOKE only. Everything past it is deliberately opt-in:
# each is a multi-hour GPU process and must be a conscious choice.
#
# LLAMA3 ONLY for now. pv7's invariants are token-id facts about the Llama-3.1
# tokenizer -- the anchor tail is token 220 and the candidates are 32-35 -- and
# neither has been audited on Qwen2.5, whose tokenizer may not even encode a
# bare trailing space as one token. The runtime audit would fail fast, but a
# usage line offering `qwen25` implies support that does not exist. Adding it
# is a real task: re-audit both invariants, then re-freeze the state bank.
#
# Usage:  bash run_bandit_pv7.sh [llama3] [SMOKE|A0_EASY|GATE|A0_HARD]

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="${DATA:-data1}"
BASE_DIR="${BASE_DIR:-/${DATA}/paveen/Dopamine/components}"
# `python`, NOT `python3.10`: the server conda env names its interpreter
# `python` and exits 127 before anything runs otherwise -- a nohup job then
# dies silently. `python3.10` is the LOCAL analysis-box convention.
PY="${PY:-python}"

MODEL=${1:-llama3}
ONLY_STEP=${2:-SMOKE}

case "$MODEL" in
  llama3)
    MODEL_NAME="llama3"; MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_SIZE="8B";     HS_PREFIX="llama3";  LAYERS="11-20" ;;
  qwen25)
    echo "pv7 does not support qwen25 yet." >&2
    echo "  Its anchor (token 220) and candidates (32-35) are Llama-3.1" >&2
    echo "  tokenizer facts and have NOT been audited on Qwen2.5." >&2
    echo "  Adding it means re-auditing both, then re-freezing the state" >&2
    echo "  bank -- not a launcher edit. Use pv6 for Qwen cross-model work." >&2
    exit 1 ;;
  *) echo "unknown model '$MODEL' (pv7 supports: llama3)"; exit 1 ;;
esac

OUT_ROOT="${BASE_DIR}/${MODEL_NAME}/bandit/pv7"

# Smoke seeds are DISJOINT from the formal 20-seed bank, so a smoke can never
# contaminate a gate cell. Same values as pv6's smoke bank -- the environment
# and its seed banks are unchanged, which is what makes the frozen algorithmic
# baselines reusable for pv7's gate.
SMOKE_SEEDS="6 12 13"
FORMAL_SEEDS="$($PY -c "import bandit_reference as br; \
print(' '.join(map(str, br.build_seed_bank(br.get_environment('easy')))))")"
FORMAL_SEEDS_HARD="$($PY -c "import bandit_reference as br; \
print(' '.join(map(str, br.build_seed_bank(br.get_environment('hard')))))")"

run_cell () {
  local tag="$1" env="$2" seeds="$3" out="$4"; shift 4
  if [ "$ONLY_STEP" != "$tag" ]; then return; fi
  echo ""
  echo "######################################################################"
  echo "# STEP ${tag}: env=${env}  alpha=0  out=${out}"
  echo "# seeds: ${seeds}"
  echo "######################################################################"
  "$PY" -u run_bandit_pv7_episodes.py \
      --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
      --type non --percentage 0.5 --mask_type nmd \
      --layers "${LAYERS}" \
      --reference_environment "${env}" \
      --seeds ${seeds} \
      --rationale_alpha 0 --action_alpha 0 \
      --base_dir "${BASE_DIR}" \
      --ans_file "${out}" \
      "$@"
}

# ── SMOKE (N=3): protocol correctness ONLY ────────────────────────────────
# DO NOT read behaviour off this. N=3 cannot separate a policy from seed luck
# and the smoke seeds are not counterbalanced. Check afterwards:
#   - steering_fires == {rationale: 0, action: 0}   (alpha=0 registers no hook)
#   - invalid_rate == 0.0                            (structural; nonzero = bug)
#   - attestation.round_0.tokens_{rationale,action}.injection_token_id == 220
#   - attestation.round_0.tokens_*.double_bos == false
#   - the two rendered prompts read as intended, and share one OPTIONS order
#   - policy_parse_rate: how often Stage 1 emitted a parseable Policy
#   - action_follows_policy_rate: S1 predicts ~0.99 on frozen states
#   - wall-clock per episode -> the real N=20 budget
run_cell SMOKE easy "${SMOKE_SEEDS}" "${OUT_ROOT}/pv7_easy_bare_smoke" --attest

# ── A0_EASY (N=20, T=100): the competence-gate cell ───────────────────────
run_cell A0_EASY easy "${FORMAL_SEEDS}" "${OUT_ROOT}/pv7_easy_bare"

# ── A0_HARD (N=20): optional second anchor ────────────────────────────────
# pv6's Hard-bare failed rule 1 on coverage (10/20 episodes never tried all 5
# arms). Whether pv7's interface changes that is a real question, but Hard is
# NOT required for the alpha line -- Easy is the anchor.
run_cell A0_HARD hard "${FORMAL_SEEDS_HARD}" "${OUT_ROOT}/pv7_hard_bare"

# ── GATE: no GPU. Reads the stored A0_EASY cell. ──────────────────────────
if [ "$ONLY_STEP" = "GATE" ]; then
  echo ""
  echo "######################################################################"
  echo "# STEP GATE: frozen rules via the pv7 loader wrapper"
  echo "######################################################################"
  "$PY" -u evaluate_competence_gate_pv7.py \
      --result "${OUT_ROOT}/pv7_easy_bare" \
      --json "${OUT_ROOT}/pv7_gate_verdict.json"
  echo ""
  echo "A pass = competence under a STRUCTURED, PARSER-ASSISTED interface with"
  echo "Policy-following constrained action. NOT native free generation:"
  echo "P1b's native termination still fails and the Policy is extractor-"
  echo "recovered. Only after this passes does the Stage-1 alpha sweep run."
fi
