#!/bin/bash
#
# ================ BANDIT pv6 F-REFERENCE ================
# The clean-slate redesign (BanditExperiment_LiteratureReview.md §3). It does
# NOT continue the pv1-pv5 line: environment, prompt, decoding, reward model
# and metrics are all different, and pre-2026-07-28 Bandit results are void
# (position leakage + permissive parser), so nothing here is comparable to a
# legacy cell. Separate output tree, separate CSV schema, separate protocol
# version — by design, so the two can never be pooled by accident.
#
# WHAT THIS MEASURES: a CAPABILITY BOUNDARY, and whether alpha moves it. Not
# "make the task work". Krishnamurthy et al. (NeurIPS 2024) found exactly ONE
# configuration that explores successfully across all their experiments
# (GPT-4 BSSC0); Llama2 failed on its entire restricted sweep. A 7-8B model
# failing here is the literature's expectation, not a harness bug — which is
# why the competence gate is pre-registered rather than chosen after seeing
# the numbers.
#
# PROTOCOL (frozen, §3.3):
#   Stage 1  free rationale, <=64 tokens, temperature 0
#   Stage 2  state + sanitized rationale + `Choice: Button` anchor; the arm is
#            chosen by candidate-only sequence log-probability argmax
#   => invalid_rate is STRUCTURALLY 0. There is no parser and no random
#      fallback, so a "format failure" cannot be confused with a bad choice.
#
# WHERE ALPHA LANDS is set by --steering_scope; the injection SHAPE is frozen
# for both scopes (once, at that pass's last prefill token, prefill_only,
# tail_len=1 — decode is never steered):
#   scope=action  Stage 1 gets NO alpha (vc.generate, no hook registered at
#                 all); only Stage 2 is steered. This is the original pv6
#                 semantics and is now the mechanism ABLATION: it measures
#                 whether alpha moves the arm logits GIVEN a fixed rationale.
#   scope=both    Stage 1 is ALSO steered, at its own last prefill token, so
#                 alpha reaches the choice by two routes (alpha -> rationale
#                 text -> action, and alpha -> action logits). This is the B1
#                 MAIN experiment: the question is whether alpha moves the
#                 whole information-seeking policy, not just the readout.
#   alpha=0       identical under both scopes — no hook is registered either
#                 way, so "unsteered" is not "steered by zero". This is what
#                 makes the stored Track A alpha=0 cells reusable for B1.
#
# INTERFACES. Two, and they answer different questions:
#   bare  = RSN-aligned. The NMD mask was extracted on bare-string prompts, so
#           this is the ONLY interface eligible for the B1 alpha main
#           experiment (see CLAUDE.md on chat-template misalignment).
#   chat  = the literature capability comparator (Krishnamurthy et al. use
#           chat models). Run it to report capability, not to sweep alpha.
#
# ENVIRONMENTS (§3.2): mu* = 0.5 + D/2, mu = 0.5 - D/2.
#   easy          K=4  .75/.25x3   T=100   D=0.50   competence-gate eligible
#   hard          K=5  .60/.40x4   T=100   D=0.20   competence-gate eligible
#   native_floor  K=2  .70/.30     T=50    minimal stochastic-adaptation
#                 diagnostic ONLY — NOT a competence anchor, and no B1 alpha
#                 main experiment runs on it.
#
# BEFORE running anything here:
#   python3.10 freeze_bandit_baseline.py --check
#   python3.10 run_bandit_algorithmic_baseline.py --reference_environment easy
# The frozen manifest fixes the gate's comparison basis (seed bank +
# Random/Greedy/Oracle + bootstrap CIs) BEFORE any model behaviour is seen.
#
# Usage:
#   bash run_bandit_reference.sh llama3 SMOKE       # N=3, easy+hard, bare+chat
#   bash run_bandit_reference.sh llama3 A0_BARE     # Track A alpha=0, bare
#   bash run_bandit_reference.sh llama3 A0_BARE_EASY  # one formal cell only
#   bash run_bandit_reference.sh llama3 A0_BARE_HARD  # one formal cell only
#   bash run_bandit_reference.sh llama3 A0_CHAT     # Track A alpha=0, chat
#   bash run_bandit_reference.sh llama3 A0_CHAT_EASY  # one formal cell only
#   bash run_bandit_reference.sh llama3 A0_CHAT_HARD  # one formal cell only
#   bash run_bandit_reference.sh llama3             # every non-smoke step
#
#   bash run_bandit_reference.sh llama3 B1_BOTH_EASY  # alpha sweep, easy-bare
#   bash run_bandit_reference.sh llama3 B1_BOTH_HARD  # alpha sweep, hard-bare
#
# B1 (the alpha sweep) was deliberately unwired until the competence gate had
# been evaluated on reference-bare, because the verdict determines what an
# alpha effect can be CALLED. Gate result, 2026-08-04, N=20:
#   Easy-bare PASS (4/4 rules)  -> competence anchor; alpha may be discussed
#                                  as moving a competent policy.
#   Hard-bare FAIL (rule 1)     -> alpha there is FAILURE-MODE CHARACTERIZATION
#                                  only. The words capability-effect / rescue /
#                                  improvement are NOT available for hard.
#
# The main intervention is --steering_scope both (alpha in BOTH the rationale
# and the action pass), because the research question is whether alpha moves
# the whole information-seeking policy, not just the arm logits given a fixed
# rationale. scope=action is its mechanism ABLATION and runs later, only if
# `both` shows an effect.
#
# alpha=0 is NOT re-run: no hook is registered at alpha=0 under either scope,
# so the stored Track A cells ARE the alpha=0 cells. Adding 0 to these configs
# would burn 7 GPU-hours reproducing a file that already exists.
#
# Each alpha gets its OWN --ans_file. Two reasons, both load-bearing:
#   1. the pv6 detail JSON name is bandit_pv6_{env}_{size}_{TOP}_{ls}_{le}.json
#      — no alpha and no scope in it, so two scopes in one dir would overwrite.
#   2. summary_{model}_{size}.csv is per-ans_file, so parallel alpha processes
#      sharing a dir would interleave appends and race the header migration.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="${DATA:-data1}"
BASE_DIR="${BASE_DIR:-/${DATA}/paveen/Dopamine/components}"
PY="${PY:-python}"

MODEL=${1:-llama3}
ONLY_STEP=${2:-}

case "$MODEL" in
  llama3)
    MODEL_NAME="llama3"; MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_SIZE="8B";     HS_PREFIX="llama3";  LAYERS="11-20" ;;
  qwen25)
    MODEL_NAME="qwen2.5"; MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SIZE="7B";      HS_PREFIX="qwen2.5"; LAYERS="16-22" ;;
  *) echo "unknown model '$MODEL' (want: llama3 | qwen25)"; exit 1 ;;
esac

# Smoke seeds are DISJOINT from the formal 20-seed bank (§3.2): the formal runs
# must stay unobserved until they are run for real. N=3 is never counterbalance
# evidence — it only exercises more than one arm layout.
#   easy -> build_smoke_bank = 6 12 13     hard -> 1 2 4
SMOKE_SEEDS_EASY="6 12 13"
SMOKE_SEEDS_HARD="1 2 4"

# The formal bank is generated by bandit_reference.build_seed_bank and frozen
# in bandit_pv6_baseline_manifest.json. Listed here so the launcher is
# self-describing; run freeze_bandit_baseline.py --check to verify they match.
BANK_EASY="0 1 2 3 4 5 8 11 14 19 22 23 26 31 32 46 48 50 53 57"
BANK_HARD="0 3 6 7 9 11 12 13 17 18 19 25 30 33 37 42 45 64 74 172"

run_step () {
  local tag="$1"; local env="$2"; local seeds="$3"; local alpha_config="$4"
  local ans="$5"; local cell_tag; shift 5
  case "$env" in
    easy) cell_tag="${tag}_EASY" ;;
    hard) cell_tag="${tag}_HARD" ;;
    *) cell_tag="${tag}_${env}" ;;
  esac
  # B1 has TWO cells per environment (one per alpha), and they are meant to run
  # as separate processes in separate dirs. So they also answer an alpha-level
  # selector: B1_BOTH_EASY_AM4 / _AP4. Derived from the config so the alpha
  # cannot drift out of sync with the tag.
  local alpha_tag="${alpha_config%%-*}"
  case "$alpha_tag" in
    neg*) alpha_tag="AM${alpha_tag#neg}" ;;
    *)    alpha_tag="AP${alpha_tag}" ;;
  esac
  local alpha_cell="${cell_tag}_${alpha_tag}"
  if [ -n "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$tag" ] \
      && [ "$ONLY_STEP" != "$cell_tag" ] \
      && [ "$ONLY_STEP" != "$alpha_cell" ]; then return; fi
  echo ""
  echo "######################################################################"
  echo "# STEP ${tag}: env=${env}  configs=${alpha_config}  $*"
  echo "# seeds: ${seeds}"
  echo "# out:   ${ans}"
  echo "######################################################################"
  "$PY" get_answer_bandit.py \
      --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
      --type non --percentage 0.5 --mask_type nmd \
      --reference_environment "${env}" \
      --configs ${alpha_config} \
      --seeds ${seeds} \
      --data "${DATA}" --base_dir "${BASE_DIR}" \
      --ans_file "${ans}" \
      "$@"
}

# ── SMOKE (N=3): protocol correctness ONLY ────────────────────────────────
# Covers easy+hard x bare+chat so both K=4 and K=5 candidate tokenization are
# exercised and both interfaces are attested. DO NOT read behaviour off this:
# N=3 cannot separate a policy from seed luck, and the smoke seeds are not
# counterbalanced. What to check afterwards, per run:
#   - invalid_rate == 0.0 in every cell (structural; nonzero means a bug)
#   - attestation.candidate_tokenization.single_token / scoring_mode
#   - attestation.round_0.tokens.double_bos == false
#   - attestation.round_0.tokens.injection_is_anchor_tail == true
#   - the rendered prompts in attestation.round_0 read as intended
#   - wall-clock per episode -> the real N=20 budget
# ── B1 ENGINEERING SMOKE (N=1, alpha=+4, scope=both) ──────────────────────
# The existing SMOKE above only ever exercised scope=action, so the steered
# rationale path has never touched a GPU. This is a PLUMBING check (~10 min),
# NOT a behavioural one: at N=1 nothing about exploration can be read. Verify
# in the detail JSON afterwards:
#   - config.steering_scope == "both", steered_rationale == true
#   - config.iface contains `scbothsv1`
#   - invalid_rate still 0.0
#   - attestation.round_0.rationale_clean is non-empty and readable
#   - attestation.round_0.tokens.injection_is_anchor_tail == true
#   - candidate_scores / choice_margins present and finite
if [ "$ONLY_STEP" = "B1_SMOKE" ]; then
  run_step B1_SMOKE easy "6" "4-${LAYERS}" \
    "pv6_smoke_easy_bare_both" --steering_scope both
  echo ""
  echo "B1 engineering smoke done. Check the scope fields listed above"
  echo "BEFORE launching the formal B1 cells. N=1 reads no behaviour."
  exit 0
fi

if [ "$ONLY_STEP" = "SMOKE" ]; then
  run_step SMOKE easy "$SMOKE_SEEDS_EASY" "0-${LAYERS}" "pv6_smoke_easy_bare"
  run_step SMOKE easy "$SMOKE_SEEDS_EASY" "0-${LAYERS}" "pv6_smoke_easy_chat" --use_chat
  run_step SMOKE hard "$SMOKE_SEEDS_HARD" "0-${LAYERS}" "pv6_smoke_hard_bare"
  run_step SMOKE hard "$SMOKE_SEEDS_HARD" "0-${LAYERS}" "pv6_smoke_hard_chat" --use_chat
  echo ""
  echo "Smoke done. Verify the checklist above BEFORE running Track A."
  exit 0
fi

# ── TRACK A (N=20, alpha=0): the capability measurement ───────────────────
# reference-bare is the RSN-aligned interface and the ONLY one the competence
# gate is evaluated on; reference-chat is the literature comparator.
run_step A0_BARE easy "$BANK_EASY" "0-${LAYERS}" "pv6_easy_bare"
run_step A0_BARE hard "$BANK_HARD" "0-${LAYERS}" "pv6_hard_bare"
run_step A0_CHAT easy "$BANK_EASY" "0-${LAYERS}" "pv6_easy_chat" --use_chat
run_step A0_CHAT hard "$BANK_HARD" "0-${LAYERS}" "pv6_hard_chat" --use_chat

# ── B1 (N=20, alpha!=0, scope=both): the main alpha experiment ────────────
# B1 never runs from a bare `run_bandit_reference.sh llama3` invocation: each
# alpha is a separate 7-hour process writing its own dir, so lumping all four
# into one sequential process (and alongside Track A) is never what is wanted.
# Naming a step is therefore required to reach them.
if [ -z "$ONLY_STEP" ]; then
  echo ""
  echo "Track A steps done. B1 (alpha sweep) is not run by the no-arg form —"
  echo "name a cell, e.g. B1_BOTH_EASY_AM4 / B1_BOTH_EASY_AP4."
  exit 0
fi

# bare only (the NMD mask is bare-string), one ans_file per alpha, alpha=0
# reused from Track A. See the header for why 0 is absent and why the dirs are
# split. Easy = anchored on a PASSED gate; Hard = failure-mode only.
run_step B1_BOTH easy "$BANK_EASY" "neg4-${LAYERS}" \
  "pv6_easy_bare_both_am4" --steering_scope both
run_step B1_BOTH easy "$BANK_EASY" "4-${LAYERS}" \
  "pv6_easy_bare_both_ap4" --steering_scope both
run_step B1_BOTH hard "$BANK_HARD" "neg4-${LAYERS}" \
  "pv6_hard_bare_both_am4" --steering_scope both
run_step B1_BOTH hard "$BANK_HARD" "4-${LAYERS}" \
  "pv6_hard_bare_both_ap4" --steering_scope both

echo ""
echo "Track A done. Next: evaluate the competence gate on reference-bare"
echo "K=4/K=5 ONLY (4 pre-registered mechanical rules, §3.7), comparing"
echo "against the FROZEN manifest — not against numbers recomputed now."
echo "The gate's verdict determines how B1 proceeds; do not start an alpha"
echo "sweep before it is evaluated."
