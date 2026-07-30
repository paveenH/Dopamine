#!/bin/bash
#
# ================ BANDIT pv5 (E-direct / E-CoT) CAPABILITY LADDER ================
# Post-D2 redesign, per BanditExperiment_LiteratureReview.md §4. D2 fixed the
# format (invalid 0.001) but collapsed to pure greedy (adherence 1.000, 18/20
# α=−4 seeds byte-identical to α=0). pv5 restores D's exploration-affordance
# sentence, drops D2's long epistemics paragraph, and splits state into
# structural TRIED/UNTRIED blocks. See get_answer_bandit.py's PROTOCOL_VERSION
# comment block (pv5) for the exact prompt diff.
#
# THIS SCRIPT DOES NOT RUN α — every step here is α=0. It answers "does this
# interface work at all" before any sweep is worth running, same discipline as
# run_bandit_validity.sh's A/B/C/D/D2/D3 arms.
#
# ORDER (do not reorder — each step's interpretation depends on the previous
# one having been read first):
#   1. E-direct, K=2, α=0            — capability floor
#   2. E-direct, K=3, α=0            — capability floor
#   3. E-direct, K=5, warm_start=2   — utilization given forced discovery
#   4. E-direct vs E-CoT, K=5, free exploration, α=0  — the real comparison
#   5. (as needed) E-CoT on whichever of 1-3 failed under E-direct
#   6. only after 4/5 → pick an interface and move to an α sweep (separate script)
#
# WHY direct-first, CoT-only-on-failure for steps 1-3 (not paired from the
# start): those three steps ask "can the model do SOMETHING at this
# difficulty/condition", not "does CoT help" — that second question only has a
# clean answer at K=5 free exploration (step 4), where E-direct vs E-CoT is run
# PAIRED on the same seeds by design. Running E-CoT on a K=2 step that already
# passes under E-direct would spend generation budget without adding
# information: if direct already works, no comparison is needed to know CoT
# isn't required THERE. (K=2/K=3 do not distinguish "task-level capability
# floor" from "interface-level floor" on their own — see the step 1/2 note
# below on what a pass/fail there does and does not establish.)
#
# WHAT WARM-START (step 3) DOES AND DOES NOT SHOW. warm_start_pulls=2 forces
# 2 pulls per arm (10 of 50 rounds) before the model chooses anything — this
# makes discovery/coverage trivially near-ceiling BY CONSTRUCTION, so those
# metrics are NOT informative here. The one thing it measures is UTILIZATION:
# given a fully-populated TRIED table with no UNTRIED rows to weigh, does the
# model track and use the best-supported arm? Read empirical_best_adherence,
# late_opt_frac, and n_model_rounds-relative regret — never coverage. A pass
# here does NOT establish autonomous exploration; a fail here means the model
# cannot even exploit clean information, which would make step 4's free-
# exploration result uninterpretable (can't ask "does it explore" if it can't
# "use what it has" in the first place) — so step 3 gates step 4, not the
# reverse.
#
# Usage:
#   bash run_bandit_e_protocol.sh llama3           # steps 1-4, all seeds
#   bash run_bandit_e_protocol.sh llama3 K2         # step 1 only
#   bash run_bandit_e_protocol.sh llama3 K3         # step 2 only
#   bash run_bandit_e_protocol.sh llama3 WARM       # step 3 only
#   bash run_bandit_e_protocol.sh llama3 MAIN       # step 4 only (E-direct)
#   bash run_bandit_e_protocol.sh llama3 MAINCOT    # step 4 only (E-CoT)
#   SEEDS=20 bash run_bandit_e_protocol.sh llama3 MAIN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/Dopamine/components"

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

# α=0 ONLY throughout this script — see header.
CONFIGS="0-${LAYERS}"
NUM_ROUNDS=50

# Seed sets are PER-K, not shared. shuffle_arms' best-arm-position distribution
# is a different shape at every K (K=2 has only 2 possible positions, so a set
# balanced for K=5 is not balanced for K=2 — e.g. K=5's default 5-seed set
# "0 3 4 9 37" is 4:1 at K=2, and even the 20-seed set "0..19" is 15:5 at K=2).
# Each 5-seed set below was hand-picked from `position_of_best(seed,
# num_arms=k)` + best-arm NAME to balance both — same discipline as K=5's
# original counterbalancing (run_bandit_validity.sh's SEED COUNTERBALANCING
# note), re-derived per K. Resolved PER STEP inside run_step (not once
# globally) so a no-argument full-ladder run gives every step its own correct
# set, not whichever K happened to be checked first.
SEEDS_MODE=${SEEDS:-5}
seeds_for_step () {
  local step="$1"
  if [ "$SEEDS_MODE" == "20" ]; then
    echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
    return
  fi
  case "$step" in
    K2) echo "0 4 5 6 7" ;;      # position [1,2,1,2,2], both names present
    K3) echo "0 2 4 9 21" ;;     # position [3,2,1,2,3], all 3 names present
    *)  echo "0 3 4 9 37" ;;     # K=5 (WARM/MAIN/MAINCOT) — matches run_bandit_validity.sh
  esac
}

run_step () {
  local tag="$1"; shift
  if [ -n "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$tag" ]; then return; fi
  local step_seeds
  step_seeds=$(seeds_for_step "$tag")
  echo ""
  echo "######################################################################"
  echo "# STEP ${tag}: $*"
  echo "# seeds: ${step_seeds}"
  echo "######################################################################"
  python get_answer_bandit.py \
      --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
      --type non --percentage 0.5 --mask_type nmd \
      --configs ${CONFIGS} \
      --seeds ${step_seeds} --num_rounds ${NUM_ROUNDS} \
      --no_role --untried_semantics \
      --summary_history --answer_anchor --use_chat \
      --data "${DATA}" --base_dir "${BASE_DIR}" \
      "$@"
}

# ── Step 1/2: capability floor, E-direct only, K=2 then K=3 ──────────────
# A pass at K=2/K=3 under E-direct establishes the model can do SOME Bandit
# adaptation at this interface; it does NOT establish K=5 will also pass (K=5
# is a harder discovery problem — see run_bandit_algorithmic_baseline.py:
# even Thompson Sampling's late OptFrac at K=5/50 rounds is only 0.585
# [0.498,0.665], well short of ceiling). A FAIL at K=2 under E-direct is the
# one result that DOES generalize forward: if the model cannot adapt to the
# easiest possible version of this task, K=3/K=5 will not rescue it, and
# E-CoT should be tried on K=2 before concluding anything about K=5.
run_step K2 --ans_file "e_K2_direct" \
          --prompt_variant E-direct --num_arms 2 --max_new_tokens 24

run_step K3 --ans_file "e_K3_direct" \
          --prompt_variant E-direct --num_arms 3 --max_new_tokens 24

# ── Step 3: K=5 warm-start, E-direct only, α=0 ────────────────────────────
# --warm_start_pulls 2: 10 of 50 rounds are the environment's forced pulls
# (2 per arm), the model gets the remaining 40. See the header note on what
# this does and does not show. GATES step 4 (see header).
run_step WARM --ans_file "e_K5_warmstart2_direct" \
          --prompt_variant E-direct --num_arms 5 --warm_start_pulls 2 \
          --max_new_tokens 24

# ── Step 4: K=5 free exploration, E-direct vs E-CoT, PAIRED, α=0 ─────────
# THE comparison this whole script exists to run. Same seeds, same K, same
# warm_start_pulls=0 (free exploration) for both — the only difference is the
# output interface (Choice: prefill vs brief rationale + final Choice line).
# Steering-token caveat: the two are DIFFERENT interventions (E-direct injects
# on the prefilled anchor's trailing space; E-CoT injects on the last PROMPT
# token before free generation, since there is no prefill to anchor to) — this
# matters once an α sweep starts, not for this α=0 comparison, but keep it in
# mind when this script's output feeds into that decision.
run_step MAIN --ans_file "e_K5_direct" \
          --prompt_variant E-direct --num_arms 5 --max_new_tokens 24

# E-CoT needs more budget: it must fit "briefly compare... two short
# sentences" AND a final Choice line, unlike E-direct's bare name.
run_step MAINCOT --ans_file "e_K5_cot" \
          --prompt_variant E-CoT --num_arms 5 --max_new_tokens 64

echo ""
echo "Done. Analyse with the mechanism metrics (coverage/adherence/late_opt_frac"
echo "per step 1-3; E-direct vs E-CoT paired contrast for step 4) — NOT OptFrac"
echo "alone anywhere in this ladder. See BanditExperiment_LiteratureReview.md §6"
echo "for the judgment order."
