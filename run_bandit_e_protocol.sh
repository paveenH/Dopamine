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
#   1.  E-direct, K=2, α=0                  — capability floor
#   1b. E-CoT,    K=2, α=0                  — mechanism-break rescue (K2 direct failed)
#   2.  E-direct, K=3, α=0                  — mechanism replication across K/labels/positions
#   2b. E-CoT,    K=3, α=0                  — paired: does reasoning break the confirmation loop
#   3.  E-direct + E-CoT, K=5, warm_start=2 — utilization given forced, rate-legible discovery
#   4.  E-direct vs E-CoT, K=5, free exploration, α=0  — the real comparison
#   5.  only after 3/4 → pick an interface and move to an α sweep (separate script)
#
# 2026-07-30 UPDATE: K2 E-direct failed the capability floor — not via a
# pass/fail gate, but diagnostically: it is an early-outcome-dependent greedy
# confirmation loop (display position 1 wins the initial tie-break under
# total uncertainty; whichever arm gets the FIRST reward=1 becomes a
# self-reinforcing incumbent). K2COT (mt128, clean) showed CoT does NOT
# reliably break this — Velvet occupancy was actually HIGHER in the clean
# CoT data (83.2%) than in the earlier fallback-contaminated run. K3 E-direct
# replicated the SAME mechanism with the label/position tying broken (5/5
# seeds' first choice = position 1; 5/5 seeds' locked arm = whichever got the
# first reward=1, not a fixed name) — this is stronger evidence than K2 alone
# because it rules out a pure "Velvet" label prior as the root cause. Given
# this, steps 3/4 no longer treat CoT-on-failure as optional per K: K3COT
# (paired, step 2b) and WARMCOT (step 3) were added so every remaining step
# gets a direct/CoT contrast, since CoT's effect is not yet established
# either way at K=3 or under forced-discovery conditions.
#
# WHAT WARM-START (step 3) DOES AND DOES NOT SHOW. warm_start_pulls=2 forces
# 2 pulls per arm (10 of 50 rounds) before the model chooses anything — this
# makes discovery/coverage trivially near-ceiling BY CONSTRUCTION AND
# equalizes trial counts across arms, which is what makes it the direct test
# of the confirmation-loop mechanism: with no trial-count imbalance to lean
# on, does the model track and use reward RATE? Read whether round 1 matches
# the warm-start empirical best, early/late empirical-best adherence, and
# late_opt_frac — never coverage (ceiling by construction), and treat OptFrac
# alone as secondary (2 pulls/arm is still noisy re: which arm is truly
# best). A pass here does NOT establish autonomous exploration; a fail means
# the model cannot even exploit clean, rate-legible information, which would
# make step 4's free-exploration result uninterpretable — so step 3 gates
# step 4, not the reverse.
#
# Usage:
#   bash run_bandit_e_protocol.sh llama3           # all steps, all seeds
#   bash run_bandit_e_protocol.sh llama3 K2         # step 1    (E-direct, K=2)
#   bash run_bandit_e_protocol.sh llama3 K2COT      # step 1b   (E-CoT,    K=2)
#   bash run_bandit_e_protocol.sh llama3 K3         # step 2    (E-direct, K=3)
#   bash run_bandit_e_protocol.sh llama3 K3COT      # step 2b   (E-CoT,    K=3)
#   bash run_bandit_e_protocol.sh llama3 WARM       # step 3    (E-direct, K=5 warm-start)
#   bash run_bandit_e_protocol.sh llama3 WARMCOT    # step 3    (E-CoT,    K=5 warm-start)
#   bash run_bandit_e_protocol.sh llama3 MAIN       # step 4    (E-direct, K=5 free)
#   bash run_bandit_e_protocol.sh llama3 MAINCOT    # step 4    (E-CoT,    K=5 free, mt128)
#   bash run_bandit_e_protocol.sh llama3 MAINCOT192 # step 4    (E-CoT,    K=5 free, mt192 — clean invalid rerun)
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
    K2)    echo "0 4 5 6 7" ;;   # position [1,2,1,2,2], both names present
    K2COT) echo "0 4 5 6 7" ;;   # same seeds as K2 — paired rescue comparison
    K3)    echo "0 2 4 9 21" ;;  # position [3,2,1,2,3], all 3 names present
    K3COT) echo "0 2 4 9 21" ;;  # same seeds as K3 — paired mechanism-break comparison
    *)     echo "0 3 4 9 37" ;;  # K=5 (WARM/WARMCOT/MAIN/MAINCOT) — matches run_bandit_validity.sh
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

# ── Step 1b: K=2 CoT rescue (only run after a K2 E-direct FAIL) ──────────
# K2 E-direct result (2026-07-30, seeds 0/4/5/6/7): NOT a position lock — it
# is a NAME lock. 245/250 choices across all 5 seeds picked "Velvet Vogue
# Jacket" regardless of its displayed position or true probability (seed 7:
# Velvet at position 2, still 48/50 picks). OptFrac only looked good on seeds
# where Velvet happened to be the optimal arm. This fails the K=2 capability
# floor under E-direct. Per the direct-first/CoT-on-failure design, rescue
# with E-CoT on the SAME seeds before touching K3: if forcing a brief
# trial-count/rate comparison breaks the Velvet lock, the floor failure was
# interface-level (bare Choice: token has no room to reason from evidence);
# if the lock persists under CoT too, it is label-level and K3/K5 need a
# neutral-label control before any further step is informative.
#
# mt64 → mt128 (2026-07-30): the first e_K2_cot run (max_new_tokens=64) DID
# loosen the lock (Velvet 245/250 -> 171/250) but at invalid_rate 22-36%
# (72/250 = 28.8% overall) — raws show good reasoning ("Silk Serenity Dress
# has 80% estimated rate... Velvet Vogue Jacket's 20% rate") truncated
# mid-sentence or mid-Choice-line before the final line could complete. That
# is a token-budget failure, not a prompt/parser/protocol failure — pv5 stays
# pv5, only the generation budget changes. Bumped to 128. Kept as a SEPARATE
# --ans_file (mt64's config.seeds only reaches _iface_tag's resume key, not
# the output filename, so reusing e_K2_cot would silently overwrite the mt64
# raws/rationales in the same JSON) so both budgets stay on disk. Do NOT
# offline-patch the mt64 invalid rounds — each invalid round's fallback pick
# altered that seed's subsequent reward history, so only a full episode
# re-run is valid. Do not read mt64's OptFrac as a CoT verdict (fallback-
# contaminated); judge the Velvet-lock question from mt128 once invalid is
# back near 0. If mt128 confirms ~0% invalid, port max_new_tokens=128 to
# MAINCOT below too (K=5's TRIED table is longer, so 64 there is likely even
# tighter than it was here).
run_step K2COT --ans_file "e_K2_cot_mt128" \
          --prompt_variant E-CoT --num_arms 2 --max_new_tokens 128

# ── Step 2b: K=3 CoT paired contrast, same seeds as K3 direct ────────────
# K3 E-direct result (2026-07-30, seeds 0/2/4/9/21): the mechanism is NOT a
# fixed-label lock (5/5 seeds' locked-on arm varies: Urban/Silk/Urban/Velvet/
# Velvet) — it is an early-outcome-dependent greedy confirmation loop.
# 5/5 seeds' FIRST choice was display position 1 (position controls the
# initial tie-break under total uncertainty); the arm that received the
# first reward=1 became the incumbent and was re-picked overwhelmingly
# afterward (seed 0: 3 rounds of reward=0 on 3 different arms before Urban's
# round-4 reward=1 locked it in; seed 21 locked onto position-1 Velvet after
# its round-1 reward=1 despite 15 further reward=0s, only self-correcting to
# the true-best arm around round 17). Position initializes, first success
# consolidates — neither name nor position alone explains the final lock.
# This step asks whether explicit reasoning (comparing trial counts AND
# rates, not just re-picking the last winner) breaks that consolidation step.
# 128 tokens from the start (K2's mt64 truncation lesson) — E-CoT's rationale
# is the same length regardless of K, only the TRIED/UNTRIED table grows.
run_step K3COT --ans_file "e_K3_cot" \
          --prompt_variant E-CoT --num_arms 3 --max_new_tokens 128

# ── Step 3: K=5 warm-start, E-direct AND E-CoT, α=0 ───────────────────────
# --warm_start_pulls 2: 10 of 50 rounds are the environment's forced pulls
# (2 per arm), the model gets the remaining 40. This equalizes trial counts
# across arms BY CONSTRUCTION, which is the direct test of the K2/K3
# confirmation-loop mechanism: with no UNTRIED rows and no trial-count
# imbalance to lean on, does the model track and use reward RATE? Read
# whether round 1's choice matches the warm-start empirical best (not
# necessarily the true best — only 2 pulls/arm is still noisy), plus
# early/late empirical-best adherence and late_opt_frac — NOT coverage
# (trivially near-ceiling by construction) and NOT OptFrac alone (noisy at
# n=2/arm). Both interfaces run here (not E-direct only, per the K2/K3
# finding that E-direct's bare Choice: token gave no room to weigh evidence
# — the same may be true here even with rate-legible state). GATES step 4.
run_step WARM --ans_file "e_K5_warmstart2_direct" \
          --prompt_variant E-direct --num_arms 5 --warm_start_pulls 2 \
          --max_new_tokens 24

# WARM (E-direct) result (2026-07-30, seeds 0/3/4/9/37): PASSED conditional
# utilization. 5/5 seeds' round-1 model choice fell in the warm-start
# empirical-best SET (seed 9's Velvet was tied at 0.5 with two other arms —
# not a miss). Overall empirical_best_adherence=0.925: the model reads and
# tracks UPDATED point estimates as new rounds come in (seed 0 switches
# Urban<->Silk as their rates change, converging on true-best Urban; seed 4
# leaves Retro once its rate drops below competitors' frozen 0.5). The
# remaining gap is NOT integration failure — it is pure point-estimate
# greedy exploitation with no uncertainty bonus: seed 9 stays on Velvet
# (frozen empirical rate ~0.585, genuinely the highest KNOWN rate throughout)
# and never re-samples Retro (n=2, high variance, true rate 0.7) because
# nothing in a point-estimate policy gives it a reason to. seed 3/37 are not
# failures either — both started on the true-best arm and stayed correctly.
# So WARM-direct's ceiling = exploitation is solid, autonomous
# uncertainty-driven exploration is the open gap. WARMCOT tests whether
# explicit reasoning adds that missing ingredient — read specifically:
# (1) does it PRESERVE the 0.925 first-choice/adherence accuracy (a CoT that
# only adds churn without keeping this would be worse, not better); (2) does
# it voluntarily re-sample low-n/high-uncertainty arms (e.g. does seed 9 ever
# revisit Retro); (3) does late_opt_frac/true-best discovery improve over
# direct's seed-9/seed-4 shortfalls; (4) is any added switching TARGETED
# (uncertainty-driven) rather than non-novel oscillation. Ran at 128 tokens
# (invalid stayed low: 2/200 rounds, one truncation each on seeds 3/4 — see
# result note below; a later 160-token bump was proposed but never re-run,
# since 128 already gave clean-enough data to settle the question).
run_step WARMCOT --ans_file "e_K5_warmstart2_cot" \
          --prompt_variant E-CoT --num_arms 5 --warm_start_pulls 2 \
          --max_new_tokens 128

# WARMCOT result (2026-07-30, same seeds): SETTLED — CoT does not add
# reliable uncertainty-aware exploration on top of WARM-direct's exploitation
# floor. seed 9 (the clean single-variable test: Retro is n=2/high-variance/
# true-best 0.7, frozen at empirical 0.5, tied with Velvet/Silk) is BYTE-
# IDENTICAL to direct — 40/40 Velvet, Retro never revisited. seed 37, which
# direct had CORRECTLY converged on the true-best arm (late_opt 1.000),
# instead oscillates Silk<->Celestial 20/20 under CoT (late_opt 0.400) even
# though Celestial's warm-start rate was 0/2, the worst in the set — a
# genuine regression, not just noise (invalid_rate=0 on this seed, so it
# isn't a truncation artifact). Raw rationale traces confirm this is not
# empty churn: the model verbalizes an explore-for-more-evidence narrative
# but applies it inconsistently (never systematically covers the OTHER
# equally-unproven arms) and makes repeated numeric misreads while doing so
# (e.g. calling a LOWER rate/lower-evidence arm "most rewards per trial" /
# "highest amount of evidence"). Net effect across seeds: first-choice
# accuracy held (5/5 in the empirical-best set, same as direct) but mean
# empirical_best_adherence dropped ~0.925 -> ~0.69, and late_opt_frac fell or
# stayed flat on every seed but one. CAPABILITY BOUNDARY established by
# steps 1-3: the model CAN read a structured summary, identify the
# empirical-best set, and track updated point estimates (greedy
# exploitation) — it CANNOT reliably assign an uncertainty bonus, sample
# systematically among under-observed arms, or reconverge cleanly after
# exploring. E-CoT's extra text sometimes narrates exploration intent but
# does not execute it consistently, and its numeric errors make it an
# unreliable exploration mechanism rather than a fix. Step 4 (free
# exploration, no warm-start) is not expected to do better than this ceiling
# on either interface — it is run to confirm that, and to characterize
# regret/novel-vs-non-novel churn for the write-up, not to look for a
# rescue.

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

# 64 -> 128 (2026-07-30): same token-truncation lesson as K2COT — K5's
# TRIED/UNTRIED table is longer than K2's, so 64 tokens is even tighter here.
# New --ans_file (not overwriting e_K5_cot) for the same reason as K2COT: the
# resume key's mt128 tag forces a fresh run, but the output filename doesn't
# encode token budget, so reusing the old dir would silently overwrite the
# mt64 raws in place.
run_step MAINCOT --ans_file "e_K5_cot_mt128" \
          --prompt_variant E-CoT --num_arms 5 --max_new_tokens 128

# MAINCOT (mt128) result (2026-07-30, same seeds): invalid was low in
# aggregate (5/250 = 2%) but NOT evenly distributed — seed3 round4, seed9
# rounds 4/17/49, seed37 round33. Round 4 falls inside the incumbent-
# establishment window every seed shows (compare seed4's clean 0-invalid
# trace: all 5 arms sampled exactly once in rounds 0-4, incumbent forms
# after). A fallback pull there injects a random arm + reward into that
# window, which can redirect which arm becomes incumbent for the rest of the
# episode — so seed3's and seed9's apparent CoT-induced destabilization
# (late_opt_frac 1.000->0.700 and 0.967->0.867) is CONFOUNDED by this, not
# cleanly attributable to CoT's exploration behavior itself. Only seed4 (0
# invalid throughout) is a clean CoT observation: it systematically sampled
# all 5 arms once each in rounds 0-4, then converged on the true-best
# Celestial (32/50, late_opt 0.000->0.767) — proof CoT CAN break first-
# success lock-in, at least sometimes. mt192 reruns at a wider budget to
# push invalid toward 0 so seed3/seed9's trajectories can be read cleanly
# before any final verdict on CoT's net effect. New --ans_file per the usual
# rule (mt128's resume key won't collide, but the output filename doesn't
# encode token budget either).
run_step MAINCOT192 --ans_file "e_K5_cot_mt192" \
          --prompt_variant E-CoT --num_arms 5 --max_new_tokens 192

# MAINCOT192 result (2026-07-30): DEAD END, do not repeat. mt192's choices
# are byte-identical to mt128 for all 5 seeds — the invalid rounds were never
# a truncation issue (greedy decoding under bf16 never reached anywhere near
# 128 tokens on these rounds), so widening max_new_tokens further will not
# help. Inspecting the raw text of every invalid round confirms the real
# cause: the model's REASONING was often substantively fine (e.g. seed9
# round48's raw ends "...this option should be tried: Retro Revival
# Sneakers" — a clear, arguably-correct intended decision) but it did not
# terminate in the strict_anchor parser's required exact `Choice: <name>`
# form, so it fell to fallback_rng instead. Per the 2026-07-30 methodology
# note: do NOT loosen strict_anchor to rescue these offline, and do NOT
# hand-recover an "intended" choice from the raw text — a differing fallback
# pull at that round changes every downstream round's TRIED/UNTRIED state
# and reward history for the rest of that episode, so there is no way to
# reconstruct the counterfactual trajectory the model would have produced
# had it committed cleanly. seed3(round4)/seed9(round4,17,49)/seed37(round33)
# stay confounded and are NOT usable as evidence of CoT-induced
# destabilization — only seed4 (0 invalid throughout: systematic 5-arm
# coverage in rounds 0-4, then correct convergence on Celestial, late_opt
# 0.000->0.767) is a clean, attributable CoT observation.
#
# ══════════ CAPABILITY LADDER VERDICT (2026-07-30, FROZEN) ══════════
# E-direct: format-stable (invalid ~0 everywhere in this ladder), and its
# failure mode is well-characterized across K=2/K=3/K=5-warm/K=5-free — an
# early-outcome-dependent greedy confirmation loop (display position 1 wins
# the initial tie-break under total uncertainty; whichever arm earns the
# FIRST reward=1 becomes a self-reinforcing incumbent; late performance is
# bimodal on whether that first-success arm happens to be the true best).
# K5-warm-start additionally showed this is not pure point-estimate
# greediness — incumbent persistence can override a competing arm with a
# CURRENTLY HIGHER empirical estimate (K5-free seed4's direct run: Retro hit
# 1/1 right after Velvet's first success, yet Velvet stayed modal 48/50).
#
# E-CoT: occasionally enables systematic initial coverage and can rescue a
# total non-discovery failure (K5-free seed4 is the one clean case), i.e.
# the model IS capable of something like uncertainty-driven sampling under
# some conditions. But format adherence to the strict_anchor protocol is not
# reliable at K=5 (~2-6% invalid per seed, unevenly distributed, sometimes
# landing in the incumbent-formation window), and every apparent case of
# CoT "destabilizing" an otherwise-correct seed (K3COT seed2, WARMCOT
# seed37, MAINCOT seed3/seed9) is at least partially confounded by either
# generation-content churn or fallback contamination — NOT a clean net-
# negative verdict, just not a reliable net-positive either. Net: CoT's
# marginal value over direct is UNESTABLISHED, not negative, not positive.
#
# DECISION: the α sweep runs on E-direct, not E-CoT. E-direct is format-
# stable, fallback-free in this data, and its steering-token semantics are
# simpler (injects on the prefilled anchor's trailing space, vs E-CoT's
# injection on the last prompt token before free generation). A clean CoT
# vs direct comparison remains possible LATER via a two-stage interface
# (generate rationale freely, then append `Choice:` and pick among the K
# legal arm names via constrained decoding — eliminating invalid/fallback
# entirely) but that is a NEW protocol, not a pv5 rerun, and is deferred.
#
# NEXT: E-direct, K=5 free exploration, α ∈ {−4, 0, +4}, 20 paired seeds,
# NEW --ans_file (do not overwrite this 5-seed baseline). Primary reads,
# in order: novel-arm exploration / best_never_tried, whether exploration
# continues past an incorrect incumbent, non-novel churn among already-
# tried arms, persistence/late empirical-best adherence — OptFrac/regret
# last, as an outcome summary, not the primary mechanism evidence.

echo ""
echo "Done. Analyse with the mechanism metrics (coverage/adherence/late_opt_frac"
echo "per step 1-3; E-direct vs E-CoT paired contrast for step 4) — NOT OptFrac"
echo "alone anywhere in this ladder. See BanditExperiment_LiteratureReview.md §6"
echo "for the judgment order."
