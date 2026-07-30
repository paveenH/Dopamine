#!/bin/bash
#
# ==================== BANDIT α=0 TASK-VALIDITY PILOT ====================
# Default (ALPHA unset) is α=0 only. This asks ONE question before any sweep
# is worth running:
#
#   under a given prompt/parser interface, does a 7-8B model do Bandit
#   learning at all?
#
# ALPHA=pm4 turns arm D into a small 0/±4 PROBE (see the ALPHA block below).
# That is still not the 9-α sweep — it is a check on whether α moves the
# discovery layer at all, read on coverage/first_best_index, never OptFrac.
#
# WHY THIS EXISTS (2026-07-29)
# The shuffle fix removed the old "best arm is always displayed first" shortcut.
# What it exposed is that the previous high OptFrac was largely position bias,
# not learning. Post-fix α=0 pilot (2 runs × 50 rounds, bare-string, raw history,
# substring parser):
#   Llama3-8B : OptFrac 0.12, invalid 0.32-0.40. Many "valid" replies still
#               contain lists/code/explanation and merely happen to hit exactly
#               one name. Document completion, not decision-making.
#   Qwen2.5-7B: OptFrac 0.02-0.04, invalid 0.00-0.06 — format is clean, but it
#               locks the FIRST-LISTED arm 94-96% of rounds and only ever
#               touches 3 distinct arms. Below the 0.20 chance floor.
# Neither is Bandit learning. Do NOT restore the old numbers by reverting the
# shuffle — the shuffle is correct; the interface is what is untested.
#
# ALSO: the −8 cell's higher OptFrac (Qwen 0.30-0.50) is NOT an improvement —
# invalid there is 0.58-0.70, so most of those "choices" are the uniform random
# fallback, which by construction hits the best arm 20% of the time.
#
# SEED COUNTERBALANCING
# The 2-run pilot used seeds 0,1 — which happen to share BOTH the same best arm
# name (Urban Mystique Jeans) AND the same position (2), so it was not position-
# balanced at all. --seeds "0 3 4 9 37" puts the best arm at display positions
# 2,4,5,3,1 with five DISTINCT best names, so neither position nor name can be
# confounded with OptFrac. (Over 30 seeds position is already near-uniform:
# {1:3, 2:8, 3:9, 4:8, 5:2}; the explicit set matters for small-n pilots.)
#
# THE THREE INTERFACE CHANGES UNDER TEST (all opt-in; legacy path unchanged)
#   --summary_history : EVOLvE SummaryContextLayerMAB — per-arm "N times,
#                       average reward R" instead of a 50-line raw log. Copied
#                       verbatim from banditbench/agents/context.py. This is how
#                       EVOLvE actually feeds frontier models; the raw log makes
#                       the model integrate 50 lines itself, which is likely the
#                       real ask that 7-8B fails.
#   --answer_anchor   : prefill "Choice: " so the next token IS the arm name
#                       (same 施力点 logic as betting's "Bet: " / CGT's
#                       "Answer: "), and parse STRICTLY: the committed line must
#                       be EXACTLY an arm name. "Choice: X because it's best" is
#                       INVALID — trailing prose means the model explained rather
#                       than adopted the decision protocol, and that shows up as
#                       (invalid, n_matched=1), distinct from menu restatement
#                       (n_matched>1) and from silence (0). This separation is
#                       what the substring parser could not provide.
#   --use_chat        : chat template. Closest to EVOLvE's API-model usage.
#                       CAVEAT: moves steering off the bare distribution the NMD
#                       mask was extracted in. Harmless at α=0 (no injection),
#                       but see CLAUDE.md before sweeping α under chat.
#
# ⚠ B AND C ARE NOT A SINGLE-VARIABLE DIFF FROM A. --answer_anchor changes THREE
# things in build_prompt(), not just the prefill+parser: it also swaps in a
# SHORTER exploration/exploitation paragraph ("Balance exploration—…—with
# exploitation—…—to maximize total reward.", one sentence vs the three-sentence
# EVOLvE wording) and a shorter closing question. So A→B differs by
# {summary history, prefill+strict parser, explore/exploit wording, closing
# wording} and A→C adds chat on top. If B or C passes where A fails, the gain
# canNOT be attributed to summary/anchor alone — the reframed wording is a live
# co-explanation. Isolating it needs a fourth arm (anchor wording, legacy
# history/parser), which is deliberately NOT run here: the pilot's question is
# "does ANY interface make this task work on a 7-8B model", not "which knob did
# it". Attribute only after the task is known to be viable at all.
#
# GO / NO-GO for spending the 13,500-generation sweep. ALL must hold at α=0:
#   1. invalid_rate low (< ~0.10)
#   2. late OptFrac clearly above the 0.20 chance floor
#   3. late OptFrac > early OptFrac  (i.e. it LEARNS, not just picks well)
# Plus a DESCRIPTIVE check (not a gate): OptFrac across the five counterbalanced
# position/name cells. With one run per cell, and position confounded with
# best-name, that spread canNOT establish independence from position or name —
# it only catches a blatant first-option lock. Proving independence needs
# several runs per (position, name) cell, which is a different design.
# If arm C (chat+summary+strict) fails the three gates, Bandit is unsuitable for
# models at this scale — record it as a boundary point like ScienceWorld and
# stop. That is a legitimate finding, not a failure to tune.
#
# Usage:
#   bash run_bandit_validity.sh llama3    # arms A/B/C on Llama3-8B
#   bash run_bandit_validity.sh qwen25    # arms A/B/C on Qwen2.5-7B
#   bash run_bandit_validity.sh llama3 C  # single arm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/Dopamine/components"

MODEL=${1:-llama3}
ONLY_ARM=${2:-}

case "$MODEL" in
  llama3)
    MODEL_NAME="llama3"; MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_SIZE="8B";     HS_PREFIX="llama3";  LAYERS="0-11-20" ;;
  qwen25)
    MODEL_NAME="qwen2.5"; MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
    MODEL_SIZE="7B";      HS_PREFIX="qwen2.5"; LAYERS="0-16-22" ;;
  *) echo "unknown model '$MODEL' (want: llama3 | qwen25)"; exit 1 ;;
esac

# α=0 by default. This pilot measures the task, not the intervention.
#
# ALPHA=pm4 adds the ±4 cells (α=0 / +4 / −4, same layers). This is a PROBE,
# not a dose-response: D's OptFrac is bimodal (sd ≈ mean at n=20), and the
# α effect is almost certainly smaller than the C→D protocol repair, whose
# paired CI was already [+0.019, +0.431]. So a positive result says "α moves
# exploration, keep going"; a null does NOT say α is inert — at this n it is
# not separable from seed noise.
#
# READ THE DISCOVERY LAYER, NOT OptFrac: coverage, best_never_tried,
# first_best_index (median), n_explore_untried — these are continuous and
# unimodal. Keep empirical_best_adherence as the control: if +4 raises
# coverage only by making choices more random, adherence falls with it.
#
# CAVEAT: α here is prefill-only on the last prompt token, and D runs under
# --use_chat, so injection sits in the chat distribution rather than the bare
# one the NMD mask was extracted in. Dilution is a live explanation for a null
# — do not read a null as "α does not move exploration".
#   ALPHA=pm4 SEEDS=20 bash run_bandit_validity.sh llama3 D
ALPHA_MODE=${ALPHA:-0}
LAYER_RANGE="${LAYERS#0-}"          # "0-11-20" -> "11-20" (qwen25: "16-22")
if [ "$ALPHA_MODE" == "pm4" ]; then
    CONFIGS="${LAYERS} 4-${LAYER_RANGE} neg4-${LAYER_RANGE}"
else
    CONFIGS="${LAYERS}"
fi
NUM_ROUNDS=50

# Seed set. Default = the 5 counterbalanced seeds (best arm at display
# positions 2,4,5,3,1 with five distinct names) — the cheap smoke test.
#
# SEEDS=20 switches to 0..19, which is what the D-vs-C comparison REQUIRES:
# the two protocols must be compared PAIRED on the same seeds, and C's 20-seed
# baseline (bandit_validity_C20_llama3) used 0..19. Comparing D on the 5-seed
# set against that baseline would be an unpaired comparison of different arm
# layouts — at n=5 with a bimodal OptFrac that is uninterpretable.
#   SEEDS=20 bash run_bandit_validity.sh llama3 D
# SEEDS=fail is the D2 targeted smoke test: the five 20-seed D runs whose
# failure mode D2 is designed to fix, so a cheap 5-cell run answers "did the
# mechanism move" before spending the full 20.
#   seed 0  — best arm's first pull returned 0; a different arm's first pull
#             returned 1 and was then taken 46 times
#   seed 2  — best arm's first pull returned 0, never tried again
#   seed 3  — ~35 rounds on a mid arm before the best arm was tried
#   seed 14 — same late-discovery pattern
#   seed 12 — held an arm down to observed 0.16 without ever exploring
# NOTE these seeds were SELECTED ON THE OUTCOME, so their pass rate is not an
# unbiased estimate of anything — it is a mechanism probe. A verdict needs the
# paired 20-seed run against D (SEEDS=20), which is the comparison the analyzer
# gates on.
SEEDS_MODE=${SEEDS:-5}
if [ "$SEEDS_MODE" == "20" ]; then
    SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
elif [ "$SEEDS_MODE" == "fail" ]; then
    SEEDS="0 2 3 12 14"
else
    SEEDS="0 3 4 9 37"
fi

run_arm () {
  local tag="$1"; shift
  if [ -n "$ONLY_ARM" ] && [ "$ONLY_ARM" != "$tag" ]; then return; fi
  echo ""
  echo "######################################################################"
  echo "# ARM ${tag}: $*"
  echo "######################################################################"
  python get_answer_bandit.py \
      --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
      --type non --percentage 0.5 --mask_type nmd \
      --configs ${CONFIGS} \
      --seeds ${SEEDS} --num_rounds ${NUM_ROUNDS} \
      --no_role \
      --data "${DATA}" --base_dir "${BASE_DIR}" \
      "$@"
}

# A = baseline control: current interface, counterbalanced seeds only. Tells us
#     how much of the pilot's failure was just the seed collision.
run_arm A --ans_file "bandit_validity_A_raw_substr"

# B = bare-string + summary history + exact Choice parser. Keeps the activation
#     distribution the NMD mask lives in, so if THIS works the α sweep needs no
#     chat exception.
run_arm B --ans_file "bandit_validity_B_bare_summary_anchor" \
          --summary_history --answer_anchor --max_new_tokens 24

# C = chat + summary history + exact parser (RUN THIS ONE FIRST if picking one).
#     Closest to how EVOLvE actually uses LLMs; the most likely to elicit
#     learning. If C fails, the task is out of reach at this scale.
run_arm C --ans_file "bandit_validity_C_chat_summary_anchor" \
          --summary_history --answer_anchor --use_chat --max_new_tokens 24

# D = C + the pv2 task-representation prompt (--untried_semantics).
#
# WHY D EXISTS. C reached invalid=0.000 on both models, so the interface is
# solved — but the 20-seed Llama run showed the TASK DESCRIPTION was wrong:
# an arm with 0 pulls rendered as "average reward 0.00", i.e. tied-worst or
# worse than every tried arm. 6 of 20 seeds never tried the best arm and scored
# OptFrac exactly 0.000, while the 14 that did averaged 0.434; one seed held an
# arm it had measured at 0.14 for all 50 rounds because the four it had never
# touched all read 0.00. That is a suppressed-discovery artifact of the table,
# not a model limitation — utilization was 70–100% throughout.
#
# D repairs only what the model is TOLD (UNTRIED≠0, fixed-but-unknown reward
# probability, round/horizon, names/positions arbitrary). It does NOT prescribe
# a strategy: no "try every option once", no explore/exploit round split, no
# forced initialization. Whether to explore must stay the model's decision,
# because that decision is the channel α is meant to move — scripting it would
# delete the dependent variable (cf. IGT v4, where externally supplied
# deliberation returned every value/risk readout to n.s.).
#
# READ THE DISCOVERY/UTILIZATION METRICS, NOT OptFrac. OptFrac multiplies
# "ever found the best arm" by "used it once found"; at n=20 it was bimodal
# (ten runs ≈0, ten runs 0.30–0.94, sd 0.333 ≈ mean 0.304), which no dose
# response can sit on. Targets: mean coverage 3.55 → ≥4.5, best-never-tried
# 6/20 → ≤2/20, and empirical-best adherence MUST NOT fall (otherwise D only
# bought exploration by giving up on using what was learned).
run_arm D --ans_file "bandit_validity_D_untried" \
          --summary_history --answer_anchor --use_chat --max_new_tokens 24 \
          --untried_semantics

# D2 = D + sampling-uncertainty FACTS. THE MAIN LINE after D.
#
# WHY D2 EXISTS. D opened the discovery gate (best-never-tried 6/20 → 1/20) but
# left coverage flat (3.55 → 3.75) and OptFrac bimodal. Reading the 20-seed
# generations shows the model had understood D's repairs — it knows there are
# five options, that UNTRIED should be tried, and it does pick the highest
# observed mean. What it does NOT represent is estimation uncertainty under a
# random reward: it treats an n=1 observation as a settled property. seeds 0
# and 2 abandoned the best arm permanently after ONE reward=0; seeds 3 and 14
# spent 35–40 rounds on a mid arm before first touching the best one; seed 12
# held an arm measured at 0.16 without ever exploring. (seeds 1/6/8/18 scored
# well only by picking the best arm first and never moving — a good number, not
# evidence of learning, which is why OptFrac alone must not be the readout.)
#
# D2 adds three FACTS and no policy: rewards are random draws, a few-trial
# result is uncertain in BOTH directions (a single 1 is as uninformative as a
# single 0 — stated symmetrically so it is not a nudge to explore), and trial
# count is evidence strength, with the table rendered "k rewards / n trials
# (observed rate r)" so n is read before the rate. It also renames "This is
# choice N of 50" → "Round N of 50" and adds "Do not add a number", targeting
# D's invalid replies (9 total, nearly all round 1, of the form "1. ...", i.e.
# list continuation induced by the enumerated wording).
#
# READ THE MECHANISM METRICS, not coverage: coverage=1 can just mean the first
# pick happened to be the best arm.
#   best_arm_pulled_le1   — best arm tried ≤1 time            (D: 6/20)
#   late_best_discovery   — first best-arm pull after round 30 (D: 3/20)
#   revisit_after_first_zero — best arm tried again after its first 0
#
#   SEEDS=fail bash run_bandit_validity.sh llama3 D2     # targeted probe first
#   SEEDS=20   bash run_bandit_validity.sh llama3 D2     # then the paired run
run_arm D2 --ans_file "bandit_validity_D2_uncertainty" \
          --summary_history --answer_anchor --use_chat --max_new_tokens 24 \
          --untried_semantics --prompt_variant D2

# D3 = D2 + "explore early, exploit late". DIAGNOSTIC CONTROL — run only if D2
# fails, and never sweep α on it.
#
# Those two sentences are strategy, not environment: they hand over the
# explore→exploit schedule, and WHEN the model stops exploring is exactly the
# behaviour α is hypothesised to modulate. Scripting it deletes the dependent
# variable — the IGT v4 result is the precedent (externally supplying the
# deliberation returned every value/risk/RPE readout to n.s., which is what
# localised the +α effect to engagement rather than valuation).
#
# So D3's only job is to split a D2 failure in two: if D3 succeeds where D2
# fails, the model understands the task but cannot plan the schedule itself; if
# D3 also fails, the deficit is not strategic and Bandit is a boundary point at
# this scale (record it like ScienceWorld and stop).
run_arm D3 --ans_file "bandit_validity_D3_scaffold" \
          --summary_history --answer_anchor --use_chat --max_new_tokens 24 \
          --untried_semantics --prompt_variant D3

# NOTE (2026-07-29): a Dbare arm (D without --use_chat) was dropped. The
# α sweep runs under chat, accepting the mask-distribution mismatch as a known
# cost — the same stance as Betting, where a bare re-run collapsed the effect
# and chat was kept deliberately (CLAUDE.md: surviving the harder, mask-
# divergent distribution is the STRONGER generalization claim). If a chat
# dose-response appears, add a bare cell as a robustness check, not a gate.

echo ""
echo "Done. Analyse with:  python3.10 analyze_bandit_validity.py"
