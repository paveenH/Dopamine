#!/bin/bash
#
# ==================== BANDIT α=0 TASK-VALIDITY PILOT ====================
# NOT an α sweep. This asks ONE question before any sweep is worth running:
#
#   under a given prompt/parser interface, does a 7-8B model do Bandit
#   learning at all?
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

# α=0 ONLY. This pilot measures the task, not the intervention.
CONFIGS="${LAYERS}"
SEEDS="0 3 4 9 37"
NUM_ROUNDS=50

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

echo ""
echo "Done. Analyse with:  python3.10 analyze_bandit_validity.py"
