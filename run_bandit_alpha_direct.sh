#!/bin/bash
#
# ================ BANDIT pv5 E-direct α SWEEP (K=5, free exploration) ================
# The α experiment proper. Only runs after the full capability-ladder
# diagnosis in run_bandit_e_protocol.sh (K2 -> K3 -> K5-warm -> K5-free,
# direct vs CoT at every step) reached a FROZEN verdict (2026-07-30, see that
# script's tail comment): E-direct is format-stable everywhere in the ladder
# (~0 invalid, no fallback contamination) and its failure mode is
# well-characterized as an early-outcome-dependent greedy confirmation loop
# — display position 1 wins the initial tie-break under total uncertainty,
# whichever arm earns the FIRST reward=1 becomes a self-reinforcing
# incumbent (sometimes strong enough to override a competing arm with a
# CURRENTLY HIGHER empirical estimate — K5-free seed4), and late performance
# is bimodal on whether that first-success arm happens to be the true best.
# E-CoT's marginal value over this was left UNESTABLISHED (not negative, not
# positive — occasional rescue, unreliable format adherence, confounded
# apparent destabilization) and is deliberately NOT the α interface here.
#
# THIS SCRIPT DOES NOT re-litigate capability — it asks whether α modulates
# the ALREADY-CHARACTERIZED confirmation-loop mechanism itself:
#   - does +α / −α change whether novel (never-yet-tried) arms get explored,
#     particularly past an incorrect incumbent (best_never_tried cases)?
#   - does α change non-novel churn among already-tried arms (switching that
#     revisits arms already sampled, vs new discovery)?
#   - does α change incumbent persistence / late empirical-best adherence
#     (i.e. policy stability, not just which arm ends up modal)?
#   - only after the above: does α move OptFrac/regret, and in which
#     direction — this is an outcome summary, not the primary mechanism
#     evidence (per the same discipline used throughout the capability
#     ladder; see run_bandit_e_protocol.sh header notes on WARM/MAIN).
#
# α ∈ {−4, 0, +4}, layers 11-20 — the same three-point grid as the GSM8K /
# Betting / CGT main lines, run first before deciding whether a wider
# −8..+8 scan is warranted (Bandit's readout is not obviously
# saturating-monotone like Betting's mean_bet, so do not assume the same
# scan shape applies without seeing this three-point result first).
#
# K=5, FREE EXPLORATION ONLY (warm_start_pulls=0) — the warm-start condition
# already showed the model CAN exploit cleanly once trial counts are
# equalized; the open question α might speak to is autonomous
# exploration/persistence under free discovery, which only exists in this
# condition. A warm-start α control is a possible follow-up (see bottom
# note) but is not run by default here.
#
# SEEDS: 0..19 (20 seeds, SEEDS_MODE fixed at 20 in this script — this is
# a paired-seed α design, not a capability-floor spot check, so it always
# uses the wider set; there is no 5-seed mode here). This reuses the same
# 20-seed convention as run_bandit_e_protocol.sh's SEEDS=20 fallback set,
# NOT the 5-seed capability-ladder counterbalanced sets (those were
# hand-picked per-K for position/name balance at small n; at n=20 the
# continuous 0..19 set is close enough to balanced and keeps this
# comparable to any other 20-seed Bandit run in the project).
#
# Usage:
#   bash run_bandit_alpha_direct.sh llama3            # all 3 α cells
#   bash run_bandit_alpha_direct.sh llama3 A0          # α=0 only
#   bash run_bandit_alpha_direct.sh llama3 APOS4       # α=+4 only
#   bash run_bandit_alpha_direct.sh llama3 ANEG4       # α=-4 only

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

NUM_ROUNDS=50
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

run_step () {
  local tag="$1"; local alpha_config="$2"; shift 2
  if [ -n "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$tag" ]; then return; fi
  echo ""
  echo "######################################################################"
  echo "# STEP ${tag}: configs=${alpha_config}  $*"
  echo "# seeds: ${SEEDS}"
  echo "######################################################################"
  python get_answer_bandit.py \
      --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
      --type non --percentage 0.5 --mask_type nmd \
      --configs ${alpha_config} \
      --seeds ${SEEDS} --num_rounds ${NUM_ROUNDS} \
      --no_role --untried_semantics \
      --summary_history --answer_anchor --use_chat \
      --prompt_variant E-direct --num_arms 5 --max_new_tokens 24 \
      --data "${DATA}" --base_dir "${BASE_DIR}" \
      "$@"
}

# ── α cells, K=5 free exploration, E-direct, 20 seeds ─────────────────────
run_step A0    "0-${LAYERS}"      --ans_file "e_K5_direct_alpha0"
run_step APOS4 "4-${LAYERS}"      --ans_file "e_K5_direct_alpha4"
run_step ANEG4 "neg4-${LAYERS}"   --ans_file "e_K5_direct_alphaneg4"

echo ""
echo "Done. Read PAIRED across the 3 cells (same 20 seeds): best_never_tried /"
echo "novel-arm exploration first, then non-novel churn among tried arms, then"
echo "incumbent persistence / late empirical_best_adherence, and OptFrac/regret"
echo "LAST as an outcome summary — not the primary mechanism evidence. See"
echo "run_bandit_e_protocol.sh's frozen capability-ladder verdict for the"
echo "confirmation-loop mechanism this sweep is testing whether α modulates."
echo ""
echo "NOTE: α=0 here is a FRESH 20-seed run, not the same output file as the"
echo "5-seed e_K5_direct capability-ladder baseline — do not treat the two"
echo "as interchangeable even though both are α=0/E-direct/K=5/free."
