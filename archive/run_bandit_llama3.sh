#!/bin/bash
#
# *** READ FIRST — ALL PRE-2026-07-28 BANDIT RESULTS ARE VOID ***
# Two design faults were found and fixed in get_answer_bandit.py. They are not
# noise; they break the main readout:
#   (A) POSITION LEAKAGE. shuffle_arms() shuffled the arm NAMES then zipped them
#       against [0.7,0.5,0.4,0.3,0.1], so the best arm sat at display position 1
#       and the worst at position 5 for EVERY seed (checked 0-29: best position
#       was 1 in all 30). Only which *name* was best varied. OptFrac therefore
#       could not be distinguished from a first-option bias.
#   (B) PERMISSIVE PARSER. parse_choice() returned the first arm name appearing
#       anywhere in the output, scanning in display order. Together with (A) a
#       reply that merely echoed the option list was recorded as a VALID choice
#       of the BEST arm (verified directly). OptFrac was inflated and
#       invalid_rate understated by an unknown amount.
# Consequence: every earlier Bandit number (incl. the published Llama inverted-U
# with peak +2 and the +8 collapse to 0.515) is not comparable to post-fix runs,
# and cannot be salvaged offline — no raw text was stored. Treat post-fix runs as
# a NEW baseline, not as a validation of the old curve.
# Now recorded per round: raw generation, valid flag, n_matched; plus valid-only
# OptFrac/regret and mean_best_position (should average ~3.0 with K=5; a value
# pinned at 1.0 means position leakage is back).
#
# ==================== Multi-Armed Bandit (EVOLvE ClothesShopping) ====================
# Exploration/Exploitation experiment (§3.2 / §4.7). De-roled port of the EVOLvE
# boutique MAB (Nie et al. ICML 2025): K=5 semantic arms, Bernoulli rewards
# [0.7,0.5,0.4,0.3,0.1] shuffled per run, T=50 rounds, bare-string (NO chat).
#
# *** LLAMA3 RE-RUN UNDER THE RNG FIXES (2026-07-28) ***
# Same experiment / same layers / same No-Role prompt as the authoritative
# neutral_0616 sweep. The ONLY difference is that get_answer_bandit.py now:
#   1. draws the invalid-parse fallback arm from a seeded per-run `fallback_rng`
#      (it used the UNSEEDED global `random` before), and
#   2. seeds torch per (run, round) so temperature=1.0 sampling noise is MATCHED
#      across α instead of drifting with generation order.
# Neither changes the task, prompt, mask, or layers.
#
# WHY RE-RUN AT ALL
#   Llama's invalid_rate is strongly α-dependent (−4: 0.201, −8: 0.157, −6: 0.115
#   vs ~0.00 on the +α side). At α=−4 roughly one in five recorded "choices" was
#   an unseeded coin flip that then fed the reward draw AND the history used to
#   build the next prompt. So the NEGATIVE arm — exactly the arm the cross-model
#   comparison with Qwen turns on — rests on that noise. This run gives a Llama
#   baseline produced by the SAME pipeline as the Qwen port, so the two curves can
#   be compared without crossing the fix boundary.
#
# ISOLATION: writes to answer_bandit_norole_rngfix/, NOT answer_bandit_norole/.
#   This is load-bearing. get_answer_bandit.py RESUMES by reading the existing
#   summary CSV and SKIPPING every alpha already listed in it — pointing this at
#   the old dir would silently skip all 9 cells and produce nothing, or worse,
#   append to the authoritative file.
#
# EXPECTED OUTCOME: the +α side should barely move (invalid ~0 there, so the
#   fallback fix is a no-op and only the sampling seed changes). Any real change
#   should concentrate on −8/−6/−4. If the −α cells shift materially, the old
#   negative arm was partly fallback noise; if they do not, the published shape
#   is confirmed under a cleaner pipeline. Both outcomes are worth having.
#   NOTE this will NOT reproduce neutral_0616 byte-for-byte — it is not meant to.
#
# Runtime: 9 alphas × 30 runs × 50 rounds ≈ 13,500 generations ≈ 1.1–2.3 h.
# Usage:
#   bash run_bandit_rngfix.sh            # full sweep (30 runs)
#   bash run_bandit_rngfix.sh --pilot    # 2 runs × 3 alphas, quick sanity

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/Dopamine/components"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Full −8→+8 dose sweep at layers 11–20 (No-Role).
ANS_FILE="answer_bandit_norole_v2"

CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

NUM_RUNS=30
NUM_ROUNDS=50

# Pilot: 2 runs over baseline + both extremes, to its own dir.
if [ "$1" == "--pilot" ]; then
    NUM_RUNS=2
    CONFIGS="neg8-11-20 0-11-20 8-11-20"
    ANS_FILE="answer_bandit_norole_v2_pilot"
    echo "[PILOT] 2 runs × 3 alphas -> ${ANS_FILE}"
fi

echo "Llama3 Bandit re-run (RNG fixes) -> ${ANS_FILE}"
echo "Start: $(date)"

python get_answer_bandit.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --num_runs ${NUM_RUNS} \
    --num_rounds ${NUM_ROUNDS} \
    --no_role \
    --ans_file "${ANS_FILE}" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}"
