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
# *** QWEN2.5-7B CROSS-MODEL PORT (2026-07-28) ***
# Second model for §3.2, mirroring the Betting cross-model extension (§3.1.2).
# Scan runs on the **No-Role** prompt (--no_role), matching Llama's authoritative
# neutral_0616 sweep, so the two models are directly comparable.
#
# Layers 16–21 (--start 16 --end 22, exclusive) = Qwen2.5-7B's own mid-band; the
# selection criterion matches Llama's 11–20 (onset of the layer-wise Expert /
# Non-Expert Pearson descent) and agrees with the MMLU-E layer scan's best cell
# 4[16,21]. Requires: components/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy
#
# WHAT TO EXPECT / WHAT WOULD BE NEWS
#   NOTE: the old Llama curve (inverted-U, plateau α=0..+6, peak +2, +8 collapse)
#   is VOID per the banner above — it was measured under position leakage, so it
#   is not a prediction to test against. Run the fixed Llama sweep
#   (run_bandit_llama3.sh) and compare against THAT.
#   The live question, carried over from Betting §3.1.2: there the failing arm was
#   MODEL-specific (Llama degraded on −α, Qwen on +α). Bandit now asks whether
#   "which arm breaks first" is a stable property of the model across tasks.
#   Do NOT assume any particular peak position transfers.
#
# BARE-STRING (no --use_chat): unlike Betting, Bandit deliberately runs on the
# same bare distribution the NMD mask was extracted in. Keep it that way — the
# chat exception in §3.1 is betting-only.
#
# *** TWO RNG FIXES LANDED WITH THIS PORT (get_answer_bandit.py, 2026-07-28) ***
#   1. parse_choice's invalid fallback used the UNSEEDED global `random`. At
#      Llama α=−4 the invalid rate is 0.20, so ~1 in 5 recorded choices was an
#      unreproducible coin flip that fed the reward draw and the next prompt's
#      history. It now uses a per-run `fallback_rng`, kept as a SEPARATE stream
#      from the reward rng so that conditions with different invalid counts still
#      see the SAME reward sequence (verified).
#   2. temperature=1.0 sampling had no per-round seed, so α conditions never
#      faced matched sampling noise. Now seeded per (run, round), independent of
#      α, so sampling luck is matched across the sweep.
#   CONSEQUENCE: Llama's neutral_0616 numbers predate these fixes AND the two
#   design fixes in the banner above, so they are superseded outright — not
#   merely non-byte-reproducible. Get the Llama comparison from a fixed-code run.
#
# Runtime: 9 alphas × 30 runs × 50 rounds ≈ 13,500 generations ≈ 1.1–2.3 h.
# Usage:
#   bash run_bandit_qwen25.sh            # full sweep (30 runs)
#   bash run_bandit_qwen25.sh --pilot    # 2 runs × 3 alphas, timing + sanity

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/Dopamine/components"

MODEL_NAME="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
MODEL_SIZE="7B"
TYPE="non"
HS_PREFIX="qwen2.5"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Full −8→+8 dose sweep at layers 16–21 (No-Role).
CONFIGS="neg8-16-22 neg6-16-22 neg4-16-22 neg2-16-22 0-16-22 2-16-22 4-16-22 6-16-22 8-16-22"

NUM_RUNS=30
NUM_ROUNDS=50
ANS_FILE="answer_bandit_norole_v2"

# Pilot: 2 runs over the three decisive cells (baseline + both extremes). Writes
# to its OWN dir so it can never contaminate the authoritative sweep's resume.
if [ "$1" == "--pilot" ]; then
    NUM_RUNS=2
    CONFIGS="neg8-16-22 0-16-22 8-16-22"
    ANS_FILE="answer_bandit_norole_v2_pilot"
    echo "[PILOT] 2 runs × 3 alphas -> ${ANS_FILE}"
fi

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
