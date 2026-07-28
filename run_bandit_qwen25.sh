#!/bin/bash
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
#   Llama (neutral_0616) gave a clean inverted-U: high plateau α=0..+6 (OptFrac
#   0.84–0.89, sample peak +2 = 0.891), +8 collapses to 0.515, −α degrades
#   monotonically. Betting's cross-model port (§3.1.2) found the FAILING ARM is
#   model-specific (Llama fails on −α, Qwen on +α), so the open question here is
#   whether Qwen's Bandit curve also shifts its failure to the +α side — i.e.
#   whether "which arm breaks first" is a stable property of the MODEL rather
#   than of the task. Do NOT assume Llama's +2 peak transfers.
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
#   CONSEQUENCE: Llama's existing neutral_0616 numbers were produced WITHOUT
#   these fixes and will NOT reproduce byte-identically if re-run. They remain
#   the authoritative Llama result; if a same-pipeline Llama baseline is needed
#   for a strict cross-model claim, re-run Llama with the fixed code rather than
#   comparing across the fix boundary on the noisiest (−α) cells.
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
ANS_FILE="answer_bandit_norole"

# Pilot: 2 runs over the three decisive cells (baseline + both extremes). Writes
# to its OWN dir so it can never contaminate the authoritative sweep's resume.
if [ "$1" == "--pilot" ]; then
    NUM_RUNS=2
    CONFIGS="neg8-16-22 0-16-22 8-16-22"
    ANS_FILE="answer_bandit_norole_pilot"
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
