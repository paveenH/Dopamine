#!/bin/bash
# ==================== Multi-Armed Bandit (EVOLvE ClothesShopping) ====================
# Exploration/Exploitation experiment (§3.2 / §4.7). De-roled port of the EVOLvE
# boutique MAB (Nie et al. ICML 2025): K=5 semantic arms, Bernoulli rewards
# [0.7,0.5,0.4,0.3,0.1] shuffled per run, T=50 rounds, bare-string (NO chat).
#
# *** ALPHA-SCAN EXTENSION (2026-06-15, user decision) ***
# Bandit is the FIRST multi-round dose-response curve in the project. Unlike
# single-step Betting (which can't reach the Yerkes-Dodson right arm), a T=50
# sequence CAN show +α overload (lock-in / behavioral rigidity), so we sweep the
# full −8→+8 dose here. Scan runs on the **No-Role** prompt (--no_role): its
# baseline (~0.816) is clean (no role-suppression confound) and already near
# ceiling, so any +6/+8 drop is unambiguously overload, not "repair topping out".
# Assistant-Role keeps its ±4-only §4.7 result — NOT re-scanned.
#
# WORK_DIR migrated RolePlaying → Dopamine (2026-06-15).
# Output dir keyed by --ans_file: "answer_bandit_norole" keeps No-Role results
# fully separate from any Assistant-Role "answer_bandit" dir.
#
# Runtime estimate: 9 alphas × 30 runs × 50 rounds × 1 gen/round (max_new_tokens=20,
# bs=1 serial) ≈ 13,500 generations ≈ 1.1–2.3 h on one Llama3-8B card. Run a
# 2-run pilot first (NUM_RUNS=2) to get a real s/round before the full sweep.

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
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

NUM_RUNS=30
NUM_ROUNDS=50

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
    --ans_file "answer_bandit_norole" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}"
