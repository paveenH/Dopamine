#!/bin/bash
# ==================== GPQA Confidence Betting Experiment ====================
# Experiment: Incentive Salience under RSN Steering
#
# Design:
#   Each question asks the model to bet 0/2/5/10 points before answering.
#   Mean bet / bet distribution = proxy for incentive salience (wanting).
#
# Conditions:
#   orig    (no steering)          → baseline betting behavior
#   α=+4   (expert direction)      → prediction: higher bets, fewer bet=0
#   α=−4   (non-expert direction)  → prediction: lower bets, more bet=0
#
# Model  : Llama3-8B-IT
# Task   : GPQA main + diamond only (micro accuracy)
# Layers : 11–20 (mid-layer RSNs)
#
# Output:
#   gpqa_bet_8B_summary.csv      — acc / mean_bet / bet distribution per condition
#   gpqa_bet_8B_per_sample.csv   — per-sample bet and answer
#   gpqa_bet_8B_results.json     — full generated texts
#
# Usage:
#   bash run_gpqa_bet.sh              # full run
#   bash run_gpqa_bet.sh --pilot      # 20-sample pilot (orig only)
#   bash run_gpqa_bet.sh --skip_orig  # skip orig, re-run steered only

# ==================== Shared config ====================
PERCENTAGE=0.5
MASK_TYPE="nmd"
CONFIGS="4-11-20 neg4-11-20"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

MODEL="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

DATA_FILE="${BASE_DIR}/benchmark/gpqa_train.json"
MASK_DIR="${BASE_DIR}/mask/${MODEL}_non_logits"
OUT_DIR="${BASE_DIR}/${MODEL}/gpqa_bet"

MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
BATCH_SIZE=8

# ==================== Pilot vs. Full ====================
LIMIT=0
EXTRA_CONFIGS="--configs ${CONFIGS}"
SKIP_ORIG=""

if [ "$1" == "--pilot" ]; then
    LIMIT=20
    EXTRA_CONFIGS=""
    echo "[PILOT MODE] 20 samples, orig condition only"
fi

if [ "$1" == "--skip_orig" ] || [ "$2" == "--skip_orig" ]; then
    SKIP_ORIG="--skip_orig"
    echo "[SKIP ORIG] Loading orig results from existing JSON"
fi

# ==================== Run ====================
echo "=================================================="
echo "GPQA Confidence Betting Experiment"
echo "Model  : ${MODEL}-${SIZE}"
if [ "$1" == "--pilot" ]; then
    echo "Mode   : PILOT (limit=20, orig only)"
else
    echo "Mode   : FULL  (all main+diamond, configs=${CONFIGS})"
fi
echo "Output : ${OUT_DIR}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_gpqa_bet.py \
    --model          "${MODEL}" \
    --model_dir      "${MODEL_DIR}" \
    --size           "${SIZE}" \
    --data_file      "${DATA_FILE}" \
    --out_dir        "${OUT_DIR}" \
    --mask_dir       "${MASK_DIR}" \
    --mask_type      "${MASK_TYPE}" \
    --percentage     "${PERCENTAGE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature    "${TEMPERATURE}" \
    --top_p          "${TOP_P}" \
    --batch_size     "${BATCH_SIZE}" \
    --limit          "${LIMIT}" \
    --out_prefix     "gpqa_bet" \
    --keep_tasks     "GPQA (gpqa_main)" "GPQA (gpqa_diamond)" \
    --use_chat \
    ${SKIP_ORIG} \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] GPQA Confidence Betting — ${MODEL} ${SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] GPQA Confidence Betting — ${MODEL} ${SIZE}"
    exit 1
fi

echo "=================================================="
echo "GPQA Confidence Betting finished at: $(date)"
echo "=================================================="
