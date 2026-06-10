#!/bin/bash
# ============ GPQA Confidence Betting — RUNNING-SCORE variant (Llama) ============
# Reward-history sub-experiment for §4.6. The MAIN version (run_gpqa_bet.sh) holds
# Current score: 0 fixed → i.i.d. bets, clean wanting–knowing dissociation, and is
# the headline. THIS variant feeds the REAL running total into each prompt
# (Current score: N) to test reward-history sensitivity: does +α amplify
# "chase after a loss / push after a win", does −α blunt sensitivity to balance?
#
# *** It does NOT replace the score=0 main version — it is a control alongside it. ***
#
# Mechanics (--running_score in get_answer_gpqa_bet.py):
#   - strictly serial (bs=1): prompt i depends on running total after i-1, so it
#     CANNOT be batched. --batch_size is ignored / forced to 1.
#   - the score fed in is the running total BEFORE question i (model never sees
#     the outcome of its current bet).
#   - per-sample CSV gains a `score_before` column = the running total the model
#     saw that turn (use it offline to bin bets by win/loss state × α).
#
# Bet/Answer still parsed from generated text (Bet:/Answer: regex), bet ∈
# {0,2,5,10} discrete — NOT a logit readout. ACC reported offline, not from
# inline correct_* fields.
#
# Same-machine rule: bf16 greedy is NOT byte-reproducible across GPUs — keep the
# whole orig/+4/−4 set on ONE box. n=646 (GPQA main+diamond) × serial is slower
# than the batched main version; budget accordingly.
#
# Conditions:
#   orig    (no steering)          → reward-history baseline
#   α=+4   (expert direction)      → predict: stronger chase-after-loss / push-after-win
#   α=−4   (non-expert direction)  → predict: blunted sensitivity to running balance
#
# Output (separate dir, never overwrites the score=0 main results):
#   gpqa_bet_run_8B_summary.csv     — acc / mean_bet / bet dist per condition
#   gpqa_bet_run_8B_per_sample.csv  — per-sample bet, answer, score_before
#   gpqa_bet_run_8B_results.json    — summaries
#
# Usage:
#   bash run_gpqa_bet_running.sh              # full run (orig + ±4)
#   bash run_gpqa_bet_running.sh --pilot      # 20-sample pilot (orig only)
#   bash run_gpqa_bet_running.sh --skip_orig  # skip orig, re-run steered only

# ==================== Shared config ====================
PERCENTAGE=0.5
MASK_TYPE="nmd"
CONFIGS="4-11-20 neg4-11-20"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

DATA_FILE="${BASE_DIR}/benchmark/gpqa_train.json"
MASK_DIR="${BASE_DIR}/mask/${MODEL}_non_logits"
OUT_DIR="${BASE_DIR}/${MODEL}/gpqa_bet_running"

MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
BATCH_SIZE=1          # forced serial; --running_score ignores any other value

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
echo "GPQA Confidence Betting — RUNNING-SCORE variant"
echo "Model  : ${MODEL}-${SIZE}  (serial, real running score)"
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
    --out_prefix     "gpqa_bet_run" \
    --keep_tasks     "GPQA (gpqa_main)" "GPQA (gpqa_diamond)" \
    --use_chat \
    --running_score \
    ${SKIP_ORIG} \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] GPQA Betting (running-score) — ${MODEL} ${SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] GPQA Betting (running-score) — ${MODEL} ${SIZE}"
    exit 1
fi

echo "=================================================="
echo "GPQA Confidence Betting (running-score) finished at: $(date)"
echo "=================================================="
