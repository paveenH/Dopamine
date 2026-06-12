#!/bin/bash
# ======= GPQA Confidence Betting — BARE + "Bet: " PREFILL decomposition =======
# Mechanism decomposition experiment for the chat-vs-bare betting puzzle.
#
# Background: run_gpqa_bet.sh (chat) shows a strong wanting→bet effect (α=+4 lifts
# mean_bet 5.05→7.65, bet10 8.8%→53%). A pure bare-string re-run KILLED it
# (mean_bet flat 5.31/5.18/5.28, bet10 ~13-16%). Two candidate causes:
#   (1) the "Bet: " prefill (a clean施力点: next token = bet digit) — present in
#       chat (build_prompt appends "Bet: " on the chat branch), absent in bare.
#   (2) the chat role scaffold itself (<|system|>/<|user|>/<|assistant|> turns).
#
# THIS run isolates (1): bare-string prompt + the SAME "Bet: " primer, but WITHOUT
# the chat scaffold (--bet_prefill, no --use_chat). Decision rule:
#   • if α=+4 lifts mean_bet/bet10 back toward the chat numbers → the PREFILL is
#     the active ingredient; the betting effect is real wanting once α has a clean
#     next-token施力点, and chat is just a convenient way to provide it.
#   • if it stays null (≈ pure-bare 5.31/5.18/5.28) → the chat ROLE STRUCTURE is
#     load-bearing, and the §4.6 headline is an α×chat-scaffold joint product.
#
# Compare against:
#   chat (authoritative): gpqa_bet/        — re-run 5.07→7.58, bet10 9.4→51.5%
#   pure bare (no prefill): the prior bare run, mean_bet 5.31/5.18/5.28 (null)
#
# Same config as run_gpqa_bet.sh except: -use_chat +bet_prefill, separate OUT_DIR.
#
# Usage:
#   bash run_gpqa_bet_prefill.sh              # full run (orig + ±4, n=646)
#   bash run_gpqa_bet_prefill.sh --pilot      # 20-sample pilot (orig only)

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
OUT_DIR="${BASE_DIR}/${MODEL}/gpqa_bet_prefill"

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
echo "GPQA Betting — BARE + 'Bet: ' prefill (mechanism decomposition)"
echo "Model  : ${MODEL}-${SIZE}  (bare-string, prefill ON, no chat)"
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
    --out_prefix     "gpqa_bet_prefill" \
    --keep_tasks     "GPQA (gpqa_main)" "GPQA (gpqa_diamond)" \
    --bet_prefill \
    ${SKIP_ORIG} \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] GPQA Betting (bare+prefill) — ${MODEL} ${SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] GPQA Betting (bare+prefill) — ${MODEL} ${SIZE}"
    exit 1
fi

echo "=================================================="
echo "GPQA Betting (bare+prefill) finished at: $(date)"
echo "=================================================="
