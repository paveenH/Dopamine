#!/bin/bash
# IGT — Iowa Gambling Task (get_answer_igt.py). 100 trials, 4 decks, multi-turn
# chat with trial-by-trial reward/penalty feedback so the model can LEARN to avoid
# the disadvantageous decks (A/B). Primary readout = net score = P(C+D)−P(A+B),
# by 20-trial block (learning curve). RSN α prediction: α+ → more A/B → lower net.
#
# Usage (verify baseline FIRST: confirm α=0 is not near-random, format parses):
#   bash run_igt.sh --verify     # α=0/±4, 5 runs
#   bash run_igt.sh --full       # full −8→+8 sweep, 20 runs (after verify)
#   nohup bash run_igt.sh --full > igt_full.log 2>&1 &
#
# Injection: tail=1 (validated CGT strength); inject_turn deliberately OFF.
# chat is REQUIRED (trial-by-trial learning needs the multi-turn dialogue) — same
# chat-vs-bare steering caveat as CGT, treated as a feature per the betting stance.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== Paths ====================
DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# ==================== Defaults ====================
CONFIGS="0-11-20 neg2-11-20 2-11-20 neg4-11-20 4-11-20 neg6-11-20 6-11-20 neg8-11-20 8-11-20"
NUM_RUNS=20
MAX_NEW_TOKENS=200       # <reasoning>…</reasoning><choice>N</choice> needs room
TEMPERATURE=1.0
TOP_P=0.9
ANS_FILE="answer_igt"
SAVE_RAW_FLAG="--save_all_raw"
USE_CHAT_FLAG="--use_chat"
PROMPT_VER="v4"          # v4 = v2's "Round N of 100" anchor + NO end/final/conclude
                         # words + plain "Chest: N" + explicit "First reason … then
                         # give" so reasoning survives. (v3's <Chest:N> tag fixed
                         # invalid but collapsed reasoning → round-robin; see get_answer_igt.py)
ANCHOR="default"

# ==================== Modes ====================
if [ "$1" == "--verify" ]; then
    CONFIGS="0-11-20 neg4-11-20 4-11-20"; NUM_RUNS=5
    ANS_FILE="answer_igt_v4_verify"
    echo "[VERIFY] ${PROMPT_VER}, α=0/±4, 5 runs"
elif [ "$1" == "--full" ]; then
    ANS_FILE="answer_igt"
    echo "[FULL] ${PROMPT_VER}, full −8→+8 sweep, ${NUM_RUNS} runs"
else
    echo "Usage: bash run_igt.sh {--verify|--full}"
    exit 1
fi

echo "=================================================="
echo "IGT — ${MODEL_NAME} ${MODEL_SIZE}"
echo "Configs: ${CONFIGS}"
echo "Output : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_igt.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --prompt_ver "${PROMPT_VER}" \
    --anchor "${ANCHOR}" \
    --num_runs "${NUM_RUNS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --ans_file "${ANS_FILE}" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    ${USE_CHAT_FLAG} \
    ${SAVE_RAW_FLAG}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] IGT — finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] IGT"
    exit 1
fi
