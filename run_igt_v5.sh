#!/bin/bash
# IGT v5 — same task as run_igt.sh, but with prompt_ver=v5.
# v5 keeps the multi-turn feedback setting and "Round N of 100" anchor, removes
# end/final/conclude format words, and does NOT explicitly ask for reasoning:
#   "Which chest do you open? Answer as Chest: N."
#
# Usage:
#   bash run_igt_v5.sh --verify
#   nohup bash run_igt_v5.sh --full > igt_v5_full.log 2>&1 &

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
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
TOP_P=0.9
ANS_FILE="answer_igt_v5"
SAVE_RAW_FLAG="--save_all_raw"
USE_CHAT_FLAG="--use_chat"
PROMPT_VER="v5"
ANCHOR="default"

# ==================== Modes ====================
if [ "$1" == "--verify" ]; then
    CONFIGS="0-11-20 neg4-11-20 4-11-20"; NUM_RUNS=5
    ANS_FILE="answer_igt_v5_verify"
    echo "[VERIFY] ${PROMPT_VER}, α=0/±4, 5 runs"
elif [ "$1" == "--full" ]; then
    ANS_FILE="answer_igt_v5"
    echo "[FULL] ${PROMPT_VER}, full −8→+8 sweep, ${NUM_RUNS} runs"
else
    echo "Usage: bash run_igt_v5.sh {--verify|--full}"
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
    echo "[✓ Done] IGT v5 — finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] IGT v5"
    exit 1
fi
