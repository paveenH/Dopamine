#!/bin/bash
# CGT-Sequential — ascending/descending delay-aversion task (get_answer_cgt_seq.py).
# Sibling of run_cgt.sh (which is the SIMULTANEOUS task). This one reveals the bet
# one tier at a time; the model accepts/waits, and the ACCEPT STEP is the primary
# readout (delay aversion / impulsivity). asc and desc are separate run-level
# conditions — run BOTH, then delay_aversion_index = mean_bet_desc − mean_bet_asc.
#
# Usage (verify baseline format FIRST before any α-scan):
#   bash run_cgt_seq.sh --verify_asc    # α=0/±6, 5 runs, ascending  (5→95)
#   bash run_cgt_seq.sh --verify_desc   # α=0/±6, 5 runs, descending (95→5)
#   bash run_cgt_seq.sh --asc           # full −8→+8 sweep, ascending  (after verify)
#   bash run_cgt_seq.sh --desc          # full −8→+8 sweep, descending (after verify)
#   nohup bash run_cgt_seq.sh --asc > cgtseq_asc.log 2>&1 &
#
# Injection: tail=1 (the validated simultaneous-CGT strength; tail=4 over-steered
# and broke qdm). inject_turn deliberately OFF.

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
# each round = 1 colour gen + up to 5 accept/wait gens, each short (one word) →
# 64 tokens is plenty and keeps per-round cost down (≈6 short gens vs 1 long).
MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
PRESENTATION=""          # set by the mode flag below (required)
ANS_FILE=""
SAVE_RAW_FLAG="--save_all_raw"
USE_CHAT_FLAG="--use_chat"
PROMPT_VER="v1"          # v1 = validated e55b132 prompt; v2 = symmetrised format
ANCHOR="default"         # default | answer | none (see get_answer_cgt_seq.py)

# ==================== Modes ====================
# --- v1 (validated e55b132 baseline) ---
if [ "$1" == "--verify_asc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_verify"
    echo "[VERIFY_ASC] v1, α=0/±6, 5 runs, ascending (5→95)"
elif [ "$1" == "--verify_desc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_verify"
    echo "[VERIFY_DESC] v1, α=0/±6, 5 runs, descending (95→5)"
elif [ "$1" == "--asc" ]; then
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc"
    echo "[ASC] v1, full −8→+8 sweep, ${NUM_RUNS} runs, ascending (5→95)"
elif [ "$1" == "--desc" ]; then
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc"
    echo "[DESC] v1, full −8→+8 sweep, ${NUM_RUNS} runs, descending (95→5)"

# --- v2a: symmetrised prompt + "Answer: " anchor on BOTH steps ---
elif [ "$1" == "--verify_v2a_asc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_v2a_verify"
    PROMPT_VER="v2"; ANCHOR="answer"
    echo "[VERIFY_V2A_ASC] v2 + Answer: anchor, α=0/±6, 5 runs, asc"
elif [ "$1" == "--verify_v2a_desc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_v2a_verify"
    PROMPT_VER="v2"; ANCHOR="answer"
    echo "[VERIFY_V2A_DESC] v2 + Answer: anchor, α=0/±6, 5 runs, desc"

# --- v2b: symmetrised prompt + NO anchor on either step ---
elif [ "$1" == "--verify_v2b_asc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_v2b_verify"
    PROMPT_VER="v2"; ANCHOR="none"
    echo "[VERIFY_V2B_ASC] v2 + no anchor, α=0/±6, 5 runs, asc"
elif [ "$1" == "--verify_v2b_desc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_v2b_verify"
    PROMPT_VER="v2"; ANCHOR="none"
    echo "[VERIFY_V2B_DESC] v2 + no anchor, α=0/±6, 5 runs, desc"

# --- v2a / v2b full sweeps (only after a verify A/B picks the winner) ---
elif [ "$1" == "--v2a_asc" ]; then
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_v2a"; PROMPT_VER="v2"; ANCHOR="answer"
    echo "[V2A_ASC] v2 + Answer: anchor, full −8→+8, ${NUM_RUNS} runs, asc"
elif [ "$1" == "--v2a_desc" ]; then
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_v2a"; PROMPT_VER="v2"; ANCHOR="answer"
    echo "[V2A_DESC] v2 + Answer: anchor, full −8→+8, ${NUM_RUNS} runs, desc"
elif [ "$1" == "--v2b_asc" ]; then
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_v2b"; PROMPT_VER="v2"; ANCHOR="none"
    echo "[V2B_ASC] v2 + no anchor, full −8→+8, ${NUM_RUNS} runs, asc"
elif [ "$1" == "--v2b_desc" ]; then
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_v2b"; PROMPT_VER="v2"; ANCHOR="none"
    echo "[V2B_DESC] v2 + no anchor, full −8→+8, ${NUM_RUNS} runs, desc"

# --- v3 (A1): v2b winner (symmetrised, no anchor) + EXPLICIT next-offer hint at
#     each bet tier. Tests whether asc early-stop is strategic-delay vs impulsivity:
#     v2b showed asc Wait reasoning NEVER mentions "wait = bigger bet" (0/539 at α=0),
#     so the model didn't know the rule. v3 states it. If asc STILL early-stops → real
#     impulsivity; if it starts strategic-waiting → v2b's early-stop was rule-ignorance.
elif [ "$1" == "--verify_v3_asc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_v3_verify"
    PROMPT_VER="v3"; ANCHOR="none"
    echo "[VERIFY_V3_ASC] v3 (v2b + next-offer hint), α=0/±6, 5 runs, asc"
elif [ "$1" == "--verify_v3_desc" ]; then
    CONFIGS="0-11-20 neg6-11-20 6-11-20"; NUM_RUNS=5
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_v3_verify"
    PROMPT_VER="v3"; ANCHOR="none"
    echo "[VERIFY_V3_DESC] v3 (v2b + next-offer hint), α=0/±6, 5 runs, desc"
elif [ "$1" == "--v3_asc" ]; then
    PRESENTATION="asc"; ANS_FILE="answer_cgtseq_asc_v3"; PROMPT_VER="v3"; ANCHOR="none"
    echo "[V3_ASC] v3 (v2b + next-offer hint), full −8→+8, ${NUM_RUNS} runs, asc"
elif [ "$1" == "--v3_desc" ]; then
    PRESENTATION="desc"; ANS_FILE="answer_cgtseq_desc_v3"; PROMPT_VER="v3"; ANCHOR="none"
    echo "[V3_DESC] v3 (v2b + next-offer hint), full −8→+8, ${NUM_RUNS} runs, desc"
else
    echo "Usage: bash run_cgt_seq.sh {--verify_asc|--verify_desc|--asc|--desc}"
    echo "       v2a (Answer: anchor): --verify_v2a_asc|--verify_v2a_desc|--v2a_asc|--v2a_desc"
    echo "       v2b (no anchor):      --verify_v2b_asc|--verify_v2b_desc|--v2b_asc|--v2b_desc"
    echo "       v3 (v2b + hint):      --verify_v3_asc|--verify_v3_desc|--v3_asc|--v3_desc"
    exit 1
fi

echo "=================================================="
echo "CGT-Sequential — ${MODEL_NAME} ${MODEL_SIZE} | presentation=${PRESENTATION}"
echo "Configs: ${CONFIGS}"
echo "Output : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_cgt_seq.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --presentation "${PRESENTATION}" \
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
    echo "[✓ Done] CGT-Sequential ${PRESENTATION} — finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] CGT-Sequential ${PRESENTATION}"
    exit 1
fi
