#!/bin/bash
# ==================== GPQA Delay Discounting Experiment ====================
# Experiment: Adaptive CoT Length under RSN Steering
#
# Design:
#   Prompt lets the model decide when to commit ("Final Answer: X").
#   CoT token length before commit = proxy for delay tolerance.
#
# Conditions:
#   orig    (no steering)          → baseline CoT length
#   α=+4   (expert direction)      → prediction: longer CoT, later commit
#   α=−4   (non-expert direction)  → prediction: shorter CoT, earlier commit
#
# Model  : Llama3-8B-IT
# Task   : GPQA (diamond + main + extended, ~590 samples)
# Layers : 11–20 (mid-layer RSNs)
#
# NOTE (2026-06): runs BARE-STRING (no --use_chat). The NMD mask / diff vectors
# were extracted on bare prompts (run_hidden_mmlu.sh, run_mean_diff.sh, run_nmd.sh),
# so steering must inject into the same activation distribution; apply_chat_template
# prepends <|start_header_id|>system… control tokens that shift residual-stream
# geometry away from where the wanting direction was measured, diluting the steer.
# build_prompt(use_chat=False) returns the bare PROMPT_TEMPLATE unchanged. See the
# chat-template alignment caveat in CLAUDE.md.
#
# Output:
#   gpqa_delay_8B_summary.csv     — acc / cot_mean / cot_std / commit_rate per condition
#   gpqa_delay_8B_per_sample.csv  — per-sample cot_tokens
#   gpqa_delay_8B_results.json    — full generated texts
#
# Usage:
#   bash run_gpqa_cot_delay.sh           # full run
#   bash run_gpqa_cot_delay.sh --pilot   # 20-sample pilot (orig only)

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
OUT_DIR="${BASE_DIR}/${MODEL}/gpqa_delay"

MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
BATCH_SIZE=8
SAMPLE=200
SEED=42

# ==================== Pilot vs. Full ====================
LIMIT=0
EXTRA_CONFIGS="--configs ${CONFIGS}"
SKIP_ORIG=""

if [ "$1" == "--pilot" ]; then
    LIMIT=20
    SAMPLE=0
    EXTRA_CONFIGS=""   # pilot: orig only, no steering
    echo "[PILOT MODE] 20 samples, orig condition only"
fi

if [ "$1" == "--skip_orig" ] || [ "$2" == "--skip_orig" ]; then
    SKIP_ORIG="--skip_orig"
    echo "[SKIP ORIG] Loading orig results from existing JSON"
fi

# ==================== Run ====================
echo "=================================================="
echo "GPQA Delay Discounting Experiment"
echo "Model  : ${MODEL}-${SIZE}"
if [ "$1" == "--pilot" ]; then
    echo "Mode   : PILOT (limit=20, orig only)"
else
    echo "Mode   : FULL  (all samples, configs=${CONFIGS})"
fi
echo "Output : ${OUT_DIR}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_gpqa_cot_delay.py \
    --model         "${MODEL}" \
    --model_dir     "${MODEL_DIR}" \
    --size          "${SIZE}" \
    --data_file     "${DATA_FILE}" \
    --out_dir       "${OUT_DIR}" \
    --mask_dir      "${MASK_DIR}" \
    --mask_type     "${MASK_TYPE}" \
    --percentage    "${PERCENTAGE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature   "${TEMPERATURE}" \
    --batch_size    "${BATCH_SIZE}" \
    --limit         "${LIMIT}" \
    --sample        "${SAMPLE}" \
    --seed          "${SEED}" \
    ${SKIP_ORIG} \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] GPQA Delay Discounting — ${MODEL} ${SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] GPQA Delay Discounting — ${MODEL} ${SIZE}"
    exit 1
fi

echo ""
echo "=================================================="
echo "GPQA Delay Discounting finished at: $(date)"
echo "=================================================="
