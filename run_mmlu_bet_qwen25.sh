#!/bin/bash
# ==================== MMLU Confidence Betting Experiment ====================
# Same design as GPQA Confidence Betting (§4.6), applied to MMLU.
# Higher baseline accuracy (~67%) makes wanting–knowing dissociation cleaner.
#
# Conditions: orig / α=+4 / α=−4  (layers 16–21, i.e. CONFIGS uses the exclusive
#             end 16-22; NMD mask nmd_0.5_16_22_7B.npy. NOT Llama's 11–20.)
# Model     : Qwen2.5-7B-Instruct
# Data      : All 57 MMLU subjects merged (~14k samples)
#
# NOTE (2026-06): KEEPS --use_chat, matching run_gpqa_bet.sh — the §4.6 betting
# headline was collected under the chat wrapper, where α=+4 produces the
# wanting→bet dose-response. A bare-string re-run collapsed the effect (mean_bet
# flat across α), so chat is required for this experiment despite the default-bare
# convention (CLAUDE.md chat-template caveat). build_prompt(use_chat=True) appends
# the "+ Bet: " primer on the chat branch.
#
# Usage:
#   bash run_mmlu_bet_qwen25.sh              # full run
#   bash run_mmlu_bet_qwen25.sh --pilot      # 50-sample pilot (orig only)
#   bash run_mmlu_bet_qwen25.sh --skip_orig  # skip orig, re-run steered only

# ==================== Shared config ====================
PERCENTAGE=0.5
MASK_TYPE="nmd"
CONFIGS="4-16-22 neg4-16-22"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SIZE="7B"

MMLU_DIR="${BASE_DIR}/mmlu"
DATA_FILE="${BASE_DIR}/benchmark/mmlu_all.json"
MASK_DIR="${BASE_DIR}/mask/${MODEL}_non_logits"
OUT_DIR="${BASE_DIR}/${MODEL}/mmlu_bet"

MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
BATCH_SIZE=8

# ==================== Merge MMLU subjects if needed ====================
if [ ! -f "${DATA_FILE}" ]; then
    echo "[Merging] Building ${DATA_FILE} from ${MMLU_DIR}/*.json ..."
    python3 - <<EOF
import json, glob, os
files = sorted(glob.glob('${MMLU_DIR}/*.json'))
all_samples = []
for f in files:
    all_samples.extend(json.load(open(f)))
os.makedirs(os.path.dirname('${DATA_FILE}'), exist_ok=True)
with open('${DATA_FILE}', 'w') as out:
    json.dump(all_samples, out)
print(f"Merged {len(files)} subjects, {len(all_samples)} samples -> ${DATA_FILE}")
EOF
fi

# ==================== Pilot vs. Full ====================
LIMIT=0
EXTRA_CONFIGS="--configs ${CONFIGS}"
SKIP_ORIG=""

if [ "$1" == "--pilot" ]; then
    LIMIT=50
    EXTRA_CONFIGS=""
    echo "[PILOT MODE] 50 samples, orig condition only"
fi

if [ "$1" == "--skip_orig" ] || [ "$2" == "--skip_orig" ]; then
    SKIP_ORIG="--skip_orig"
    echo "[SKIP ORIG] Loading orig results from existing JSON"
fi

# ==================== Run ====================
echo "=================================================="
echo "MMLU Confidence Betting Experiment"
echo "Model  : ${MODEL}-${SIZE}"
if [ "$1" == "--pilot" ]; then
    echo "Mode   : PILOT (limit=50, orig only)"
else
    echo "Mode   : FULL  (all 57 subjects, configs=${CONFIGS})"
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
    --out_prefix     "mmlu_bet" \
    --use_chat \
    --save_all_raw \
    ${SKIP_ORIG} \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] MMLU Confidence Betting — ${MODEL} ${SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] MMLU Confidence Betting — ${MODEL} ${SIZE}"
    exit 1
fi

echo "=================================================="
echo "MMLU Confidence Betting finished at: $(date)"
echo "=================================================="
