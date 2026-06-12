#!/bin/bash
# ==================== MMLU Confidence Betting Experiment ====================
# Same design as GPQA Confidence Betting (§4.6), applied to MMLU.
# Higher baseline accuracy (~67%) makes wanting–knowing dissociation cleaner.
#
# Conditions: orig / α=+4 / α=−4  (layers 11–20, TOP=20, NMD mask)
# Model     : Llama3-8B-IT
# Data      : All 57 MMLU subjects merged (~14k samples)
#
# NOTE (2026-06): runs BARE-STRING (no --use_chat). The NMD mask / diff vectors
# were extracted on bare prompts, so steering must inject into the same activation
# distribution; apply_chat_template prepends <|start_header_id|>system… control
# tokens that shift residual-stream geometry away from where the wanting direction
# was measured, diluting the steer. build_prompt(use_chat=False) returns the bare
# PROMPT_TEMPLATE (which already ends with the "Bet:/Answer:" format spec, so the
# chat-branch's "+ Bet: " primer is not needed). See CLAUDE.md chat-template caveat.
# Prior §4.6 headline numbers were collected under chat — re-run before citing.
#
# Usage:
#   bash run_mmlu_bet.sh              # full run
#   bash run_mmlu_bet.sh --pilot      # 50-sample pilot (orig only)
#   bash run_mmlu_bet.sh --skip_orig  # skip orig, re-run steered only

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
