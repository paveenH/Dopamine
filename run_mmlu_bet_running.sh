#!/bin/bash
# ============ MMLU Confidence Betting — RUNNING-SCORE variant (Llama) ============
# Reward-history sub-experiment, MMLU counterpart of run_gpqa_bet_running.sh.
# The MAIN MMLU version (run_mmlu_bet.sh) holds Current score: 0 fixed (i.i.d.,
# n=14,042) and is the headline (§4.6 結果二). THIS variant feeds the REAL
# running total into each prompt to test reward-history sensitivity.
#
# *** PER-TASK reset (--per_task_reset): MMLU has 57 subjects. A single 14k-long
# continuous score series is meaningless (balance drifts to the thousands), so
# the running score resets to 0 at each subject boundary — each subject is one
# independent game (~300–4600 questions). Within a subject the score accumulates;
# the position index also restarts per subject. Samples are grouped by task first.
#
# Mechanics (--running_score in get_answer_gpqa_bet.py):
#   - strictly serial (bs=1): prompt i depends on the running total after i-1.
#   - score fed in = running total BEFORE question i (model never sees the
#     outcome of its current bet); resets at each task boundary.
#   - per-sample CSV gains a `score_before` column = the running total the model
#     saw that turn (offline: regress bet ~ score_before per subject / per alpha).
#
# *** This is SLOW: 14,042 × 3 conditions serial (~42k generations, bs=1). Budget
# accordingly; run under nohup. Same-machine rule applies (keep all 3 on one box).
#
# Bet/Answer parsed from generated text (regex), bet ∈ {0,2,5,10} discrete — NOT
# a logit readout. ACC reported offline, not from inline correct_* fields.
#
# NOTE (2026-06): runs BARE-STRING (no --use_chat), matching run_mmlu_bet.sh —
# steering must sit on the same bare activation distribution the NMD mask was
# extracted in. build_prompt(use_chat=False) returns the bare PROMPT_TEMPLATE.
# See CLAUDE.md chat-template caveat. Re-run before citing prior chat numbers.
#
# Output (separate dir, never overwrites the score=0 main MMLU results):
#   mmlu_bet_run_8B_summary.csv     — acc / mean_bet / bet dist per condition
#   mmlu_bet_run_8B_per_sample.csv  — per-sample bet, answer, task, score_before
#   mmlu_bet_run_8B_results.json    — summaries
#
# Usage:
#   bash run_mmlu_bet_running.sh              # full run (orig + ±4)
#   bash run_mmlu_bet_running.sh --pilot      # 50-sample pilot (orig only)
#   bash run_mmlu_bet_running.sh --skip_orig  # skip orig, re-run steered only

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
OUT_DIR="${BASE_DIR}/${MODEL}/mmlu_bet_running"

MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
BATCH_SIZE=1          # forced serial; --running_score ignores any other value

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
echo "MMLU Confidence Betting — RUNNING-SCORE variant (per-task reset)"
echo "Model  : ${MODEL}-${SIZE}  (serial, real running score, reset per subject)"
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
    --out_prefix     "mmlu_bet_run" \
    --running_score \
    --per_task_reset \
    ${SKIP_ORIG} \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] MMLU Betting (running-score) — ${MODEL} ${SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] MMLU Betting (running-score) — ${MODEL} ${SIZE}"
    exit 1
fi

echo "=================================================="
echo "MMLU Confidence Betting (running-score) finished at: $(date)"
echo "=================================================="
