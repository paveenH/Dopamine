#!/bin/bash
# ==================== GPQA Confidence Betting Experiment ====================
# Experiment: Incentive Salience under RSN Steering
#
# Design:
#   Each question asks the model to bet 0/2/5/10 points before answering.
#   Mean bet / bet distribution = proxy for incentive salience (wanting).
#
# Conditions (2026-06-16: extended to a full −8→+8 dose scan):
#   orig (=α=0)  → baseline betting behavior (run via the orig branch, not CONFIGS)
#   α<0          → prediction: lower bets, more bet=0 (under-wanting)
#   α>0          → prediction: higher bets, fewer bet=0, saturating toward bet=10
# Unlike GSM8K/Bandit (inverted-U on a performance metric), betting's readout is
# mean_bet — a SATURATING MONOTONE quantity (no "overload collapse"; wanting just
# rises until it hits the bet=10 ceiling). The scan's job is to confirm the
# dose-response is monotone and locate the saturation point, giving a SECOND
# "needs-engagement → peak-on-positive-α" curve (alongside Bandit) for the
# motivation-knob cross-task argument. GPQA (n=646, already chat) is the cheap
# carrier for the 9-cell scan; MMLU stays ±4-only (large-n dissociation, no curve).
#
# Model  : Qwen2.5-7B-Instruct   (CROSS-MODEL extension of the Llama3-8B result)
# Task   : GPQA main + diamond only (micro accuracy)
# Layers : 16–21 (i.e. --start 16 --end 22, exclusive) — Qwen2.5-7B's own mid-band,
#          chosen by the same criterion as Llama3's 11–20: the onset of the
#          layer-wise Expert/Non-Expert Pearson descent (Qwen2.5-7B has 28 decoder
#          layers, H=3584; Pearson falls from L16 and bottoms ~L20-25), and matching
#          the MMLU-E layer scan's best cell 4[16,21] (66.18%).
#          Mask MUST be generated over the same range:
#            nmd_0.5_16_22_7B.npy in components/mask/qwen2.5_non_logits/
#
# NOTE (2026-06): KEEPS --use_chat — this is a deliberate EXCEPTION to the
# default-bare convention (CLAUDE.md chat-template caveat). On Llama3 a bare-string
# re-run (n=646) COLLAPSED the betting effect: mean_bet flat at 5.18/5.28/5.31
# across α=+4/−4/orig and bet10 flat (~13–16%), vs the chat §3.1 headline where
# α=+4 lifts mean_bet to 7.65 and bet10 to 53.1%. So the wanting→bet dose-response
# lives only under the chat wrapper there; running bare makes the experiment null.
# CAVEAT for this cross-model port: that bare-vs-chat test was run on LLAMA only.
# Qwen2.5's chat template differs more from the bare distribution the NMD mask was
# extracted in, so if the Qwen curve comes out flat, a bare-vs-chat control
# (--pilot-scale, α=0/±4) is the FIRST thing to run before calling it a null.
#
# Output:
#   gpqa_bet_7B_summary.csv      — acc / mean_bet / bet distribution per condition
#   gpqa_bet_7B_per_sample.csv   — per-sample bet and answer
#   gpqa_bet_7B_results.json     — full generated texts
#
# Usage:
#   bash run_gpqa_bet_qwen25.sh              # full run
#   bash run_gpqa_bet_qwen25.sh --pilot      # 20-sample pilot (orig only)
#   bash run_gpqa_bet_qwen25.sh --skip_orig  # skip orig, re-run steered only

# ==================== Shared config ====================
PERCENTAGE=0.5
MASK_TYPE="nmd"
# Full −8→+8 scan (orig=α=0 runs via the orig branch; CONFIGS holds the 8 steered cells).
CONFIGS="neg8-16-22 neg6-16-22 neg4-16-22 neg2-16-22 2-16-22 4-16-22 6-16-22 8-16-22"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SIZE="7B"

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
