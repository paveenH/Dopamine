#!/bin/bash
# ============ GPQA Betting — RAW DIAGNOSTIC re-run (Qwen2.5-7B, α=+6/+8) ============
#
# WHY THIS EXISTS
#   The Qwen2.5-7B 9-α sweep (components/qwen2.5/gpqa_bet/) reproduced the Llama
#   betting dose-response on the band −8…+4 (mean_bet 4.02 → 7.06, Spearman
#   rho=0.464, accuracy flat 32.8–34.7%), but the two top cells are UNINTERPRETABLE
#   from the parsed CSV alone:
#
#     α=+6 : mean_bet 5.031, std 0.392, bet5=99.4%, bet_entropy 0.038, invalid 0%
#            → the bet distribution COLLAPSED TO A CONSTANT (baseline is 88% bet-5
#              + 12% bet-10, so this is NOT "back to baseline"). Answering is
#              unaffected (acc 33.75%, pred distribution normal). Only the betting
#              dimension degenerated.
#     α=+8 : bet_invalid 76.6% — only 151/646 produced a parseable bet, and those
#            survivors are positively selected (acc|valid 35.8% vs acc|invalid
#            27.7%), so its mean_bet/accuracy are NOT comparable to other cells.
#
#   Two incompatible explanations, opposite conclusions:
#     (a) REAL behavioral collapse → Yerkes-Dodson right arm; Qwen overloads
#         earlier than Llama (whose mean_bet was still 7.62 at +8).
#     (b) PARSE ARTIFACT → the model starts reasoning before committing, so the
#         digit falls outside the 64-token window / off the "Bet: " prefill, and
#         parse_output silently fails or returns a default.
#   Same failure class as the HaluEval +8 `startYN%` collapse (reasoning-first
#   swallowing the verdict), which was only diagnosable from raw text.
#
#   The parsed `bet` column cannot separate (a) from (b). This script re-runs ONLY
#   those two cells with --save_all_raw so the generated text is on disk.
#
# ISOLATION
#   Writes to gpqa_bet_raw/ (NOT gpqa_bet/) so the authoritative sweep is never
#   overwritten. Also re-runs orig as an in-run reference: it is the same greedy-ish
#   sampled config, so orig's raw text shows what a HEALTHY reply looks like under
#   the identical decoding settings, which is what +6 must be compared against.
#
# NOTE temperature=1.0 (inherited from the main sweep) means generation is NOT
#   deterministic — the re-run's numbers will not byte-match the original sweep.
#   That is fine for a diagnostic: we are reading the SHAPE of the text, not
#   re-deriving the table. Do NOT cite numbers from this directory; cite the main
#   sweep and use this only to classify the failure mode.
#
# Usage:
#   bash run_gpqa_bet_qwen25_raw.sh            # orig + α=+6 + α=+8, full n=646
#   bash run_gpqa_bet_qwen25_raw.sh --pilot    # 40 samples, orig only (smoke test)

PERCENTAGE=0.5
MASK_TYPE="nmd"
# Only the two anomalous cells. Layers 16-22 = Qwen2.5-7B mid-band (see
# run_gpqa_bet_qwen25.sh for the layer-selection rationale).
CONFIGS="6-16-22 8-16-22"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SIZE="7B"

DATA_FILE="${BASE_DIR}/benchmark/gpqa_train.json"
MASK_DIR="${BASE_DIR}/mask/${MODEL}_non_logits"
OUT_DIR="${BASE_DIR}/${MODEL}/gpqa_bet_raw"

MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
BATCH_SIZE=8

LIMIT=0
EXTRA_CONFIGS="--configs ${CONFIGS}"

if [ "$1" == "--pilot" ]; then
    LIMIT=40
    EXTRA_CONFIGS=""
    echo "[PILOT MODE] 40 samples, orig only"
fi

echo "=================================================="
echo "GPQA Betting — RAW DIAGNOSTIC (${MODEL}-${SIZE})"
echo "Cells  : orig + ${CONFIGS}"
echo "Output : ${OUT_DIR}   (isolated; does NOT touch gpqa_bet/)"
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
    --out_prefix     "gpqa_bet_raw" \
    --keep_tasks     "GPQA (gpqa_main)" "GPQA (gpqa_diamond)" \
    --use_chat \
    --save_all_raw \
    ${EXTRA_CONFIGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "[Done] raw diagnostic finished at: $(date)"
    echo "Next: download ${OUT_DIR}/gpqa_bet_raw_7B_per_sample.csv and inspect the"
    echo "      'raw' column for orig vs alpha_6 vs alpha_8."
else
    echo ""
    echo "[Failed] raw diagnostic"
    exit 1
fi
