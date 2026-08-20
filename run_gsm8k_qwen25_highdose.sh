#!/bin/bash
# ============ GSM8K Qwen2.5 HIGH-DOSE sequence: 0 / +6 / +8 / +10 / +12 ============
# SEPARATE launcher and SEPARATE output dir, deliberately. This is an EXTENSION,
# not part of the frozen nine-alpha main replication -- mixing it into
# answer_mdf_gsm8k would blur the boundary between the frozen curve and an
# exploratory probe, and would silently regenerate finished mdf_0/6/8 cells.
#
# WHAT IT TESTS (one falsifiable prediction, pre-registered):
#   Qwen's accuracy rises monotonically to +8 (60.3 -> 86.0 first_acc) while the
#   generation order flips (answer-first 94.3% -> 4.0%) and contamination rises
#   (17.3% -> 60.3%). If that is the LEFT arm of a Yerkes-Dodson curve, then
#   past some dose accuracy must PEAK AND FALL.
#
#   *** The right arm is established by ACCURACY FALLING, not by contamination
#   rising. +8 already has 60.3% contamination and it is NOT the source of the
#   gain (clean subset at +8 = 83.97% vs alpha=0 clean = 70.40%). Rising
#   contamination alone is NOT evidence of overload. ***
#
# alpha=0 / +6 / +8 are RE-RUN here rather than reused from the frozen dir: this
# sequence must be internally paired on ONE card, and the frozen cells' device
# provenance is not recorded in the artifacts (the summary CSV has no device
# field). Re-running is cheap next to a provenance guess.
#
# Two-digit alpha parses correctly (utils.parse_configs: int(parts[0]) ->
# 10-16-22 gives alpha=10) and the output dirs are mdf_10 / mdf_12, which cannot
# collide with mdf_1 / mdf_2. Verified before writing this file.
#
# Usage:  CUDA_VISIBLE_DEVICES=<n> bash run_gsm8k_qwen25_highdose.sh

MODEL_NAME="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
MODEL_SIZE="7B"
TYPE="non"
HS_PREFIX="qwen2.5"
DATA="data1"

SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"
WORDING="plain"
ROLES_NEUTRAL="neutral"

LS=16
LE=22

# Own dir: never write into the frozen answer_mdf_gsm8k tree.
ANS_HIGH="answer_mdf_gsm8k_highdose"

CONFIGS="0-${LS}-${LE} 6-${LS}-${LE} 8-${LS}-${LE} 10-${LS}-${LE} 12-${LS}-${LE}"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_HIGH}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[warn] CUDA_VISIBLE_DEVICES is unset -- device_map=auto claims EVERY"
  echo "       visible GPU. Pin one card; the whole sequence must share it."
fi

echo "=================================================="
echo "GSM8K Qwen2.5 HIGH-DOSE (extension, NOT the frozen nine)"
echo "Layers : ${LS}-${LE} (exclusive end, L=6)"
echo "Configs: ${CONFIGS}"
echo "Output : ${OUT}/mdf_<alpha>   (separate from answer_mdf_gsm8k)"
echo "GPU    : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start  : $(date)"
echo "=================================================="
echo "Prediction under test: accuracy PEAKS AND FALLS past +8."
echo "Contamination rising is NOT sufficient evidence of a right arm."
echo ""

cd "${WORK_DIR}" || exit 1

python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    ${CONFIGS} \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "${ANS_HIGH}" \
    --suite      "${SUITE}" \
    --fmt_wording "${WORDING}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
rc=$?

if [ $rc -eq 0 ]; then
  echo ""
  echo "[Done] high-dose sequence — $(date)"
  echo "Read it with the Qwen analyzer pointed at this dir, e.g."
  echo "  python3.10 analyze_first_last_acc_qwen.py --nocot_dir ${ANS_HIGH}"
  echo "(its DOSE list will need the mdf_10 / mdf_12 cells added)"
else
  echo "[FAILED] high-dose sequence — $(date)"
  exit 1
fi
