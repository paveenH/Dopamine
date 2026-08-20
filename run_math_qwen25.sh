#!/bin/bash
# ==================== MATH — Qwen2.5-7B-Instruct cross-model ====================
# Direct replication of the Llama MATH curve, extended to the full nine alphas.
# SEPARATE launcher on purpose: run_math.sh is Llama-only and its non-Cartesian
# role x alpha matrix is frozen -- editing it to add a model risks that line.
#
# Same benchmark file, driver, templates and \boxed{} directive as run_math.sh.
# MATH generation parameters are INHERITED FROM MATH, not from the GSM8K port:
# MAX_NEW_TOKENS=2048 and BATCH_SIZE=8 (GSM8K uses 768/24). MATH items are longer;
# reusing GSM8K's 768 would truncate solutions and manufacture an extraction floor.
#
# Qwen-specific facts (identical to the GSM8K port, none inherited from Llama):
#   * layers 16-21, written as the EXCLUSIVE end "16-22" (L=6). Llama's 11-20
#     (L=9) does NOT transfer.
#   * mask nmd_0.5_16_22_7B.npy under components/mask/qwen2.5_non_logits/
#   * size=7B (28 decoder layers)
#
# *** Same-card rule: bf16 greedy is NOT byte-reproducible across GPUs, and
# cross-alpha contrasts are paired by question index. Every alpha of one curve
# must run on ONE card. Do not split a curve across devices. ***
#
# alpha=0 loads a REAL all-zero matrix (get_answer_regenerate_math.py:129 does
# `np.load(mask_path) * alpha` unconditionally), so a missing mask file blocks
# the baseline cell too -- same as GSM8K, unlike get_action_regenerate_gsm8k.py.
#
# What the alpha=0 probe is for: MATH commits with \boxed{}, whose LaTeX
# convention puts it at the END of a solution. If Qwen's alpha=0 \boxed{} is
# already terminal, there is NO ordering to flip and MATH cannot test the GSM8K
# commitment-ordering mechanism -- it can still test accuracy / stopping /
# perseveration. Read the boxed position BEFORE interpreting the steered cells.
#
# Usage:
#   bash run_math_qwen25.sh --baseline   # alpha=0 only, then read boxed position
#   bash run_math_qwen25.sh --nocot      # the other eight alphas
#   bash run_math_qwen25.sh --full       # all nine, one card, sequential

MODEL_NAME="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
MODEL_SIZE="7B"
TYPE="non"
HS_PREFIX="qwen2.5"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=2048        # MATH value, NOT GSM8K's 768
TEMPERATURE=0.0
BATCH_SIZE=8               # MATH value, NOT GSM8K's 24
N_SAMPLES=300
MATH_FILE="benchmark/math_test_sample.json"

ROLES_NEUTRAL="neutral"    # neutral only: this is a dose curve, not a role study

LS=16
LE=22

ANS_NOCOT="answer_math"

CONFIG_BASELINE="0-${LS}-${LE}"
# --nocot excludes alpha=0 so a completed --baseline is never regenerated.
CONFIGS_NOCOT_REST="neg8-${LS}-${LE} neg6-${LS}-${LE} neg4-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE} 4-${LS}-${LE} 6-${LS}-${LE} 8-${LS}-${LE}"
CONFIGS_FULL="neg8-${LS}-${LE} neg6-${LS}-${LE} neg4-${LS}-${LE} neg2-${LS}-${LE} ${CONFIG_BASELINE} 2-${LS}-${LE} 4-${LS}-${LE} 6-${LS}-${LE} 8-${LS}-${LE}"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}"

case "$1" in
  --baseline) SEL="${CONFIG_BASELINE}";     LBL="--baseline (alpha=0 only)" ;;
  --nocot)    SEL="${CONFIGS_NOCOT_REST}";  LBL="--nocot (the eight non-zero alphas)" ;;
  --full)     SEL="${CONFIGS_FULL}";        LBL="--full (all nine alphas)"
              # --full includes alpha=0. If --baseline already wrote mdf_0, the
              # intended follow-up is --nocot (which excludes it); --full here
              # would regenerate a finished cell for no gain. Refuse rather than
              # silently redo hours of compute.
              if [ -d "${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}/mdf_0" ]; then
                echo "STOP: mdf_0 already exists -- --baseline has run."
                echo "      Use --nocot (the eight non-zero alphas) on the SAME card."
                echo "      Re-running --full would regenerate alpha=0."
                echo "      dir: ${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}/mdf_0"
                exit 2
              fi ;;
  *)
    echo "Usage: bash run_math_qwen25.sh {--baseline|--nocot|--full}"
    echo ""
    echo "  --baseline  alpha=0 -> ${OUT}/mdf_0, then READ the boxed position"
    echo "  --nocot     the other eight alphas (alpha=0 excluded, never re-run)"
    echo "  --full      all nine on one card, sequential"
    echo ""
    echo "Pin a card: CUDA_VISIBLE_DEVICES=<n> bash $0 --full"
    exit 1 ;;
esac

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[warn] CUDA_VISIBLE_DEVICES is unset -- device_map=auto will claim EVERY"
  echo "       visible GPU. Pin one card, or a concurrent job will collide."
fi

echo "=================================================="
echo "MATH Qwen2.5 cross-model | ${LBL}"
echo "Layers  : ${LS}-${LE} (exclusive end, L=6) | mask ${MASK_TYPE}_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
echo "Configs : ${SEL}"
echo "Gen     : max_new_tokens=${MAX_NEW_TOKENS} batch=${BATCH_SIZE} temp=${TEMPERATURE} (MATH values)"
echo "Output  : ${OUT}/mdf_<alpha>"
echo "GPU     : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start   : $(date)"
echo "=================================================="

cd "${WORK_DIR}" || exit 1

python get_answer_regenerate_math.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    ${SEL} \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${MATH_FILE}" \
    --ans_file   "${ANS_NOCOT}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --n_samples      ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}
rc=$?

if [ $rc -eq 0 ]; then
  echo ""
  echo "[Done] MATH ${MODEL_NAME} — $(date)"
  echo "ACC offline (first-#### analogue = last \\boxed{}):"
  echo "  python3.10 analyze_first_last_acc.py   # MATH branch, in RoleAnswer/"
  if [ "$1" == "--baseline" ]; then
    echo ""
    echo "NEXT: read the \\boxed{} POSITION in ${OUT}/mdf_0 before running --nocot."
    echo "  terminal boxed  -> no ordering to flip; MATH tests accuracy/stopping only"
    echo "  early boxed     -> the GSM8K ordering mechanism is testable here"
  fi
else
  echo "[FAILED] MATH ${MODEL_NAME} — $(date)"
  exit 1
fi
