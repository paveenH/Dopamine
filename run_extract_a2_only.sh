#!/bin/bash
set -euo pipefail

# Phase 1b — OFFLINE extraction for a SUBSET of alpha cells only (default ±2).
#
# run_extract_signal.sh globs the WHOLE phase1b_eot dir (12 h5) and re-extracts
# every cell. When only a couple of new cells were collected (e.g. the ±2
# backfill), that re-runs 10 already-done cells for nothing — and the entropy
# step (lm_head matmul) is slow. This script instead symlinks ONLY the target
# h5 into a scratch dir and points the same three directory-level extractors at
# it, so exactly the requested cells are (re-)extracted. Outputs land beside the
# existing _a4/_a6/... json (same SIG_OUT/RAND_OUT), so nothing else is touched.
#
# Zero code / run_extract_signal.sh changes; idempotent; safe to re-run.
#
# Override which cells via CELLS (space-separated h5 basenames w/o .h5), e.g.
#   CELLS="hs_gsm8k_8B_nocot_a2_L11-20 hs_gsm8k_8B_nocot_aneg2_L11-20" bash run_extract_a2_only.sh
# Skip the random-mask (axis-C) step with SKIP_RANDOM=1.

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TASK="gsm8k"
HS_PREFIX="llama3"
TYPE="non"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5
LAYER_START=11
LAYER_END=20
EMA_ALPHA=0.95

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
RUN_TAG="${RUN_TAG:-phase1b_eot}"

H5_DIR="${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}"
MASK_DIR="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits"
NMD_MASK="${MASK_DIR}/${MASK_TYPE}_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"
RAND_MASK="${MASK_DIR}/diff_random_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"
SIG_OUT="${BASE_DIR}/${MODEL_NAME}/signal/${RUN_TAG}"
RAND_OUT="${BASE_DIR}/${MODEL_NAME}_random/signal/${RUN_TAG}"

DEVICE="${DEVICE:-cuda}"
SKIP_RANDOM="${SKIP_RANDOM:-0}"

# Which cells to extract (h5 basenames without the .h5 suffix). Default = ±2.
CELLS="${CELLS:-hs_gsm8k_8B_nocot_a2_L11-20 hs_gsm8k_8B_nocot_aneg2_L11-20}"

# Scratch dir holding ONLY the target h5 (symlinks — no copy, no disk cost).
SCRATCH="${BASE_DIR}/hidden_states/${TASK}/_subset_extract"

echo "=================================================="
echo "Phase 1b: OFFLINE signal extraction — SUBSET only"
echo "Source   : ${H5_DIR}"
echo "Cells    : ${CELLS}"
echo "NMD mask : ${NMD_MASK}"
echo "SIG_OUT  : ${SIG_OUT}"
echo "Device   : ${DEVICE} (entropy step only)"
echo "Start    : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

# ── Build the scratch dir with only the requested h5 ──────────────────────
rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"
for cell in ${CELLS}; do
  src="${H5_DIR}/${cell}.h5"
  if [ ! -f "${src}" ]; then
    echo "[error] h5 not found: ${src}"
    rm -rf "${SCRATCH}"
    exit 1
  fi
  ln -sf "${src}" "${SCRATCH}/"
done
echo "[info] Linked $(compgen -G "${SCRATCH}/hs_*.h5" | wc -l | tr -d ' ') h5 into scratch:"
ls -l "${SCRATCH}"

# Cleanup scratch on any exit (success or error).
trap 'rm -rf "${SCRATCH}"' EXIT

# ── Pre-flight: NMD mask alignment (CLAUDE.md layer-offset warning) ───────
echo ""
echo "[sanity] NMD mask indexing check"
python sanity_mask_indexing.py \
    --mask_path "${NMD_MASK}" \
    --expect_layer_start ${LAYER_START} \
    --expect_layer_end ${LAYER_END} \
    --skip_model

# ── Step 2a: NMD-mask signal ──────────────────────────────────────────────
echo ""
echo "[Step 2a] extract_signal_json.py  (NMD mask → ${SIG_OUT})"
python extract_signal_json.py \
    --h5_dir      "${SCRATCH}" \
    --mask_path   "${NMD_MASK}" \
    --out_dir     "${SIG_OUT}" \
    --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
echo "[✓] NMD signal"

# ── Step 2b: random-mask signal (RSN-specificity, axis C) ─────────────────
echo ""
if [ "${SKIP_RANDOM}" = "1" ]; then
  echo "[skip] random-mask step disabled (SKIP_RANDOM=1)"
elif [ -f "${RAND_MASK}" ]; then
  echo "[Step 2b] extract_signal_json_remask.py  (random mask → ${RAND_OUT})"
  python extract_signal_json_remask.py \
      --h5_dir      "${SCRATCH}" \
      --mask_path   "${RAND_MASK}" \
      --out_dir     "${RAND_OUT}" \
      --out_prefix  random_signal \
      --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
  echo "[✓] random signal"
else
  echo "[skip] random mask not found at ${RAND_MASK} — generate it first for axis C, then re-run."
fi

# ── Step 3: entropy / confidence (loads norm + lm_head only, not full 8B) ──
echo ""
echo "[Step 3] extract_entropy_confidence.py  (entropy/top1/margin/info_gain → ${SIG_OUT})"
python extract_entropy_confidence.py \
    --h5_dir      "${SCRATCH}" \
    --model_dir   "${MODEL_DIR}" \
    --out_dir     "${SIG_OUT}" \
    --device      "${DEVICE}" \
    --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
echo "[✓] entropy/confidence"

echo ""
echo "=================================================="
echo "Subset extraction done: $(date)"
echo "NMD signal + entropy → ${SIG_OUT}/"
echo "random signal        → ${RAND_OUT}/"
echo "Cells extracted      : ${CELLS}"
echo "Next: scp the new *_a2 / *_aneg2 *.json to local RoleAnswer/"
echo "=================================================="
