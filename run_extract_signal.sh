#!/bin/bash
set -euo pipefail

# Phase 1b round 2 — OFFLINE extraction only (Step 2/3).
# All 12 HS-collection runs are already done under phase1b_eot:
#   neutral No-CoT / CoT, expert, non_expert, primary_teacher,
#   neutral ±4 / ±6 / ±8 (prefill-only), CoT −4.
# This script does NOT load the 8B model for generation — it only re-projects
# the stored HDF5 hidden states against masks and computes entropy/confidence.
#
# It is the `START_FROM=13` tail of run_track_hidden_states.sh extracted into a
# standalone script so re-extraction never risks truncating the H5 (no mode="w"
# generation path here). Re-runnable any number of times.
#
# All three extractors are DIRECTORY-level (glob hs_*.h5) → each runs ONCE over
# every HDF5 in H5_DIR, no per-role loop.
#
# Outputs (to be scp'd to local RoleAnswer/ for analyze_multi_metric.py):
#   ${SIG_OUT}/dopamine_signal_*.json   — NMD-mask signal (x_prefill / x_decode / ema)
#   ${RAND_OUT}/random_signal_*.json    — random-mask signal (RSN-specificity, axis C)
#   ${SIG_OUT}/metrics_*.json           — entropy / top1 / margin / info_gain

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

# DEVICE for entropy extraction (lm_head matmul). cpu is fine if the GPU is busy.
DEVICE="${DEVICE:-cuda}"

echo "=================================================="
echo "Phase 1b: OFFLINE signal extraction (Step 2/3, no generation)"
echo "H5 dir   : ${H5_DIR}"
echo "NMD mask : ${NMD_MASK}"
echo "Rand mask: ${RAND_MASK}"
echo "Device   : ${DEVICE} (entropy step only)"
echo "Start    : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

# Pre-flight: H5 dir must exist and contain files.
if [ ! -d "${H5_DIR}" ] || ! compgen -G "${H5_DIR}/hs_*.h5" > /dev/null; then
  echo "[error] No hs_*.h5 found under ${H5_DIR} — nothing to extract."
  exit 1
fi
N_H5=$(compgen -G "${H5_DIR}/hs_*.h5" | wc -l | tr -d ' ')
echo "[info] Found ${N_H5} HDF5 file(s) to process."

# Pre-flight: NMD mask alignment (cheap; reuses the layer-offset sanity check).
echo ""
echo "[sanity] NMD mask indexing check (CLAUDE.md layer-offset warning)"
python sanity_mask_indexing.py \
    --mask_path "${NMD_MASK}" \
    --expect_layer_start ${LAYER_START} \
    --expect_layer_end ${LAYER_END} \
    --skip_model

# ── Step 2a: NMD-mask signal ──────────────────────────────────────────────
echo ""
echo "[Step 2a] extract_signal_json.py  (NMD mask → ${SIG_OUT})"
python extract_signal_json.py \
    --h5_dir      "${H5_DIR}" \
    --mask_path   "${NMD_MASK}" \
    --out_dir     "${SIG_OUT}" \
    --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
echo "[✓] NMD signal"

# ── Step 2b: random-mask signal (RSN-specificity, axis C) ─────────────────
echo ""
echo "[Step 2b] extract_signal_json_remask.py  (random mask → ${RAND_OUT})"
if [ -f "${RAND_MASK}" ]; then
  python extract_signal_json_remask.py \
      --h5_dir      "${H5_DIR}" \
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
    --h5_dir      "${H5_DIR}" \
    --model_dir   "${MODEL_DIR}" \
    --out_dir     "${SIG_OUT}" \
    --device      "${DEVICE}" \
    --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
echo "[✓] entropy/confidence"

echo ""
echo "=================================================="
echo "All extraction done: $(date)"
echo "NMD signal + entropy → ${SIG_OUT}/"
echo "random signal        → ${RAND_OUT}/"
echo "Next: scp the *.json to local RoleAnswer/ , then run analyze_multi_metric.py"
echo "=================================================="
