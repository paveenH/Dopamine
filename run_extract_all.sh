#!/bin/bash
# Parse all HDF5 hidden-state files in gsm8k/ into lightweight JSON, three ways:
#   1) NMD projection      → dopamine_signal_*.json   (extract_signal_json.py)
#   2) Random projection   → random_signal_*.json     (extract_signal_json_remask.py)
#   3) Entropy/confidence  → metrics_*.json           (extract_entropy_confidence.py)
#
# Run on the server after run_track_hidden_states.sh has produced the .h5 files.
# Output JSON are small (few MB) — scp them to local ./signal/ for analysis.

set -e

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

H5_DIR="${BASE_DIR}/hidden_states/gsm8k"
MASK_DIR="${BASE_DIR}/mask/llama3_non_logits"
NMD_MASK="${MASK_DIR}/nmd_0.5_11_20_8B.npy"
RANDOM_MASK="${MASK_DIR}/diff_random_0.5_11_20_8B.npy"

OUT_NMD="${BASE_DIR}/llama3"          # dopamine_signal_* + metrics_*
OUT_RANDOM="${BASE_DIR}/llama3_random" # random_signal_*

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
LAYER_START=11
LAYER_END=20
EMA_ALPHA=0.95

cd ${WORK_DIR}

echo "=================================================="
echo "Extract-all: parsing HDF5 → JSON (3 passes)"
echo "H5 dir   : ${H5_DIR}"
echo "NMD mask : ${NMD_MASK}"
echo "Rnd mask : ${RANDOM_MASK}"
echo "Start: $(date)"
echo "=================================================="

# Fail fast if the saved masks do not match the fixed HF-hidden-state offset.
python sanity_mask_indexing.py \
  --mask_path "${NMD_MASK}" \
  --expect_layer_start ${LAYER_START} \
  --expect_layer_end ${LAYER_END} \
  --skip_model
if [ -f "${RANDOM_MASK}" ]; then
  python sanity_mask_indexing.py \
    --mask_path "${RANDOM_MASK}" \
    --expect_layer_start ${LAYER_START} \
    --expect_layer_end ${LAYER_END} \
    --skip_model
fi

# ── Pass 1: NMD projection ──
echo ""
echo "[1/3] NMD projection → dopamine_signal_*.json"
python extract_signal_json.py \
  --h5_dir ${H5_DIR} \
  --mask_path ${NMD_MASK} \
  --out_dir ${OUT_NMD} \
  --layer_start ${LAYER_START} \
  --layer_end ${LAYER_END} \
  --ema_alpha ${EMA_ALPHA}
echo "[Done] NMD signal"

# ── Pass 2: Random projection (re-projects ALL scalars, not just per-layer) ──
echo ""
echo "[2/3] Random projection → random_signal_*.json"
python extract_signal_json_remask.py \
  --h5_dir ${H5_DIR} \
  --mask_path ${RANDOM_MASK} \
  --out_dir ${OUT_RANDOM} \
  --out_prefix random_signal \
  --layer_start ${LAYER_START} \
  --layer_end ${LAYER_END} \
  --ema_alpha ${EMA_ALPHA}
echo "[Done] Random signal"

# ── Pass 3: Entropy / confidence (norm + lm_head, no full model load) ──
echo ""
echo "[3/3] Entropy/confidence → metrics_*.json"
python extract_entropy_confidence.py \
  --h5_dir ${H5_DIR} \
  --model_dir ${MODEL_DIR} \
  --out_dir ${OUT_NMD} \
  --layer_start ${LAYER_START} \
  --layer_end ${LAYER_END} \
  --ema_alpha ${EMA_ALPHA}
echo "[Done] Entropy/confidence"

echo ""
echo "=================================================="
echo "All done: $(date)"
echo "NMD + metrics JSON → ${OUT_NMD}"
echo "Random JSON        → ${OUT_RANDOM}"
echo ""
echo "Next: scp to local analysis dir, e.g."
echo "  scp '<server>:${OUT_NMD}/dopamine_signal_gsm8k_8B_*.json' \\"
echo "      ~/Downloads/RSNResult/RoleAnswer/llama3/dopamine/signal/"
echo "  scp '<server>:${OUT_NMD}/metrics_gsm8k_8B_*.json' \\"
echo "      ~/Downloads/RSNResult/RoleAnswer/llama3/dopamine/signal/"
echo "  scp '<server>:${OUT_RANDOM}/random_signal_gsm8k_8B_*.json' \\"
echo "      ~/Downloads/RSNResult/RoleAnswer/llama3/dopamine/signal/"
echo "=================================================="
