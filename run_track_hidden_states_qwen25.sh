#!/bin/bash
set -euo pipefail

# ============================================================================
# Qwen2.5-7B-Instruct HDF5 hidden-state backfill for the signal line.
#
# WHY A SEPARATE LAUNCHER: run_track_hidden_states.sh is Llama-frozen — band
# 11-20, llama3 mask, RUN_TAG=phase1b_eot, and a 14-run matrix whose runs 1-7
# are already collected on that machine. Editing it to take Qwen parameters
# would fork the 口径 of every stored Llama H5. Same reason
# run_track_dopamine_signal_qwen25.sh is separate from its Llama counterpart.
#
# WHAT THIS IS: a representative-cell backfill for the lightweight signal batch
# already collected under RUN_TAG=qwen25_signal_v1 (11 No-CoT α + 2 CoT). The
# lightweight path discards hidden states after projecting, so the HS cannot be
# recovered offline — these cells must be re-run.
#
# *** NOT THE SAME TRAJECTORY. *** Same card + same seed usually reproduces
# closely, but bf16 greedy can still diverge at a critical token and one
# divergence changes the whole chain. This is a SAME-PROTOCOL REPRESENTATIVE
# RE-RUN, so an H5 batch carries its OWN signal readouts and must NOT be
# cross-cited per-question with the lightweight batch. Each H5 stores
# question_idx + generated + correct; report the AGREEMENT RATE against the
# lightweight cell rather than assuming identity.
#   -> the agreement check is not yet written; compare question_idx +
#      generated + correct against the matching dopamine_signal_*.json
#      cell and REPORT THE RATE. Do not assume identity.
#
# SEVEN CELLS (was six; +12 added 2026-08-22):
#   No-CoT  α = -8 / 0 / +6 / +8 / +12
#   CoT     α =  0 / +6
# +12 is load-bearing: the headline finding is that entry gain stays linear
# through +12 while the decode slow state stops rising after +8, so the
# plateau region needs its own geometry. With only +8 the manifold work cannot
# observe the plateau at all.
#
# ORDER MATTERS: run this WITHOUT changing code, model, GPU or environment
# since the lightweight curves finished. The agreement rate is only meaningful
# under an unchanged pipeline.
#
# Steps:  CHECK | G1 | G2 | G3  (token-balanced, one card each)
#         NOCOT | COT | ALL  (legacy groupings, kept)
# ============================================================================

MODEL_NAME="qwen2.5"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SIZE="7B"
TASK="gsm8k"
HS_PREFIX="qwen2.5"
TYPE="non"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5
# Qwen band. NOT Llama's 11-20. Exclusive end, so decoder_layer_range(16,22)
# = range(15,21) -> 6 layers. Matches the mask nmd_0.5_16_22_7B.npy and the
# signal / GSM8K / betting / CGT-seq Qwen ports.
LAYER_START=16
LAYER_END=22

EMA_ALPHA=0.95
MAX_NEW_TOKENS=768
N_SAMPLES=300
ROLE="neutral"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
# Same tag as the lightweight batch: these ARE that batch's representative
# cells, and keeping one tag is what makes the pairing discoverable later.
RUN_TAG="${RUN_TAG:-qwen25_signal_v1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
# Passed through to the tracker, which refuses to truncate an existing HDF5
# without it (h5py opens mode="w"). An HS cell is hours of GPU time.
OVERWRITE_ARG=""
[ "${ALLOW_OVERWRITE}" = "1" ] && OVERWRITE_ARG="--allow_overwrite"

H5_DIR="${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}"
MASK_DIR="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits"
NMD_MASK="${MASK_DIR}/${MASK_TYPE}_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# Server convention: the conda env names its interpreter `python`. `python3.10`
# is the LOCAL analysis-box convention and does NOT exist here — it exits 127
# before anything runs, which under nohup looks like a job that silently died.
PY="${PY:-python}"

# --- steps -------------------------------------------------------------------
# NOCOT/COT is the natural 2-way split but is badly unbalanced: 488k vs 208k
# decode tokens (2.35x), so the CoT card would idle for hours.
#
# HS cells are INDEPENDENT, which is what makes a free regrouping legal here:
# each cell's agreement check is against its OWN lightweight cell, and every
# alpha contrast is read from the lightweight batch (already collected on one
# card). No HS cell is ever paired per-question with another HS cell, so the
# one-curve-one-GPU rule does not bind across cells -- only WITHIN a cell,
# which is trivially satisfied since a cell never splits.
#
# G1/G2/G3 are a token-balanced 3-way split (282k/207k/207k, 1.36x):
#   G1: cot+0, nocot+12, nocot+8
#   G2: nocot+0, cot+6
#   G3: nocot-8, nocot+6
STEP="${1:-}"
case "${STEP}" in
  CHECK|NOCOT|COT|ALL|G1|G2|G3) ;;
  *) echo "usage: bash $0 {CHECK|G1|G2|G3|NOCOT|COT|ALL}"; exit 2 ;;
esac

# --- one curve, one card -----------------------------------------------------
# bf16 greedy is not byte-reproducible across GPUs and the H5 meta carries no
# device field, so a split batch mixes the machine difference in irrecoverably.
require_single_card () {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[x] CUDA_VISIBLE_DEVICES is unset. llms.py loads with device_map=auto"
    echo "    and would claim every visible card. Pin ONE card explicitly."
    exit 2
  fi
  if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[x] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards."
    echo "    All cells of one batch must share one card."
    exit 2
  fi
}
[[ "${STEP}" != "CHECK" ]] && require_single_card

cd "${WORK_DIR}"

echo "=================================================="
echo "Qwen2.5 HS backfill | step=${STEP} | RUN_TAG=${RUN_TAG}"
echo "  band ${LAYER_START}-${LAYER_END} (L=6) | ${N_SAMPLES} samples | mnt=${MAX_NEW_TOKENS}"
echo "  card CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  H5_DIR=${H5_DIR}"
echo "=================================================="

# --- interpreter + deps self-check ------------------------------------------
if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "[x] interpreter '${PY}' not found. Override with PY=<interpreter>."
  exit 127
fi
"${PY}" - <<'EOF' || { echo "[x] interpreter cannot import numpy/torch/h5py"; exit 127; }
import numpy, torch, h5py
print(f"[ok] python deps: numpy {numpy.__version__}, torch {torch.__version__}, h5py {h5py.__version__}")
EOF

if [[ ! -f "${NMD_MASK}" ]]; then
  echo "[x] mask not found: ${NMD_MASK}"
  exit 2
fi
echo "[ok] mask: ${NMD_MASK}"

if [[ "${STEP}" == "CHECK" ]]; then
  echo
  # Measured from the lightweight batch's decode-token counts:
  # 34.9 GB decode_hs + ~2 GB prefill_hs uncompressed, x gzip-4 (~0.65-0.8).
  echo "Disk estimate: ~24-30 GB total for 7 cells under ${H5_DIR}"
  df -h "${BASE_DIR}" 2>/dev/null | tail -1
  echo
  echo "[ok] CHECK passed. Launch with NOCOT / COT (pin one card each)."
  exit 0
fi

run_one () {
  local alpha="$1" cot_flag="$2" label="$3"
  echo
  echo "--- cell ${label} (alpha=${alpha}) ---"
  # shellcheck disable=SC2086
  "${PY}" -u track_hidden_states.py \
    --task "${TASK}" \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --mask_type "${MASK_TYPE}" \
    --percentage "${PERCENTAGE}" \
    --layer_start "${LAYER_START}" \
    --layer_end "${LAYER_END}" \
    --ema_alpha "${EMA_ALPHA}" \
    --alpha "${alpha}" \
    --test_file "${GSM8K_FILE}" \
    --n_samples "${N_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --base_dir "${BASE_DIR}" \
    --data "${DATA}" \
    --role "${ROLE}" \
    --save_dir "${H5_DIR}" \
    ${OVERWRITE_ARG} \
    ${cot_flag} \
    || { echo "[x] cell ${label} FAILED. A partially collected batch is not"; \
         echo "    representative data — fix and re-run this cell before continuing."; exit 1; }
  echo "[ok] cell ${label} done"
}

if [[ "${STEP}" == "NOCOT" || "${STEP}" == "ALL" ]]; then
  for a in -8 0 6 8 12; do
    run_one "${a}" "" "nocot_a${a}"
  done
fi

if [[ "${STEP}" == "COT" || "${STEP}" == "ALL" ]]; then
  for a in 0 6; do
    run_one "${a}" "--cot" "cot_a${a}"
  done
fi

# --- token-balanced 3-way split (heaviest cell first in each group, so a
# --- failure surfaces early rather than after two cheap cells) ---------------
if [[ "${STEP}" == "G1" ]]; then
  run_one 0  "--cot" "cot_a0"
  run_one 12 ""      "nocot_a12"
  run_one 8  ""      "nocot_a8"
fi
if [[ "${STEP}" == "G2" ]]; then
  run_one 0  ""      "nocot_a0"
  run_one 6  "--cot" "cot_a6"
fi
if [[ "${STEP}" == "G3" ]]; then
  run_one -8 ""      "nocot_aneg8"
  run_one 6  ""      "nocot_a6"
fi

echo
echo "=================================================="
echo "[ok] step ${STEP} complete."
echo "Next: report the AGREEMENT RATE against the lightweight batch."
echo "This is a same-protocol representative re-run, NOT the same trajectory."
echo "=================================================="
