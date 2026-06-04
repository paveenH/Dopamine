#!/bin/bash
set -euo pipefail

# Phase 1b round 2: Hidden-state recording with paper-aligned roles + neutral
# baselines. HS storage is now selective (middle 9 layers + final layer = 10
# layers total) — see track_hidden_states.py HiddenStateRecorder.
#
# 12 runs total:
#   1) GSM8K No-CoT  role=expert            "an expert"
#   2) GSM8K No-CoT  role=non_expert        "a non expert"
#   3) GSM8K No-CoT  role=neutral           (no character)
#   4) GSM8K CoT     role=neutral           (no character)
#   5) GSM8K No-CoT  role=primary_teacher   "a primary school teacher"
#   6) GSM8K No-CoT  role=neutral  α=+4     (prefill-only static steering)
#   7) GSM8K No-CoT  role=neutral  α=-4     (prefill-only static steering)
#   8) GSM8K No-CoT  role=neutral  α=+6     (extends the dose-response trajectory)
#   9) GSM8K No-CoT  role=neutral  α=-6     (α=-6 = acc peak / oscillation trough)
#  10) GSM8K No-CoT  role=neutral  α=+8
#  11) GSM8K No-CoT  role=neutral  α=-8     (α=-8 = oscillation collapse; §2.3)
#  12) GSM8K CoT     role=neutral  α=-4     (CoT steering trajectory)
#
# Runs 1–7 are ALREADY COLLECTED on this machine (phase1b_eot). The new cells
# 8–12 extend the ±α dose-response axis (±6/±8) and add a CoT −4 trajectory so
# the §2.3 oscillation collapse (α=-8) and §1.1 acc peak (α=-6) have HS to back
# the text-only findings. Use START_FROM=8 to collect only the new runs.
#
# Output: HDF5 files under ${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}/
# (α=0 runs untagged; steered runs carry _a<k> / _aneg<k>; CoT carries _cot.)

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TASK="gsm8k"
HS_PREFIX="llama3"
TYPE="non"
DATA="data1"

MASK_TYPE="nmd"           # only used as the sanity-projection mask in HDF5
PERCENTAGE=0.5
LAYER_START=11
LAYER_END=20
EMA_ALPHA=0.95

MAX_NEW_TOKENS_GSM=768
N_SAMPLES=300

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
# Keep each expensive HS collection isolated. Override RUN_TAG explicitly for
# a new replicate; set ALLOW_OVERWRITE=1 only when replacing that exact run.
#
# RUN_TAG = phase1b_eot: FULL re-collection under the <|eot_id|> terminator fix
# (llms._build_terminators). The earlier phase1b_plain HDF5 were generated with
# the buggy single-eos config and are NOT comparable — kept only as reference,
# not overwritten (new tag = new dir).
RUN_TAG="${RUN_TAG:-phase1b_eot}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
# Default 1 = re-collect ALL 12 runs on this machine with the terminator fix.
# The generation path (vc.generate_one) inherits the fix via self.terminators
# automatically — no code change to track_hidden_states.py.
#   START_FROM=8  → skip the done runs 1–7, collect ONLY the new ±6/±8/CoT-4 runs.
#   START_FROM=13 → SKIP all HS collection, run only offline Step 2/3 extraction
#                   over HDF5 already in H5_DIR.
START_FROM="${START_FROM:-1}"
# SKIP_EXTRACT=1 → run ONLY the HS-collection runs and STOP before Step 2/3
# offline extraction (so you can re-collect a single run, e.g. START_FROM=11,
# without re-extracting JSON for all H5). Extract later with START_FROM=13.
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
H5_DIR="${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}"
MASK_DIR="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits"
NMD_MASK="${MASK_DIR}/${MASK_TYPE}_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"
RAND_MASK="${MASK_DIR}/diff_random_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"
SIG_OUT="${BASE_DIR}/${MODEL_NAME}/signal/${RUN_TAG}"            # NMD signal + entropy staging area
RAND_OUT="${BASE_DIR}/${MODEL_NAME}_random/signal/${RUN_TAG}"    # random signal staging area

echo "=================================================="
echo "Phase 1b round 2: HS recording (12 runs, GSM8K, <|eot_id|> terminator fix)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE}) | Middle layers: ${LAYER_START}-${LAYER_END}"
echo "HS storage: middle [${LAYER_START},${LAYER_END}) + final layer = 10 stored layers"
echo "Sanity mask: ${MASK_TYPE} (offline reanalysis can use any mask)"
echo "Run tag: ${RUN_TAG}"
echo "Start from run: ${START_FROM}"
echo "Start: $(date)"
echo "=================================================="

cd ${WORK_DIR}

# HDF5 files are opened with mode="w", so an accidental rerun would truncate
# completed or partially collected files. Require an explicit override.
if [ "${START_FROM}" = "1" ] && [ "${ALLOW_OVERWRITE}" != "1" ] && [ -d "${H5_DIR}" ] && compgen -G "${H5_DIR}/hs_*.h5" > /dev/null; then
  echo "[error] Existing HDF5 files found under ${H5_DIR}"
  echo "        Use a new RUN_TAG, or set ALLOW_OVERWRITE=1 to replace this exact run."
  exit 1
fi

# Pre-flight: confirm mask rows match the fixed HF-hidden-state offset before
# spending time on the five slow HDF5 recording runs.
echo ""
echo "[sanity] NMD mask indexing check (see CLAUDE.md layer-offset warning)"
python sanity_mask_indexing.py \
    --mask_path "${NMD_MASK}" \
    --expect_layer_start ${LAYER_START} \
    --expect_layer_end ${LAYER_END} \
    --skip_model

if [ -f "${RAND_MASK}" ]; then
  echo ""
  echo "[sanity] random mask indexing check"
  python sanity_mask_indexing.py \
      --mask_path "${RAND_MASK}" \
      --expect_layer_start ${LAYER_START} \
      --expect_layer_end ${LAYER_END} \
      --skip_model
else
  echo ""
  echo "[warn] random mask not found at ${RAND_MASK} — Axis C will be skipped"
fi

BASE_ARGS="
  --task ${TASK}
  --model ${MODEL_NAME}
  --model_dir ${MODEL_DIR}
  --hs ${HS_PREFIX}
  --size ${MODEL_SIZE}
  --type ${TYPE}
  --mask_type ${MASK_TYPE}
  --percentage ${PERCENTAGE}
  --layer_start ${LAYER_START}
  --layer_end ${LAYER_END}
  --ema_alpha ${EMA_ALPHA}
  --test_file benchmark/gsm8k_test_sample.json
  --n_samples ${N_SAMPLES}
  --max_new_tokens ${MAX_NEW_TOKENS_GSM}
  --temperature 0.0
  --base_dir ${BASE_DIR}
  --save_dir ${H5_DIR}
"

# ── Run 1: expert ──
if [ "${START_FROM}" -le 1 ]; then
  echo ""
  echo "[1/12] GSM8K No-CoT | role=expert ('an expert')"
  python track_hidden_states.py ${BASE_ARGS} --role expert
  echo "[Done] expert"
fi

# ── Run 2: non_expert ──
if [ "${START_FROM}" -le 2 ]; then
  echo ""
  echo "[2/12] GSM8K No-CoT | role=non_expert ('a non expert')"
  python track_hidden_states.py ${BASE_ARGS} --role non_expert
  echo "[Done] non_expert"
fi

# ── Run 3: neutral (No-CoT) ──
if [ "${START_FROM}" -le 3 ]; then
  echo ""
  echo "[3/12] GSM8K No-CoT | role=neutral"
  python track_hidden_states.py ${BASE_ARGS} --role neutral
  echo "[Done] neutral No-CoT"
fi

# ── Run 4: neutral (CoT) ──
if [ "${START_FROM}" -le 4 ]; then
  echo ""
  echo "[4/12] GSM8K CoT    | role=neutral"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --cot
  echo "[Done] neutral CoT"
fi

# ── Run 5: primary_teacher (task-matched extra role) ──
if [ "${START_FROM}" -le 5 ]; then
  echo ""
  echo "[5/12] GSM8K No-CoT | role=primary_teacher ('a primary school teacher')"
  python track_hidden_states.py ${BASE_ARGS} --role primary_teacher
  echo "[Done] primary_teacher"
fi

# ── Run 6: neutral + α=+4 (prefill-only static-steering trajectory) ──
# Same neutral prompt as Run 3, but inject +4 × NMD mask into the last prompt
# token (prefill-only, matches get_answer_regenerate_gsm8k.py). Output file
# carries the _a4 tag so it sits beside the α=0 neutral baseline. Forms the
# −4 / 0 / +4 steering axis for offline signal comparison.
if [ "${START_FROM}" -le 6 ]; then
  echo ""
  echo "[6/12] GSM8K No-CoT | role=neutral | α=+4 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --alpha 4
  echo "[Done] neutral α=+4"
fi

# ── Run 7: neutral + α=-4 ──
if [ "${START_FROM}" -le 7 ]; then
  echo ""
  echo "[7/12] GSM8K No-CoT | role=neutral | α=-4 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --alpha -4
  echo "[Done] neutral α=-4"
fi

# ── Runs 8–11: extend the ±α dose-response trajectory (±6, ±8) ──
# Same neutral prompt as Run 3, prefill-only steering at larger magnitudes.
# α=-6 is the acc peak / oscillation trough (§1.1, §2.3); α=-8 is the
# oscillation collapse we now want HS trajectory for. Files tag _a6/_aneg6/
# _a8/_aneg8, sitting beside the existing ±4 runs.
if [ "${START_FROM}" -le 8 ]; then
  echo ""
  echo "[8/12] GSM8K No-CoT | role=neutral | α=+6 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --alpha 6
  echo "[Done] neutral α=+6"
fi

if [ "${START_FROM}" -le 9 ]; then
  echo ""
  echo "[9/12] GSM8K No-CoT | role=neutral | α=-6 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --alpha -6
  echo "[Done] neutral α=-6"
fi

if [ "${START_FROM}" -le 10 ]; then
  echo ""
  echo "[10/12] GSM8K No-CoT | role=neutral | α=+8 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --alpha 8
  echo "[Done] neutral α=+8"
fi

if [ "${START_FROM}" -le 11 ]; then
  echo ""
  echo "[11/12] GSM8K No-CoT | role=neutral | α=-8 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --alpha -8
  echo "[Done] neutral α=-8"
fi

# ── Run 12: neutral + α=-4 under CoT (CoT steering trajectory) ──
# CoT counterpart to Run 7. File tags _cot_aneg4. Lets the CoT steering axis
# (CoT α=-4) be compared against No-CoT α=-4 on the HS level.
if [ "${START_FROM}" -le 12 ]; then
  echo ""
  echo "[12/12] GSM8K CoT    | role=neutral | α=-4 (prefill-only)"
  python track_hidden_states.py ${BASE_ARGS} --role neutral --cot --alpha -4
  echo "[Done] neutral CoT α=-4"
fi

# ==================================================================
# Step 2/3: offline re-projection (SOP §3.1b). All three extractors are
# DIRECTORY-level (glob hs_*.h5), so each runs ONCE over ALL HDF5 files
# (now 12: neutral/role α=0 + neutral ±4/±6/±8 + CoT −4) — no per-role loop.
# ==================================================================

if [ "${SKIP_EXTRACT}" = "1" ]; then
  echo ""
  echo "[skip] SKIP_EXTRACT=1 — HS collection done, stopping before Step 2/3."
  echo "       Run extraction later with: START_FROM=13 bash run_track_hidden_states.sh"
  echo "Done (HS only): $(date)"
  exit 0
fi

echo ""
echo "[Step 2a] extract_signal_json.py  (NMD mask → ${SIG_OUT})"
python extract_signal_json.py \
    --h5_dir     "${H5_DIR}" \
    --mask_path  "${NMD_MASK}" \
    --out_dir    "${SIG_OUT}" \
    --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
echo "[✓] NMD signal"

echo ""
echo "[Step 2b] extract_signal_json_remask.py  (random mask → ${RAND_OUT}, RSN-specificity axis C)"
if [ -f "${RAND_MASK}" ]; then
  python extract_signal_json_remask.py \
      --h5_dir     "${H5_DIR}" \
      --mask_path  "${RAND_MASK}" \
      --out_dir    "${RAND_OUT}" \
      --out_prefix random_signal \
      --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
  echo "[✓] random signal"
else
  echo "[skip] random mask not found at ${RAND_MASK} — generate it first, then re-run this step for axis C"
fi

echo ""
echo "[Step 3] extract_entropy_confidence.py  (entropy/top1/margin/info_gain → ${SIG_OUT})"
python extract_entropy_confidence.py \
    --h5_dir     "${H5_DIR}" \
    --model_dir  "${MODEL_DIR}" \
    --out_dir    "${SIG_OUT}" \
    --layer_start ${LAYER_START} --layer_end ${LAYER_END} --ema_alpha ${EMA_ALPHA}
echo "[✓] entropy/confidence"

echo ""
echo "=================================================="
echo "All done: $(date)"
echo "HS     → ${H5_DIR}/"
echo "NMD signal + entropy → ${SIG_OUT}/"
echo "random signal        → ${RAND_OUT}/"
echo "Next: scp the *.json to local ./signal/, then run analyze_multi_metric.py"
echo "=================================================="
