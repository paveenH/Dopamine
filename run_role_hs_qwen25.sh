#!/bin/bash
set -euo pipefail

# ================= Role Hidden-State Extraction: Qwen2.5 =================
# Runs extract_role_hidden_states.py --run_all for Qwen2.5-7B-Instruct
# across all 4 tasks {gsm8k, math, gsm_hard, mmlue (57 subjects)} x 2 role
# conditions {expert, non_expert}: 3*2=6 reasoning cells + 57*2=114 mmlue
# cells = 120 cells total.
#
# STANDALONE. No steering, no generation -- prefill-only forward passes
# only. Does NOT touch any existing Chat/Bare/Confidence extraction output,
# script, mask, or output tree in this repo.
#
# MODEL LOADED EXACTLY ONCE for this whole run. This launcher invokes
# `python extract_role_hidden_states.py --run_all ...` a single time; the
# script itself validates every one of the 120 cells (samples loaded,
# pairing digests cross-checked, output/meta files confirmed absent) BEFORE
# loading the model, then loads the model ONCE and processes all 120 cells
# sequentially in that one process, reusing the same loaded model object
# for every cell's forward passes. This launcher no longer loops over
# cells in bash or re-invokes python per cell (that was the prior design;
# it reloaded the 8B model up to 120 times per run and has been replaced).
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_role_hs_qwen25.sh > role_hs_qwen25.log 2>&1 &
#
# Syntax check only (no GPU/model needed): bash -n run_role_hs_qwen25.sh

MODEL="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
MMLU_DIR="${MMLU_DIR:-${BASE_DIR}/mmlu}"

cd "${WORK_DIR}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"

echo "=================================================="
echo "Role Hidden-State Extraction: ${MODEL}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

# ==================== Cheap pre-flight (before invoking python) ====================
# These two checks are deliberately kept in bash and run before python
# starts at all, so a missing script/interpreter/dependency/MMLU_DIR fails
# in under a second rather than after python has already begun the (still
# model-free) per-cell validation pass. Every other check -- the one-time
# whole-model all-120-cell overwrite guard, the MMLU-E per-subject file
# presence check, the 14,042-total assertion -- now lives INSIDE
# extract_role_hidden_states.py's --run_all PASS 1 (see that script's
# run_all() docstring/comments), not here, since it needs the same
# task-list/sample-loading logic the script already has and duplicating it
# in bash would risk the two checks drifting apart.

[ -f "extract_role_hidden_states.py" ] || { echo "[REFUSE] extract_role_hidden_states.py not found in ${WORK_DIR}"; exit 1; }
"${PY}" -c "import numpy, torch, h5py" || { echo "[REFUSE] ${PY} cannot import numpy/torch/h5py"; exit 1; }
[ -d "${MMLU_DIR}" ] || { echo "[REFUSE] MMLU_DIR not found: ${MMLU_DIR}"; exit 1; }

# ==================== Single invocation: model loads ONCE ====================
"${PY}" extract_role_hidden_states.py \
    --model "${MODEL}" \
    --run_all \
    --model_dir "${MODEL_DIR}" \
    --base_dir "${BASE_DIR}" \
    --mmlu_dir "${MMLU_DIR}"

echo "=================================================="
echo "[DONE] All reasoning + MMLU-E cells finished for ${MODEL}."
echo "Finish: $(date)"
echo "=================================================="
