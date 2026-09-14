#!/bin/bash
set -euo pipefail

# ================= Interface Hidden-State Extraction: Llama3 =================
# Runs extract_interface_hidden_states.py for Llama-3.1-8B-Instruct across all
# 3 tasks x 2 conditions (6 cells): {gsm8k, gsm_hard, math} x {chat, bare}.
#
# STANDALONE. No steering, no generation -- prefill-only forward passes only.
# Does NOT touch any frozen generation/steering pipeline, mask, or output tree
# in this repo.
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_interface_hs_llama3.sh > interface_hs_llama3.log 2>&1 &
#
# Syntax check only (no GPU/model needed): bash -n run_interface_hs_llama3.sh

MODEL="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

TASKS=(gsm8k gsm_hard math)
CONDITIONS=(chat bare)

cd "${WORK_DIR}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"

# ==================== Pre-flight (before model load) ====================
# All 6 target files (.h5 AND .manifest.json) are checked up front so a
# fresh model run refuses before any GPU time is spent if a stray file is
# already there. extract_interface_hidden_states.py itself only checks the
# ONE cell it is about to write (see that script's own preflight) -- it must
# NOT re-check all 6, since cells 2..6 legitimately run after cell 1's files
# already exist. This launcher-level check is therefore the only "no cell
# has ever run for this model" gate; it runs once, before task 1's
# pairing-digest step.
echo "=================================================="
echo "Interface Hidden-State Extraction: ${MODEL}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

MISSING_EXTRACTOR=0
[ -f "extract_interface_hidden_states.py" ] || { echo "[REFUSE] extract_interface_hidden_states.py not found in ${WORK_DIR}"; MISSING_EXTRACTOR=1; }
[ "${MISSING_EXTRACTOR}" -eq 0 ] || exit 1

"${PY}" -c "import numpy, torch, h5py" || { echo "[REFUSE] ${PY} cannot import numpy/torch/h5py"; exit 1; }

EXISTING=0
for t in "${TASKS[@]}"; do
    for c in "${CONDITIONS[@]}"; do
        SUFFIX="8B"
        for f in \
            "${BASE_DIR}/hidden_states/${MODEL}/${t}/${c}_${SUFFIX}.h5" \
            "${BASE_DIR}/hidden_states/${MODEL}/${t}/${c}_${SUFFIX}.manifest.json"; do
            if [ -f "${f}" ]; then
                echo "[REFUSE] already exists: ${f}"
                EXISTING=1
            fi
        done
    done
done
if [ "${EXISTING}" -ne 0 ]; then
    echo "Refusing to run any cell for ${MODEL} -- delete the existing file(s)"
    echo "deliberately first if a re-run is truly intended."
    exit 1
fi

for t in "${TASKS[@]}"; do
    echo "--- pairing digest check: task=${t} ---"
    DIGEST=$("${PY}" extract_interface_hidden_states.py \
        --model "${MODEL}" --task "${t}" --condition chat \
        --base_dir "${BASE_DIR}" --print_pairing_digest_only | tail -1)
    echo "  ${t} ordered_sample_identity_sha256=${DIGEST}"

    for c in "${CONDITIONS[@]}"; do
        echo "=================================================="
        echo "Running: model=${MODEL} task=${t} condition=${c}"
        echo "Time: $(date)"
        echo "=================================================="
        "${PY}" extract_interface_hidden_states.py \
            --model "${MODEL}" \
            --task "${t}" \
            --condition "${c}" \
            --model_dir "${MODEL_DIR}" \
            --base_dir "${BASE_DIR}" \
            --expect_pairing_digest "${DIGEST}"
    done
done

echo "=================================================="
echo "[DONE] All 6 cells finished for ${MODEL}."
echo "Finish: $(date)"
echo "=================================================="
