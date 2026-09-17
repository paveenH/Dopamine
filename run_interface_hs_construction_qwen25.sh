#!/bin/bash
set -euo pipefail

# ======= Interface Hidden-State Extraction (CONSTRUCTION SPLIT): Qwen2.5 =======
# Runs extract_interface_hidden_states_construction.py for
# Qwen2.5-7B-Instruct on the LARGER GSM8K construction split
# (components/benchmark/gsm8k_construction_split.json, n~1019), GSM8K ONLY,
# {chat, bare} conditions (2 cells).
#
# WHY: to build a Chat direction (Chat - Bare) that is sample-composition-
# matched to the formal construction-split ARRSN (same 1019 questions, same
# order) -- comparing against the frozen 300-question chat/bare files would
# confound the comparison with sample-composition differences (the 300 are
# the RESERVED questions, deliberately excluded from the construction split).
#
# PREREQUISITE (run once, before this launcher, NOT part of this script):
#   python build_gsm8k_construction_split.py --base_dir "${BASE_DIR}"
# This produces:
#   ${BASE_DIR}/benchmark/gsm8k_construction_split.json
#   ${BASE_DIR}/benchmark/gsm8k_construction_split_manifest.json
# This launcher does NOT build the split itself -- it only checks the two
# files exist before running.
#
# STANDALONE. No steering, no generation -- prefill-only forward passes only.
# Does NOT touch extract_interface_hidden_states.py, run_interface_hs_qwen25.sh,
# or any of their (300-question, 3-task) output.
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_interface_hs_construction_qwen25.sh \
#       > logs/interface_hs_construction_qwen25.log 2>&1 &
#
# Syntax check only (no GPU/model needed):
#   bash -n run_interface_hs_construction_qwen25.sh
#
# Output: components/hidden_states/qwen2.5/gsm8k_construction/
#   chat_7B.h5, chat_7B.manifest.json
#   bare_7B.h5, bare_7B.manifest.json
#
# CRASH-RECOVERY CAVEAT: if this run dies after chat_7B.h5 is written but
# before bare_7B.h5 is, the overwrite guard will REFUSE the next invocation.
# Resuming requires a DELIBERATE manual step: inspect
#   components/hidden_states/qwen2.5/gsm8k_construction/
# and if incomplete, delete its contents before re-running.

MODEL="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

CONDITIONS=(chat bare)

cd "${WORK_DIR}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"

echo "=================================================="
echo "Interface Hidden-State Extraction (CONSTRUCTION SPLIT): ${MODEL}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

# ==================== Cheap pre-flight ====================
[ -f "extract_interface_hidden_states_construction.py" ] || { echo "[REFUSE] extract_interface_hidden_states_construction.py not found in ${WORK_DIR}"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_construction_split.json" ] || { echo "[REFUSE] gsm8k_construction_split.json not found under ${BASE_DIR}/benchmark/ -- run build_gsm8k_construction_split.py first"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_construction_split_manifest.json" ] || { echo "[REFUSE] gsm8k_construction_split_manifest.json not found under ${BASE_DIR}/benchmark/ -- run build_gsm8k_construction_split.py first"; exit 1; }

"${PY}" -c "import numpy, torch, h5py" || { echo "[REFUSE] ${PY} cannot import numpy/torch/h5py"; exit 1; }

EXISTING=0
SUFFIX="7B"
for c in "${CONDITIONS[@]}"; do
    for f in \
        "${BASE_DIR}/hidden_states/${MODEL}/gsm8k_construction/${c}_${SUFFIX}.h5" \
        "${BASE_DIR}/hidden_states/${MODEL}/gsm8k_construction/${c}_${SUFFIX}.manifest.json"; do
        if [ -f "${f}" ]; then
            echo "[REFUSE] already exists: ${f}"
            EXISTING=1
        fi
    done
done
if [ "${EXISTING}" -ne 0 ]; then
    echo "Refusing to run any cell for ${MODEL} -- delete the existing file(s)"
    echo "deliberately first if a re-run is truly intended."
    exit 1
fi

echo "--- pairing digest check ---"
DIGEST=$("${PY}" extract_interface_hidden_states_construction.py \
    --model "${MODEL}" --condition chat \
    --base_dir "${BASE_DIR}" --print_pairing_digest_only | tail -1)
echo "  gsm8k_construction ordered_sample_identity_sha256=${DIGEST}"

for c in "${CONDITIONS[@]}"; do
    echo "=================================================="
    echo "Running: model=${MODEL} task=gsm8k_construction condition=${c}"
    echo "Time: $(date)"
    echo "=================================================="
    "${PY}" extract_interface_hidden_states_construction.py \
        --model "${MODEL}" \
        --condition "${c}" \
        --model_dir "${MODEL_DIR}" \
        --base_dir "${BASE_DIR}" \
        --expect_pairing_digest "${DIGEST}"
done

echo "=================================================="
echo "[DONE] Both cells finished for ${MODEL} (gsm8k_construction, chat+bare)."
echo "Finish: $(date)"
echo "=================================================="
