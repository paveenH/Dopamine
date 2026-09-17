#!/bin/bash
set -euo pipefail

# = GSM8K Abstention-Enabled Role HS Extraction (CONSTRUCTION SPLIT): Qwen2.5 =
# Runs extract_gsm8k_abstention_role_hs_construction.py for
# Qwen2.5-7B-Instruct on the LARGER GSM8K construction split (full test set
# minus the existing frozen 300-question benchmark, n~1019 -- exact count
# read from the split's own manifest, never hardcoded), expert + non_expert
# conditions, SAME "honest" + abstention-enabled prompt as the existing
# 300-question pilot extraction.
#
# PREREQUISITE (run once, before this launcher, NOT part of this script):
#   python build_gsm8k_construction_split.py --base_dir "${BASE_DIR}"
# This produces:
#   ${BASE_DIR}/benchmark/gsm8k_construction_split.json
#   ${BASE_DIR}/benchmark/gsm8k_construction_split_manifest.json
# This launcher does NOT build the split itself -- it only checks the two
# files exist before running.
#
# MINIMAL SCOPE: GSM8K only. No MATH, no GSM-Hard, no MMLU-E, no Chat, no
# generation/answer audit, no steering, no layer band, no NMD mask, no
# RRSN/MRSN similarity analysis. STANDALONE -- does not touch
# extract_gsm8k_abstention_role_hs.py, run_gsm8k_abstention_role_qwen25.sh,
# or any of their (300-question) output.
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm8k_abstention_role_construction_qwen25.sh \
#       > logs/gsm8k_abstention_role_construction_qwen25.log 2>&1 &
#
# Verify prompts only (no GPU, no model load):
#   bash run_gsm8k_abstention_role_construction_qwen25.sh --verify_only
#
# Syntax check only: bash -n run_gsm8k_abstention_role_construction_qwen25.sh
#
# Output: components/hidden_states/qwen2.5/gsm8k_abstention_role_construction/
#   expert_7B.h5
#   non_expert_7B.h5
#   manifest.json
#
# CRASH-RECOVERY CAVEAT: same as the 300-question launcher -- if this run
# dies after expert_7B.h5 is written but before non_expert_7B.h5/manifest.json
# are, the overwrite guard will REFUSE the next invocation. Resuming requires
# a DELIBERATE manual step: inspect
#   components/hidden_states/qwen2.5/gsm8k_abstention_role_construction/
# and if incomplete, delete its contents before re-running.

MODE="${1:-}"

MODEL="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

cd "${WORK_DIR}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"

echo "=================================================="
echo "GSM8K Abstention-Enabled Role HS Extraction (CONSTRUCTION SPLIT): ${MODEL}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

# ==================== Cheap pre-flight ====================
[ -f "extract_gsm8k_abstention_role_hs_construction.py" ] || { echo "[REFUSE] extract_gsm8k_abstention_role_hs_construction.py not found in ${WORK_DIR}"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_construction_split.json" ] || { echo "[REFUSE] gsm8k_construction_split.json not found under ${BASE_DIR}/benchmark/ -- run build_gsm8k_construction_split.py first"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_construction_split_manifest.json" ] || { echo "[REFUSE] gsm8k_construction_split_manifest.json not found under ${BASE_DIR}/benchmark/ -- run build_gsm8k_construction_split.py first"; exit 1; }

if [[ "${MODE}" == "--verify_only" ]]; then
  "${PY}" extract_gsm8k_abstention_role_hs_construction.py \
      --model "${MODEL}" \
      --model_dir "${MODEL_DIR}" \
      --base_dir "${BASE_DIR}" \
      --verify_only
  exit $?
fi

"${PY}" -c "import numpy, torch, h5py" || { echo "[REFUSE] ${PY} cannot import numpy/torch/h5py"; exit 1; }

# ==================== Single invocation: model loads ONCE ====================
"${PY}" extract_gsm8k_abstention_role_hs_construction.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --base_dir "${BASE_DIR}"

echo "=================================================="
echo "[DONE] ${MODEL} GSM8K abstention-role (construction split) extraction finished."
echo "Finish: $(date)"
echo "=================================================="
