#!/bin/bash
set -euo pipefail

# ========= GSM8K Abstention-Enabled Role HS Extraction: Llama3 =========
# Runs extract_gsm8k_abstention_role_hs.py for Llama-3.1-8B-Instruct on the
# fixed 300-question GSM8K benchmark, expert + non_expert conditions, with
# the "honest" + abstention-enabled prompt (see that script's module
# docstring for the exact prompt construction and the two rendered prompts).
#
# MINIMAL SCOPE: GSM8K only. No MATH, no GSM-Hard, no MMLU-E, no Chat, no
# generation/answer audit, no steering, no layer band, no NMD mask, no
# RRSN/MRSN similarity analysis. STANDALONE -- does not touch
# extract_role_hidden_states.py, run_role_hs_llama3.sh, or any of their
# output.
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm8k_abstention_role_llama3.sh \
#       > logs/gsm8k_abstention_role_llama3.log 2>&1 &
#
# Verify prompts only (no GPU, no model load):
#   bash run_gsm8k_abstention_role_llama3.sh --verify_only
#
# Syntax check only: bash -n run_gsm8k_abstention_role_llama3.sh
#
# Output: components/hidden_states/llama3/gsm8k_abstention_role/
#   expert_8B.h5
#   non_expert_8B.h5
#   manifest.json   (ONE experiment-level manifest, not per-role)
#
# CRASH-RECOVERY CAVEAT: if this run dies after expert_8B.h5 is written but
# before non_expert_8B.h5/manifest.json are, the extraction script's
# overwrite guard will REFUSE the next invocation (expert_8B.h5 already
# exists) even though the cell as a whole is incomplete. Resuming requires a
# DELIBERATE manual step: inspect
#   components/hidden_states/llama3/gsm8k_abstention_role/
# and if incomplete, delete its contents before re-running -- do not assume
# a bare re-run will resume cleanly.

MODE="${1:-}"

MODEL="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

cd "${WORK_DIR}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"

echo "=================================================="
echo "GSM8K Abstention-Enabled Role HS Extraction: ${MODEL}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

# ==================== Cheap pre-flight ====================
[ -f "extract_gsm8k_abstention_role_hs.py" ] || { echo "[REFUSE] extract_gsm8k_abstention_role_hs.py not found in ${WORK_DIR}"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_test_sample.json" ] || { echo "[REFUSE] GSM8K sample file not found under ${BASE_DIR}/benchmark/"; exit 1; }

if [[ "${MODE}" == "--verify_only" ]]; then
  "${PY}" extract_gsm8k_abstention_role_hs.py \
      --model "${MODEL}" \
      --model_dir "${MODEL_DIR}" \
      --base_dir "${BASE_DIR}" \
      --verify_only
  exit $?
fi

"${PY}" -c "import numpy, torch, h5py" || { echo "[REFUSE] ${PY} cannot import numpy/torch/h5py"; exit 1; }

# ==================== Single invocation: model loads ONCE ====================
"${PY}" extract_gsm8k_abstention_role_hs.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --base_dir "${BASE_DIR}"

echo "=================================================="
echo "[DONE] ${MODEL} GSM8K abstention-role extraction finished."
echo "Finish: $(date)"
echo "=================================================="
