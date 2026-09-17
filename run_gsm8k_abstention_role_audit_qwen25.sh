#!/bin/bash
set -euo pipefail

# ========= GSM8K Abstention-Role NO-STEERING Behavioral Audit: Qwen2.5 =========
# Runs get_answer_gsm8k_abstention_role_audit.py for Qwen2.5-7B-Instruct:
# free-form generation (VicundaModel.generate, NO hooks, NO mask, NO
# steering) on the fixed 300-question GSM8K benchmark, expert vs non_expert
# roles, using the BYTE-IDENTICAL "honest" + abstention-enabled prompt the
# ARRSN HS extraction used.
#
# STANDALONE. Does NOT touch extract_gsm8k_abstention_role_hs.py,
# get_answer_regenerate_gsm8k.py, run_gsm8k_qwen25.sh, or any of their
# output. This is a pure behavioral generation-and-parse step -- no HS, no
# steering, no dose sweep, no MRSN/ARRSN/Chat similarity analysis.
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm8k_abstention_role_audit_qwen25.sh \
#       > logs/gsm8k_abstention_role_audit_qwen25.log 2>&1 &
#
# Verify prompt + pairing digest only (no GPU, no model load):
#   bash run_gsm8k_abstention_role_audit_qwen25.sh --verify_only
#
# Syntax check only: bash -n run_gsm8k_abstention_role_audit_qwen25.sh
#
# Output: components/qwen2.5/answer_gsm8k_abstention_role_audit/
#   expert_7B.json
#   non_expert_7B.json
#   run_meta_7B.json
#
# CRASH-RECOVERY CAVEAT: if this run dies after expert_7B.json is written but
# before non_expert_7B.json/run_meta_7B.json are, the overwrite guard will
# REFUSE the next invocation. Resuming requires a DELIBERATE manual step:
# inspect components/qwen2.5/answer_gsm8k_abstention_role_audit/ and if
# incomplete, delete its contents before re-running.

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
echo "GSM8K Abstention-Role NO-STEERING Behavioral Audit: ${MODEL}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

# ==================== Cheap pre-flight ====================
[ -f "get_answer_gsm8k_abstention_role_audit.py" ] || { echo "[REFUSE] get_answer_gsm8k_abstention_role_audit.py not found in ${WORK_DIR}"; exit 1; }
[ -f "extract_gsm8k_abstention_role_hs.py" ] || { echo "[REFUSE] extract_gsm8k_abstention_role_hs.py not found in ${WORK_DIR} (prompt is imported from it)"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_test_sample.json" ] || { echo "[REFUSE] gsm8k_test_sample.json not found under ${BASE_DIR}/benchmark/"; exit 1; }

if [[ "${MODE}" == "--verify_only" ]]; then
  "${PY}" get_answer_gsm8k_abstention_role_audit.py \
      --model "${MODEL}" \
      --model_dir "${MODEL_DIR}" \
      --base_dir "${BASE_DIR}" \
      --verify_only
  exit $?
fi

"${PY}" -c "import numpy, torch" || { echo "[REFUSE] ${PY} cannot import numpy/torch"; exit 1; }

# ==================== Single invocation: model loads ONCE, both roles ====================
"${PY}" get_answer_gsm8k_abstention_role_audit.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --base_dir "${BASE_DIR}"

echo "=================================================="
echo "[DONE] ${MODEL} GSM8K abstention-role behavioral audit finished."
echo "Finish: $(date)"
echo "=================================================="
