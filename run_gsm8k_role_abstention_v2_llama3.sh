#!/bin/bash
# ==================== GSM8K role-abstention prompt v2 -- Llama3.1-8B pilot ====
# Standalone, NO STEERING. Runs get_answer_gsm8k_role_abstention_v2.py, which
# calls VicundaModel.generate() only (no diff_matrices). Two roles: expert /
# non_expert, identical prompt except the role phrase. Fixed 300-question
# GSM8K benchmark, same order as every other GSM8K experiment in this repo.
#
# Steps:
#   bash run_gsm8k_role_abstention_v2_llama3.sh --check   local-only pairing
#                                                          check (question
#                                                          digest, prompt
#                                                          diff). No model
#                                                          load, no GPU.
#   bash run_gsm8k_role_abstention_v2_llama3.sh --run     full generation,
#                                                          both roles, one
#                                                          model load.
#
# Output: components/llama3/answer_gsm8k_role_abstention_v2/ (NEW dir, does
# not touch any existing answer_* tree).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"

GSM8K_FILE="benchmark/gsm8k_test_sample.json"
MAX_NEW_TOKENS=768
BATCH_SIZE=24
ANS_FILE="answer_gsm8k_role_abstention_v2"

DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

MODE="${1:-}"

banner () {
  echo "=================================================="
  echo "GSM8K role-abstention v2 | ${MODEL_NAME} (${MODEL_SIZE})"
  echo "step        : $1"
  echo "benchmark   : ${BASE_DIR}/${GSM8K_FILE}"
  echo "output      : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/"
  echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"
  echo "Start: $(date)"
  echo "=================================================="
}

case "${MODE}" in
  --check)
    banner "check"
    ${PY} get_answer_gsm8k_role_abstention_v2.py \
        --model      "${MODEL_NAME}" \
        --model_dir  "${MODEL_DIR}" \
        --size       "${MODEL_SIZE}" \
        --test_file  "${GSM8K_FILE}" \
        --base_dir   "${BASE_DIR}" \
        --ans_file   "${ANS_FILE}" \
        --check
    ;;
  --run)
    banner "run"
    ${PY} get_answer_gsm8k_role_abstention_v2.py \
        --model      "${MODEL_NAME}" \
        --model_dir  "${MODEL_DIR}" \
        --size       "${MODEL_SIZE}" \
        --test_file  "${GSM8K_FILE}" \
        --base_dir   "${BASE_DIR}" \
        --ans_file   "${ANS_FILE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --batch_size     "${BATCH_SIZE}"
    ;;
  *)
    echo "usage: bash run_gsm8k_role_abstention_v2_llama3.sh check-or-run"
    exit 1
    ;;
esac
