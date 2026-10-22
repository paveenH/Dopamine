#!/bin/bash
# ==================== GSM8K Role Prompt - RSN behavioural correspondence (Qwen2.5) ====
# Protocol gsm8k-role-correspondence-v1. ONE model load, three roles in sequence:
#   "an expert" / "a non expert" -> templates["neg"] ("Now you are {role}."), neutral -> templates["neutral"]
# alpha=0 ONLY, Bare, zero-shot, No-CoT, suite=default, wording=plain, greedy, 768 tokens, bs=24.
# 300 fixed questions x 3 roles = 900 replies. No honest / chat / CoT / demo / abstention option.
# Reuses get_answer_regenerate_gsm8k.py unchanged except the opt-in --save_prompt_audit.
#
# Output (isolated; never touches answer_mdf_gsm8k):
#   components/qwen2.5/answer_gsm8k_role_correspondence_v1/mdf_0/
#
# Steps:
#   bash run_gsm8k_role_correspondence_qwen25.sh --run     generate (skips only if an existing cell VERIFIES)
#   bash run_gsm8k_role_correspondence_qwen25.sh --verify  integrity check of the finished cell
# GPU placement is not constrained; CUDA_VISIBLE_DEVICES / host / sharding are recorded in run_meta.
# Accuracy and early-candidate are computed OFFLINE in RoleAnswer/, never from inline correct_*.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="qwen2.5"; MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"; MODEL_SIZE="7B"; HS_PREFIX="qwen2.5"; TYPE="non"
MASK_TYPE="nmd"; PERCENTAGE=0.5; LS=16; LE=22     # mask is only needed for the run_meta fail-closed check; alpha=0
GSM8K_FILE="benchmark/gsm8k_test_sample.json"
ANS_FILE="answer_gsm8k_role_correspondence_v1"
ROLES="an expert,a non expert,neutral"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/mdf_0"

MODE="${1:-}"
case "${MODE}" in
  --verify)
    ${PY} verify_role_correspondence_cell.py --cell_dir "${OUT}" --test_file "${BASE_DIR}/${GSM8K_FILE}"
    exit $? ;;
  --run)
    if [ -e "${OUT}" ]; then
      echo "[info] ${OUT} exists -> verifying provenance/integrity before skipping"
      if ${PY} verify_role_correspondence_cell.py --cell_dir "${OUT}" --test_file "${BASE_DIR}/${GSM8K_FILE}"; then
        echo "[skip] existing cell verified; nothing to do"; exit 0
      fi
      echo "[x] existing cell FAILED verification; refusing to overwrite or skip. Inspect/move it by hand."; exit 1
    fi
    echo "role-correspondence-v1 | ${MODEL_DIR} | roles: ${ROLES} | out: ${OUT}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} host=$(hostname) $(date)"
    ${PY} get_answer_regenerate_gsm8k.py \
        --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" --hs "${HS_PREFIX}" --size "${MODEL_SIZE}" \
        --type "${TYPE}" --percentage "${PERCENTAGE}" --configs "0-${LS}-${LE}" --mask_type "${MASK_TYPE}" \
        --test_file "${GSM8K_FILE}" --ans_file "${ANS_FILE}" --suite default --fmt_wording plain \
        --base_dir "${BASE_DIR}" --roles "${ROLES}" \
        --max_new_tokens 768 --temperature 0.0 --batch_size 24 \
        --save_gen_meta --expect_n 300 --save_prompt_audit
    rc=$?
    if [ $rc -eq 0 ]; then
      ${PY} verify_role_correspondence_cell.py --cell_dir "${OUT}" --test_file "${BASE_DIR}/${GSM8K_FILE}"; rc=$?
    fi
    [ $rc -eq 0 ] && echo "[ok] finished + verified: ${OUT}" || echo "[x] FAILED (rc=$rc)"
    exit $rc ;;
  *)
    echo "Usage: bash run_gsm8k_role_correspondence_qwen25.sh [--run|--verify]"; exit 1 ;;
esac
