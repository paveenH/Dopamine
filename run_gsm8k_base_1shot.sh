#!/bin/bash
# ==================== GSM8K Llama-3.1-8B BASE, 1-shot with a worked solution ====
# Protocol gsm8k-base-1shot-v1. EXPLORATORY follow-up to the zero-shot Base check
# (run_gsm8k_base.sh, left untouched): same 300 questions, same IT mask, same band
# [11,20), prefill-only tail=1, greedy, 768 tokens, bs=24 -- plus ONE fixed GSM8K
# train demonstration (question + full official answer, final '#### N' kept).
# Not identical to the zero-shot No-CoT condition: the demo also shows the
# compute-then-submit order. alpha=0 is RE-RUN under this prompt (no old baseline).
#
# Steps (single positional arg):
#   prep   download + freeze the demo (refuses to overwrite; use prep-check to re-derive)
#   check  technical pre-flight on the REAL 1-shot prompts (loads the model)
#   run    ONE process, ONE model load, alpha -6, 0, +6 sequentially
#
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base_1shot.sh prep
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base_1shot.sh check
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base_1shot.sh run
#
# Output (refuses if ANY of the three cells exists):
#   components/llama3_base/answer_mdf_gsm8k_1shot_v1/mdf_{-6,0,6}/
# ACC is computed offline (RoleAnswer/analyze_gsm8k_base_1shot.py), never from inline fields.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEP="${1:-}"
MODEL_NAME="llama3_base"; MODEL_DIR="meta-llama/Llama-3.1-8B"; HS_PREFIX="llama3"
MODEL_SIZE="8B"; LS=11; LE=20; CHECK_ALPHA=-6
CONFIGS="neg6-11-20 0-11-20 6-11-20"; ALPHA_DIRS="mdf_-6 mdf_0 mdf_6"
TYPE="non"; SUITE="default"; MASK_TYPE="nmd"; PERCENTAGE=0.5
MAX_NEW_TOKENS=768; TEMPERATURE=0.0; BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"; WORDING="plain"; ROLES="neutral"
DEMO_FILE="benchmark/gsm8k_1shot_demo_v1.json"
ANS_FILE="answer_mdf_gsm8k_1shot_v1"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

MASK="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/${MASK_TYPE}_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
DATA="${BASE_DIR}/${GSM8K_FILE}"
DEMO="${BASE_DIR}/${DEMO_FILE}"
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"

case "${STEP}" in
  prep)       ${PY} prepare_gsm8k_1shot_demo.py --base_dir "${BASE_DIR}"; exit $? ;;
  prep-check) ${PY} prepare_gsm8k_1shot_demo.py --base_dir "${BASE_DIR}" --check; exit $? ;;
  check)
    [ -f "${DEMO}" ] || { echo "[x] demo missing -- run: bash run_gsm8k_base_1shot.sh prep"; exit 1; }
    ${PY} check_gsm8k_qwen.py \
        --base_dir "${BASE_DIR}" --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" --type "${TYPE}" --size "${MODEL_SIZE}" \
        --mask_type "${MASK_TYPE}" --percentage "${PERCENTAGE}" \
        --layer_start ${LS} --layer_end ${LE} --test_file "${GSM8K_FILE}" \
        --ans_file "${ANS_FILE}" --fmt_wording "${WORDING}" --alpha ${CHECK_ALPHA} \
        --fewshot_file "${DEMO_FILE}"
    rc=$?
    [ $rc -eq 0 ] && echo "[ok] check passed" || echo "[x] check failed (rc=$rc)"
    exit $rc ;;
  run) ;;
  *) echo "usage: run_gsm8k_base_1shot.sh prep-or-prep-check-or-check-or-run"; exit 2 ;;
esac

# ---------- run: fail-closed pre-conditions ----------
[ -f "${MASK}" ] || { echo "[x] mask missing: ${MASK}"; exit 1; }
[ -f "${DATA}" ] || { echo "[x] benchmark missing: ${DATA}"; exit 1; }
[ -f "${DEMO}" ] || { echo "[x] demo missing -- run: bash run_gsm8k_base_1shot.sh prep"; exit 1; }
for d in ${ALPHA_DIRS}; do
  [ -e "${OUT}/${d}" ] && { echo "[x] ${OUT}/${d} already exists -- refusing to overwrite"; exit 1; }
done

echo "GSM8K BASE 1-shot | ${MODEL_NAME} <- ${MODEL_DIR} | cells ${ALPHA_DIRS}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset)}  (recorded, not enforced)"

mkdir -p "${BASE_DIR}/${MODEL_NAME}"
REC="${BASE_DIR}/${MODEL_NAME}/base_1shot_run_record_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "protocol=gsm8k-base-1shot-v1 model_dir=${MODEL_DIR} configs=${CONFIGS}"
  echo "max_new_tokens=${MAX_NEW_TOKENS} temperature=${TEMPERATURE} bs=${BATCH_SIZE}"
  echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} date=$(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "gpu=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "git=$(git rev-parse HEAD 2>/dev/null)"
  echo "mask_sha256=$(sha256sum "${MASK}" | cut -d' ' -f1)"
  echo "data_sha256=$(sha256sum "${DATA}" | cut -d' ' -f1)"
  echo "demo_sha256=$(sha256sum "${DEMO}" | cut -d' ' -f1)"
} | tee "${REC}"

${PY} -u get_answer_regenerate_gsm8k.py \
    --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" --type "${TYPE}" --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} --mask_type "${MASK_TYPE}" --test_file "${GSM8K_FILE}" \
    --ans_file "${ANS_FILE}" --suite "${SUITE}" --fmt_wording "${WORDING}" \
    --base_dir "${BASE_DIR}" --roles "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} --temperature ${TEMPERATURE} \
    --batch_size ${BATCH_SIZE} --save_gen_meta --expect_n 300 \
    --fewshot_file "${DEMO_FILE}"
rc=$?
echo "EXIT_STATUS=${rc} $(date +%Y-%m-%dT%H:%M:%S%z)" | tee -a "${REC}"
exit $rc
