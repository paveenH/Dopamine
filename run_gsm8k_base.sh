#!/bin/bash
# ==================== GSM8K Base-checkpoint check (Qwen2.5-7B / Llama-3.1-8B) ====
# Same Bare No-CoT zero-shot protocol as the IT main line, run on the BASE
# checkpoints with the existing IT masks (NOT re-extracted). Per model: one
# model load, two cells -- alpha=0 and the frozen workpoint (Qwen +8, Llama -6).
#
# SEPARATE from run_gsm8k.sh / run_gsm8k_qwen25.sh (frozen). The driver is reused
# unchanged: get_answer_regenerate_gsm8k.py. What each flag actually does there:
#   --model      output dir name only (components/<model>/<ans_file>/) + CSV field
#   --hs         mask dir only (mask/<hs>_non_logits/) -- the IT mask is reused
#   --model_dir  the HF repo id loaded by VicundaModel
# Terminators: llms._build_terminators() unions tokenizer eos, <|eot_id|>,
# <|end_of_turn|>, <|im_end|> and generation_config eos. Base ships only
# <|endoftext|> / <|end_of_text|> as eos; the extra ids exist in the vocab and are
# harmless (a Base model is not expected to emit them). Not changed.
#
# GPU: not enforced. The two cells of a model run in ONE process (one model load);
# per-cell run_meta_gsm8k_*.json records CUDA_VISIBLE_DEVICES, device names and
# whether the model was sharded, plus steering_fires vs expected (L x 300).
#
# Usage (model = qwen or llama; step = check or run):
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base.sh qwen  check
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base.sh qwen  run
#   CUDA_VISIBLE_DEVICES=1 bash run_gsm8k_base.sh llama check
#   CUDA_VISIBLE_DEVICES=1 bash run_gsm8k_base.sh llama run
#
# Output (isolated; refuses if either target cell already exists):
#   components/qwen2.5_base/answer_mdf_gsm8k/mdf_{0,8}/
#   components/llama3_base/answer_mdf_gsm8k/mdf_{0,-6}/
# ACC is computed offline (RoleAnswer/analyze_gsm8k_base.py), never from the
# driver's inline correct_* fields.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WHICH="${1:-}"
STEP="${2:-}"

case "${WHICH}" in
  qwen)
    MODEL_NAME="qwen2.5_base"; MODEL_DIR="Qwen/Qwen2.5-7B"; HS_PREFIX="qwen2.5"
    MODEL_SIZE="7B"; CHECK_ALPHA=8; LS=16; LE=22; ALPHA_DIRS="mdf_0 mdf_8"
    CONFIGS="0-16-22 8-16-22" ;;
  llama)
    MODEL_NAME="llama3_base"; MODEL_DIR="meta-llama/Llama-3.1-8B"; HS_PREFIX="llama3"
    MODEL_SIZE="8B"; CHECK_ALPHA=-6; LS=11; LE=20; ALPHA_DIRS="mdf_0 mdf_-6"
    CONFIGS="0-11-20 neg6-11-20" ;;
  *)
    echo "usage: run_gsm8k_base.sh qwen-or-llama check-or-run"; exit 2 ;;
esac
case "${STEP}" in
  check|run) ;;
  *) echo "usage: run_gsm8k_base.sh qwen-or-llama check-or-run"; exit 2 ;;
esac

TYPE="non"; SUITE="default"; MASK_TYPE="nmd"; PERCENTAGE=0.5
MAX_NEW_TOKENS=768; TEMPERATURE=0.0; BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"; WORDING="plain"; ROLES="neutral"
ANS_FILE="answer_mdf_gsm8k"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

MASK="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/${MASK_TYPE}_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
DATA="${BASE_DIR}/${GSM8K_FILE}"
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"

echo "=================================================="
echo "GSM8K BASE | ${MODEL_NAME} <- ${MODEL_DIR}   step=${STEP}"
echo "mask (IT, reused): ${MASK}"
echo "benchmark        : ${DATA}"
echo "output           : ${OUT}/{${ALPHA_DIRS// /,}}"
echo "configs          : ${CONFIGS}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset)}  (recorded, not enforced)"
echo "=================================================="
# GPU placement is NOT enforced (repo policy). Device, host and sharding are
# recorded per cell in run_meta_gsm8k_*.json. Both cells of a model are run in
# ONE process, so they share whatever devices that process loaded onto.

if [ "${STEP}" = "check" ]; then
  # Technical wiring check only (tokenizer/terminators, prompt + injection token,
  # mask band, benchmark, steering fires). No accuracy or format threshold.
  ${PY} check_gsm8k_qwen.py \
      --base_dir "${BASE_DIR}" --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" \
      --hs "${HS_PREFIX}" --type "${TYPE}" --size "${MODEL_SIZE}" \
      --mask_type "${MASK_TYPE}" --percentage "${PERCENTAGE}" \
      --layer_start ${LS} --layer_end ${LE} --test_file "${GSM8K_FILE}" \
      --ans_file "${ANS_FILE}" --fmt_wording "${WORDING}" --alpha ${CHECK_ALPHA}
  rc=$?
  [ $rc -eq 0 ] && echo "[ok] check passed" || echo "[x] check failed (rc=$rc)"
  exit $rc
fi

# ---------- run: fail-closed pre-conditions ----------
[ -f "${MASK}" ] || { echo "[x] mask missing: ${MASK}"; exit 1; }
[ -f "${DATA}" ] || { echo "[x] benchmark missing: ${DATA}"; exit 1; }
for d in ${ALPHA_DIRS}; do
  if [ -e "${OUT}/${d}" ]; then
    echo "[x] ${OUT}/${d} already exists -- refusing to overwrite"; exit 1
  fi
done

mkdir -p "${BASE_DIR}/${MODEL_NAME}"
REC="${BASE_DIR}/${MODEL_NAME}/base_run_record_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "model_name=${MODEL_NAME} model_dir=${MODEL_DIR} hs=${HS_PREFIX} size=${MODEL_SIZE}"
  echo "configs=${CONFIGS} max_new_tokens=${MAX_NEW_TOKENS} temperature=${TEMPERATURE} bs=${BATCH_SIZE}"
  echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} date=$(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "gpu=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "git=$(git rev-parse HEAD 2>/dev/null)"
  echo "mask_sha256=$(sha256sum "${MASK}" | cut -d' ' -f1)"
  echo "data_sha256=$(sha256sum "${DATA}" | cut -d' ' -f1)"
} | tee "${REC}"

${PY} -u get_answer_regenerate_gsm8k.py \
    --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" --type "${TYPE}" --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} --mask_type "${MASK_TYPE}" --test_file "${GSM8K_FILE}" \
    --ans_file "${ANS_FILE}" --suite "${SUITE}" --fmt_wording "${WORDING}" \
    --base_dir "${BASE_DIR}" --roles "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} --temperature ${TEMPERATURE} \
    --batch_size ${BATCH_SIZE} --save_gen_meta
rc=$?
echo "EXIT_STATUS=${rc} $(date +%Y-%m-%dT%H:%M:%S%z)" | tee -a "${REC}"
exit $rc
