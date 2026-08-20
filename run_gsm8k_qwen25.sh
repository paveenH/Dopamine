#!/bin/bash
# ==================== GSM8K — Qwen2.5-7B-Instruct cross-model replication ======
# Direct replication of the Llama3 GSM8K dose-response (AdaDopamine_gsm8k.md
# §1.2): No-CoT neutral over the full -8 -> +8 sweep, plus CoT at -4/0/+4.
#
# SEPARATE from run_gsm8k.sh on purpose: that script's plain/pushy x 4-role x
# machine-182-backfill matrix is Llama-specific and its results are frozen.
# Editing it to add a model risks the Llama main line for no benefit. This
# follows run_gsm8k_mistral.sh, the existing cross-model precedent.
#
# FOUR Qwen-specific facts, none inherited from Llama:
#   * layers 16-21 -> --configs uses the EXCLUSIVE end "16-22" (L=6), matching
#     the betting / CGT-seq / IGT ports. Llama's 11-20 (L=9) is NOT transferable.
#   * mask nmd_0.5_16_22_7B.npy under mask/qwen2.5_non_logits/
#   * size 7B (28 decoder layers), not 8B/32
#   * Qwen ends its turn with <|im_end|> and ships a LIST eos in
#     generation_config; llms._build_terminators() already unions both.
#
# Everything else is held identical to the Llama main line so the two are
# comparable: same 300-question benchmark, same driver, same templates, neutral
# role only, plain wording, greedy (temperature=0), max_new_tokens=768, bs=24.
#
# *** SAME-MACHINE / SAME-GPU RULE ***
# bf16 greedy is NOT byte-reproducible across GPUs (different cuBLAS
# accumulation order -> logit ties flip -> the whole chain diverges; a Llama +4
# cross-machine re-run gave 205/300 sample-text mismatches). Every alpha of this
# curve must run on ONE machine and ONE card. Pin CUDA_VISIBLE_DEVICES and keep
# it fixed across --baseline / --nocot / --cot. The script prints it every run.
#
# ==================== Steps ====================
#   bash run_gsm8k_qwen25.sh --check      technical pre-flight only, no data.
#                                         Verifies model/tokenizer/mask/layers/
#                                         benchmark/output paths + steering
#                                         fires. Makes NO judgement about
#                                         results.
#   bash run_gsm8k_qwen25.sh --baseline   alpha=0 only -> No-CoT mdf_0.
#                                         This IS the final mdf_0 cell, not a
#                                         throwaway pilot. Read its raw output
#                                         before deciding anything.
#   bash run_gsm8k_qwen25.sh --nocot      the remaining 8 alphas (mdf_0 is NOT
#                                         re-run, so nothing is overwritten).
#                                         Together with --baseline this forms
#                                         the 9-point curve.
#   bash run_gsm8k_qwen25.sh --cot        CoT at -4/0/+4 -> its own ans_file.
#
# --baseline and --nocot deliberately do not overlap: the driver writes one dir
# per alpha and would otherwise regenerate mdf_0 a second time (wasted hours and
# a needless overwrite of a cell already inspected).
#
# Output isolation: the driver's output path does NOT encode --cot, so CoT must
# use its own --ans_file or it would overwrite the No-CoT cells of the same
# alpha. Same split as Llama/Mistral.
#   No-CoT -> components/qwen2.5/answer_mdf_gsm8k/mdf_<alpha>/
#   CoT    -> components/qwen2.5/answer_mdf_gsm8k_cot/mdf_<alpha>/
#
# NOT automated on purpose: prompt, CoT flag, dose range and answer extraction
# are all fixed here. If alpha=0 looks wrong, inspect the raw generations and
# decide by hand -- this script will not adapt anything on its own.
#
# ACC is computed offline (first-#### / last-####) in RoleAnswer/, never from
# the driver's inline correct_* fields, which are process state only.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== Model ====================
MODEL_NAME="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
MODEL_SIZE="7B"
HS_PREFIX="qwen2.5"
TYPE="non"

# ==================== Shared config (identical to the Llama main line) ========
SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"
WORDING="plain"
ROLES_NEUTRAL="neutral"

# Layer band: Qwen 16-21, written with the EXCLUSIVE end 16-22.
LS=16
LE=22

ANS_NOCOT="answer_mdf_gsm8k"
ANS_COT="answer_mdf_gsm8k_cot"

# alpha=0 is run by --baseline; --nocot covers the remaining 8.
CONFIG_BASELINE="0-${LS}-${LE}"
CONFIGS_NOCOT_REST="neg8-${LS}-${LE} neg6-${LS}-${LE} neg4-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE} 4-${LS}-${LE} 6-${LS}-${LE} 8-${LS}-${LE}"
CONFIGS_COT="neg4-${LS}-${LE} 0-${LS}-${LE} 4-${LS}-${LE}"
# --cot6 extends the CoT curve to -6/+6. SEPARATE step, not merged into
# --cot: that sweep is already running, and re-running it would regenerate
# three finished cells. Must go on the SAME card as --cot (same curve).
CONFIGS_COT6="neg6-${LS}-${LE} 6-${LS}-${LE}"
# --cot-rest completes the CoT curve to the SAME nine alphas as No-CoT. It is
# the union of --cot6 and the four cells neither --cot nor --cot6 covers, so a
# run that already did --cot6 will simply regenerate -6/+6; use --cot-rest4 to
# skip them. Rationale for completing it: No-CoT's only significant cells are
# +6/+8, and +-4 is non-significant there too (p_adj=.973), so the current
# -4/0/+4 CoT triple cannot distinguish "CoT suppresses the effect" from
# "CoT was never tested in the range where the effect lives".
CONFIGS_COT_REST="neg8-${LS}-${LE} neg6-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE} 6-${LS}-${LE} 8-${LS}-${LE}"
CONFIGS_COT_REST4="neg8-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE} 8-${LS}-${LE}"

# ==================== Paths ====================
DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
cd "${WORK_DIR}" || { echo "[✗] cannot cd ${WORK_DIR}"; exit 1; }

MODE="${1:-}"

banner () {
  echo "=================================================="
  echo "GSM8K Qwen2.5 cross-model | ${MODEL_NAME} (${MODEL_SIZE})"
  echo "step           : $1"
  echo "layers         : ${LS}-${LE}  (decoder ${LS}..$((LE-1)), L=$((LE-LS)))"
  echo "mask           : ${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/${MASK_TYPE}_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
  echo "benchmark      : ${BASE_DIR}/${GSM8K_FILE}"
  echo "output         : $2"
  echo "configs        : $3"
  echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"
  echo "Start: $(date)"
  echo "=================================================="
  if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "[warn] CUDA_VISIBLE_DEVICES is unset. llms.py loads with"
    echo "       device_map=\"auto\", which claims every VISIBLE card. Pin one"
    echo "       card and keep it fixed across ALL alphas of this curve."
  fi
}

run_sweep () {   # $1=configs  $2=ans_file  $3=extra flags
  ${PY} get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    $1 \
      --mask_type  "${MASK_TYPE}" \
      --test_file  "${GSM8K_FILE}" \
      --ans_file   "$2" \
      --suite      "${SUITE}" \
      --fmt_wording "${WORDING}" \
      --base_dir   "${BASE_DIR}" \
      --roles      "${ROLES_NEUTRAL}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature    ${TEMPERATURE} \
      --batch_size     ${BATCH_SIZE} \
      $3
}

case "${MODE}" in
  --check)
    banner "--check (technical pre-flight, no data written)" \
           "${BASE_DIR}/${MODEL_NAME}/{${ANS_NOCOT},${ANS_COT}}" "(none)"
    ${PY} check_gsm8k_qwen.py \
        --base_dir   "${BASE_DIR}" \
        --model      "${MODEL_NAME}" \
        --model_dir  "${MODEL_DIR}" \
        --hs         "${HS_PREFIX}" \
        --type       "${TYPE}" \
        --size       "${MODEL_SIZE}" \
        --mask_type  "${MASK_TYPE}" \
        --percentage "${PERCENTAGE}" \
        --layer_start ${LS} \
        --layer_end   ${LE} \
        --test_file  "${GSM8K_FILE}" \
        --ans_file   "${ANS_NOCOT}" \
        --ans_file_cot "${ANS_COT}" \
        --fmt_wording "${WORDING}"
    rc=$?
    [ $rc -eq 0 ] && echo "[✓] check passed" || echo "[✗] check failed (rc=$rc)"
    exit $rc
    ;;

  --baseline)
    OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}/mdf_0"
    banner "--baseline (alpha=0 -> the FINAL No-CoT mdf_0 cell)" \
           "${OUT}" "${CONFIG_BASELINE}"
    echo "This is not a throwaway pilot: it is the alpha=0 cell of the final"
    echo "9-point curve. Read its raw generations before running --nocot."
    echo ""
    run_sweep "${CONFIG_BASELINE}" "${ANS_NOCOT}" ""
    rc=$?
    ;;

  --nocot)
    OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_NOCOT}"
    banner "--nocot (the remaining 8 alphas; mdf_0 NOT re-run)" \
           "${OUT}/mdf_<alpha>" "${CONFIGS_NOCOT_REST}"
    if [ ! -d "${OUT}/mdf_0" ]; then
      echo "[warn] ${OUT}/mdf_0 does not exist yet."
      echo "       --baseline normally runs first, so the 9-point curve is"
      echo "       complete. Continuing anyway (this step never writes mdf_0)."
      echo ""
    fi
    run_sweep "${CONFIGS_NOCOT_REST}" "${ANS_NOCOT}" ""
    rc=$?
    ;;

  --cot)
    OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}"
    banner "--cot (CoT at -4/0/+4, own ans_file)" \
           "${OUT}/mdf_<alpha>" "${CONFIGS_COT}"
    run_sweep "${CONFIGS_COT}" "${ANS_COT}" "--cot"
    rc=$?
    ;;

  --cot6)
    OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}"
    banner "--cot6 (CoT -6/+6; extends the -4/0/+4 curve)" \
           "${OUT}/mdf_<alpha>" "${CONFIGS_COT6}"
    echo "Run this on the SAME card as --cot: -6/+6 join that curve, and bf16"
    echo "greedy is not byte-reproducible across GPUs. Only -6/+6 are written,"
    echo "so the finished -4/0/+4 cells are untouched."
    if [ ! -d "${OUT}/mdf_0" ]; then
      echo "[warn] ${OUT}/mdf_0 not present -- --cot has not finished yet."
      echo "       Continuing (this step writes only mdf_-6 / mdf_6), but the"
      echo "       curve is incomplete until --cot lands."
    fi
    echo ""
    run_sweep "${CONFIGS_COT6}" "${ANS_COT}" "--cot"
    rc=$?
    ;;

  --cot-rest|--cot-rest4)
    OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}"
    if [ "$1" == "--cot-rest4" ]; then
      SEL="${CONFIGS_COT_REST4}"; LBL="--cot-rest4 (CoT -8/-2/+2/+8; assumes --cot6 already ran)"
    else
      SEL="${CONFIGS_COT_REST}"; LBL="--cot-rest (CoT -8/-6/-2/+2/+6/+8)"
    fi
    banner "${LBL}" "${OUT}/mdf_<alpha>" "${SEL}"
    echo "MUST run on the SAME card as --cot: these join that curve, and bf16"
    echo "greedy is not byte-reproducible across GPUs. The finished -4/0/+4"
    echo "cells are not in this config list, so they are never overwritten."
    if [ ! -d "${OUT}/mdf_0" ]; then
      echo "[warn] ${OUT}/mdf_0 not present -- --cot has not finished yet."
    fi
    echo ""
    run_sweep "${SEL}" "${ANS_COT}" "--cot"
    rc=$?
    ;;

  *)
    echo "Usage: bash run_gsm8k_qwen25.sh {--check|--baseline|--nocot|--cot|--cot6|--cot-rest|--cot-rest4}"
    echo ""
    echo "  --check     technical pre-flight only (model/mask/layers/paths/fires)"
    echo "  --baseline  alpha=0  -> No-CoT mdf_0   (the final cell, then READ it)"
    echo "  --nocot     the other 8 alphas -> completes the 9-point curve"
    echo "  --cot       CoT -4/0/+4 -> answer_mdf_gsm8k_cot"
    echo "  --cot6      CoT -6/+6 only -> extends that same curve (same card)"
    echo "  --cot-rest  CoT -8/-6/-2/+2/+6/+8 -> completes the nine-alpha curve"
    echo "  --cot-rest4 CoT -8/-2/+2/+8 -> same, when --cot6 already ran"
    echo ""
    echo "Pin CUDA_VISIBLE_DEVICES and keep it identical across all three runs:"
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_qwen25.sh --baseline"
    exit 1
    ;;
esac

echo ""
echo "=================================================="
if [ $rc -eq 0 ]; then
  echo "[✓] ${MODE} finished: $(date)"
  echo "Output: ${OUT}"
  echo "ACC is computed offline (first-####) in RoleAnswer/, not from the"
  echo "driver's inline correct_* fields."
else
  echo "[✗] ${MODE} FAILED (rc=$rc): $(date)"
fi
echo "=================================================="
exit $rc
