#!/bin/bash
# ==================== RRSN -> GSM8K own-direction positive control =========
# Question: does the RRSN (Reasoning RSN, = unweighted mean of the 3 per-task
# GSM8K/MATH/GSM-Hard Expert-Non-expert directions, raw sign), reduced to its
# exact-NMD sparse support and per-layer L2-norm-matched to this model's own
# MRSN exact-NMD support over the EXISTING MMLU-E-matched band, have causal
# activity on GSM8K when steered the same way the model's own MRSN is?
#
# This is an IN-DOMAIN positive control for RRSN, not a cross-task
# generalization test -- GSM8K is one of the 3 tasks RRSN was constructed
# from. A positive result here shows matched-band RRSN CAN move GSM8K
# behaviour; it does NOT show cross-task transfer, and it does NOT by
# itself establish that RRSN and MRSN share a functional mechanism (see
# analyze_mrsn_rrsn_exact_nmd.py's frozen scope-limit wording). A null
# result must first be checked against band/dose/injection validity before
# being read as "RRSN and MRSN differ functionally".
#
# INDEPENDENT of run_gsm8k.sh / run_gsm8k_qwen25.sh -- neither is modified,
# neither is imported. This launcher drives ONLY
# gsm8k_rrsn_positive_control/get_answer_gsm8k_rrsn_positive_control.py,
# which loads the pre-built norm-matched RRSN mask from the SERVER's standard
# mask tree, using the SAME filename convention every other steering
# launcher already uses for its MRSN mask:
#   ${BASE_DIR}/mask/{model}_non_logits/rrsn_normmatched_0.5_<start>_<end>_<size>.npy
# The mask is built OFFLINE, LOCALLY, BEFORE this launcher (see
# RoleHidden/build_rrsn_gsm8k_positive_control_mask.py), then DEPLOYED to the
# server by copying its output array (rrsn_scaled_{model}_{size}.npy in the
# local archive tree) to the path above under a NEW filename -- this NEVER
# overwrites the existing nmd_0.5_..._.npy MRSN mask files in that same
# directory, and requires no rebuild of the mask itself.
#
# This launcher and the driver it calls have NO runtime dependency on the
# local build's provenance JSON or on any RoleHidden-relative path.
# Mask validation is performed automatically at the beginning of every
# formal generation invocation (--baseline/--sweep/--full), directly from
# the deployed mask's own array content on the server (shape, finiteness,
# exact band/top_k structure) -- there is no separate preflight/check
# command. Its actual sha256 is recorded in each cell's run_config.json for
# offline cross-checking, e.g. against the local archive's array sha256,
# after upload.
#
# Everything else is held IDENTICAL to each model's existing frozen No-CoT
# GSM8K sweep: same 300-question benchmark + order, same neutral/Bare/No-CoT
# plain-wording prompt, same post-<|eot_id|> pipeline, same prefill-only
# last-token injection, same greedy decoding / batch_size=24 /
# max_new_tokens=768, and the SAME per-model alpha grid as that model's
# frozen sweep (Llama: -8..+8 step 2; Qwen: -8..+12, +10/+12 included).
#
# *** SAME-MACHINE / SAME-GPU RULE (same reason as every other GSM8K line in
# this repo) *** bf16 greedy is not byte-reproducible across GPUs. EVERY
# alpha of ONE model's curve, INCLUDING alpha=0, must run on ONE machine and
# ONE card in this NEW tree -- alpha=0 is re-run here fresh, never copied
# from the existing frozen tree, so the whole curve (including its own
# baseline) is same-GPU and internally paired.
#
# ==================== Steps ====================
#   bash run_gsm8k_rrsn_positive_control.sh --baseline <model>
#       alpha=0 only. Re-run in THIS tree (not copied), so this curve's own
#       baseline is same-card with every steered cell.
#   bash run_gsm8k_rrsn_positive_control.sh --sweep <model>
#       the remaining alphas of that model's frozen grid (alpha=0 excluded;
#       --baseline must have run first on the SAME card).
#   bash run_gsm8k_rrsn_positive_control.sh --full <model>
#       the FULL frozen grid including alpha=0, sequentially, one card. Use
#       this instead of --baseline+--sweep when starting fresh on an empty
#       output tree.
#
# Every mode above loads and validates the mask automatically before the
# model or any generation starts (see get_answer_gsm8k_rrsn_positive_control.py's
# load_and_verify_mask()) -- there is no separate preflight/check command.
# If the mask is missing or structurally wrong, the run exits immediately,
# before the model is loaded.
#
# Output: components/{model}/gsm8k_rrsn_positive_control/mdf_<alpha>/
#   gsm8k_rrsn_pc_<size>_answers_<top_k>_<start>_<end>.json
#   run_config.json
#
# ACC is computed OFFLINE by eval_gsm8k_rrsn_positive_control.py using the
# frozen GSM8K extractor -- never from this script's inline accuracy field.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${1:-}"
MODEL="${2:-}"

if [[ "${MODEL}" != "llama3" && "${MODEL}" != "qwen2.5" ]]; then
  echo "Usage: bash run_gsm8k_rrsn_positive_control.sh {--baseline|--sweep|--full} {llama3|qwen2.5}"
  exit 1
fi

if [[ "${MODEL}" == "llama3" ]]; then
  MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
  ALPHAS_BASELINE="0"
  ALPHAS_SWEEP="-8 -6 -4 -2 2 4 6 8"
  ALPHAS_FULL="-8 -6 -4 -2 0 2 4 6 8"
else
  MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
  ALPHAS_BASELINE="0"
  ALPHAS_SWEEP="-8 -6 -4 -2 2 4 6 8 10 12"
  ALPHAS_FULL="-8 -6 -4 -2 0 2 4 6 8 10 12"
fi

# ==================== Paths ====================
DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
PY="${PY:-python}"
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

cd "${WORK_DIR}" || { echo "[✗] cannot cd ${WORK_DIR}"; exit 1; }

banner () {
  echo "=================================================="
  echo "RRSN -> GSM8K positive control | ${MODEL}"
  echo "step                : $1"
  echo "model_dir            : ${MODEL_DIR}"
  echo "alphas this step     : $2"
  echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"
  echo "Start: $(date)"
  echo "=================================================="
  if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "[warn] CUDA_VISIBLE_DEVICES is unset. Pin ONE card and keep it fixed"
    echo "       across --baseline/--sweep/--full for this model."
  fi
}

run_driver () {   # $1 = space-separated alphas, $2 = extra flags
  ${PY} gsm8k_rrsn_positive_control/get_answer_gsm8k_rrsn_positive_control.py \
      --model          "${MODEL}" \
      --model_dir      "${MODEL_DIR}" \
      --base_dir       "${BASE_DIR}" \
      --gsm8k_file     "${GSM8K_FILE}" \
      --alphas         $1 \
      $2
}

case "${MODE}" in
  --baseline)
    OUT="${BASE_DIR}/${MODEL}/gsm8k_rrsn_positive_control/mdf_0"
    banner "--baseline (alpha=0, re-run fresh in this tree)" "${ALPHAS_BASELINE}"
    echo "Output: ${OUT}"
    run_driver "${ALPHAS_BASELINE}" ""
    rc=$?
    ;;

  --sweep)
    OUT="${BASE_DIR}/${MODEL}/gsm8k_rrsn_positive_control"
    banner "--sweep (remaining alphas, mdf_0 NOT re-run)" "${ALPHAS_SWEEP}"
    if [ ! -d "${OUT}/mdf_0" ]; then
      echo "[warn] ${OUT}/mdf_0 does not exist yet. Run --baseline FIRST on the"
      echo "       SAME card, or the alpha=0 pairing for this curve is missing."
    fi
    echo "Output: ${OUT}/mdf_<alpha>"
    run_driver "${ALPHAS_SWEEP}" ""
    rc=$?
    ;;

  --full)
    OUT="${BASE_DIR}/${MODEL}/gsm8k_rrsn_positive_control"
    banner "--full (ALL alphas incl. 0, sequential, one card)" "${ALPHAS_FULL}"
    echo "Output: ${OUT}/mdf_<alpha>"
    run_driver "${ALPHAS_FULL}" ""
    rc=$?
    ;;

  *)
    echo "Usage: bash run_gsm8k_rrsn_positive_control.sh {--baseline|--sweep|--full} {llama3|qwen2.5}"
    echo ""
    echo "  --baseline  alpha=0 only -> mdf_0 (run first, THIS tree's own baseline)"
    echo "  --sweep     remaining alphas of the frozen grid -> completes the curve"
    echo "  --full      the ENTIRE frozen grid incl. alpha=0, one shot, one card"
    echo ""
    echo "Each mode validates the mask automatically before the model loads --"
    echo "there is no separate preflight/check command."
    echo ""
    echo "Pin CUDA_VISIBLE_DEVICES and keep it identical across all steps for one model:"
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_rrsn_positive_control.sh --baseline llama3"
    exit 1
    ;;
esac

echo ""
echo "=================================================="
if [ $rc -eq 0 ]; then
  echo "[✓] ${MODE} ${MODEL} finished: $(date)"
else
  echo "[✗] ${MODE} ${MODEL} FAILED (rc=$rc): $(date)"
fi
echo "=================================================="
exit $rc
