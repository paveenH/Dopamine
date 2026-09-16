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
# local build's provenance JSON or on any RoleHidden-relative path -- both
# --check and formal generation verify the deployed mask directly from its
# own array content (shape, finiteness, exact band/top_k structure) and
# record its actual sha256 for offline cross-checking, e.g. against the
# local archive's array sha256, after upload.
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
#   bash run_gsm8k_rrsn_positive_control.sh --check   <model>
#       technical pre-flight: mask presence+shape+sha256, benchmark file,
#       output paths. No generation.
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
  echo "Usage: bash run_gsm8k_rrsn_positive_control.sh {--check|--baseline|--sweep|--full} {llama3|qwen2.5}"
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
  --check)
    banner "--check (technical pre-flight, no generation)" "(none)"
    MASK_DIR="${BASE_DIR}/mask/${MODEL}_non_logits"
    if [[ "${MODEL}" == "llama3" ]]; then
      MASK_FILE="${MASK_DIR}/rrsn_normmatched_0.5_11_20_8B.npy"
      BAND_START=11; BAND_END=20; TOP_K=20
    else
      MASK_FILE="${MASK_DIR}/rrsn_normmatched_0.5_16_22_7B.npy"
      BAND_START=16; BAND_END=22; TOP_K=17
    fi
    echo "mask file       : ${MASK_FILE}"
    if [[ ! -f "${MASK_FILE}" ]]; then
      echo "[✗] mask file not found -- deploy it: copy the locally-built"
      echo "    rrsn_scaled_${MODEL}_*.npy (from"
      echo "    RoleHidden/build_rrsn_gsm8k_positive_control_mask.py's output,"
      echo "    AdaResult/8.rrsn_gsm8k_positive_control/masks/) to"
      echo "    ${MASK_FILE} on this machine. Do NOT overwrite the existing"
      echo "    nmd_0.5_*.npy MRSN mask files in the same directory."
      exit 1
    fi
    echo "benchmark       : ${BASE_DIR}/${GSM8K_FILE}"
    if [[ ! -f "${BASE_DIR}/${GSM8K_FILE}" ]]; then
      echo "[✗] benchmark file not found: ${BASE_DIR}/${GSM8K_FILE}"
      exit 1
    fi
    echo "output          : ${BASE_DIR}/${MODEL}/gsm8k_rrsn_positive_control/mdf_<alpha>/"
    # No external provenance file is read here -- shape, finiteness, band
    # alignment and exact per-layer top_k are all verified directly from the
    # deployed array's own content, and its actual sha256 is printed for
    # offline cross-checking (e.g. against the local archive copy) rather
    # than checked against a server-side provenance record.
    ${PY} - "${MASK_FILE}" "${BAND_START}" "${BAND_END}" "${TOP_K}" <<'PYEOF'
import sys, hashlib
import numpy as np
mask_file, band_start, band_end, top_k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
mask = np.load(mask_file)
sha = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
finite_ok = bool(np.all(np.isfinite(mask)))
nz_rows = sorted(int(i) for i in np.flatnonzero(np.any(mask != 0, axis=1)))
expected_rows = list(range(band_start - 1, band_end - 1))
rows_ok = (nz_rows == expected_rows)
topk_ok = True
for r in range(mask.shape[0]):
    nnz = int(np.count_nonzero(mask[r]))
    expect = top_k if r in expected_rows else 0
    if nnz != expect:
        topk_ok = False
        print(f"  [row {r}] nnz={nnz} expected={expect} -- MISMATCH")
print(f"mask shape      : {mask.shape}")
print(f"mask sha256     : {sha}")
print(f"finite          : {finite_ok}")
print(f"band (raw)      : ({band_start}, {band_end}), top_k={top_k}")
print(f"nonzero rows    : {nz_rows}")
print(f"expected rows   : {expected_rows}")
print(f"rows match      : {rows_ok}")
print(f"exact top_k     : {topk_ok}")
ok = finite_ok and rows_ok and topk_ok
sys.exit(0 if ok else 1)
PYEOF
    rc=$?
    [ $rc -eq 0 ] && echo "[✓] check passed" || echo "[✗] check failed (rc=$rc)"
    exit $rc
    ;;

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
    echo "Usage: bash run_gsm8k_rrsn_positive_control.sh {--check|--baseline|--sweep|--full} {llama3|qwen2.5}"
    echo ""
    echo "  --check     technical pre-flight (mask sha256+shape/paths), no generation"
    echo "  --baseline  alpha=0 only -> mdf_0 (run first, THIS tree's own baseline)"
    echo "  --sweep     remaining alphas of the frozen grid -> completes the curve"
    echo "  --full      the ENTIRE frozen grid incl. alpha=0, one shot, one card"
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
