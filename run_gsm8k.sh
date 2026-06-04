#!/bin/bash
# ==================== GSM8K regenerate — INCREMENTAL top-up (2026-06-03) ======
# Single entry-point: get_answer_regenerate_gsm8k.py. Non-Cartesian matrix —
# multi-role only at alpha=0, steering only on neutral.
#
# *** This run ONLY tops up the NEW conditions. Already-run cells are SKIPPED. ***
# Plain only (pushy full set already exists). Already done (do NOT rerun):
#   - alpha=0 No-CoT all roles  / alpha=0 CoT neutral  (steps [1][2] below, kept
#     commented for reference)
#   - alpha=±4 No-CoT neutral   (mdf_4 / mdf_-4 already exist)
#   - the full pushy matrix
#
# NEW cells this run:
#   [3] alpha=±2/±6/±8  No-CoT (neutral)  — fill the dose-response sweep (±4 done)
#   [4] alpha=±4        CoT    (neutral)  — CoT steering axis (was α=0 CoT only)
#
# Prompt: roles passed as full character strings. GSM8K has no E-option → NO
# "honest" framing. get_answer_regenerate_gsm8k.py routes neutral→neutral and
# any other role→neg ("Now you are {role}."), matching track_dopamine_signal.py.
# All 6 GSM8K templates now carry the "#### <number>" final-answer directive.
#
# Output isolation: --ans_file differs for CoT vs No-CoT (the .py output path
# does NOT encode cot), so the two alpha=0 calls don't overwrite each other.
#   No-CoT → answer_mdf_gsm8k[_pushy]        (mdf_0 / mdf_4 / mdf_-4)
#   CoT    → answer_mdf_gsm8k_cot[_pushy]    (mdf_0)
#
# WORDING knob (set below): "plain" (main line) | "pushy" (early-#### 抢答
# positive-control ablation). Same matrix either way; pushy writes to _pushy
# dirs so it never overwrites the neutral main line. Run both, compare wording
# effect vs the internal α-steering lever.
#
# Usage: bash run_gsm8k.sh   (plain only — the incremental top-up)
#        To re-run a previously-completed cell, uncomment step [1]/[2] or add
#        ±4 back into step [3]'s configs.

# ==================== Model ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"

# ==================== Shared config ====================
SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# Full character strings (NOT short names) so prompts align with track.
ROLES_ALL="an expert,a non expert,a primary school teacher,neutral"
ROLES_NEUTRAL="neutral"

# #### directive wording. "plain" = main line; "pushy" = early-#### 抢答
# positive-control ablation. Same matrix either way — only this knob differs,
# so accuracy differences attribute cleanly to wording. "pushy" outputs go to a
# separate _pushy dir so they don't overwrite the plain main line.
# (Named "plain" not "neutral" to avoid colliding with the neutral role.)
#
# Incremental top-up = plain only (pushy full set already exists).
WORDINGS="${WORDING:-plain}"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "GSM8K regenerate re-run | ${MODEL_NAME} (${MODEL_SIZE}) | wordings: ${WORDINGS}"
echo "Start: $(date)"
echo "=================================================="
cd "${WORK_DIR}"

for WORDING in ${WORDINGS}; do
  if [ "${WORDING}" = "pushy" ]; then ANS_SFX="_pushy"; else ANS_SFX=""; fi
  ANS_NOCOT="answer_mdf_gsm8k${ANS_SFX}"
  ANS_COT="answer_mdf_gsm8k_cot${ANS_SFX}"

  echo ""
  echo "########## WORDING = ${WORDING}  (out: ${ANS_NOCOT} / ${ANS_COT}) ##########"

  # ==================== [1] alpha=0, No-CoT, all roles — DONE, SKIPPED ==========
  # Already collected. Uncomment to re-run.
  # python get_answer_regenerate_gsm8k.py \
  #     --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" --hs "${HS_PREFIX}" \
  #     --size "${MODEL_SIZE}" --type "${TYPE}" --percentage "${PERCENTAGE}" \
  #     --configs 0-11-20 --mask_type "${MASK_TYPE}" --test_file "${GSM8K_FILE}" \
  #     --ans_file "${ANS_NOCOT}" --suite "${SUITE}" --fmt_wording "${WORDING}" \
  #     --base_dir "${BASE_DIR}" --roles "${ROLES_ALL}" \
  #     --max_new_tokens ${MAX_NEW_TOKENS} --temperature ${TEMPERATURE} --batch_size ${BATCH_SIZE}

  # ==================== [2] alpha=0, CoT, neutral — DONE, SKIPPED ===============
  # Already collected. Uncomment to re-run.
  # python get_answer_regenerate_gsm8k.py \
  #     --model "${MODEL_NAME}" --model_dir "${MODEL_DIR}" --hs "${HS_PREFIX}" \
  #     --size "${MODEL_SIZE}" --type "${TYPE}" --percentage "${PERCENTAGE}" \
  #     --configs 0-11-20 --mask_type "${MASK_TYPE}" --test_file "${GSM8K_FILE}" \
  #     --ans_file "${ANS_COT}" --suite "${SUITE}" --fmt_wording "${WORDING}" \
  #     --base_dir "${BASE_DIR}" --roles "${ROLES_NEUTRAL}" \
  #     --max_new_tokens ${MAX_NEW_TOKENS} --temperature ${TEMPERATURE} --batch_size ${BATCH_SIZE} --cot

  # ==================== [3] alpha sweep ±2/±6/±8, No-CoT, neutral ===============
  # NEW: fills the dose-response curve. ±4 already exists (mdf_4/mdf_-4) → omitted.
  echo ""
  echo "[3/4] alpha sweep ±2/±6/±8 No-CoT — neutral (±4 already done)"
  python get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    2-11-20 neg2-11-20 6-11-20 neg6-11-20 8-11-20 neg8-11-20 \
      --mask_type  "${MASK_TYPE}" \
      --test_file  "${GSM8K_FILE}" \
      --ans_file   "${ANS_NOCOT}" \
      --suite      "${SUITE}" \
      --fmt_wording "${WORDING}" \
      --base_dir   "${BASE_DIR}" \
      --roles      "${ROLES_NEUTRAL}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature    ${TEMPERATURE} \
      --batch_size     ${BATCH_SIZE}
  [ $? -eq 0 ] && echo "[✓] ${WORDING} step 3" || { echo "[✗] ${WORDING} step 3"; exit 1; }

  # ==================== [4] alpha=+4 / -4, CoT, neutral ====================
  # CoT steering axis (was α=0 CoT only). --cot + the separate _cot ans_file so it
  # lands in mdf_{4,-4} under answer_mdf_gsm8k_cot[_pushy], beside the α=0 CoT.
  echo ""
  echo "[4/4] alpha=+4/-4 CoT — neutral"
  python get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    4-11-20 neg4-11-20 \
      --mask_type  "${MASK_TYPE}" \
      --test_file  "${GSM8K_FILE}" \
      --ans_file   "${ANS_COT}" \
      --suite      "${SUITE}" \
      --fmt_wording "${WORDING}" \
      --base_dir   "${BASE_DIR}" \
      --roles      "${ROLES_NEUTRAL}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature    ${TEMPERATURE} \
      --batch_size     ${BATCH_SIZE} \
      --cot
  [ $? -eq 0 ] && echo "[✓] ${WORDING} step 4" || { echo "[✗] ${WORDING} step 4"; exit 1; }
done

echo ""
echo "=================================================="
echo "All GSM8K regenerate runs finished: $(date)"
echo "=================================================="
