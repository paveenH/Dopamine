#!/bin/bash
# ==================== GSM8K regenerate — SAME-MACHINE backfill (2026-06-04) ====
# Single entry-point: get_answer_regenerate_gsm8k.py. Non-Cartesian matrix —
# multi-role only at alpha=0, steering only on neutral.
#
# *** WHY: bf16 greedy is NOT byte-reproducible across GPUs (different cuBLAS
# accumulation order → logit ties flip → whole CoT chain diverges; the 6/3 +4
# verify showed 205/300 sample-text mismatches across machines). So the
# dose-response curve must live on ONE machine. This backfills, on the NEW
# machine (182), every cell that was previously run on the OLD machine, so the
# whole GSM8K table is same-GPU and horizontally comparable. ***
#
# Already on the new machine (182) — do NOT rerun:
#   plain: alpha=+4 No-CoT neutral (mdf_4); alpha=±2/±6/±8 No-CoT neutral; CoT ±4
#   (Everything else below was OLD-machine and is being replaced.)
#
# Cells this run (per wording; plain vs pushy differ — see WORDINGS loop):
#   plain  [1] alpha=0  No-CoT  all 4 roles          → mdf_0
#   plain  [2] alpha=-4 No-CoT  neutral              → mdf_-4   (+4 already on 182)
#   plain  [3] alpha=0  CoT     neutral              → mdf_0_cot
#   pushy  [1] alpha=0  No-CoT  all 4 roles          → mdf_0_pushy
#   pushy  [2] alpha=±4 No-CoT  neutral              → mdf_4_pushy / mdf_-4_pushy
#   pushy  [3] alpha=0  CoT     neutral              → mdf_0_pushy_cot
# (pushy = full re-run of the OLD pushy matrix, no new conditions added.)
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
# positive-control ablation). pushy writes to _pushy dirs so it never overwrites
# the plain main line.
#
# Usage: bash run_gsm8k.sh              (plain + pushy backfill, default)
#        WORDING=plain bash run_gsm8k.sh   (just one wording)
#
# ==================== MACHINE switch (184 cross-machine control) =============
# MACHINE=182 (default) → the full same-machine backfill matrix above (UNCHANGED).
# MACHINE=184           → cross-machine REPRODUCIBILITY CONTROL ONLY. Runs just
#   neutral No-CoT alpha=0 and alpha=+4 and writes to *_184 dirs, so the 182
#   authoritative cells are never overwritten. Purpose: byte-compare 184 vs 182
#   to quantify the bf16-greedy cross-GPU divergence (prior +4 cross-machine run
#   = 205/300 sample-text mismatches). *** THIS IS A CONTROL, NOT PAPER DATA ***
#   — paper ACC stays the 182 single-machine dose table only. Do NOT merge the
#   184 cells into the dose-response curve.
#
#   Usage on the 184 box:  MACHINE=184 bash run_gsm8k.sh
#   (WORDING is forced to "plain" under 184 — wording ablation is a 182-only thing.)

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
# Same-machine backfill = BOTH wordings (the whole table moves to machine 182).
WORDINGS="${WORDING:-plain pushy}"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

# ==================== MACHINE switch ====================
MACHINE="${MACHINE:-182}"
cd "${WORK_DIR}"

if [ "${MACHINE}" = "184" ]; then
  # ---------- 184 cross-machine CONTROL: neutral No-CoT alpha=0 / +4 only ----------
  # Writes to *_184 dirs (never overwrites the 182 authoritative cells).
  echo "=================================================="
  echo "GSM8K 184 CROSS-MACHINE CONTROL | ${MODEL_NAME} (${MODEL_SIZE})"
  echo "neutral No-CoT, plain wording, alpha=0 + alpha=+4 → answer_mdf_gsm8k_184"
  echo "*** CONTROL, NOT PAPER DATA — byte-compare vs 182 mdf_0 / mdf_4 ***"
  echo "Start: $(date)"
  echo "=================================================="

  ANS_184="answer_mdf_gsm8k_184"
  echo ""
  echo "[1/1] alpha=0 + alpha=+4 No-CoT — neutral → ${ANS_184}/mdf_0, mdf_4"
  python get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    0-11-20 4-11-20 \
      --mask_type  "${MASK_TYPE}" \
      --test_file  "${GSM8K_FILE}" \
      --ans_file   "${ANS_184}" \
      --suite      "${SUITE}" \
      --fmt_wording "plain" \
      --base_dir   "${BASE_DIR}" \
      --roles      "${ROLES_NEUTRAL}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature    ${TEMPERATURE} \
      --batch_size     ${BATCH_SIZE}
  [ $? -eq 0 ] && echo "[✓] 184 control" || { echo "[✗] 184 control"; exit 1; }

  echo ""
  echo "=================================================="
  echo "184 control finished: $(date)"
  echo "Compare:  ${ANS_184}/mdf_0 vs 182 mdf_0 ; ${ANS_184}/mdf_4 vs 182 mdf_4"
  echo "=================================================="
  exit 0
fi

# ---------- MACHINE=182 (default): full same-machine backfill ----------
echo "=================================================="
echo "GSM8K regenerate re-run | ${MODEL_NAME} (${MODEL_SIZE}) | wordings: ${WORDINGS}"
echo "Start: $(date)"
echo "=================================================="

for WORDING in ${WORDINGS}; do
  if [ "${WORDING}" = "pushy" ]; then ANS_SFX="_pushy"; else ANS_SFX=""; fi
  ANS_NOCOT="answer_mdf_gsm8k${ANS_SFX}"
  ANS_COT="answer_mdf_gsm8k_cot${ANS_SFX}"

  echo ""
  echo "########## WORDING = ${WORDING}  (out: ${ANS_NOCOT} / ${ANS_COT}) ##########"

  # No-CoT steering configs differ by wording: plain skips +4 (already on 182),
  # pushy re-runs the full ±4 (the whole OLD pushy matrix moves to 182).
  if [ "${WORDING}" = "pushy" ]; then
    NOCOT_STEER_CONFIGS="4-11-20 neg4-11-20"
  else
    NOCOT_STEER_CONFIGS="neg4-11-20"
  fi

  # ==================== [1] alpha=0, No-CoT, all 4 roles ====================
  # OLD-machine baseline → re-run on 182. Multi-role only at alpha=0.
  echo ""
  echo "[1/3] alpha=0 No-CoT — all roles → ${ANS_NOCOT}/mdf_0"
  python get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    0-11-20 \
      --mask_type  "${MASK_TYPE}" \
      --test_file  "${GSM8K_FILE}" \
      --ans_file   "${ANS_NOCOT}" \
      --suite      "${SUITE}" \
      --fmt_wording "${WORDING}" \
      --base_dir   "${BASE_DIR}" \
      --roles      "${ROLES_ALL}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature    ${TEMPERATURE} \
      --batch_size     ${BATCH_SIZE}
  [ $? -eq 0 ] && echo "[✓] ${WORDING} step 1" || { echo "[✗] ${WORDING} step 1"; exit 1; }

  # ==================== [2] No-CoT steering, neutral ====================
  # plain: -4 only (+4 already on 182). pushy: ±4 (full re-run).
  echo ""
  echo "[2/3] No-CoT steering (${NOCOT_STEER_CONFIGS}) — neutral → ${ANS_NOCOT}"
  python get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    ${NOCOT_STEER_CONFIGS} \
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
  [ $? -eq 0 ] && echo "[✓] ${WORDING} step 2" || { echo "[✗] ${WORDING} step 2"; exit 1; }

  # ==================== [3] alpha=0, CoT, neutral ====================
  # OLD-machine CoT baseline → re-run on 182. --cot + the separate _cot ans_file
  # so it lands in mdf_0 under answer_mdf_gsm8k_cot[_pushy].
  echo ""
  echo "[3/3] alpha=0 CoT — neutral → ${ANS_COT}/mdf_0"
  python get_answer_regenerate_gsm8k.py \
      --model      "${MODEL_NAME}" \
      --model_dir  "${MODEL_DIR}" \
      --hs         "${HS_PREFIX}" \
      --size       "${MODEL_SIZE}" \
      --type       "${TYPE}" \
      --percentage "${PERCENTAGE}" \
      --configs    0-11-20 \
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
  [ $? -eq 0 ] && echo "[✓] ${WORDING} step 3" || { echo "[✗] ${WORDING} step 3"; exit 1; }
done

echo ""
echo "=================================================="
echo "All GSM8K regenerate runs finished: $(date)"
echo "=================================================="
