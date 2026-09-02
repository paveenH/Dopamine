#!/bin/bash
# ====== GSM8K CoT fixed-workpoint backfill: Llama3-8B alpha=-6 (CoT) ======
#
# PURPOSE. The Llama GSM8K No-CoT curve covers all nine alphas and its optimum
# is alpha=-6 -- the workpoint every later phase (P2/P3/P4/P4b) reads from the
# frozen record. The CoT side stops at {-4, 0, +4}, so the workpoint has never
# been run under CoT on the very task where it was established. This adds
# exactly ONE cell. Same shape as run_math_cot_llama3_wp.sh, which closed the
# identical gap on MATH.
#
# THIS IS NOT A DOSE SEARCH. alpha is read from the frozen record, NOT
# re-searched. One pre-declared contrast, -6 vs the stored CoT alpha=0, paired
# per question.
#
# *** CROSS-RUN PAIRING -- STATE THIS WITH THE RESULT. *** The stored CoT cells
# (mdf_0 / mdf_4 / mdf_neg4 under llama3/answer_mdf_gsm8k_cot/) carry no device
# provenance: summary_gsm8k_*.csv has fields model/size/alpha/start/end/TOP/
# task/role/correct/total/accuracy_percentage/suite/cot -- NO device field. So
# `-6_cot vs 0_cot` mixes a possible cross-GPU bf16 difference into the alpha
# effect and may not be reported as a same-card paired contrast. MATH carries
# the same caveat for the same reason. If you need a clean same-card CoT dose
# family, that is a FOUR-cell rerun into a fresh dir (the Qwen --cot-full
# pattern), not this script.
#
# HELD CONSTANT with the stored CoT cells, verified against run_gsm8k.sh and
# against the stored mdf_0_cot JSON (n=300, acc 69.0%, plain wording):
#   same 300 problems (benchmark/gsm8k_test_sample.json)
#   max_new_tokens=768, temperature=0.0, batch_size=24, greedy
#   GSM8K budget 768/bs=24 -- NOT MATH's 2048/bs=8
#   band 11-20 (L=9), mask nmd_0.5_11_20_8B.npy, prefill-only, tail_len=1
#   role neutral, suite=default, --fmt_wording plain, --cot
#   template: "Let's think step by step." + "#### " directive + "Answer: " anchor
#
# STATISTICS. A Llama CoT dose family would be its OWN Holm m=3
# (-6/-4/+4 vs 0). It is never pooled with the No-CoT family, nor with P4's
# cross-model m=2. Adding this cell does not modify any frozen artifact.
#
# OUTPUT ISOLATION. get_answer_regenerate_gsm8k.py's output path does NOT
# encode cot, so a CoT run MUST use its own --ans_file or it would overwrite
# the No-CoT cell of the same alpha. CoT -> answer_mdf_gsm8k_cot (mdf_neg6).
# Unlike MATH, the summary CSV lives INSIDE the cell dir (mdf_neg6/), so this
# run writes its own CSV and cannot clobber the stored cells' CSVs -- no
# snapshot/merge step is needed here.
#
# ONE CARD. bf16 greedy is not byte-reproducible across GPUs. Pin
# CUDA_VISIBLE_DEVICES even though the pairing is already cross-run: an
# unpinned multi-card run would add a second, avoidable source of divergence.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm8k_cot_llama3_wp.sh > gsm8k_wp_cot_llama.log 2>&1 &
#   cat gsm8k_wp_cot_llama.log     # immediately -- a wrong PY exits 127 before anything runs

set -u

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[x] refusing to run: CUDA_VISIBLE_DEVICES is unset."
    echo "    This cell is paired per question against a stored alpha=0 cell;"
    echo "    an unpinned or multi-card run adds a further device difference"
    echo "    on top of the cross-run caveat this comparison already carries."
    exit 1
fi
case "${CUDA_VISIBLE_DEVICES}" in
    *,*) echo "[x] refusing to run: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names more than one card."; exit 1 ;;
esac

PY="${PY:-python}"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=768         # GSM8K value, NOT MATH's 2048
TEMPERATURE=0.0
BATCH_SIZE=24              # GSM8K value, NOT MATH's 8
WORDING="plain"            # main line; pushy is a separate ablation tree
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

ROLES_NEUTRAL="neutral"
ANS_COT="answer_mdf_gsm8k_cot"
CONFIG="neg6-11-20"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
OUT_DIR="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}/mdf_neg6"
MASK="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/nmd_${PERCENTAGE}_11_20_${MODEL_SIZE}.npy"

echo "=================================================="
echo "GSM8K CoT fixed-workpoint | ${MODEL_NAME} alpha=-6 | card ${CUDA_VISIBLE_DEVICES}"
echo "Start: $(date)"
echo "=================================================="
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

# interpreter names itself rather than failing obscurely under nohup
"${PY}" -c "import sys, numpy, torch; print('[py]', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)" \
    || { echo "[x] PY='${PY}' does not resolve or cannot import numpy/torch."; exit 1; }

# cheap checks BEFORE 8B of weights load
if [ ! -f "${MASK}" ]; then
    echo "[x] mask not found: ${MASK}"; exit 1
fi
# NOTE: --test_file is resolved against BASE_DIR by the generator
# (get_answer_regenerate_gsm8k.py:251, DATA_DIR = join(BASE, args.test_file)),
# NOT against the CWD. Check the resolved path, not the relative one.
if [ ! -f "${BASE_DIR}/${GSM8K_FILE}" ]; then
    echo "[x] benchmark not found: ${BASE_DIR}/${GSM8K_FILE}"; exit 1
fi
# The stored alpha=0 CoT cell is the comparison target; without it this run has
# nothing to be paired against.
A0="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}/mdf_0"
if [ ! -d "${A0}" ]; then
    echo "[x] stored CoT alpha=0 cell not found: ${A0}"
    echo "    This backfill is a paired contrast against it."
    exit 1
fi
# TOP is computed at runtime (percentage/100 * hidden_size), so the output
# filename is not known here -- guard on the cell DIRECTORY being non-empty.
if [ -d "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
    echo "[x] refusing to overwrite a non-empty cell dir: ${OUT_DIR}"
    echo "    Delete it deliberately if this is a re-run."
    exit 1
fi

echo ""
echo "[1/1] alpha=-6 CoT — neutral"
"${PY}" get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    "${CONFIG}" \
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
[ $? -eq 0 ] && echo "[v] alpha=-6 CoT done" || { echo "[x] alpha=-6 CoT failed"; exit 1; }

echo ""
echo "=================================================="
echo "Done: $(date)"
echo "CoT -> ${OUT_DIR}/"
echo ""
echo "Authoritative accuracy is NOT the inline number: recompute offline with"
echo "  analyze_first_last_acc.py --gsm8k_root llama3/gsm8k"
echo "and report -6_cot vs 0_cot as a CROSS-RUN paired contrast."
echo "=================================================="
