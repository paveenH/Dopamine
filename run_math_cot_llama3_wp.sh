#!/bin/bash
# ====== MATH CoT fixed-workpoint transfer: Llama3-8B alpha=-6 (CoT) ======
#
# PURPOSE. The MATH No-CoT curve now covers -6/-4/0/+4 and alpha=-6 -- the
# workpoint read from the frozen GSM8K record -- is its optimum (+6.67pp vs
# alpha=0, McNemar p=.0245). The CoT side stops at {-4, 0, +4}, so the
# workpoint has never been run under CoT and the CoT x workpoint interaction
# cannot be formed. This adds exactly ONE cell.
#
# THIS IS NOT A DOSE SEARCH. alpha is read from the frozen record, NOT
# re-searched. It is a single pre-declared contrast, -6 vs the stored CoT
# alpha=0, paired per question -- the same shape as the P3 CoT supplement.
#
# HELD CONSTANT with the stored CoT cells (mdf_0_cot / mdf_4_cot /
# mdf_neg4_cot under llama3/math/math_eot/), verified against run_math.sh:
#   same 300 problems (benchmark/math_test_sample.json, n_samples=300)
#   max_new_tokens=2048, temperature=0.0, batch_size=8, greedy
#   MATH budget 2048/bs=8 -- NOT GSM8K's 768/bs=24
#   band 11-20 (L=9), mask nmd_0.5_11_20_8B.npy, prefill-only, tail_len=1
#   role neutral, --cot, \boxed{} directive
#
# OUTPUT ISOLATION. get_answer_regenerate_math.py's output path does NOT
# encode cot, so a CoT run MUST use its own --ans_file or it would overwrite
# the No-CoT cell of the same alpha. CoT -> answer_math_cot (mdf_neg6).
#
# ONE CARD. bf16 greedy is not byte-reproducible across GPUs and this cell is
# compared per-question against a stored alpha=0. Pin CUDA_VISIBLE_DEVICES.
# The stored cells' physical GPU is unrecoverable (summary CSV carries no
# device field), so -6 vs 0 is a CROSS-RUN pairing -- state it with the result.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_math_cot_llama3_wp.sh > math_wp_cot_llama.log 2>&1 &
#   cat math_wp_cot_llama.log     # a wrong PY exits 127 before anything runs

set -u

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[x] refusing to run: CUDA_VISIBLE_DEVICES is unset."
    echo "    This cell is paired per question against a stored alpha=0 cell;"
    echo "    an unpinned or multi-card run mixes device differences into the"
    echo "    alpha effect irrecoverably."
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

MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=2048        # MATH value, NOT GSM8K's 768
TEMPERATURE=0.0
BATCH_SIZE=8
N_SAMPLES=300
MATH_FILE="benchmark/math_test_sample.json"

ROLES_NEUTRAL="neutral"
ANS_COT="answer_math_cot"
CONFIG="neg6-11-20"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
OUT="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}/mdf_neg6/math_${MODEL_SIZE}_11_20.json"
MASK="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/nmd_${PERCENTAGE}_11_20_${MODEL_SIZE}.npy"

echo "=================================================="
echo "MATH CoT fixed-workpoint | ${MODEL_NAME} alpha=-6 | card ${CUDA_VISIBLE_DEVICES}"
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
# (get_answer_regenerate_math.py:212, DATA_DIR = join(BASE, args.test_file)),
# NOT against the CWD. Check the resolved path, not the relative one.
if [ ! -f "${BASE_DIR}/${MATH_FILE}" ]; then
    echo "[x] benchmark not found: ${BASE_DIR}/${MATH_FILE}"; exit 1
fi
if [ -e "${OUT}" ]; then
    echo "[x] refusing to overwrite an existing cell: ${OUT}"
    echo "    Delete it deliberately if this is a re-run."
    exit 1
fi

# The generator rewrites the summary CSV from scratch with only this run's row.
# Snapshot it so the pre-existing CoT cells can be merged back afterwards.
CSV="${BASE_DIR}/${MODEL_NAME}/${ANS_COT}/summary_math_${MODEL_NAME}_${MODEL_SIZE}.csv"
CSV_BAK=""
if [ -e "${CSV}" ]; then
    CSV_BAK="${CSV}.bak.$(date +%Y%m%d_%H%M%S)"
    cp -p "${CSV}" "${CSV_BAK}" || { echo "[x] cannot snapshot ${CSV}"; exit 1; }
    echo "[i] summary CSV snapshot -> ${CSV_BAK}"
else
    echo "[i] no existing summary CSV; nothing to preserve"
fi

echo ""
echo "[1/1] alpha=-6 CoT — neutral"
"${PY}" get_answer_regenerate_math.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    "${CONFIG}" \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${MATH_FILE}" \
    --ans_file   "${ANS_COT}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    --n_samples      ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE} \
    --cot
[ $? -eq 0 ] && echo "[v] alpha=-6 CoT done" || { echo "[x] alpha=-6 CoT failed"; exit 1; }

# Merge the pre-run rows back in: the generator just replaced the file with a
# single row. Dedup on (alpha, start, end, role); the NEW row wins on conflict.
if [ -n "${CSV_BAK}" ] && [ -e "${CSV}" ]; then
    "${PY}" - "${CSV_BAK}" "${CSV}" <<'PYMERGE'
import csv, sys
bak, cur = sys.argv[1], sys.argv[2]
key = lambda r: (r["alpha"], r["start"], r["end"], r["role"])
with open(bak, newline="", encoding="utf-8") as f:
    old = list(csv.DictReader(f))
with open(cur, newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    new = list(rdr)
    fields = list(rdr.fieldnames)
merged = {key(r): r for r in old}
merged.update({key(r): r for r in new})
with open(cur, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in merged.values():
        w.writerow({k: r.get(k, "") for k in fields})
print(f"[i] summary CSV merged: {len(old)} old + {len(new)} new -> {len(merged)} rows")
PYMERGE
fi

echo ""
echo "=================================================="
echo "Done: $(date)"
echo "CoT -> ${BASE_DIR}/${MODEL_NAME}/${ANS_COT}/mdf_neg6/"
echo "=================================================="
