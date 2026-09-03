#!/bin/bash
# ====== workpoint-stability supplement (wps-v0): Llama3-8B neighbour cells ======
#
# docs/PREREG_WORKPOINT_STABILITY.md, frozen BEFORE these cells were generated.
#
# PURPOSE. Each cell adds a NEIGHBOUR to an already-reported dose so that the
# reported point has data on BOTH sides. This is NOT a dose search and NOT a
# re-opening of any closed result: no frozen workpoint may be redefined by what
# comes out of it, and no better neighbour becomes a new headline accuracy.
#
#   gsm8k_cot_neg2 : GSM8K CoT  alpha=-2  -- is -4 a local peak, or a -4/-2 plateau
#   math_neg8      : MATH No-CoT alpha=-8 -- left neighbour of the boundary best -6
#   math_cot_neg8  : MATH CoT   alpha=-8  -- is -6 still in the peak region under CoT
#
# BUDGETS ARE INHERITED, NOT CHOSEN. GSM8K is 768/bs=24; MATH is 2048/bs=8.
# Mixing them would fork the caliber against every stored cell of that tree.
#
# OUTPUT ISOLATION. get_answer_regenerate_{gsm8k,math}.py's output path does NOT
# encode cot, so a CoT run MUST use its own --ans_file or it would overwrite the
# No-CoT cell of the same alpha.
#
# ONE CARD. bf16 greedy is not byte-reproducible across GPUs and every cell is
# compared per question against a stored alpha=0. The stored cells' physical GPU
# is unrecoverable (summary CSV carries no device field), so each contrast is a
# CROSS-RUN pairing -- state that with the result.
#
# SERVER IS SCRATCH. Finished cells are downloaded to the local RoleAnswer/
# tree; nothing here requires the comparison target to be present on the server.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_llama3.sh gsm8k_cot_neg2 > wps_l1.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_llama3.sh math_neg8      > wps_l6.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_wps_llama3.sh math_cot_neg8  > wps_l7.log 2>&1 &
#   cat wps_l1.log     # immediately -- a wrong PY exits 127 before anything runs

set -u

CELL="${1:-}"
case "${CELL}" in
    gsm8k_cot_neg2|math_neg8|math_cot_neg8) ;;
    *) echo "usage: $0 {gsm8k_cot_neg2|math_neg8|math_cot_neg8}" >&2; exit 2 ;;
esac

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
TEMPERATURE=0.0
ROLES_NEUTRAL="neutral"
# --n_samples exists ONLY on get_answer_regenerate_math.py; the GSM8K
# generator takes its 300 from the test file and argparse REJECTS the flag.
N_SAMPLES_ARG=""

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
MASK="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/nmd_${PERCENTAGE}_11_20_${MODEL_SIZE}.npy"

case "${CELL}" in
  gsm8k_cot_neg2)
    SCRIPT="get_answer_regenerate_gsm8k.py"
    TEST_FILE="benchmark/gsm8k_test_sample.json"
    ANS_FILE="answer_mdf_gsm8k_cot"
    CONFIG="neg2-11-20"
    TAG="mdf_neg2"
    MAX_NEW_TOKENS=768         # GSM8K value, NOT MATH's 2048
    BATCH_SIZE=24              # GSM8K value, NOT MATH's 8
    COT_FLAG="--cot"
    N_SAMPLES_ARG=""          # GSM8K generator has no --n_samples
    OUT_FILE="gsm8k_${MODEL_SIZE}_answers_20_11_20.json"
    CSV="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/mdf_neg2/summary_gsm8k_${MODEL_NAME}_${MODEL_SIZE}_20_11_20.csv"
    DESC="GSM8K CoT alpha=-2 (neighbour of -4)"
    ;;
  math_neg8)
    SCRIPT="get_answer_regenerate_math.py"
    TEST_FILE="benchmark/math_test_sample.json"
    ANS_FILE="answer_math"
    CONFIG="neg8-11-20"
    TAG="mdf_neg8"
    MAX_NEW_TOKENS=2048        # MATH value, NOT GSM8K's 768
    BATCH_SIZE=8
    COT_FLAG=""
    N_SAMPLES_ARG="--n_samples 300"
    OUT_FILE="math_${MODEL_SIZE}_11_20.json"
    CSV="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/summary_math_${MODEL_NAME}_${MODEL_SIZE}.csv"
    DESC="MATH No-CoT alpha=-8 (neighbour of -6)"
    ;;
  math_cot_neg8)
    SCRIPT="get_answer_regenerate_math.py"
    TEST_FILE="benchmark/math_test_sample.json"
    ANS_FILE="answer_math_cot"
    CONFIG="neg8-11-20"
    TAG="mdf_neg8"
    MAX_NEW_TOKENS=2048
    BATCH_SIZE=8
    COT_FLAG="--cot"
    N_SAMPLES_ARG="--n_samples 300"
    OUT_FILE="math_${MODEL_SIZE}_11_20.json"
    CSV="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/summary_math_${MODEL_NAME}_${MODEL_SIZE}.csv"
    DESC="MATH CoT alpha=-8 (neighbour of -6)"
    ;;
esac

OUT_DIR="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}/${TAG}"

echo "=================================================="
echo "wps-v0 | ${DESC}"
echo "  cell ${CELL} | card ${CUDA_VISIBLE_DEVICES} | mnt=${MAX_NEW_TOKENS} bs=${BATCH_SIZE}"
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
# --test_file is resolved against BASE_DIR by the generator, NOT against the CWD.
if [ ! -f "${BASE_DIR}/${TEST_FILE}" ]; then
    echo "[x] benchmark not found: ${BASE_DIR}/${TEST_FILE}"; exit 1
fi
if [ -d "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
    echo "[x] refusing to overwrite a non-empty cell dir: ${OUT_DIR}"
    echo "    Delete it deliberately if this is a re-run."
    exit 1
fi

# The generator rewrites the summary CSV from scratch with only this run's row.
# Snapshot it so pre-existing rows in the same tree can be merged back.
CSV_BAK=""
if [ -e "${CSV}" ]; then
    CSV_BAK="${CSV}.bak.$(date +%Y%m%d_%H%M%S)"
    cp -p "${CSV}" "${CSV_BAK}" || { echo "[x] cannot snapshot ${CSV}"; exit 1; }
    echo "[i] summary CSV snapshot -> ${CSV_BAK}"
else
    echo "[i] no existing summary CSV in this tree; nothing to preserve"
fi

echo ""
echo "[1/1] ${DESC}"
"${PY}" "${SCRIPT}" \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    "${CONFIG}" \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${TEST_FILE}" \
    --ans_file   "${ANS_FILE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES_NEUTRAL}" \
    ${N_SAMPLES_ARG} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE} \
    ${COT_FLAG}
[ $? -eq 0 ] && echo "[v] ${CELL} done" || { echo "[x] ${CELL} failed"; exit 1; }

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
echo "cell -> ${OUT_DIR}/"
echo ""
echo "NEXT: download this cell to the local RoleAnswer/ tree (the server is"
echo "scratch). Recompute accuracy OFFLINE with analyze_first_last_acc.py --"
echo "never cite the inline number."
echo "=================================================="
