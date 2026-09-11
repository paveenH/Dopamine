#!/bin/bash
# ==================== MATH No-CoT chat-template interface sweep =============
# INDEPENDENT of the frozen bare-string MATH line (run_math.sh /
# run_math_llama3_wp.sh). Does NOT touch, read, or overwrite anything under
# answer_math/ or answer_math_cot/.
#
# Question: does wrapping the SAME neutral No-CoT MATH prompt in the HF chat
# template change the generation phenotype, and does the alpha=-6 workpoint
# (the frozen MATH No-CoT curve is monotone -6 > -4 > 0 > +4) survive the
# interface change?
#
# All nine alpha of the fixed dose set (-8,-6,-4,-2,0,+2,+4,+6,+8) run in ONE
# invocation, ONE model load -- so every cell of the curve shares a machine and
# a device set BY CONSTRUCTION. alpha=0 is generated fresh under the chat
# wrapper (NOT copied from the bare baseline, which used a different prompt).
#
# SCOPE NOTE: the frozen bare MATH tree carries only FIVE No-CoT doses
# (-8/-6/-4/0/+4). Four chat cells (-2/+2/+6/+8) therefore have no bare
# counterpart. This sweep is a complete nine-point CHAT curve read against its
# own chat alpha=0; the bare-vs-chat table is a five-row descriptive subset.
#
# BUDGET IS MATH'S OWN, NOT GSM8K'S. 2048 tokens / batch 8, matching run_math.sh
# and run_math_llama3_wp.sh. Reusing GSM8K's 768/24 would truncate MATH
# solutions and manufacture an extraction floor, destroying comparability with
# the frozen bare MATH cells -- and CLAUDE.md separately records that changing
# the MATH budget forks the caliber against all seven existing neutral cells.
#
# Output (never overwritten -- the generator fails closed on an existing file):
#   components/llama3/answer_math_chat_v1/mdf_<alpha>/math_chat_8B_11_20.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 bash run_math_chat_sweep.sh
#
# Syntax check only (does NOT launch): bash -n run_math_chat_sweep.sh

set -e

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

# The SAME sample file the frozen bare MATH cells use. get_answer_regenerate_math.py
# joins a RELATIVE --test_file onto --base_dir; this script passes an ABSOLUTE
# path, so the components/ segment must be explicit.
MATH_FILE="${BASE_DIR}/benchmark/math_test_sample.json"
MASK_PATH="${BASE_DIR}/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
OUT_DIR="${BASE_DIR}/llama3/answer_math_chat_v1"

# MATH's own frozen generation budget -- NOT GSM8K's 768/24.
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
BATCH_SIZE=8
N_SAMPLES=300

# Fixed, frozen nine-point dose set.
CONFIGS="0-11-20 neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

cd "${WORK_DIR}"

# Cheap checks BEFORE the ~16GB model load.
if [ ! -f "${MATH_FILE}" ]; then
    echo "[x] MATH sample file not found: ${MATH_FILE}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
# A wrong interpreter exits 127 BEFORE anything runs and, under nohup, the log
# looks empty (CLAUDE.md: "cat the log immediately"). Name it here instead.
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

# All nine alpha run in ONE invocation with ONE model load, so the curve cannot
# be split across machines. WARNING, not a hard reject (repo-wide rule); the
# device actually used is recorded in each cell's meta.provenance.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[!] CUDA_VISIBLE_DEVICES is unset: device_map=auto will claim EVERY"
    echo "    visible card. Pin one card if another job shares this machine."
elif [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[!] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards;"
    echo "    the model will be sharded. All nine cells still share it."
fi

echo "=================================================="
echo "MATH chat-template interface sweep | llama3 (${SIZE})"
echo "9 alpha, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} n=${N_SAMPLES} band=11-20 (L=9)"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_math_chat_sweep.py \
    --model_dir  "${MODEL_DIR}" \
    --size       "${SIZE}" \
    --test_file  "${MATH_FILE}" \
    --mask_path  "${MASK_PATH}" \
    --configs    ${CONFIGS} \
    --out_dir    "${OUT_DIR}" \
    --n_samples  ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE}

echo ""
echo "=================================================="
echo "MATH chat-template sweep finished: $(date)"
LOCAL_SUBDIR="llama3/math_chat_v1"
ANALYZER="analyze_math_chat_sweep.py"
echo "Results are NOT analysed on the server: the analyzers live in the"
echo "offline workspace (~/Documents/RSNResult/RoleAnswer/), which is NOT part"
echo "of this repo and is NOT synced here. Next step -- from the ANALYSIS BOX:"
echo "  rsync -av <server>:${OUT_DIR}/ ~/Documents/RSNResult/RoleAnswer/${LOCAL_SUBDIR}/"
echo "  cd ~/Documents/RSNResult/RoleAnswer && python3.10 ${ANALYZER}"
echo "=================================================="
