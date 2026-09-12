#!/bin/bash
# ============= MATH No-CoT chat-template MATCHED-ANCHOR control ============
# INDEPENDENT STATISTICAL FAMILY from run_gsm8k_chat_matched_anchor.sh -- the
# two tasks do NOT depend on each other and may be launched CONCURRENTLY on
# separate GPUs (this script on CUDA_VISIBLE_DEVICES=0, GSM8K on
# CUDA_VISIBLE_DEVICES=1). Also INDEPENDENT of the frozen bare-string MATH
# line (run_math.sh) and the native chat sweep (run_math_chat_sweep.sh). Does
# NOT touch, read, or overwrite anything under answer_math/, answer_math_cot/,
# or answer_math_chat_v1/.
#
# (An earlier revision of this experiment used a targeted four-point family
# {-6,0,6,8} and gated GSM8K on this script completing first; both were
# replaced before any real data existed -- this is the ONLY matched-anchor
# family that has ever been generated under protocol
# "math-chat-matched-anchor-v1".)
#
# Question: get_answer_math_chat_sweep.py's own docstring records a MEASURED
# confound -- native chat wrapping moves the last prefill token (the only
# position prefill-only tail=1 steering injects into) from the bare
# condition's "Answer: " anchor (id 220, ' ') to the assistant generation
# header (id 271, '\n\n'), because apply_chat_template's Jinja `| trim`
# strips the anchor's trailing space. This experiment keeps the full chat
# template but manually re-creates the bare anchor position UNDER it, so the
# final prefill token identity and anchor are matched to the bare condition's
# -- the hidden state, surrounding context, and absolute position still
# differ, so this does NOT mean the interface and injection site are fully
# disentangled.
#
# Full nine-point alpha sweep (-8,-6,-4,-2,0,+2,+4,+6,+8) -- the SAME dose
# set as the frozen bare/native-chat MATH curves, NOT a re-search. All nine
# run in ONE invocation, ONE model load.
#
# BUDGET IS MATH'S OWN, NOT GSM8K'S: 2048 tokens / batch 8, matching
# run_math.sh, run_math_llama3_wp.sh, and run_math_chat_sweep.sh.
#
# Output (never overwritten -- the generator fails closed on an existing
# file, and ALL NINE paths are checked before any cell runs):
#   components/llama3/answer_math_chat_matched_anchor_v1/mdf_<alpha_tag>/
#       math_chat_matched_anchor_8B_11_20.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 bash run_math_chat_matched_anchor.sh
#
# Syntax check only (does NOT launch): bash -n run_math_chat_matched_anchor.sh

set -e

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

MATH_FILE="${BASE_DIR}/benchmark/math_test_sample.json"
MASK_PATH="${BASE_DIR}/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
OUT_DIR="${BASE_DIR}/llama3/answer_math_chat_matched_anchor_v1"

# MATH's own frozen generation budget -- NOT GSM8K's 768/24.
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
BATCH_SIZE=8
N_SAMPLES=300

# Fixed, frozen NINE-point dose set -- the SAME family as the frozen bare/
# native-chat MATH curves.
CONFIGS="0-11-20 neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

cd "${WORK_DIR}"

# Cheap checks BEFORE the ~16GB model load.
if [ ! -f "${MATH_FILE}" ]; then
    echo "[x] MATH sample file not found: ${MATH_FILE}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

# Nine-way pre-check: refuse the WHOLE run if ANY of the nine output paths
# already exists, mirroring the generator's own internal check but catching
# it before the model even loads.
declare -a OUT_TAGS=(0 neg8 neg6 neg4 neg2 2 4 6 8)
EXISTING=0
for tag in "${OUT_TAGS[@]}"; do
    f="${OUT_DIR}/mdf_${tag}/math_chat_matched_anchor_${SIZE}_11_20.json"
    if [ -f "${f}" ]; then
        echo "[x] already exists: ${f}"
        EXISTING=1
    fi
done
if [ "${EXISTING}" -eq 1 ]; then
    echo "[x] refusing to run ANY cell of this nine-point family. Delete the"
    echo "    existing file(s) deliberately first if a re-run is intended."
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[!] CUDA_VISIBLE_DEVICES is unset: device_map=auto will claim EVERY"
    echo "    visible card. Pin one card if another job shares this machine."
elif [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[!] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards;"
    echo "    the model will be sharded. All nine cells still share it."
fi

echo "=================================================="
echo "MATH chat MATCHED-ANCHOR control | llama3 (${SIZE})"
echo "9 alpha {-8,-6,-4,-2,0,2,4,6,8}, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} n=${N_SAMPLES} band=11-20 (L=9)"
echo "protocol=math-chat-matched-anchor-v1"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_math_chat_matched_anchor.py \
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
echo "MATH chat matched-anchor control finished: $(date)"
LOCAL_SUBDIR="llama3/math_chat_matched_anchor_v1"
ANALYZER="analyze_chat_matched_anchor.py --task math"
echo "Results are NOT analysed on the server. Next step -- from the"
echo "ANALYSIS BOX (~/Documents/RSNResult/RoleAnswer/, NOT part of this repo):"
echo "  rsync -av <server>:${OUT_DIR}/ ~/Documents/RSNResult/RoleAnswer/${LOCAL_SUBDIR}/"
echo "  cd ~/Documents/RSNResult/RoleAnswer && python3.10 ${ANALYZER}"
echo ""
echo "GSM8K's matched-anchor sweep is an INDEPENDENT family (own Holm m=8,"
echo "own output tree) and does not need to wait for this one -- it can run"
echo "concurrently on another GPU via run_gsm8k_chat_matched_anchor.sh."
echo "=================================================="
