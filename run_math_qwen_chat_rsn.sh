#!/bin/bash
# ============= MATH Qwen2.5-7B Native Chat RSN four-point verification ======
# INDEPENDENT of: the frozen bare Qwen MATH line (run_math_qwen25.sh), the
# Llama MATH chat sweep, and the matched-anchor family. Does NOT touch, read,
# or overwrite any of them.
#
# Runs on its OWN GPU, in parallel with the GSM8K and GSM-Hard Qwen Native
# Chat RSN launchers on two other GPUs (per the task brief's expected launch
# commands: GSM8K GPU 0, this script GPU 1, GSM-Hard GPU 2). This task's own
# four alpha ALWAYS run in ONE invocation / ONE model load.
#
# Four-point verification family alpha in {-8, 0, +6, +8} -- NOT the full
# nine-point bare/Llama-chat dose set. No Qwen chat-matched-anchor is run
# here or anywhere in this launcher.
#
# Frozen configuration (identical to the bare Qwen MATH line):
#   Model: Qwen/Qwen2.5-7B-Instruct
#   Mask:  components/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy
#   Band:  [16,22), L=6
#   Prefill-only steering, tail=1, greedy (temperature=0), n=300
#   max_new_tokens=2048, batch_size=8 -- MATH's OWN budget, NOT GSM8K's
#
# Output (never overwritten -- the generator fails closed on an existing
# file, and ALL FOUR paths are checked before any cell runs):
#   components/qwen2.5/answer_math_chat_v1/mdf_<alpha_tag>/
#       math_chat_7B_16_22.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=1 bash run_math_qwen_chat_rsn.sh
#
# Syntax check only (does NOT launch): bash -n run_math_qwen_chat_rsn.sh

set -e

MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SIZE="7B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

MATH_FILE="${BASE_DIR}/benchmark/math_test_sample.json"
MASK_PATH="${BASE_DIR}/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy"
OUT_DIR="${BASE_DIR}/qwen2.5/answer_math_chat_v1"

# MATH's own frozen generation budget -- NOT GSM8K's 768/24.
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
BATCH_SIZE=8
N_SAMPLES=300

# Fixed, frozen FOUR-point verification family.
CONFIGS="neg8-16-22 0-16-22 6-16-22 8-16-22"

cd "${WORK_DIR}"

# Cheap checks BEFORE the ~15GB model load.
if [ ! -f "${MATH_FILE}" ]; then
    echo "[x] MATH sample file not found: ${MATH_FILE}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

# Four-way pre-check: refuse the WHOLE run if ANY of the four output paths
# already exists, mirroring the generator's own internal check but catching
# it before the model even loads.
declare -a OUT_TAGS=(neg8 0 6 8)
EXISTING=0
for tag in "${OUT_TAGS[@]}"; do
    f="${OUT_DIR}/mdf_${tag}/math_chat_${SIZE}_16_22.json"
    if [ -f "${f}" ]; then
        echo "[x] already exists: ${f}"
        EXISTING=1
    fi
done
if [ "${EXISTING}" -eq 1 ]; then
    echo "[x] refusing to run ANY cell of this four-point family. Delete the"
    echo "    existing file(s) deliberately first if a re-run is intended."
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[!] CUDA_VISIBLE_DEVICES is unset: device_map=auto will claim EVERY"
    echo "    visible card. Pin one card if another job shares this machine."
elif [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[!] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards;"
    echo "    the model will be sharded. All four cells still share it."
fi

echo "=================================================="
echo "MATH Qwen2.5 Native Chat RSN four-point verification | (${SIZE})"
echo "4 alpha {-8,0,+6,+8}, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} n=${N_SAMPLES} band=16-22 (L=6)"
echo "protocol=math-qwen-chat-rsn-v1"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_math_qwen_chat_rsn.py \
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
echo "MATH Qwen Native Chat RSN finished: $(date)"
LOCAL_SUBDIR="qwen2.5/math/chat_rsn"
echo "Results are NOT analysed on the server. Next step -- from the"
echo "ANALYSIS BOX (~/Documents/RSNResult/RoleAnswer/, NOT part of this repo):"
echo "  rsync -av <server>:${OUT_DIR}/ ~/Documents/RSNResult/RoleAnswer/${LOCAL_SUBDIR}/"
echo ""
echo "GSM8K and GSM-Hard's Qwen Native Chat RSN sweeps are INDEPENDENT"
echo "families (own Holm m=3, own output tree) and do not need to wait for"
echo "this one -- they run concurrently on their own GPUs."
echo "=================================================="
