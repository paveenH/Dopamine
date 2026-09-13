#!/bin/bash
# ============= GSM8K Qwen2.5-7B Native Chat RSN four-point verification =====
# INDEPENDENT of: the frozen bare Qwen GSM8K line (run_gsm8k_qwen25.sh), the
# Llama GSM8K chat sweep (run_gsm8k_chat_sweep.sh, if present), and the
# matched-anchor family. Does NOT touch, read, or overwrite any of them.
#
# Runs on its OWN GPU, in parallel with the MATH and GSM-Hard Qwen Native
# Chat RSN launchers on two other GPUs (this script: GPU 0; MATH: GPU 1;
# GSM-Hard: GPU 2, per the task brief's expected launch commands). This
# task's own four alpha ALWAYS run in ONE invocation / ONE model load, so the
# same-machine/same-GPU pairing rule holds structurally.
#
# Four-point verification family alpha in {-8, 0, +6, +8} -- NOT the full
# nine-point bare/Llama-chat dose set. No Qwen chat-matched-anchor is run
# here or anywhere in this launcher.
#
# Frozen configuration (identical to the bare Qwen GSM8K line):
#   Model: Qwen/Qwen2.5-7B-Instruct
#   Mask:  components/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy
#   Band:  [16,22), L=6 (Llama's [11,20)/L=9 does NOT transfer)
#   Prefill-only steering, tail=1, greedy (temperature=0), n=300
#   max_new_tokens=768, batch_size=24
#
# Output (never overwritten -- the generator fails closed on an existing
# file, and ALL FOUR paths are checked before any cell runs):
#   components/qwen2.5/answer_mdf_gsm8k_chat_v1/mdf_<alpha>/
#       gsm8k_chat_7B_answers_16_22.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_qwen_chat_rsn.sh
#
# Syntax check only (does NOT launch): bash -n run_gsm8k_qwen_chat_rsn.sh

set -e

MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SIZE="7B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

GSM8K_FILE="${BASE_DIR}/benchmark/gsm8k_test_sample.json"
MASK_PATH="${BASE_DIR}/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy"
OUT_DIR="${BASE_DIR}/qwen2.5/answer_mdf_gsm8k_chat_v1"

MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
N_SAMPLES=300

# Fixed, frozen FOUR-point verification family.
CONFIGS="neg8-16-22 0-16-22 6-16-22 8-16-22"

cd "${WORK_DIR}"

# Cheap checks BEFORE the model load.
if [ ! -f "${GSM8K_FILE}" ]; then
    echo "[x] GSM8K sample file not found: ${GSM8K_FILE}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }
"${PY}" -c "
import json
d = json.load(open('${GSM8K_FILE}', encoding='utf-8'))
n = len(d)
assert n == ${N_SAMPLES}, f'expected {n} == ${N_SAMPLES} GSM8K samples'
print(f'[ok] {n} GSM8K samples')
" || exit 1

# Four-way pre-check: refuse the WHOLE run if ANY of the four output paths
# already exists.
declare -a OUT_TAGS=(-8 0 6 8)
EXISTING=0
for tag in "${OUT_TAGS[@]}"; do
    f="${OUT_DIR}/mdf_${tag}/gsm8k_chat_${SIZE}_answers_16_22.json"
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
echo "GSM8K Qwen2.5 Native Chat RSN four-point verification | (${SIZE})"
echo "4 alpha {-8,0,+6,+8}, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} n=${N_SAMPLES} band=16-22 (L=6)"
echo "protocol=gsm8k-qwen-chat-rsn-v1"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_gsm8k_qwen_chat_rsn.py \
    --model_dir  "${MODEL_DIR}" \
    --size       "${SIZE}" \
    --test_file  "${GSM8K_FILE}" \
    --mask_path  "${MASK_PATH}" \
    --configs    ${CONFIGS} \
    --out_dir    "${OUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE}

echo ""
echo "=================================================="
echo "GSM8K Qwen Native Chat RSN finished: $(date)"
LOCAL_SUBDIR="qwen2.5/gsm8k/chat_rsn"
echo "Results are NOT analysed on the server. Next step -- from the"
echo "ANALYSIS BOX (~/Documents/RSNResult/RoleAnswer/, NOT part of this repo):"
echo "  rsync -av <server>:${OUT_DIR}/ ~/Documents/RSNResult/RoleAnswer/${LOCAL_SUBDIR}/"
echo ""
echo "MATH and GSM-Hard's Qwen Native Chat RSN sweeps are INDEPENDENT"
echo "families (own Holm m=3, own output tree) and do not need to wait for"
echo "this one -- they run concurrently on their own GPUs."
echo "=================================================="
