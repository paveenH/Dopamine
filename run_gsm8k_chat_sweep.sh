#!/bin/bash
# ==================== GSM8K No-CoT chat-template interface sweep ============
# INDEPENDENT of the frozen bare-string GSM8K main line (run_gsm8k.sh). Does
# NOT touch, read, or overwrite anything under answer_mdf_gsm8k/ or
# answer_mdf_gsm8k_cot/.
#
# Question: does wrapping the SAME neutral No-CoT GSM8K prompt in the HF chat
# template reduce the loop baseline CLAUDE.md documents for the bare sweep
# (raw loop rate 74-88%, no clean alpha trend), and if so does the alpha=-6
# region commitment/accuracy workpoint survive the interface change?
#
# All nine alpha of the SAME frozen dose set (-8,-6,-4,-2,0,+2,+4,+6,+8) run
# in ONE invocation, ONE model load, ONE GPU -- so the same-machine pairing a
# dose curve requires holds structurally, not by convention. alpha=0 is
# generated fresh under the chat wrapper here (NOT copied from the bare
# baseline -- the bare alpha=0 cell used a different prompt string).
#
# Output (never overwritten -- get_answer_gsm8k_chat_sweep.py fails closed on
# an existing file):
#   components/llama3/answer_mdf_gsm8k_chat_v1/mdf_<alpha>/
#       gsm8k_chat_8B_answers_11_20.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_chat_sweep.sh
#
# Local smoke test (no GPU sweep -- do NOT run this against the full 9-alpha
# CONFIGS list without explicit confirmation):
#   bash -n run_gsm8k_chat_sweep.sh

set -e

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

# NOTE: the benchmark tree lives under components/, NOT under WORK_DIR
# directly. run_gsm8k.sh passes the RELATIVE "benchmark/gsm8k_test_sample.json"
# because get_answer_regenerate_gsm8k.py joins it onto --base_dir; this script
# takes an ABSOLUTE --test_file, so the components/ segment must be explicit.
GSM8K_FILE="${BASE_DIR}/benchmark/gsm8k_test_sample.json"
MASK_PATH="${BASE_DIR}/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
OUT_DIR="${BASE_DIR}/llama3/answer_mdf_gsm8k_chat_v1"

MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24

# Fixed, frozen — the SAME 9-point No-CoT dose set as the bare main line.
CONFIGS="0-11-20 neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

cd "${WORK_DIR}"

# Cheap checks BEFORE the ~16GB model load, so a wrong path names itself
# instead of surfacing as an unrelated HFValidationError hours later.
if [ ! -f "${GSM8K_FILE}" ]; then
    echo "[x] benchmark not found: ${GSM8K_FILE}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
# A wrong interpreter exits 127 BEFORE anything runs and, under nohup, the log
# looks empty (CLAUDE.md: "cat the log immediately"). Name it here instead.
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

# All nine alpha run in ONE invocation with ONE model load, so they are on the
# same machine and the same device set BY CONSTRUCTION -- the curve cannot be
# split. This is a WARNING, not a hard reject: CLAUDE.md's repo-wide rule says a
# launcher must not require a particular GPU nor refuse an unset/multi-card
# CUDA_VISIBLE_DEVICES. It is warned about only because llms.py loads with
# device_map="auto", so an unset value claims every visible card and could
# collide with a concurrent job. Whatever is used is recorded in each cell's
# meta.provenance.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[!] CUDA_VISIBLE_DEVICES is unset: device_map=auto will claim EVERY"
    echo "    visible card. Pin one card if another job shares this machine."
elif [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[!] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards;"
    echo "    the model will be sharded. All nine cells still share it."
fi

echo "=================================================="
echo "GSM8K chat-template interface sweep | llama3 (${SIZE})"
echo "9 alpha, one model load, one GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_gsm8k_chat_sweep.py \
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
echo "GSM8K chat-template sweep finished: $(date)"
echo "Next: python RoleAnswer/analyze_gsm8k_chat_sweep.py"
echo "=================================================="
