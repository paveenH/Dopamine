#!/bin/bash
# ============ GSM8K No-CoT chat-template MATCHED-ANCHOR control ============
# SECOND phase of this experiment. GATED: refuses to launch unless (a) all
# four MATH matched-anchor cells already exist AND (b) CONFIRMED=1 is set,
# acknowledging the MATH results have been reviewed. This is a real,
# mechanical gate (not only a comment) implementing "GSM8K 等 MATH 完成并确认
#后再运行" -- mirroring this repo's P4b WORKPOINT/REVERSE ordering-guard
# precedent (a later stage refuses until the earlier stage's artifact
# exists).
#
# INDEPENDENT of the frozen bare-string GSM8K main line (run_gsm8k.sh) and the
# native chat sweep (run_gsm8k_chat_sweep.sh). Does NOT touch, read, or
# overwrite anything under answer_mdf_gsm8k/, answer_mdf_gsm8k_cot/, or
# answer_mdf_gsm8k_chat_v1/.
#
# Same mechanism as the MATH sibling: the full chat template is kept, but the
# "Answer: " anchor is stripped from the user turn and re-appended after the
# assistant generation header, so prefill-only tail=1 steering injects into
# the SAME token id the bare condition uses (id 220, a single ASCII space)
# rather than the native-chat condition's header token (id 271, '\n\n').
#
# Four alpha only (-6, 0, +6, +8) -- the SAME family as the MATH sibling, not
# re-derived per task. All four run in ONE invocation, ONE model load.
#
# BUDGET IS GSM8K'S OWN, NOT MATH'S: 768 tokens / batch 24, matching
# run_gsm8k.sh and run_gsm8k_chat_sweep.sh.
#
# Output (never overwritten -- the generator fails closed on an existing
# file, and ALL FOUR paths are checked before any cell runs):
#   components/llama3/answer_mdf_gsm8k_chat_matched_anchor_v1/mdf_<alpha>/
#       gsm8k_chat_matched_anchor_8B_answers_11_20.json
#
# Usage (server, from /data1/paveen/Dopamine), ONLY after MATH is done:
#   CUDA_VISIBLE_DEVICES=0 CONFIRMED=1 bash run_gsm8k_chat_matched_anchor.sh
#
# Syntax check only (does NOT launch, and does NOT require CONFIRMED=1 or the
# MATH gate -- bash -n never executes the script body):
#   bash -n run_gsm8k_chat_matched_anchor.sh

set -e

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

GSM8K_FILE="${BASE_DIR}/benchmark/gsm8k_test_sample.json"
MASK_PATH="${BASE_DIR}/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
OUT_DIR="${BASE_DIR}/llama3/answer_mdf_gsm8k_chat_matched_anchor_v1"

# GSM8K's own frozen generation budget -- NOT MATH's 2048/8.
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24

# Fixed, frozen FOUR-point dose set -- the SAME family as the MATH sibling.
CONFIGS="0-11-20 neg6-11-20 6-11-20 8-11-20"

cd "${WORK_DIR}"

# ---- GATE 1: MATH phase must be complete -----------------------------------
MATH_OUT_DIR="${BASE_DIR}/llama3/answer_math_chat_matched_anchor_v1"
declare -a MATH_TAGS=(0 neg6 6 8)
MATH_MISSING=0
for tag in "${MATH_TAGS[@]}"; do
    f="${MATH_OUT_DIR}/mdf_${tag}/math_chat_matched_anchor_${SIZE}_11_20.json"
    if [ ! -f "${f}" ]; then
        echo "[x] missing MATH matched-anchor cell: ${f}"
        MATH_MISSING=1
    fi
done
if [ "${MATH_MISSING}" -eq 1 ]; then
    echo "[x] refusing to run GSM8K matched-anchor: this experiment's spec"
    echo "    requires MATH to complete FIRST (primary task). Run"
    echo "    run_math_chat_matched_anchor.sh and confirm its results before"
    echo "    running this launcher."
    exit 1
fi

# ---- GATE 2: explicit human confirmation ------------------------------------
if [ "${CONFIRMED:-0}" != "1" ]; then
    echo "[x] MATH matched-anchor cells all exist, but CONFIRMED=1 was not"
    echo "    set. This experiment's spec requires GSM8K to run only after"
    echo "    MATH's results have been reviewed and confirmed, not merely"
    echo "    after the files exist. Re-invoke with CONFIRMED=1 once the"
    echo "    MATH matched-anchor results (analyze_chat_matched_anchor.py"
    echo "    --task math) have been reviewed."
    exit 1
fi
echo "[ok] MATH matched-anchor gate passed (4/4 cells present, CONFIRMED=1)."

# Cheap checks BEFORE the ~16GB model load.
if [ ! -f "${GSM8K_FILE}" ]; then
    echo "[x] GSM8K sample file not found: ${GSM8K_FILE}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

# Four-way pre-check on THIS script's own output.
declare -a OUT_TAGS=(0 -6 6 8)
EXISTING=0
for tag in "${OUT_TAGS[@]}"; do
    f="${OUT_DIR}/mdf_${tag}/gsm8k_chat_matched_anchor_${SIZE}_answers_11_20.json"
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
echo "GSM8K chat MATCHED-ANCHOR control | llama3 (${SIZE})"
echo "4 alpha {-6,0,6,8}, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} n=300 band=11-20 (L=9)"
echo "protocol=gsm8k-chat-matched-anchor-v1"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_gsm8k_chat_matched_anchor.py \
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
echo "GSM8K chat matched-anchor control finished: $(date)"
LOCAL_SUBDIR="llama3/gsm8k_chat_matched_anchor_v1"
ANALYZER="analyze_chat_matched_anchor.py --task gsm8k"
echo "Results are NOT analysed on the server. Next step -- from the"
echo "ANALYSIS BOX (~/Documents/RSNResult/RoleAnswer/, NOT part of this repo):"
echo "  rsync -av <server>:${OUT_DIR}/ ~/Documents/RSNResult/RoleAnswer/${LOCAL_SUBDIR}/"
echo "  cd ~/Documents/RSNResult/RoleAnswer && python3.10 ${ANALYZER}"
echo "=================================================="
