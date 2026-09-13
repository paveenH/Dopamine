#!/bin/bash
# ========= GSM-Hard Qwen2.5-7B Native Chat RSN four-point verification =====
# INDEPENDENT of: the frozen bare Qwen GSM-Hard P3 line
# (run_wps_gsm_hard.sh / get_answer_gsm_hard_blind.py), the Llama GSM-Hard
# chat sweep, and the matched-anchor family. Does NOT touch, read, or
# overwrite any of them.
#
# Runs on its OWN GPU, in parallel with the GSM8K and MATH Qwen Native Chat
# RSN launchers on two other GPUs (per the task brief's expected launch
# commands: GSM8K GPU 0, MATH GPU 1, this script GPU 2). This task's own
# four alpha ALWAYS run in ONE invocation / ONE model load.
#
# Four-point verification family alpha in {-8, 0, +6, +8} -- NOT the full
# nine-point bare/Llama-chat dose set. No Qwen chat-matched-anchor is run
# here or anywhere in this launcher.
#
# Frozen configuration (identical to the bare Qwen GSM-Hard line):
#   Model: Qwen/Qwen2.5-7B-Instruct
#   Mask:  components/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy
#   Band:  [16,22), L=6
#   Prefill-only steering, tail=1, greedy (temperature=0), n=300
#   max_new_tokens=768, batch_size=24
#
# LABEL-FREE, STRUCTURALLY SO: reads only the label-free GSM-Hard questions
# file, never the sealed gold. Generation writes NO correctness field and
# prints NO accuracy -- scoring happens offline, separately, against the
# frozen sealed gold.
#
# Output (never overwritten -- the generator fails closed on an existing
# file, and ALL FOUR paths are checked before any cell runs):
#   components/qwen2.5/gsm_hard_chat_v1/mdf_<alpha_tag>/
#       gsm_hard_chat_7B_16_22.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=2 bash run_gsm_hard_qwen_chat_rsn.sh
#
# Syntax check only (does NOT launch): bash -n run_gsm_hard_qwen_chat_rsn.sh

set -e

MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
SIZE="7B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

QUESTIONS="${BASE_DIR}/benchmark/gsm_hard_p3_questions.json"
MASK_PATH="${BASE_DIR}/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy"
OUT_DIR="${BASE_DIR}/qwen2.5/gsm_hard_chat_v1"

MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24
N_SAMPLES=300

# Fixed, frozen FOUR-point verification family.
CONFIGS="neg8-16-22 0-16-22 6-16-22 8-16-22"

cd "${WORK_DIR}"

# Cheap checks BEFORE the model load. Label-free check runs here AND inside
# the generator (belt-and-braces, same as run_wps_gsm_hard.sh).
if [ ! -f "${QUESTIONS}" ]; then
    echo "[x] GSM-Hard questions file not found: ${QUESTIONS}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

"${PY}" - <<PYCHK
import json, sys
d = json.load(open("${QUESTIONS}", encoding="utf-8"))
if d["meta"].get("contains_labels") is not False:
    sys.exit("[x] questions file does not declare contains_labels=false")
bad = [k for s in d["data"] for k in s
       if k.lower() in ("answer","gold","gold_answer","correct","target","accuracy")]
if bad:
    sys.exit(f"[x] label fields present: {sorted(set(bad))}")
n = len(d["data"])
if n != ${N_SAMPLES}:
    sys.exit(f"[x] expected ${N_SAMPLES} questions, got {n}")
print(f"[ok] label-free check OK ({n} questions, "
      f"digest {d['meta']['questions_sha256'][:16]})")
PYCHK
[ $? -eq 0 ] || exit 1

# Four-way pre-check: refuse the WHOLE run if ANY of the four output paths
# already exists.
declare -a OUT_TAGS=(neg8 0 6 8)
EXISTING=0
for tag in "${OUT_TAGS[@]}"; do
    f="${OUT_DIR}/mdf_${tag}/gsm_hard_chat_${SIZE}_16_22.json"
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
echo "GSM-Hard Qwen2.5 Native Chat RSN four-point verification | (${SIZE})"
echo "4 alpha {-8,0,+6,+8}, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} n=${N_SAMPLES} band=16-22 (L=6)"
echo "protocol=gsm-hard-qwen-chat-rsn-v1"
echo "LABEL-FREE: no accuracy will be computed or printed by this script."
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_gsm_hard_qwen_chat_rsn.py \
    --model_dir  "${MODEL_DIR}" \
    --size       "${SIZE}" \
    --questions  "${QUESTIONS}" \
    --mask_path  "${MASK_PATH}" \
    --configs    ${CONFIGS} \
    --out_dir    "${OUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE}

echo ""
echo "=================================================="
echo "GSM-Hard Qwen Native Chat RSN finished: $(date)"
LOCAL_SUBDIR="qwen2.5/gsm_hard/chat_rsn"
echo "Results are NOT analysed on the server. Next step -- from the"
echo "ANALYSIS BOX (~/Documents/RSNResult/RoleAnswer/, NOT part of this repo):"
echo "  rsync -av <server>:${OUT_DIR}/ ~/Documents/RSNResult/RoleAnswer/${LOCAL_SUBDIR}/"
echo "Then score offline against the frozen sealed gold. first_acc is MAIN."
echo ""
echo "GSM8K and MATH's Qwen Native Chat RSN sweeps are INDEPENDENT families"
echo "(own Holm m=3, own output tree) and do not need to wait for this one --"
echo "they run concurrently on their own GPUs."
echo "=================================================="
