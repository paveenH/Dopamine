#!/bin/bash
# ==================== GSM-Hard No-CoT chat-template interface sweep =========
# INDEPENDENT of the frozen bare-string GSM-Hard P3 line (run_gsm_hard_llama3.sh).
# Does NOT touch, read, or overwrite anything under gsm_hard_p3/.
#
# Question: does wrapping the SAME neutral No-CoT GSM-Hard prompt in the HF chat
# template change the generation phenotype (CLAUDE.md records Llama GSM-Hard as
# 91-96% cap-hit at 768 -- truncation-bound), and does the alpha=-6 workpoint
# survive the interface change?
#
# All nine alpha of the fixed dose set (-8,-6,-4,-2,0,+2,+4,+6,+8) run in ONE
# invocation, ONE model load -- so every cell of the curve shares a machine and
# a device set BY CONSTRUCTION. alpha=0 is generated fresh under the chat
# wrapper (NOT copied from the bare baseline, which used a different prompt).
#
# SCOPE NOTE: the frozen bare GSM-Hard tree carries only FIVE No-CoT doses
# (-8/-6/-4/0/+4). Four chat cells (-2/+2/+6/+8) therefore have no bare
# counterpart. This sweep is a complete nine-point CHAT curve read against its
# own chat alpha=0; the bare-vs-chat table is a five-row descriptive subset.
#
# NO ACCURACY IS COMPUTED HERE. GSM-Hard gold lives in a separate sealed file
# and the generator cannot reach it (contains_labels=false + a per-row label
# scan). Accuracy is offline only.
#
# Output (never overwritten -- the generator fails closed on an existing file):
#   components/llama3/gsm_hard_chat_v1/mdf_<alpha>/gsm_hard_chat_8B_11_20.json
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 bash run_gsm_hard_chat_sweep.sh
#
# Syntax check only (does NOT launch): bash -n run_gsm_hard_chat_sweep.sh

set -e

MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
SIZE="8B"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

# The SAME label-free question file the frozen bare P3 cells use.
QUESTIONS="${BASE_DIR}/benchmark/gsm_hard_p3_questions.json"
MASK_PATH="${BASE_DIR}/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
OUT_DIR="${BASE_DIR}/llama3/gsm_hard_chat_v1"

# Identical to the frozen bare GSM-Hard cells.
MAX_NEW_TOKENS=768
TEMPERATURE=0.0
BATCH_SIZE=24

# Fixed, frozen nine-point dose set.
CONFIGS="0-11-20 neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

cd "${WORK_DIR}"

# Cheap checks BEFORE the ~16GB model load, so a wrong path names itself
# instead of surfacing hours later as an unrelated HFValidationError.
if [ ! -f "${QUESTIONS}" ]; then
    echo "[x] questions file not found: ${QUESTIONS}"; exit 1
fi
if [ ! -f "${MASK_PATH}" ]; then
    echo "[x] mask not found: ${MASK_PATH}"; exit 1
fi
# A wrong interpreter exits 127 BEFORE anything runs and, under nohup, the log
# looks empty (CLAUDE.md: "cat the log immediately"). Name it here instead.
"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }

# Fail closed: the questions file must declare itself label-free. Same check
# run_gsm_hard_llama3.sh does -- kept here so the launcher refuses before the
# model load rather than relying only on the generator's own guard.
"${PY}" - <<PYCHK
import json, sys
d = json.load(open("${QUESTIONS}", encoding="utf-8"))
if d["meta"].get("contains_labels") is not False:
    sys.exit("ERROR: questions file does not declare contains_labels=false")
bad = [k for s in d["data"] for k in s
       if k.lower() in ("answer","gold","gold_answer","correct","target","accuracy")]
if bad:
    sys.exit(f"ERROR: label fields present: {sorted(set(bad))}")
print(f"label-free check OK ({len(d['data'])} questions, "
      f"digest {d['meta']['questions_sha256'][:16]})")
PYCHK

# All nine alpha run in ONE invocation with ONE model load, so the curve cannot
# be split across machines. This is a WARNING, not a hard reject: the repo-wide
# rule says a launcher must not require a particular GPU nor refuse an unset or
# multi-card CUDA_VISIBLE_DEVICES. It is warned about only because llms.py loads
# with device_map="auto", so an unset value claims every visible card and could
# collide with a concurrent job. Whatever is used is recorded in meta.provenance.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[!] CUDA_VISIBLE_DEVICES is unset: device_map=auto will claim EVERY"
    echo "    visible card. Pin one card if another job shares this machine."
elif [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[!] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards;"
    echo "    the model will be sharded. All nine cells still share it."
fi

echo "=================================================="
echo "GSM-Hard chat-template interface sweep | llama3 (${SIZE})"
echo "9 alpha, one model load, GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "budget=${MAX_NEW_TOKENS} bs=${BATCH_SIZE} band=11-20 (L=9)"
echo "Start: $(date)"
echo "=================================================="

"${PY}" get_answer_gsm_hard_chat_sweep.py \
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
echo "GSM-Hard chat-template sweep finished: $(date)"
echo "Next: python3.10 RoleAnswer/analyze_gsm_hard_chat_sweep.py"
echo "=================================================="
