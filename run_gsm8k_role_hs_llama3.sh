#!/bin/bash
set -euo pipefail

# ============== GSM8K Role Hidden-State Extraction: gsm8k_role (llama3) =====
# Runs extract_gsm8k_role_hs.py --run_all for Llama-3.1-8B-Instruct, task
# gsm8k_role (NO abstention exit), over the FULL 1319-question GSM8K test
# set. Both conditions (expert, non_expert), ONE model load.
#
# Deliberately NOT named RRSN/ARRSN -- see CLAUDE.md's "Reasoning RSN (RRSN)
# history" entry; both prior lines were retired 2026-09-18.
#
# PREREQUISITES (run once, BEFORE this launcher, NOT part of this script):
#   python build_gsm8k_construction_split.py --base_dir "${BASE_DIR}"   (if
#       ${BASE_DIR}/benchmark/gsm8k_construction_split.json does not exist)
#   python build_gsm8k_1319_full.py --base_dir "${BASE_DIR}"
# This produces ${BASE_DIR}/benchmark/gsm8k_1319_full.json (+manifest). This
# launcher checks that file exists but does NOT build it.
#
# STANDALONE. No steering, no generation -- prefill-only forward passes only.
# Intended to run on GPU 0, IN PARALLEL with
# run_gsm8k_role_abstention_hs_llama3.sh on GPU 1 (that script runs the
# gsm8k_role_abstention task). Separate output tree, separate log.
#
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm8k_role_hs_llama3.sh \
#       > logs/gsm8k_role_hs_llama3.log 2>&1 &
#
# Syntax check only (no GPU/model needed):
#   bash -n run_gsm8k_role_hs_llama3.sh
#
# Output: components/hidden_states/llama3/gsm8k_role/
#   expert_8B.h5, expert_8B.manifest.json
#   non_expert_8B.h5, non_expert_8B.manifest.json
#
# CRASH-RECOVERY CAVEAT: if this run dies after expert_8B.h5 is written but
# before non_expert_8B.h5 is, the overwrite guard will REFUSE the next
# invocation. Resuming requires a DELIBERATE manual step: inspect
#   components/hidden_states/llama3/gsm8k_role/
# and if incomplete, delete its contents before re-running.

MODEL="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
TASK="gsm8k_role"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"

cd "${WORK_DIR}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(unset - ALL cards visible)}"

echo "=================================================="
echo "GSM8K Role Hidden-State Extraction: ${MODEL} / task=${TASK}"
echo "Model dir : ${MODEL_DIR}"
echo "Base dir  : ${BASE_DIR}"
echo "Start     : $(date)"
echo "=================================================="

# ==================== Cheap pre-flight (no model load) ====================
[ -f "extract_gsm8k_role_hs.py" ] || { echo "[REFUSE] extract_gsm8k_role_hs.py not found in ${WORK_DIR}"; exit 1; }
[ -f "get_answer_gsm8k_role_abstention_v2.py" ] || { echo "[REFUSE] get_answer_gsm8k_role_abstention_v2.py not found in ${WORK_DIR} (extract_gsm8k_role_hs.py imports its PROMPT_TEMPLATE)"; exit 1; }
[ -f "${BASE_DIR}/benchmark/gsm8k_1319_full.json" ] || { echo "[REFUSE] gsm8k_1319_full.json not found under ${BASE_DIR}/benchmark/ -- run build_gsm8k_1319_full.py first"; exit 1; }

"${PY}" -c "import numpy, torch, h5py" || { echo "[REFUSE] ${PY} cannot import numpy/torch/h5py"; exit 1; }

echo "--- pairing digest check (no model load) ---"
DIGEST=$("${PY}" extract_gsm8k_role_hs.py \
    --model "${MODEL}" --task "${TASK}" \
    --base_dir "${BASE_DIR}" --print_pairing_digest_only)
echo "  task=${TASK} ordered_sample_identity_sha256=${DIGEST}"

echo "=================================================="
echo "Running --run_all: model=${MODEL} task=${TASK} (expert + non_expert, ONE model load)"
echo "Time: $(date)"
echo "=================================================="
"${PY}" extract_gsm8k_role_hs.py \
    --model "${MODEL}" \
    --task "${TASK}" \
    --run_all \
    --model_dir "${MODEL_DIR}" \
    --base_dir "${BASE_DIR}"

echo "=================================================="
echo "[DONE] Both cells finished for ${MODEL} (task=${TASK}, expert+non_expert)."
echo "Finish: $(date)"
echo "=================================================="
