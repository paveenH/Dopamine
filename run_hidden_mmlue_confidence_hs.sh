#!/bin/bash
set -euo pipefail

# ==================== MMLU-E Confidence Hidden States Extraction ====================
# Collect RAW hidden states + answers for Llama3.1-8B-Instruct on MMLU-E
# (A/B/C/D/E, E="I am not sure"), for two conditions ONLY:
#   confident    -> "Now you are confident."
#   unconfident  -> "Now you are unconfident."
# This uses the same explicit confident/unconfident role condition and
# existing template code path (templates["neg"] + utils.construct_prompt),
# bare-string, no chat template.
#
# This stage ONLY collects HS + answers. No mean_diff / NMD / PCA / overlap /
# manifold / steering here.
#
# Does NOT touch run_hidden_mmlu.sh or its existing outputs — separate script,
# separate output directories.
#
# Runs ALL 57 tasks in detection/task_list.py, no smoke/partial mode.
# Usage: bash run_hidden_mmlue_confidence_hs.sh

# ==================== Configuration ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"

TYPE="non"
SUITE="default"
DATA="data1"

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

# MMLU per-task JSON files. Dopamine components has no mmlu/ yet as of writing;
# point at the existing RolePlaying tree (read-only, not copied/rewritten).
MMLU_DIR="${MMLU_DIR:-/data1/paveen/RolePlaying/components/mmlu}"

# Confidence-specific, independent output dirs (per task instructions).
HS_DIR="/data1/paveen/ConfSteer/HiddenStates"
TASK_NAME="mmlue_confidence"
ANS_FILE="answer_hs_mmlue_confidence"

ROLES="confident,unconfident"

HS_OUT_DIR="${HS_DIR}/${MODEL_NAME}/${TASK_NAME}"
ANS_OUT_DIR="${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"

# ==================== Change to work directory (BEFORE any preflight that ====
# imports project modules, e.g. detection.task_list, which assumes cwd is the
# project root) ====
if [ -d "${WORK_DIR}" ]; then
    cd "${WORK_DIR}"
    echo "Working directory: $(pwd)"
else
    echo "[REFUSE] WORK_DIR not found: ${WORK_DIR}"
    exit 1
fi

# ==================== Pre-flight checks (before loading the model) ====================
# All checks must pass before ANY collection starts -- no partial runs.

# (1) MMLU_DIR exists
if [ ! -d "${MMLU_DIR}" ]; then
    echo "[REFUSE] MMLU_DIR not found: ${MMLU_DIR}"
    echo "Check the real location of the per-task MMLU JSON files on the server"
    echo "and pass MMLU_DIR=... as an env var, or edit this script's default."
    exit 1
fi

# (2) every task in detection/task_list.py has a corresponding JSON file.
# Run as a plain (non-substituted) command so a Python-side failure (import
# error, etc.) gives a non-zero exit that `set -e` actually catches -- a
# `done < <(python3 ...)` process substitution would let a failed Python
# process silently leave MISSING_TASKS empty and pass preflight.
TASK_CHECK_LOG="$(mktemp)"
trap 'rm -f "${TASK_CHECK_LOG}"' EXIT

TASK_CHECK_STATUS=0
python3 - "${MMLU_DIR}" > "${TASK_CHECK_LOG}" <<'PYEOF' || TASK_CHECK_STATUS=$?
import sys
from detection.task_list import TASKS

mmlu_dir = sys.argv[1]
missing = [t for t in TASKS if not __import__("os").path.isfile(f"{mmlu_dir}/{t}.json")]
for t in missing:
    print(t)
sys.exit(1 if missing else 0)
PYEOF

if [ "${TASK_CHECK_STATUS}" -ne 0 ]; then
    echo "[REFUSE] Missing MMLU task JSON files under ${MMLU_DIR} (or task-list check failed, exit ${TASK_CHECK_STATUS}):"
    sed 's/^/  - /' "${TASK_CHECK_LOG}"
    exit 1
fi

# (3) both output directories are empty (no silent overwrite)
check_no_overwrite() {
    local dir="$1"
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "[REFUSE] Output directory already contains files, will not overwrite:"
        echo "  $dir"
        echo "Remove/move it deliberately first if you intend to re-run."
        exit 1
    fi
}

check_no_overwrite "${HS_OUT_DIR}"
check_no_overwrite "${ANS_OUT_DIR}"

echo "=================================================="
echo "MMLU-E Confidence Hidden States Extraction"
echo "Model       : ${MODEL_NAME} (${MODEL_SIZE})"
echo "Model dir   : ${MODEL_DIR}"
echo "MMLU dir    : ${MMLU_DIR}"
echo "Roles       : ${ROLES}"
echo "HS out      : ${HS_OUT_DIR}"
echo "Answers out : ${ANS_OUT_DIR}"
echo "Start       : $(date)"
echo "=================================================="

python get_answer_logits.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --ans_file   "${ANS_FILE}" \
    --suite      "${SUITE}" \
    --data       "${DATA}" \
    --base_dir   "${BASE_DIR}" \
    --mmlu_dir   "${MMLU_DIR}" \
    --hs_dir     "${HS_DIR}" \
    --task_name  "${TASK_NAME}" \
    --roles      "${ROLES}" \
    --use_E \
    --save

echo "[✓ Done] MMLU-E confidence hidden states saved"
echo "[✓ Finished] at $(date)"
