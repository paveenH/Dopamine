#!/bin/bash

# ==================== MMLU-E Confidence Hidden States Extraction ====================
# Collect RAW hidden states + answers for Llama3.1-8B-Instruct on MMLU-E
# (A/B/C/D/E, E="I am not sure"), for two conditions ONLY:
#   confident    -> "Now you are confident."
#   unconfident  -> "Now you are unconfident."
# Prompt wording matches the RSN paper's explicit-confidence validation
# (ACLARR/main.tex, app:prompt-template-confidence): bare-string, no "honest",
# via the existing templates["neg"] + utils.construct_prompt path.
#
# This stage ONLY collects HS + answers. No mean_diff / NMD / PCA / overlap /
# manifold / steering here.
#
# Does NOT touch run_hidden_mmlu.sh or its existing outputs — separate script,
# separate output directories.
#
# Usage: bash run_hidden_mmlue_confidence_hs.sh [smoke|full]
#   smoke (default): run ONE task only, for verification.
#   full: run all detection/task_list.py TASKS.

MODE="${1:-smoke}"

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

# ==================== Overwrite guard ====================
# get_answer_logits.py opens H5 with mode="w" unconditionally and does not
# check for pre-existing outputs itself, so the guard lives here: refuse to
# run (no silent overwrite) if either output directory already has files.
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

if [ ! -d "${MMLU_DIR}" ]; then
    echo "[REFUSE] MMLU_DIR not found: ${MMLU_DIR}"
    echo "Check the real location of the per-task MMLU JSON files on the server"
    echo "and pass MMLU_DIR=... as an env var, or edit this script's default."
    exit 1
fi

# ==================== Change to work directory ====================
if [ -d "${WORK_DIR}" ]; then
    cd "${WORK_DIR}"
    echo "Working directory: $(pwd)"
else
    echo "Warning: WORK_DIR not found: ${WORK_DIR}, using current directory"
fi

echo "=================================================="
echo "MMLU-E Confidence Hidden States Extraction"
echo "Mode        : ${MODE}"
echo "Model       : ${MODEL_NAME} (${MODEL_SIZE})"
echo "Model dir   : ${MODEL_DIR}"
echo "MMLU dir    : ${MMLU_DIR}"
echo "Roles       : ${ROLES}"
echo "HS out      : ${HS_OUT_DIR}"
echo "Answers out : ${ANS_OUT_DIR}"
echo "Start       : $(date)"
echo "=================================================="

CMD=(python get_answer_logits.py
    --model      "${MODEL_NAME}"
    --model_dir  "${MODEL_DIR}"
    --size       "${MODEL_SIZE}"
    --type       "${TYPE}"
    --ans_file   "${ANS_FILE}"
    --suite      "${SUITE}"
    --data       "${DATA}"
    --base_dir   "${BASE_DIR}"
    --mmlu_dir   "${MMLU_DIR}"
    --hs_dir     "${HS_DIR}"
    --task_name  "${TASK_NAME}"
    --roles      "${ROLES}"
    --use_E
    --save
)

if [ "${MODE}" == "smoke" ]; then
    # Smoke test: patch TASKS to a single task via env var consumed by a tiny
    # wrapper, so detection/task_list.py itself is never modified.
    echo "[smoke] Running ONE task only (abstract_algebra) for verification."
    SMOKE_TASK="abstract_algebra" python - "${CMD[@]:1}" <<'PYEOF'
import sys
import detection.task_list as task_list
import os
task_list.TASKS = [os.environ["SMOKE_TASK"]]
sys.argv = ["get_answer_logits.py"] + sys.argv[1:]
import runpy
runpy.run_path("get_answer_logits.py", run_name="__main__")
PYEOF
else
    echo "[full] Running all TASKS in detection/task_list.py"
    "${CMD[@]}"
fi

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[✓ Done] MMLU-E confidence hidden states saved"
else
    echo "[✗ Failed] MMLU-E confidence hidden states (exit ${STATUS})"
    exit 1
fi

echo "[✓ Finished] at $(date)"
