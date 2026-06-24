#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="data1"
BASE_DIR="/${DATA}/paveen/Dopamine/components"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
MASK_TYPE="nmd"
PERCENTAGE=0.5
# Probe one penalty-prone task before re-running the expensive full sweep.
CONFIGS="0-11-20 4-11-20 neg4-11-20"
TASK_NUMS="21"
EPISODES=5

COMMON_ARGS=(
    --model "${MODEL_NAME}"
    --model_dir "${MODEL_DIR}"
    --hs "${HS_PREFIX}"
    --size "${MODEL_SIZE}"
    --type "${TYPE}"
    --percentage "${PERCENTAGE}"
    --mask_type "${MASK_TYPE}"
    --configs ${CONFIGS}
    --task_nums ${TASK_NUMS}
    --num_episodes "${EPISODES}"
    --max_steps 50
    # action_number asks for a single integer, then maps it to valid_actions[n-1].
    --max_new_tokens 16
    --prompt_style "action_number"
    --save_trace
    --data "${DATA}"
    --base_dir "${BASE_DIR}"
)

echo "[1/2] ScienceWorld action-number RAW probe"
python get_answer_sciworld.py \
    "${COMMON_ARGS[@]}" \
    --ans_file "answer_sciworld_actionnumber_raw_probe"

echo "[2/2] ScienceWorld action-number CHAT probe"
python get_answer_sciworld.py \
    "${COMMON_ARGS[@]}" \
    --use_chat \
    --ans_file "answer_sciworld_actionnumber_chat_probe"
