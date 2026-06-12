#!/bin/bash
# GSM8K Willingness Self-Evaluation (0-9) — GENERATION variant (chat + prefill)
# Model: Llama3-8B
#
# Sibling of run_action_willingness_gsm8k.sh (the LOGITS path, §2.6 authoritative).
# THIS runs the betting-style generation readout: the model GENERATES a
# "Willingness: <number>" line (chat scaffold + "Willingness: " prefill) and the
# digit is parsed from text. Probes whether casting willingness as generation +
# chat revives an α dose-response that the bare logits readout did not show
# (cf. the betting chat-vs-bare finding). CONTROL / probe — does NOT replace the
# §2.6 logits numbers. Separate output dir (answer_action_gen_gsm8k).
#
# --use_chat is ON here on purpose: the generation form is a free continuation and
# wants the chat scaffold + prefill exactly like betting. For a bare ablation,
# drop --use_chat.

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5

# alpha=0 uses plain generate (no mask); alpha≠0 uses diff_mtx * alpha via regenerate.
# Full dose-response sweep -8 -> +8, layers 11-20.
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"
# Quick check first (orig + ±4):
# CONFIGS="0-11-20 4-11-20 neg4-11-20"

ROLES="an expert,a non expert,a primary school teacher"
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "Start: $(date)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})  | GENERATION + chat + prefill"
echo "Configs: ${CONFIGS}"
echo "Roles  : neutral, ${ROLES}"
echo "Task: GSM8K Willingness (0-9), generation readout"
echo "=================================================="

cd ${WORK_DIR}

python get_action_generate_gsm8k.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "${GSM8K_FILE}" \
    --ans_file "answer_action_gen_gsm8k" \
    --roles "${ROLES}" \
    --use_chat \
    --max_new_tokens 8 \
    --temperature 0.0 \
    --batch_size 8 \
    --data "${DATA}"

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
