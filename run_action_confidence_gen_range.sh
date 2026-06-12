#!/bin/bash
# GSM8K Confidence Self-Eval — GENERATION variant, RANGE notation "[0,9]"
# Model: Llama3-8B
#
# Confidence sibling of run_action_willingness_gen_range.sh. Identical scaffold
# (generation + chat + prefill, range "[0,9]", neutral, full dose) — the ONLY
# difference is the metric phrasing: "confidence about the question" instead of
# "reasoning willingness", and a "Confidence: " prefill/field. Lets the two §2.6
# self-reports be compared under the same generation readout that the willingness
# range run used.
#
# Single run: neutral only, no anchor, chat + "Confidence: " prefill, full dose.
# Output: answer_action_gen_gsm8k_conf_range. CONTROL evidence — does NOT rescue
# self-report as a wanting proxy (§2.6 verdict stands).

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5

# Full dose-response sweep -8 -> +8 (the run is fast).
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"

GSM8K_FILE="benchmark/gsm8k_test_sample.json"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "Start: $(date)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})  | GENERATION + chat + prefill | metric=confidence | scale=range [0,9]"
echo "Configs: ${CONFIGS}  | Roles: neutral only | no anchor"
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
    --ans_file "answer_action_gen_gsm8k_conf_range" \
    --metric confidence \
    --scale range \
    --use_chat \
    --max_new_tokens 8 \
    --temperature 0.0 \
    --batch_size 8 \
    --data "${DATA}"

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
