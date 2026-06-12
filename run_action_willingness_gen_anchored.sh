#!/bin/bash
# GSM8K Willingness Self-Eval (0-9) — GENERATION variant, ANCHORED polarity probe
# Model: Llama3-8B
#
# Sibling of run_action_willingness_gen_gsm8k.sh. The no-anchor generation runs
# showed the willingness self-report is a LIST-ORDER artifact: baseline mean swings
# ~5 (asc 0..9) → ~8.9 (desc 9..0) purely by flipping option order, so the model
# tracks token POSITION, not numeric "effort" semantics.
#
# THIS run adds an explicit polarity label "(higher score = pay more reasoning
# effort)" (--anchor) and runs BOTH orders. Decision test:
#   • if anchored asc-baseline ≈ anchored desc-baseline (both high) → stating the
#     polarity COLLAPSES the order swing → the artifact was polarity ambiguity, and
#     the model CAN express effort once told the direction (= instruction-following,
#     still not non-conscious wanting, so §2.6 verdict unchanged).
#   • if anchored asc vs desc still diverge → even an explicit label can't override
#     the position bias → no stable willingness signal exists.
# Either way this is CONTROL evidence for "oral self-report ≠ wanting proxy"; it
# does NOT rescue willingness as a proxy. Separate output dirs per order.
#
# Full -8→+8 sweep, both orders (the run is fast).

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
DATA="data1"

MASK_TYPE="nmd"
PERCENTAGE=0.5

# Full dose-response sweep -8 -> +8 (the run is fast, so no 3-point pre-check).
CONFIGS="neg8-11-20 neg6-11-20 neg4-11-20 neg2-11-20 0-11-20 2-11-20 4-11-20 6-11-20 8-11-20"
# Quick 3-point check (orig + ±4):
# CONFIGS="0-11-20 4-11-20 neg4-11-20"

GSM8K_FILE="benchmark/gsm8k_test_sample.json"

WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

cd ${WORK_DIR}

# anchored, both orders → two output dirs so asc vs desc baselines can be compared.
for ORDER in asc desc; do
    echo "=================================================="
    echo "Start: $(date)"
    echo "Model: ${MODEL_NAME} (${MODEL_SIZE})  | GENERATION + chat + prefill | ANCHORED | order=${ORDER}"
    echo "Configs: ${CONFIGS}  | Roles: neutral only"
    echo "=================================================="

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
        --ans_file "answer_action_gen_gsm8k_anchored_${ORDER}" \
        --order "${ORDER}" \
        --anchor \
        --use_chat \
        --max_new_tokens 8 \
        --temperature 0.0 \
        --batch_size 8 \
        --data "${DATA}"
done

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
