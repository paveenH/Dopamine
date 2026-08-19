#!/bin/bash
# IGT — Qwen2.5-7B-Instruct cross-model port.
#
# SEPARATE from run_igt.sh on purpose: that script's v4/v6a/v6b x verify/full
# matrix is Llama-specific and its results are frozen (AdaDopamine.md §3.3 IGT).
# Editing it to add a model risks the Llama main line for no benefit.
#
# THREE Qwen-specific facts, none inherited from Llama:
#   * layers 16-21 -> --configs uses the EXCLUSIVE end "16-22" (L=6), matching
#     the betting and CGT-seq ports. Llama's 11-20 (L=9) is NOT transferable.
#   * mask nmd_0.5_16_22_7B.npy under mask/qwen2.5_non_logits/
#   * IGT uses NO anchor by design, so injection lands on the chat template's
#     assistant header -- Llama-3.1 ends it with id 271 '\n\n', Qwen2.5 with
#     id 198 '\n'. check_igt_qwen.py reads out the real one.
#
# ORDER (do not skip the check -- a failed sweep costs hours and yields nothing):
#   bash run_igt_qwen25.sh --check      # tokenizer/anchor/mask/fires, no data
#   bash run_igt_qwen25.sh --baseline   # alpha=0 only, N=5
#   # -> evaluate the FROZEN baseline gate below; only then is the pilot unlocked.
#
# BASELINE GATE (alpha=0, N=5 aggregate, DESCRIPTIVE -- no significance testing).
# Frozen 2026-08-19, BEFORE the baseline ran. It asks one question: does Qwen
# understand and execute the task at all? Not whether steering works.
#   1. invalid_rate < 0.10
#   2. mean(net_block5) > mean(net_block1) AND last50_net > 0
#      -- a positive learning slope alone can still sit inside the disadvantageous
#         decks, so the level condition is required alongside the slope.
#   3. cycle_score < 0.80                  (not mechanical round-robin)
#   4. max_run_len != 100                  (no whole-episode constant deck)
#   5. median(delib_tok) > 0               (v3/v5 collapsed to zero reasoning)
#   6. premature_stop_rate < 0.15
# If the baseline FAILS, stop Qwen2.5-7B IGT and consider a larger Qwen; do not
# iterate the prompt (the v1->v6b chain was already run on Llama).
#
# There is deliberately NO pilot branch yet. -2/0/+2 x N=5 is unlocked only after
# the baseline passes, and N=20 only after the pilot passes -- same discipline as
# the CGT-seq port.
#
# Injection: tail=1 (validated strength); inject_turn deliberately OFF.
# chat is REQUIRED (trial-by-trial learning needs the multi-turn dialogue).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== Paths ====================
DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL_NAME="qwen2.5"
MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"
MODEL_SIZE="7B"
TYPE="non"
HS_PREFIX="qwen2.5"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Qwen band: layers 16-21, written 16-22 (exclusive end), same as the other ports.
LS=16
LE=22

# ==================== Defaults ====================
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
TOP_P=0.9
SAVE_RAW_FLAG="--save_all_raw"
USE_CHAT_FLAG="--use_chat"
PROMPT_VER="v6b"         # Llama's natural/unforced main line (invitation cue)
ANCHOR="default"         # NO anchor -- an anchor kills the reasoning span

BASELINE_CONFIGS="0-${LS}-${LE}"

# ==================== Modes ====================
if [ "$1" == "--check" ]; then
    cd "${WORK_DIR}"
    echo "=== Qwen2.5 IGT pre-flight (no data written) ==="
    python check_igt_qwen.py \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --hs "${HS_PREFIX}" \
        --type "${TYPE}" \
        --mask_type "${MASK_TYPE}" \
        --percentage "${PERCENTAGE}" \
        --layer_start "${LS}" \
        --layer_end "${LE}" \
        --alpha 4 \
        --base_dir "${BASE_DIR}" \
        --prompt_ver "${PROMPT_VER}" \
        --anchor "${ANCHOR}"
    exit $?
elif [ "$1" == "--baseline" ]; then
    CONFIGS="${BASELINE_CONFIGS}"; NUM_RUNS=5
    ANS_FILE="igt/qwen_v6b_baseline"
else
    echo "Usage: bash run_igt_qwen25.sh --check | --baseline"
    echo "       Run --check first, then the alpha=0 baseline."
    echo "       The -2/0/+2 pilot is unlocked only after the baseline gate passes"
    echo "       (see the gate in this script's header); N=20 only after that."
    exit 1
fi

echo "=================================================="
echo "IGT — ${MODEL_NAME} ${MODEL_SIZE}"
echo "Layers : ${LS}-${LE} (exclusive end) | prompt=${PROMPT_VER} | anchor=${ANCHOR}"
echo "Configs: ${CONFIGS}"
echo "Runs   : ${NUM_RUNS}"
echo "Output : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_igt.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --prompt_ver "${PROMPT_VER}" \
    --anchor "${ANCHOR}" \
    --num_runs "${NUM_RUNS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --ans_file "${ANS_FILE}" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    ${USE_CHAT_FLAG} \
    ${SAVE_RAW_FLAG}

if [ $? -eq 0 ]; then
    echo ""
    echo "[Done] IGT ${MODEL_NAME} — $(date)"
    echo "Evaluate the baseline gate (from RoleAnswer/), with Qwen's OWN tokenizer:"
    echo "  python3.10 analyze_igt.py --dir ${MODEL_NAME}/${ANS_FILE} \\"
    echo "      --tokenizer ${MODEL_DIR}"
else
    echo "[FAILED] IGT ${MODEL_NAME} — $(date)"
    exit 1
fi
