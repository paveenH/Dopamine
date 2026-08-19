#!/bin/bash
# CGT-Sequential — Qwen2.5-7B-Instruct cross-model replication.
#
# SEPARATE from run_cgt_seq.sh on purpose: that script's mode matrix (v1/v2b/v3/v4
# x asc/desc x verify/full) is Llama-specific and its results are frozen. Editing it
# to add a model would risk the Llama main line for no benefit.
#
# THREE Qwen-specific facts, none inherited from Llama:
#   * layers 16-21  -> --configs uses the EXCLUSIVE end "16-22", matching the
#     betting port (run_gpqa_bet_qwen25.sh). Llama's 11-20 is NOT transferable.
#   * mask nmd_0.5_16_22_7B.npy under mask/qwen2.5_non_logits/
#   * the chat template / final prompt token differ from Llama's. CGT-seq REQUIRES
#     --use_chat (bare modes were qdm~0.50, near random), so the injection site is
#     whatever that template ends with. check_cgt_seq_qwen.py reads it out.
#
# ORDER (do not skip the check — a failed sweep costs hours and yields nothing):
#   bash run_cgt_seq_qwen25.sh --check         # tokenizer/anchor/mask/fires, no data
#   bash run_cgt_seq_qwen25.sh --pilot_asc     # a=0/+-4, 5 runs, ascending
#   bash run_cgt_seq_qwen25.sh --pilot_desc    # a=0/+-4, 5 runs, descending
#   bash run_cgt_seq_qwen25.sh --pilot2_asc    # a=0/+-2, 5 runs (narrow-band retry)
#   bash run_cgt_seq_qwen25.sh --pilot2_desc
#   # -> analyze both; only if the pilot is valid:
#   bash run_cgt_seq_qwen25.sh --asc           # full -8..+8, 20 runs
#   bash run_cgt_seq_qwen25.sh --desc
#
# PILOT GATE (judge before spending the full sweep):
#   1. invalid_rate at a=0 is low (Llama v4 read 0.00). A high a=0 invalid means
#      the interface is broken on this tokenizer, not that Qwen is impulsive.
#   2. qdm at a=0 is clearly above 0.50. At chance the model is not using the
#      chest counts at all and every betting readout is uninterpretable.
#   3. mean_accept_step MOVES between -4 / 0 / +4. Direction should match Llama
#      (+a earlier accept), but a Qwen-specific band is expected -- betting broke
#      at OPPOSITE ends on the two models (Llama -a, Qwen +a).
#   4. empty_gen_rate / color_confusion_rate near 0 at a=0.
#
# Injection: tail=1, inject_turn OFF (Llama-validated strength; tail=4 over-steered).

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

# Qwen band: layers 16-21, written 16-22 (exclusive end), same as the betting port.
LS=16
LE=22

# ==================== Defaults ====================
NUM_RUNS=20
MAX_NEW_TOKENS=64
TEMPERATURE=1.0
TOP_P=0.9
PRESENTATION=""
ANS_FILE=""
SAVE_RAW_FLAG="--save_all_raw"
USE_CHAT_FLAG="--use_chat"
PROMPT_VER="v4"          # paper main line (direction-only hint)
ANCHOR="default"         # colour step "Color: ", bet step NO anchor (validated)

FULL_CONFIGS="0-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE} neg4-${LS}-${LE} 4-${LS}-${LE} neg6-${LS}-${LE} 6-${LS}-${LE} neg8-${LS}-${LE} 8-${LS}-${LE}"
PILOT_CONFIGS="0-${LS}-${LE} neg4-${LS}-${LE} 4-${LS}-${LE}"
# +-2 pilot: the +-4 pilot FAILED its validity gate (-4 = colour-step format
# drift, +4 = a stimulus-independent constant). This asks whether Qwen simply
# has a NARROWER usable band at these layers. Separate ANS_FILE so it cannot
# overwrite the stored +-4 pilot.
PILOT2_CONFIGS="0-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE}"

# v5 DOSE-RANGE PILOT (frozen prompt, 2026-08-19). v5 passed the alpha=0 knowing
# gate that v4 failed (qdm_major_red .719/.854 vs v4's .306/.65), so this is no
# longer prompt tuning -- it is a PRE-DECLARED dose-range pilot with the prompt
# held fixed. Seven cells so the usable band is found in one pass:
#   * EVERY cell is reported. Selecting a subset after seeing the numbers is
#     exactly the freedom the alpha=0-only gate rule exists to remove.
#   * Check the knowing/invalid gate PER CELL first, behaviour second.
#   * Usable band := the largest CONTIGUOUS SYMMETRIC range whose every cell
#     passes. +-4 / +-6 are boundary diagnostics, not conclusions.
#   * alpha=0 is re-run at N=5 with these seeds. The stored N=3 calibration is
#     NOT reused -- cross-alpha stats are paired by run index, so mixing an N=3
#     baseline into an N=5 sweep breaks the pairing.
# Still a PILOT: N=20 comes later, and only over the band selected here.
V5_DOSE_CONFIGS="0-${LS}-${LE} neg2-${LS}-${LE} 2-${LS}-${LE} neg4-${LS}-${LE} 4-${LS}-${LE} neg6-${LS}-${LE} 6-${LS}-${LE}"

# ==================== Modes ====================
if [ "$1" == "--check" ]; then
    cd "${WORK_DIR}"
    echo "=== Qwen2.5 CGT-Seq pre-flight (no data written) ==="
    python check_cgt_seq_qwen.py \
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
        --anchor "${ANCHOR}" \
        --presentation desc
    exit $?
elif [ "$1" == "--pilot_asc" ]; then
    CONFIGS="${PILOT_CONFIGS}"; NUM_RUNS=5
    PRESENTATION="asc";  ANS_FILE="cgt/seq_asc_v4_qwen_pilot"
elif [ "$1" == "--pilot_desc" ]; then
    CONFIGS="${PILOT_CONFIGS}"; NUM_RUNS=5
    PRESENTATION="desc"; ANS_FILE="cgt/seq_desc_v4_qwen_pilot"
elif [ "$1" == "--pilot2_asc" ]; then
    CONFIGS="${PILOT2_CONFIGS}"; NUM_RUNS=5
    PRESENTATION="asc";  ANS_FILE="cgt/seq_asc_v4_qwen_pilot2"
elif [ "$1" == "--pilot2_desc" ]; then
    CONFIGS="${PILOT2_CONFIGS}"; NUM_RUNS=5
    PRESENTATION="desc"; ANS_FILE="cgt/seq_desc_v4_qwen_pilot2"
elif [ "$1" == "--v5_pilot_desc" ]; then
    CONFIGS="${V5_DOSE_CONFIGS}"; NUM_RUNS=5; PROMPT_VER="v5"
    PRESENTATION="desc"; ANS_FILE="cgt/seq_desc_v5_qwen_pilot"
elif [ "$1" == "--v5_pilot_asc" ]; then
    CONFIGS="${V5_DOSE_CONFIGS}"; NUM_RUNS=5; PROMPT_VER="v5"
    PRESENTATION="asc";  ANS_FILE="cgt/seq_asc_v5_qwen_pilot"
elif [ "$1" == "--asc" ]; then
    CONFIGS="${FULL_CONFIGS}"
    PRESENTATION="asc";  ANS_FILE="cgt/seq_asc_v4_qwen"
elif [ "$1" == "--desc" ]; then
    CONFIGS="${FULL_CONFIGS}"
    PRESENTATION="desc"; ANS_FILE="cgt/seq_desc_v4_qwen"
else
    echo "Usage: bash run_cgt_seq_qwen25.sh --check"
    echo "       v4 (FROZEN as boundary evidence): --pilot_asc | --pilot_desc | --pilot2_asc | --pilot2_desc | --asc | --desc"
    echo "       v5 (current line):                --v5_pilot_asc | --v5_pilot_desc"
    echo "       Run --check first. There is deliberately NO v5 full branch:"
    echo "       N=20 is unlocked only after the v5 dose pilot passes per-cell."
    exit 1
fi

echo "=================================================="
echo "CGT-Sequential — ${MODEL_NAME} ${MODEL_SIZE} | presentation=${PRESENTATION}"
echo "Layers : ${LS}-${LE} (exclusive end) | prompt=${PROMPT_VER} | anchor=${ANCHOR}"
echo "Configs: ${CONFIGS}"
echo "Runs   : ${NUM_RUNS}"
echo "Output : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_cgt_seq.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --presentation "${PRESENTATION}" \
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
    echo "[Done] CGT-Sequential ${MODEL_NAME} ${PRESENTATION} — $(date)"
    echo "Run the OTHER presentation, then analyse both together (from RoleAnswer/):"
    echo "  python3.10 analyze_cgt_seq.py \\"
    echo "      --asc  ${MODEL_NAME}/${ANS_FILE/_desc_/_asc_} \\"
    echo "      --desc ${MODEL_NAME}/${ANS_FILE/_asc_/_desc_}"
else
    echo "[FAILED] CGT-Sequential ${MODEL_NAME} ${PRESENTATION} — $(date)"
    exit 1
fi
