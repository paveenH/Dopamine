#!/bin/bash
# MMLU-E — Qwen2.5-7B-Instruct layer-position test: [11,18) vs the existing [16,22).
#
# SEPARATE from run_regenerate_mmlue.sh on purpose: that script is Mistral3-
# configured and still points at the stale WORK_DIR=/data1/paveen/RolePlaying.
#
# WHY MMLU-E FIRST: the E option ("I am not sure") gives a wanting readout that is
# INDEPENDENT of correctness -- E-ratio should move bidirectionally with alpha
# while accuracy holds. That makes it a cheap, high-signal layer probe: if a band
# cannot move E-ratio without damaging accuracy, it is not worth spending IGT
# hours on. Read BOTH: an E-ratio shift bought with an accuracy drop is damage,
# not steering.
#
# READ THE BANDS BEFORE COMPARING -- they OVERLAP, they are not disjoint:
#   [11,18) -> decoder layers [10..16], L=7
#   [16,22) -> decoder layers [15..20], L=6
# Layers 15 and 16 are in BOTH. So a difference between them is "earlier band vs
# later band", NOT "two independent neuron sets"; check the per-layer Jaccard
# printed by check_mask_qwen.py --compare before wording any conclusion.
#
# ORDER:
#   bash run_nmd_qwen25.sh                      # build the [11,18) mask
#   python check_mask_qwen.py --layer_start 11 --layer_end 18 --compare 16 22
#   bash run_mmlue_qwen25.sh --new              # [11,18) alpha=0/+-4
#   bash run_mmlue_qwen25.sh --old              # [16,22) alpha=0/+-4, same protocol
#
# The [16,22) arm is re-run rather than cited from any stored number so both bands
# are measured under one protocol on one machine (bf16 is not bit-reproducible
# across GPUs -- keep both arms on the SAME card).
#
# Both arms include alpha=0. It costs one extra cell per arm and gives each band
# its own same-card baseline, so an E-ratio shift cannot be a baseline difference.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

ROLES="{task} expert,non {task} expert"
SUITE="default"
USE_E="--E"

if [ "$1" == "--new" ]; then
    LS=11; LE=18
    CONFIGS="0-${LS}-${LE} 4-${LS}-${LE} neg4-${LS}-${LE}"
    ANS_FILE="mmlue/qwen_nmd_11_18"
elif [ "$1" == "--new_fast" ]; then
    # FAST layer probe: +-4 only, reusing an EXISTING alpha=0 as the baseline.
    #
    # Legitimate ONLY because the reused alpha=0 is on the SAME machine (bf16 is
    # not bit-reproducible across GPUs -- the IGT split measured two same-config
    # alpha=0 cells differing by 0.100 on net_score, larger than most alpha
    # effects). If the baseline came from another card, this branch is invalid;
    # use --new instead.
    #
    # CAVEAT that survives the machine match: the reused baseline was taken with
    # the [16,22) mask. At alpha=0 the diff matrix is all-zero either way, so the
    # steering is identical -- but confirm the PROMPT/suite/roles/E-option match
    # too, or the baseline differs for a reason that has nothing to do with the
    # band. analyze_mmlue_bands.py compares each band to ITS OWN mdf_0, so point
    # --new at a dir that actually contains one.
    LS=11; LE=18
    CONFIGS="4-${LS}-${LE} neg4-${LS}-${LE}"
    ANS_FILE="mmlue/qwen_nmd_11_18"
elif [ "$1" == "--old" ]; then
    LS=16; LE=22
    CONFIGS="0-${LS}-${LE} 4-${LS}-${LE} neg4-${LS}-${LE}"
    ANS_FILE="mmlue/qwen_nmd_16_22"
else
    echo "Usage: bash run_mmlue_qwen25.sh --new | --new_fast | --old"
    echo "  --new_fast = +-4 only, reusing an existing SAME-MACHINE alpha=0."
    echo "               Copy that mdf_0 into the --new dir before analysing,"
    echo "               or the band has no baseline to be compared against."
    echo "  --new = [11,18) (build the mask first: bash run_nmd_qwen25.sh)"
    echo "  --old = [16,22) re-run under the same protocol, for comparison"
    echo "Run BOTH on the SAME GPU: bf16 is not bit-reproducible across cards,"
    echo "and the whole point is a band-vs-band contrast."
    exit 1
fi

echo "=================================================="
echo "MMLU-E — ${MODEL_NAME} ${MODEL_SIZE} | layers ${LS}-${LE} (exclusive end)"
echo "Configs: ${CONFIGS}"
echo "Output : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
echo "Start  : $(date)"
echo "=================================================="

cd ${WORK_DIR}

python get_answer_regenerate_logits.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --ans_file "${ANS_FILE}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    ${USE_E}

if [ $? -eq 0 ]; then
    echo ""
    echo "[Done] MMLU-E ${MODEL_NAME} layers ${LS}-${LE} — $(date)"
    echo "After BOTH arms, compare E-ratio and accuracy per alpha."
else
    echo "[FAILED] MMLU-E ${MODEL_NAME} layers ${LS}-${LE} — $(date)"
    exit 1
fi
