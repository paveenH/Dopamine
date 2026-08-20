#!/bin/bash
# NMD mask generation — Qwen2.5-7B-Instruct.
#
# SEPARATE from run_nmd.sh on purpose: that script is Mistral3-configured and
# still points at the stale WORK_DIR=/data1/paveen/RolePlaying tree. Editing it
# would move a path other experiments may still resolve.
#
# WHY [11,18): the existing Qwen mask is [16,22), chosen as the onset of the
# layer-wise Expert/Non-Expert Pearson descent (same criterion as Llama's 11-20).
# This generates a SECOND, EARLIER band so the layer position itself can be
# tested rather than assumed. It does NOT replace the [16,22) mask -- both files
# coexist and every stored [16,22) result stays valid.
#
# Reads pre-computed mean hidden states
#   components/hidden_states_mean/qwen2.5_non_logits/{diff_mean,none_diff_mean}_7B.npy
# so NO hidden-state re-extraction is needed; the [16,22) mask came from the same
# two files. Output:
#   components/mask/qwen2.5_non_logits/nmd_0.5_11_18_7B.npy
#
# --logits is REQUIRED: it selects the *_logits output directory, which is where
# the existing Qwen masks live. Without it the file lands in a sibling dir the
# runners do not read (see CLAUDE.md, detection/nmd.py gotcha).
#
# Usage:  bash run_nmd_qwen25.sh

DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"

MODEL="qwen2.5"
SIZE="7B"
TYPE="non"
HS_PREFIX="qwen2.5"
PERCENTAGE=0.5
MASK_TYPE="nmd"
SEED=42

# Qwen2.5-7B has 28 decoder layers. Ranges are HF hidden_states semantics
# (0 = embedding), half-open, and the saved mask drops the embedding row.
LAYER_CONFIGS=(
    "11 18"
)

cd ${WORK_DIR}/detection

for CFG in "${LAYER_CONFIGS[@]}"; do
    read -r START_LAYER END_LAYER <<< "${CFG}"
    echo "=================================================="
    echo "Generating ${MASK_TYPE} mask — ${MODEL} ${SIZE}"
    echo "Layer range: [${START_LAYER}, ${END_LAYER})"
    echo "Percentage : ${PERCENTAGE}%"
    echo "=================================================="

    python nmd.py \
        --model "${MODEL}" \
        --size "${SIZE}" \
        --type "${TYPE}" \
        --hs "${HS_PREFIX}" \
        --percentage "${PERCENTAGE}" \
        --start_layer "${START_LAYER}" \
        --end_layer "${END_LAYER}" \
        --mask_type "${MASK_TYPE}" \
        --seed "${SEED}" \
        --base_dir "${WORK_DIR}/components" \
        --logits || exit 1

    echo "[Done] mask for [${START_LAYER}, ${END_LAYER})"
done

echo ""
echo "=================================================="
echo "VERIFY before using the mask (layer alignment is the classic bug):"
echo "  python check_mask_qwen.py --layer_start 11 --layer_end 18"
echo "=================================================="
