#!/usr/bin/env bash
# RSN direction-specificity: build GENERIC-DIRECTION null distributions.
#
# Two families, 10 seeds each (orthogonal Gaussian, norm-matched, per-layer ⊥ Δ_l):
#   ortho_gauss_same : NMD's own 20 positions/layer, Gaussian weights ⊥ role-diff
#   ortho_gauss_off  : 20 random positions/layer DISJOINT from NMD, same ⊥ + norm-match
#
# For each (family, seed): build the mask (nmd.py, with acceptance guards) then
# offline re-project ONLY the 7 frozen-analysis conditions (zero-GPU).
#
# This upgrades §4.6 from a support-selection null (diff_random) toward a
# direction-specificity test. Run on the server (needs the HDF5). No GPU.
set -euo pipefail

BASE_DIR=/data1/paveen/Dopamine/components
H5_DIR=$BASE_DIR/hidden_states/gsm8k/phase1b_eot
MASK_DIR=$BASE_DIR/mask/llama3_non_logits

MODEL=llama3
SIZE=8B
TYPE=non
HS=llama3
PCT=0.5
LS=11
LE=20
EMA=0.95

SEEDS="1 2 3 4 5 6 7 8 9 10"
FAMILIES="ortho_gauss_same ortho_gauss_off"

# extract_signal_json_remask.py re-projects every H5 it finds by globbing $H5_DIR,
# so which conditions get re-projected is whatever lives in that dir (the 7 frozen
# conditions the analysis reads: nocot/cot + expert/non_expert + aneg4/aneg6/a4).

for fam in $FAMILIES; do
  case "$fam" in
    ortho_gauss_same) OUT_ROOT=$BASE_DIR/llama3_ortho_same ;;
    ortho_gauss_off)  OUT_ROOT=$BASE_DIR/llama3_ortho_off  ;;
  esac
  for s in $SEEDS; do
    echo "==================== $fam  seed=$s ===================="
    # 1) build the mask (guards inside nmd.py assert ⊥, norm-match, support)
    python detection/nmd.py \
      --mask_type "$fam" --seed "$s" --logits \
      --model "$MODEL" --size "$SIZE" --type "$TYPE" --hs "$HS" \
      --percentage "$PCT" --start_layer "$LS" --end_layer "$LE" \
      --base_dir "$BASE_DIR"

    MASK_PATH="$MASK_DIR/${fam}_${PCT}_${LS}_${LE}_${SIZE}_seed${s}.npy"

    # 2) re-project the frozen conditions against this mask
    python extract_signal_json_remask.py \
      --h5_dir "$H5_DIR" \
      --mask_path "$MASK_PATH" \
      --out_dir "$OUT_ROOT/seed${s}" \
      --out_prefix random_signal \
      --layer_start "$LS" --layer_end "$LE" --ema_alpha "$EMA"
  done
done

echo "DONE."
echo "  same-support null -> $BASE_DIR/llama3_ortho_same/seed{1..10}/"
echo "  off-support  null -> $BASE_DIR/llama3_ortho_off/seed{1..10}/"
echo "Pull both trees to the local analysis workspace, then run:"
echo "  python3.10 analyze_rsn_specificity.py --null_family ortho_gauss_same --null_root llama3/dopamine/ortho_same"
echo "  python3.10 analyze_rsn_specificity.py --null_family ortho_gauss_off  --null_root llama3/dopamine/ortho_off"
