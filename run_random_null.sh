#!/usr/bin/env bash
# RSN direction-specificity: build a diff_random NULL DISTRIBUTION.
#
# For each seed in SEEDS:
#   1) build a diff_random mask (random positions, values from the true role-diff)
#   2) offline re-project every gsm8k HDF5 against that mask (zero-GPU)
#      -> one full set of random_signal_*.json per seed
#
# The existing seed=42 draw (diff_random_0.5_11_20_8B.npy + signal/random_signal_*)
# is already the N=1 sample; these SEEDS extend it to a proper null.
#
# Run on the server (needs the HDF5 under $H5_DIR). No GPU required.
set -euo pipefail

BASE_DIR=/data1/paveen/Dopamine/components
H5_DIR=$BASE_DIR/hidden_states/gsm8k/phase1b_eot
MASK_DIR=$BASE_DIR/mask/llama3_non_logits
OUT_ROOT=$BASE_DIR/llama3_random_null

MODEL=llama3
SIZE=8B
TYPE=non
HS=llama3
PCT=0.5
LS=11
LE=20
EMA=0.95

SEEDS="1 2 3 4 5 6 7 8 9 10"

for s in $SEEDS; do
  echo "==================== seed=$s ===================="
  # 1) build the mask (nmd.py appends _seed$s when seed != 42)
  python detection/nmd.py \
    --mask_type diff_random --seed "$s" \
    --model "$MODEL" --size "$SIZE" --type "$TYPE" --hs "$HS" \
    --percentage "$PCT" --start_layer "$LS" --end_layer "$LE" \
    --base_dir "$BASE_DIR"

  MASK_PATH="$MASK_DIR/diff_random_${PCT}_${LS}_${LE}_${SIZE}_seed${s}.npy"

  # 2) re-project every HDF5 against this mask
  python extract_signal_json_remask.py \
    --h5_dir "$H5_DIR" \
    --mask_path "$MASK_PATH" \
    --out_dir "$OUT_ROOT/seed${s}" \
    --out_prefix random_signal \
    --layer_start "$LS" --layer_end "$LE" --ema_alpha "$EMA"
done

echo "DONE. Null sets under $OUT_ROOT/seed{1..10}/"
