#!/usr/bin/env bash
# ============================================================================
# Qwen2.5 RSN READOUT-SPECIFICITY null: re-project the SEVEN collected H5 cells
# onto random / orthogonal directions. NO GPU, NO re-collection.
#
# *** WHAT THIS IS AND IS NOT -- read before citing anything it produces. ***
# These hidden states were generated under RSN steering. Re-projecting them onto
# a different direction is a READOUT-SPECIFICITY control: it asks whether the
# per-layer response structure (step 4: CV 0.92/1.19, L20 sign-reversed) is
# peculiar to the RSN direction or would appear under ANY sparse direction.
# It is NOT a causal control on the steering direction. Claiming the steering
# DIRECTION is causally specific requires separately INJECTING random/orthogonal
# directions and re-collecting -- a different experiment, not this one.
#
# WHY A SEPARATE LAUNCHER: run_random_null.sh and run_generic_null.sh are
# Llama-frozen (band 11-20, llama3 mask dir, RUN_TAG=phase1b_eot). Same reason
# run_track_hidden_states_qwen25.sh is separate. Do not parameterise those.
#
# PRIMARY STATISTIC (fixed before the nulls are read): the scalar-compression
# residual, i.e. 1 - R^2 of the least-squares fit v_high ~ k*v_low across the
# six layers. On the RSN direction that reads k=0.309, residual 2.6%, so the
# high-dose response is very nearly the low-dose profile uniformly shrunk. CV
# and the layer x dose F ratio are DESCRIPTIVE companions -- F has no valid
# p-value here (the six layers are re-projections of one hidden state).
#
# Three null families, mirroring the Llama section-4.6 control matrix:
#   diff_random       support-selection null (random positions, real diff values)
#   ortho_gauss_same  NMD's own positions, Gaussian weights _|_ role-diff
#   ortho_gauss_off   positions DISJOINT from NMD, same _|_ + norm-match
# Read them as a control MATRIX, not a clean factorial. The off-support cell is
# load-bearing: positions moved OUT of NMD support AND weights _|_ role-diff.
#
# Steps: CHECK | NMD_VERIFY | DIFF_RANDOM | ORTHO | ALL
# ============================================================================
set -euo pipefail

BASE_DIR=/data1/paveen/Dopamine/components
RUN_TAG="${RUN_TAG:-qwen25_signal_v1}"
H5_DIR=$BASE_DIR/hidden_states/gsm8k/$RUN_TAG
MASK_DIR=$BASE_DIR/mask/qwen2.5_non_logits

MODEL=qwen2.5
SIZE=7B
TYPE=non
HS=qwen2.5
PCT=0.5
# Qwen band. NOT Llama's 11-20. Exclusive end -> decoder_layer_range(16,22)
# = range(15,21), L=6. Must match the mask the H5 were collected under.
LS=16
LE=22
EMA=0.95

NMD_MASK="$MASK_DIR/nmd_${PCT}_${LS}_${LE}_${SIZE}.npy"
# The frozen mask's md5, verified 2026-08-24 (mtime 2026-07-28, i.e. before the
# H5 were collected; exactly one copy exists filesystem-wide).
NMD_MD5_EXPECTED=59f5695533045b9be34a2510946890e5

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"

# Server convention: the conda env names its interpreter `python`. `python3.10`
# is the LOCAL analysis-box convention and exits 127 here, which under nohup
# looks like a job that silently died.
PY="${PY:-python}"

STEP="${1:-}"
case "$STEP" in
  CHECK|NMD_VERIFY|DIFF_RANDOM|ORTHO|ALL) ;;
  *) echo "usage: bash $0 {CHECK|NMD_VERIFY|DIFF_RANDOM|ORTHO|ALL}"; exit 2 ;;
esac

cd /data1/paveen/Dopamine

echo "=================================================="
echo "Qwen2.5 null remask | step=$STEP | RUN_TAG=$RUN_TAG"
echo "  band $LS-$LE (L=6) | seeds: $SEEDS"
echo "  H5_DIR=$H5_DIR"
echo "=================================================="

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[x] interpreter '$PY' not found. Override with PY=<interpreter>."
  exit 127
fi
"$PY" - <<'EOF' || { echo "[x] interpreter cannot import numpy/h5py"; exit 127; }
import numpy, h5py
print(f"[ok] deps: numpy {numpy.__version__}, h5py {h5py.__version__}")
EOF

# --- pre-flight --------------------------------------------------------------
n_h5=$(ls "$H5_DIR"/hs_*.h5 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_h5" != "7" ]; then
  echo "[x] expected 7 H5 cells in $H5_DIR, found $n_h5"
  echo "    extract_signal_json_remask.py GLOBS this dir, so a wrong count means"
  echo "    the null would cover a different cell set than the RSN result."
  exit 2
fi
echo "[ok] $n_h5 H5 cells"

if [ ! -f "$NMD_MASK" ]; then
  echo "[x] NMD mask not found: $NMD_MASK"
  exit 2
fi
got=$(md5sum "$NMD_MASK" | awk '{print $1}')
if [ "$got" != "$NMD_MD5_EXPECTED" ]; then
  echo "[x] NMD mask md5 $got != expected $NMD_MD5_EXPECTED"
  echo "    The collected H5 were steered with the expected mask. A different"
  echo "    mask here means the null and the RSN result do not share a basis."
  exit 2
fi
echo "[ok] NMD mask md5 matches the frozen value"

if [ "$STEP" = "CHECK" ]; then
  echo
  echo "Disk estimate: ~7 JSON per draw, few MB each; 30 draws -> well under 1 GB."
  df -h "$BASE_DIR" 2>/dev/null | tail -1
  echo
  echo "[ok] CHECK passed. Run NMD_VERIFY next -- it is the load-bearing check."
  exit 0
fi

# --- NMD_VERIFY: the null masks are only comparable if the builder, run on
# --- THIS machine with THESE role-diff files, reproduces the frozen NMD mask.
# --- A mismatch means the null draws are built from a different basis than the
# --- direction whose effect they are supposed to bound.
if [ "$STEP" = "NMD_VERIFY" ] || [ "$STEP" = "ALL" ]; then
  echo
  echo "--- NMD_VERIFY: rebuild the NMD mask and diff against the frozen file ---"
  TMPDIR_V=$(mktemp -d)
  trap 'rm -rf "$TMPDIR_V"' EXIT
  mkdir -p "$TMPDIR_V/mask/${MODEL}_${TYPE}_logits"
  # nmd.py reads hidden_states_mean/ from --base_dir, so symlink the real one in
  # and let it write its mask into the scratch tree.
  ln -s "$BASE_DIR/hidden_states_mean" "$TMPDIR_V/hidden_states_mean"
  "$PY" detection/nmd.py \
    --mask_type nmd --logits \
    --model "$MODEL" --size "$SIZE" --type "$TYPE" --hs "$HS" \
    --percentage "$PCT" --start_layer "$LS" --end_layer "$LE" \
    --base_dir "$TMPDIR_V"
  REBUILT="$TMPDIR_V/mask/${MODEL}_${TYPE}_logits/nmd_${PCT}_${LS}_${LE}_${SIZE}.npy"
  "$PY" - "$REBUILT" "$NMD_MASK" <<'EOF'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
if a.shape != b.shape:
    raise SystemExit(f"[x] shape {a.shape} != frozen {b.shape}")
if not np.array_equal(a, b):
    d = int((a != b).sum())
    raise SystemExit(
        f"[x] rebuilt NMD mask differs from the frozen file in {d} entries.\n"
        "    The null draws would be built from a different role-diff basis\n"
        "    than the direction the H5 were steered with. STOP.")
print(f"[ok] rebuilt NMD mask is byte-identical to the frozen file {a.shape}")
EOF
  rm -rf "$TMPDIR_V"; trap - EXIT
fi

build_and_project () {
  local fam="$1" seed="$2" out_root="$3"
  echo "==================== $fam seed=$seed ===================="
  "$PY" detection/nmd.py \
    --mask_type "$fam" --seed "$seed" --logits \
    --model "$MODEL" --size "$SIZE" --type "$TYPE" --hs "$HS" \
    --percentage "$PCT" --start_layer "$LS" --end_layer "$LE" \
    --base_dir "$BASE_DIR"

  # nmd.py tags the seed for ortho_gauss_* always, and for diff_random when
  # seed != 42. SEEDS never contains 42, so both families are tagged here.
  local mask_path="$MASK_DIR/${fam}_${PCT}_${LS}_${LE}_${SIZE}_seed${seed}.npy"
  [ -f "$mask_path" ] || { echo "[x] mask not written: $mask_path"; exit 1; }

  "$PY" extract_signal_json_remask.py \
    --h5_dir "$H5_DIR" \
    --mask_path "$mask_path" \
    --out_dir "$out_root/seed${seed}" \
    --out_prefix random_signal \
    --layer_start "$LS" --layer_end "$LE" --ema_alpha "$EMA"

  local n_out
  n_out=$(ls "$out_root/seed${seed}"/random_signal_*.json 2>/dev/null | wc -l | tr -d ' ')
  [ "$n_out" = "7" ] || { echo "[x] $fam seed=$seed wrote $n_out JSON, expected 7"; exit 1; }
  echo "[ok] $fam seed=$seed -> 7 JSON"
}

if [ "$STEP" = "DIFF_RANDOM" ] || [ "$STEP" = "ALL" ]; then
  for s in $SEEDS; do
    build_and_project diff_random "$s" "$BASE_DIR/qwen2.5_random_null"
  done
fi

if [ "$STEP" = "ORTHO" ] || [ "$STEP" = "ALL" ]; then
  for s in $SEEDS; do
    build_and_project ortho_gauss_same "$s" "$BASE_DIR/qwen2.5_ortho_same"
  done
  for s in $SEEDS; do
    build_and_project ortho_gauss_off "$s" "$BASE_DIR/qwen2.5_ortho_off"
  done
fi

echo
echo "=================================================="
echo "[ok] step $STEP complete."
echo "Null JSON live in qwen2.5_{random_null,ortho_same,ortho_off}/seed*/"
echo "These are READOUT-specificity controls on RSN-steered states,"
echo "NOT causal controls on the steering direction."
echo "=================================================="
