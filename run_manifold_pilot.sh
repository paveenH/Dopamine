#!/usr/bin/env bash
# Manifold Plan section 3 -- fit the alpha=0 manifold, export coordinates.
#
# SERVER-side. PY defaults to `python`: the server conda env names its
# interpreter that, while `python3.10` is the LOCAL analysis-box convention and
# does NOT exist here -- it exits 127 before anything runs, and under nohup the
# job dies silently with an empty-looking log. Always `cat` the log right after
# launching. Use `python -u` under nohup or stdout is block-buffered and an
# empty log looks identical to a job that never started.
set -euo pipefail

PY="${PY:-python}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
RUN_TAG="${RUN_TAG:-phase1b_eot}"
TASK="${TASK:-gsm8k}"
SIZE="${SIZE:-8B}"
LAYER_START="${LAYER_START:-11}"
LAYER_END="${LAYER_END:-20}"
K_MAX="${K_MAX:-20}"
MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"

H5_DIR="${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/llama3/manifold/${RUN_TAG}}"
SPLIT="${SPLIT:-split_manifest.json}"

# The four PRIMARY pilot cells. The basis is fit on nocot (alpha=0) TRAIN only;
# the steered cells are projected onto it, never fitted.
CELLS="${CELLS:-nocot,nocot_aneg6,nocot_aneg8,nocot_a6}"

echo "[env] PY=$(command -v "$PY" || echo MISSING)"
"$PY" -c "import numpy, h5py, transformers" \
  || { echo "[FAIL] wrong env: numpy/h5py/transformers missing"; exit 1; }

[[ -f "$SPLIT" ]] || {
  echo "[FAIL] split manifest not found: $SPLIT"
  echo "       Generate it locally with RoleAnswer/manifold/split_manifest.py"
  echo "       and rsync it here. It is REQUIRED -- there is no default, so a"
  echo "       run can never silently fit on all questions."
  exit 1; }

echo "[run] h5_dir=$H5_DIR"
echo "[run] out_dir=$OUT_DIR"
echo "[run] cells=$CELLS  k_max=$K_MAX  band=[${LAYER_START},${LAYER_END})"

"$PY" -u manifold_fit.py \
  --h5_dir "$H5_DIR" \
  --split_manifest "$SPLIT" \
  --out_dir "$OUT_DIR" \
  --model_dir "$MODEL_DIR" \
  --task "$TASK" --size "$SIZE" \
  --base_cell nocot \
  --cells "$CELLS" \
  --layer_start "$LAYER_START" --layer_end "$LAYER_END" \
  --k_max "$K_MAX" \
  "$@"
