#!/usr/bin/env bash
# MMLU-E off-support functional controls (mmlue-orthogonal-off-v1, baseline + off-support). NOT YET RUN.
# Prebuilt ortho_gauss_off seeds 1-3 only; masks are NEVER generated or changed here.
# Per model: ONE alpha=0 baseline (RSN mask x 0, both roles) + seeds 1-3 x alpha -4/+4 = 7 configs.
# RSN conditions are NOT run. Old baseline/RSN results are not used.
#   verify   read-only mask checks (off-support, norm, dense orthogonality, 3 distinct masks)
#   full     57 tasks, seeds 1-3, alpha -4/+4, Expert + Non-expert roles (one model load)
#   status   count completed cells (expect 7 configs x 57 tasks = 399)
set -euo pipefail
MODE="${1:-}"
MODEL="${2:-}"
case "$MODE" in verify|full|status) ;; *)
  echo "usage: bash run_mmlue_orthogonal_off_v1.sh verify-full-or-status llama3-or-qwen2.5" >&2; exit 2;;
esac
case "$MODEL" in llama3|qwen2.5) ;; *) echo "model must be llama3 or qwen2.5" >&2; exit 2;; esac

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="$WORK_DIR/components"
cd "$WORK_DIR"
PY="${PY:-python}"
if [[ "$MODEL" == "llama3" ]]; then
  MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"; SIZE="8B"; START=11; END=20; MASK_FOLDER="llama3_non_logits"
else
  MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"; SIZE="7B"; START=16; END=22; MASK_FOLDER="qwen2.5_non_logits"
fi
MASK_DIR="$BASE_DIR/mask/$MASK_FOLDER"
OUT_REL="mmlue_orthogonal_off_v1"
ROLES="{task} expert,non {task} expert"
RSN_MASK="$MASK_DIR/nmd_0.5_${START}_${END}_${SIZE}.npy"
mask() { echo "$MASK_DIR/ortho_gauss_off_0.5_${START}_${END}_${SIZE}_seed$1.npy"; }

run() {  # $1 = ans_file, rest = extra args
  local ans="$1"; shift
  "$PY" get_answer_regenerate_logits.py \
    --model "$MODEL" --model_dir "$MODEL_DIR" --hs "$MODEL" --size "$SIZE" --type non --percentage 0.5 \
    --configs neg4-${START}-${END} 4-${START}-${END} --ans_file "$ans" --suite default --E \
    --base_dir "$BASE_DIR" --roles "$ROLES" --tail_len 1 \
    --baseline_mask "$RSN_MASK" --mask_set seed1="$(mask 1)" seed2="$(mask 2)" seed3="$(mask 3)" "$@"
}

case "$MODE" in
  verify)
    [[ -f "$RSN_MASK" ]] || { echo "missing RSN mask (used x0 for baseline): $RSN_MASK" >&2; exit 1; }
    "$PY" verify_gsm8k_orthogonal_masks.py --model "$MODEL" --base_dir "$BASE_DIR" --family off --seed all
    echo "[ok] Read-only off-support mask checks passed. No mask was generated or changed." ;;
  full)  run "$OUT_REL" ;;
  status)
    n=$(find "$BASE_DIR/$MODEL/$OUT_REL" -name "*_answers_*.json" 2>/dev/null | wc -l)
    echo "$MODEL: $n / 399 cell files" ;;
esac
