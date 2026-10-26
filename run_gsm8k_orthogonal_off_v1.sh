#!/usr/bin/env bash
# GSM8K off-support orthogonal injection controls (gsm8k-orthogonal-off-v1).
# Exploratory extension of the on-support check (gsm8k-orthogonal-v1). Prebuilt
# ortho_gauss_off seeds 1-3 only; masks are NEVER generated or changed here.
#   verify    read-only mask checks (off-support, norm, dense orthogonality, 3 distinct masks)
#   full      formal 300-item cells: seed1 seed2 seed3  -> $OUT/seedN
#   refs      OPTIONAL: re-run baseline, rsn and the three on-support seeds in THIS
#             environment -> $OUT/ref/*  (only when the old section-7 cells are not comparable)
#   status    validate completed cells (add "refs" as 3rd arg for the reference tree)
set -euo pipefail

MODE="${1:-}"
MODEL="${2:-}"
SCOPE="${3:-}"
case "$MODE" in verify|full|refs|status) ;; *)
  echo "usage: bash run_gsm8k_orthogonal_off_v1.sh verify-full-refs-or-status llama3-or-qwen2.5 [refs]" >&2; exit 2;;
esac
case "$MODEL" in llama3|qwen2.5) ;; *) echo "model must be llama3 or qwen2.5" >&2; exit 2;; esac

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="$WORK_DIR/components"
cd "$WORK_DIR"
PY="${PY:-python}"
PROTOCOL="gsm8k-orthogonal-off-v1"
if [[ "$MODEL" == "llama3" ]]; then
  MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"; SIZE="8B"; START=11; END=20; ALPHA=-6
  MASK_FOLDER="llama3_non_logits"; DENSE_FOLDER="llama3"; OUT="$BASE_DIR/llama3/gsm8k_orthogonal_off_v1"
else
  MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"; SIZE="7B"; START=16; END=22; ALPHA=8
  MASK_FOLDER="qwen2.5_non_logits"; DENSE_FOLDER="qwen2.5"; OUT="$BASE_DIR/qwen2.5/gsm8k_orthogonal_off_v1"
fi
RSN="$BASE_DIR/mask/$MASK_FOLDER/nmd_0.5_${START}_${END}_${SIZE}.npy"
DENSE="$BASE_DIR/hidden_states_mean/${DENSE_FOLDER}/mmlue_expert_mean_${SIZE}.npy"
NONE_DENSE="$BASE_DIR/hidden_states_mean/${DENSE_FOLDER}/mmlue_non_expert_mean_${SIZE}.npy"
DATA="$BASE_DIR/benchmark/gsm8k_test_sample.json"
MASK_DIR="$BASE_DIR/mask/$MASK_FOLDER"

# run_cell condition family root
run_cell() {
  local condition="$1" family="$2" root="$3" mask="$RSN" alpha="$ALPHA"
  if [[ "$condition" == "baseline" ]]; then alpha=0; fi
  if [[ "$condition" == seed* ]]; then
    mask="$MASK_DIR/ortho_gauss_${family}_0.5_${START}_${END}_${SIZE}_seed${condition#seed}.npy"
  fi
  "$PY" get_answer_gsm8k_orthogonal.py \
    --model "$MODEL" --model_dir "$MODEL_DIR" --size "$SIZE" --alpha "$alpha" \
    --condition "$condition" --control_family "$family" --protocol "$PROTOCOL" \
    --mask_path "$mask" --rsn_mask_path "$RSN" \
    --dense_mean_path "$DENSE" --none_dense_mean_path "$NONE_DENSE" --test_file "$DATA" --base_dir "$BASE_DIR" \
    --out_dir "$root/$condition" --start "$START" --end "$END" \
    --batch_size 24 --max_new_tokens 768
}

case "$MODE" in
  verify)
    for f in "$RSN" "$DENSE" "$NONE_DENSE" "$DATA"; do [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }; done
    "$PY" verify_gsm8k_orthogonal_masks.py --model "$MODEL" --base_dir "$BASE_DIR" --family off --seed all
    echo "[ok] Read-only off-support mask/data checks passed. No mask was generated or changed."
    ;;
  full)
    for c in seed1 seed2 seed3; do run_cell "$c" off "$OUT"; done ;;
  refs)
    for c in baseline rsn seed1 seed2 seed3; do run_cell "$c" same "$OUT/ref"; done ;;
  status)
    if [[ "$SCOPE" == "refs" ]]; then ROOT="$OUT/ref"; CELLS="baseline rsn seed1 seed2 seed3"; FAM=same
    else ROOT="$OUT"; CELLS="seed1 seed2 seed3"; FAM=off; fi
    "$PY" - "$ROOT" "$MODEL" "$ALPHA" "$START" "$END" "$FAM" "$PROTOCOL" $CELLS <<'PY'
import hashlib, json, pathlib, sys
root, model, alpha, start, end, fam, proto = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
failed = False
for condition in sys.argv[8:]:
    d = root / condition
    try:
        result_path, manifest_path = d / "result.json", d / "manifest.json"
        result, manifest = json.loads(result_path.read_text()), json.loads(manifest_path.read_text())
        rows, cfg, run = result["data"], manifest["configuration"], manifest["run"]
        expected_alpha = 0.0 if condition == "baseline" else alpha
        expected_fires = 0 if condition == "baseline" else (end-start)*300
        qids = [hashlib.sha256(r["question"].encode()).hexdigest() for r in rows]
        mv = cfg.get("mask_validation", {})
        valid = (len(rows) == 300 and len(set(qids)) == 300 and cfg["model"] == model and cfg["protocol"] == proto
                 and cfg["condition"] == condition and cfg["alpha"] == expected_alpha
                 and cfg.get("control_family") == fam
                 and cfg["start"] == start and cfg["end"] == end
                 and cfg["question_sha256"] == qids and cfg["question_order_sha256"] == hashlib.sha256("\n".join(qids).encode()).hexdigest()
                 and run["row_count"] == 300 and run["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
                 and run["steering_fires"] == run["expected_steering_fires"] == expected_fires
                 and (condition != "seed1" and condition != "seed2" and condition != "seed3" or mv.get("direction_kind") == "orthogonal")
                 and (fam != "off" or mv.get("support_kind") == "off"))
        if not valid: raise ValueError("content, provenance, row, support or fire checks failed")
        print(f"COMPLETE {d}")
    except Exception as exc:
        print(f"INVALID {d}: {exc}", file=sys.stderr)
        failed = True
sys.exit(1 if failed else 0)
PY
    ;;
esac
