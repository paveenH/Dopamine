#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
MODEL="${2:-}"
if [[ "$MODE" != "verify" && "$MODE" != "preflight" && "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "status" ]]; then
  echo "usage: bash run_gsm8k_orthogonal_v1.sh verify|preflight|smoke|full|status llama3|qwen2.5" >&2
  exit 2
fi
if [[ "$MODEL" != "llama3" && "$MODEL" != "qwen2.5" ]]; then
  echo "model must be llama3 or qwen2.5" >&2
  exit 2
fi

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="$WORK_DIR/components"
cd "$WORK_DIR"
PY="${PY:-python}"
if [[ "$MODEL" == "llama3" ]]; then
  MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"; SIZE="8B"; START=11; END=20; ALPHA=-6
  MASK_FOLDER="llama3_non_logits"; DENSE_FOLDER="llama3"; OUT="$BASE_DIR/llama3/gsm8k_orthogonal_v1"
else
  MODEL_DIR="Qwen/Qwen2.5-7B-Instruct"; SIZE="7B"; START=16; END=22; ALPHA=8
  MASK_FOLDER="qwen2.5_non_logits"; DENSE_FOLDER="qwen2.5"; OUT="$BASE_DIR/qwen2.5/gsm8k_orthogonal_v1"
fi
RSN="$BASE_DIR/mask/$MASK_FOLDER/nmd_0.5_${START}_${END}_${SIZE}.npy"
DENSE="$BASE_DIR/hidden_states_mean/${DENSE_FOLDER}/mmlue_expert_mean_${SIZE}.npy"
NONE_DENSE="$BASE_DIR/hidden_states_mean/${DENSE_FOLDER}/mmlue_non_expert_mean_${SIZE}.npy"
DATA="$BASE_DIR/benchmark/gsm8k_test_sample.json"
MASK_DIR="$BASE_DIR/mask/$MASK_FOLDER"
conditions=(baseline rsn seed1 seed2 seed3)

run_cell() {
  local condition="$1" root="$2" smoke_n="${3:-0}" mask="$RSN" alpha="$ALPHA"
  local -a extra=()
  if [[ "$condition" == "baseline" ]]; then alpha=0; fi
  if [[ "$condition" == seed* ]]; then
    local seed="${condition#seed}"
    mask="$MASK_DIR/ortho_gauss_same_0.5_${START}_${END}_${SIZE}_seed${seed}.npy"
  fi
  if [[ "$smoke_n" != "0" ]]; then extra=(--smoke_n "$smoke_n"); fi
  "$PY" get_answer_gsm8k_orthogonal.py \
    --model "$MODEL" --model_dir "$MODEL_DIR" --size "$SIZE" --alpha "$alpha" \
    --condition "$condition" --mask_path "$mask" --rsn_mask_path "$RSN" \
    --dense_mean_path "$DENSE" --test_file "$DATA" --base_dir "$BASE_DIR" \
    --none_dense_mean_path "$NONE_DENSE" \
    --out_dir "$root/$condition" --start "$START" --end "$END" \
    --batch_size 24 --max_new_tokens 768 "${extra[@]}"
}

case "$MODE" in
  verify)
    [[ -f "$RSN" ]] || { echo "missing RSN mask: $RSN" >&2; exit 1; }
    [[ -f "$DENSE" ]] || { echo "missing dense mean source: $DENSE" >&2; exit 1; }
    [[ -f "$DATA" ]] || { echo "missing GSM8K sample: $DATA" >&2; exit 1; }
    for seed in 1 2 3; do
      "$PY" verify_gsm8k_orthogonal_masks.py --model "$MODEL" --base_dir "$BASE_DIR" --seed "$seed"
    done
    echo "[ok] Read-only mask/data checks passed. No mask was generated or changed."
    ;;
  preflight)
    # One fixed item in every condition, isolated from both smoke and formal outputs.
    for condition in "${conditions[@]}"; do run_cell "$condition" "$OUT/preflight" 1; done
    ;;
  smoke)
    root="$OUT/smoke"
    for condition in "${conditions[@]}"; do run_cell "$condition" "$root" 4; done
    ;;
  full)
    root="$OUT"
    for condition in "${conditions[@]}"; do run_cell "$condition" "$root"; done
    ;;
  status)
    "$PY" - "$OUT" "$MODEL" "$ALPHA" "$START" "$END" <<'PY'
import hashlib, json, pathlib, sys
root, model, alpha, start, end = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
failed = False
for condition in ("baseline", "rsn", "seed1", "seed2", "seed3"):
    d = root / condition
    try:
        result_path, manifest_path = d / "result.json", d / "manifest.json"
        result, manifest = json.loads(result_path.read_text()), json.loads(manifest_path.read_text())
        rows, cfg, run = result["data"], manifest["configuration"], manifest["run"]
        expected_alpha = 0.0 if condition == "baseline" else alpha
        expected_fires = 0 if condition == "baseline" else (end-start)*300
        qids = [hashlib.sha256(r["question"].encode()).hexdigest() for r in rows]
        valid = (len(rows) == 300 and len(set(qids)) == 300 and cfg["model"] == model
                 and cfg["condition"] == condition and cfg["alpha"] == expected_alpha
                 and cfg["start"] == start and cfg["end"] == end
                 and cfg["question_sha256"] == qids and cfg["question_order_sha256"] == hashlib.sha256("\n".join(qids).encode()).hexdigest()
                 and run["row_count"] == 300 and run["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
                 and run["steering_fires"] == run["expected_steering_fires"] == expected_fires)
        if not valid: raise ValueError("content, provenance, row, or fire checks failed")
        print(f"COMPLETE {d}")
    except Exception as exc:
        print(f"INVALID {d}: {exc}", file=sys.stderr)
        failed = True
sys.exit(1 if failed else 0)
PY
    ;;
esac
