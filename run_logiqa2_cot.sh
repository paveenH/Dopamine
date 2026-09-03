#!/usr/bin/env bash
# LogiQA 2.0 explicit-CoT follow-up (cot-transfer-followup-v0).
#
# 300 items x 4 cells for ONE model at max_new_tokens=1024 (inherited from
# the P4 stage-1 freeze). Generates only -- accuracy is computed afterwards.
# POST-HOC EXPLORATORY FOLLOW-UP, NOT a replication of the frozen No-CoT
# logiqa2-p4-v0 result.
#
#   bash run_logiqa2_cot.sh llama3
#   bash run_logiqa2_cot.sh qwen2.5
#
# CELLS ARE NOT REQUIRED TO SHARE A GPU (project-wide convention -- GPU
# assignment is provenance, not a constraint). host and CUDA_VISIBLE_DEVICES
# are recorded per generation file; pairing is by sample_id only, never by
# hardware identity. Because bf16 greedy may vary across hardware, a contrast
# between cells generated on different devices is reported as a CROSS-RUN
# pairing.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_logiqa2_cot.sh llama3|qwen2.5" >&2; exit 1
fi
MODEL="$1"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/logiqa2/cot_followup}"

"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the interpreter" >&2
  echo "        is 'python', not 'python3.10'." >&2; exit 1; }

case "$MODEL" in
  llama3)
    SIZE=8B
    MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
    CONFIGS=(0-11-20 neg6-11-20 neg4-11-20 4-11-20) ;;
  qwen2.5)
    SIZE=7B
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
    CONFIGS=(0-16-22 8-16-22 6-16-22 neg6-16-22) ;;
  *) echo "[FATAL] unknown model '$MODEL'" >&2; exit 1 ;;
esac

# generation reads the BLIND copy -- same P4 firewall, no new sample
FORMAL_BLIND="${FORMAL_BLIND:-$BENCH/logiqa2_p4_formal_blind.json}"
FORMAL_FILE="${FORMAL_FILE:-$BENCH/logiqa2_p4_formal.json}"
for f in "$MASK" "$FORMAL_BLIND"; do
  if [[ ! -f "$f" ]]; then echo "[FATAL] not found: $f" >&2; exit 1; fi
done
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

mkdir -p "$OUT_DIR"
echo "[cot-followup] LogiQA 2.0 model=$MODEL configs=${CONFIGS[*]}"
echo "[cot-followup] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[cot-followup] budget 1024 (inherited from P4 stage-1), 300 items, 4 cells"
echo "[cot-followup] POST-HOC EXPLORATORY -- does not replace the frozen No-CoT result"

"$PY" get_answer_logiqa2_cot.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" --mask "$MASK" \
  --configs "${CONFIGS[@]}" \
  --formal_file "$FORMAL_BLIND" \
  --out "$OUT_DIR/cot_${MODEL}.json"

echo "[cot-followup] done. After BOTH models finish, score with the shared eval script"
echo "     (NOTE: --nocot_generations needs the No-CoT alpha=0 AND workpoint"
echo "     generation files for this model -- e.g. formal_llama3.json holds"
echo "     both cells in one file, per get_answer_logiqa2.py's output shape):"
echo "     $PY eval_cot_transfer_followup.py --task logiqa2 \\"
echo "       --generations $OUT_DIR/cot_llama3.json $OUT_DIR/cot_qwen2.5.json \\"
echo "       --formal_file $FORMAL_FILE \\"
echo "       --nocot_generations \\"
echo "         $BASE_DIR/logiqa2/formal/formal_llama3.json \\"
echo "         $BASE_DIR/logiqa2/formal/formal_qwen2.5.json \\"
echo "       --nocot_evaluation docs/p4_logiqa2_evaluation.json \\"
echo "       --out docs/cot_followup_logiqa2_evaluation.json"
