#!/usr/bin/env bash
# ProofWriter-OWA chat 0-shot CoT, four cells (protocol proofwriter-owa-zs-chat-v1,
# template proofwriter-owa-zs-cot-v1). Two GPUs in parallel, one process per model:
#   llama3  : alpha 0 then -6  on LLAMA_GPU
#   qwen2.5 : alpha 0 then +8  on QWEN_GPU
# No reverse cells, no gate, no tests. Reuses the multi-cell generator
# (`--cell a0 wp`), NOT the legacy `workpoint` stage (which also runs `rev`).
#   LLAMA_GPU=0 QWEN_GPU=1 bash run_proofwriter_owa_zeroshot_cot_4cell.sh
# Run detached:  nohup env LLAMA_GPU=0 QWEN_GPU=1 bash run_proofwriter_owa_zeroshot_cot_4cell.sh > zs_cot_4cell.log 2>&1 &
set -uo pipefail

LLAMA_GPU="${LLAMA_GPU:?set LLAMA_GPU to a GPU index}"
QWEN_GPU="${QWEN_GPU:?set QWEN_GPU to a GPU index}"
[[ "$LLAMA_GPU" != "$QWEN_GPU" ]] || { echo "[FATAL] LLAMA_GPU and QWEN_GPU must differ" >&2; exit 1; }

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
BLIND="$BENCH/manifest_blind.json"
GEN=proofwriter_owa/get_answer_proofwriter_owa_zeroshot.py
"$PY" -c "import numpy, torch, transformers" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch/transformers" >&2; exit 1; }
[[ -f "$BLIND" ]] || { echo "[FATAL] $BLIND not found" >&2; exit 1; }
cd "$WORK_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$LLAMA_GPU" nohup "$PY" "$GEN" --mode run --model llama3 --condition cot \
  --cell a0 wp --manifest "$BLIND" \
  --mask_path "$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy" --out_root "$BASE_DIR" \
  > "zs_cot_llama3_${TS}.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES="$QWEN_GPU" nohup "$PY" "$GEN" --mode run --model qwen2.5 --condition cot \
  --cell a0 wp --manifest "$BLIND" \
  --mask_path "$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy" --out_root "$BASE_DIR" \
  > "zs_cot_qwen25_${TS}.log" 2>&1 &
P2=$!
echo "llama3 pid=$P1 (GPU $LLAMA_GPU) log=zs_cot_llama3_${TS}.log"
echo "qwen2.5 pid=$P2 (GPU $QWEN_GPU) log=zs_cot_qwen25_${TS}.log"

wait "$P1"; R1=$?
wait "$P2"; R2=$?
echo "llama3 exit=$R1  qwen2.5 exit=$R2"
if [[ $R1 -ne 0 || $R2 -ne 0 ]]; then echo "[FATAL] at least one model failed" >&2; exit 1; fi
echo "done. Next: bash run_proofwriter_owa_zeroshot_cot_4cell_eval.sh"
