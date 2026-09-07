#!/usr/bin/env bash
# FinQA preflight 评分（读 gold 的唯一脚本）。纯 CPU，不需要 GPU/CUDA_VISIBLE_DEVICES。
#
#   bash finqa/run_finqa_eval_preflight.sh
#
# 依赖两模型的 alpha=0 preflight 生成结果已存在（run_finqa.sh ... PREFLIGHT 跑完）。
set -euo pipefail

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"

cd "$WORK_DIR"
mkdir -p docs

"$PY" finqa/eval_finqa.py --preflight \
  --generations "$BASE_DIR/llama3/finqa/preflight/mdf_0/finqa_8B_11_20.json" \
  --gold_file "$BENCH/finqa_formal.json" \
  --out docs/finqa_preflight_llama3.json

"$PY" finqa/eval_finqa.py --preflight \
  --generations "$BASE_DIR/qwen2.5/finqa/preflight/mdf_0/finqa_7B_16_22.json" \
  --gold_file "$BENCH/finqa_formal.json" \
  --out docs/finqa_preflight_qwen2.5.json

echo "[finqa] done. see docs/finqa_preflight_llama3.json and docs/finqa_preflight_qwen2.5.json"
