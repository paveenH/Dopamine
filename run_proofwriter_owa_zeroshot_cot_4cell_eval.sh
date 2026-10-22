#!/usr/bin/env bash
# Offline scoring of the four CoT cells (reads gold; no GPU). Own output file.
set -euo pipefail
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
RES="$WORK_DIR/proofwriter_owa/results/zs_chat_v1"
cd "$WORK_DIR"; mkdir -p "$RES"
L="$BASE_DIR/llama3/proofwriter_owa_zs"; Q="$BASE_DIR/qwen2.5/proofwriter_owa_zs"
"$PY" proofwriter_owa/eval_proofwriter_owa_zeroshot.py --condition cot \
  --manifest_blind "$BENCH/manifest_blind.json" --gold "$BENCH/manifest_gold.json" \
  --cells "$L/cot_mdf_0/proofwriter_owa_zs_8B_11_20.json" "$L/cot_mdf_neg6/proofwriter_owa_zs_8B_11_20.json" \
          "$Q/cot_mdf_0/proofwriter_owa_zs_7B_16_22.json" "$Q/cot_mdf_8/proofwriter_owa_zs_7B_16_22.json" \
  --out "$RES/eval_cot.json"
