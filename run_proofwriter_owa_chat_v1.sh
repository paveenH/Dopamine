#!/usr/bin/env bash
# ProofWriter OWA, {llama3|qwen2.5}: FORMAL four-point steering sweep under
# the HF chat-template interface condition.
#   llama3   -> protocol proofwriter-owa-chat-v1
#   qwen2.5  -> protocol proofwriter-owa-chat-v11
# (Qwen support added 2026-09-08 by parameterizing this launcher over an
# optional <model> argument; `sweep`/`eval` with NO model argument default
# to llama3 and are byte-identical to this launcher's original behavior.)
#
# INDEPENDENT INTERFACE-CONDITION EXPERIMENT -- does NOT redefine either
# model's ProofWriter-OWA workpoint (bare-string sweep, COMPLETE + CLOSED
# 2026-09-05, proofwriter_owa/results/formal_sweep_v2.json) and does NOT
# overwrite or reuse any bare v2 result. Llama3's 30-item alpha=0 diagnostic
# (chat_v2_mdf_0, see diag_chat_template.py) and formal 4-point chat sweep
# (formal_sweep_chat_v1.json) showed HF chat-template wrapping sharply
# reduces llama3's no_answer_rate/loop_rate/truncation_rate, but no
# Holm-significant workpoint after that interface confound is controlled
# for. Qwen's bare-string sweep did not show llama3's severe pathology; this
# Qwen chat sweep asks the same narrower question under Qwen's own chat
# template and dose set.
#
# Held IDENTICAL to each model's own bare formal sweep: the SAME 300-item
# manifest and order, the SAME v2 1-shot Unknown exemplar, the SAME prompt
# body and "#### <Label>" marker convention, the SAME mask file and band,
# the SAME alpha set, greedy/temperature=0, max_new_tokens=1024,
# batch_size=8, prefill-only tail=1. The ONLY variable held across all four
# alpha of one model is the prompt wrapping (bare-string vs
# apply_chat_template).
#
# No other benchmark, no steering-search beyond each model's own frozen
# four-point set.
#
# TWO stages, each taking an optional model argument (default llama3):
#   sweep [llama3|qwen2.5]   run all four alpha (one call, all four via
#                            --configs), on ONE GPU (bf16 greedy is not
#                            byte-reproducible across cards, and this
#                            curve's four cells must be directly comparable)
#   eval  [llama3|qwen2.5]   score that model's four cells (the ONLY script
#                            that reads gold), via eval_proofwriter_owa.py
#                            --protocol <that model's chat protocol> (the
#                            one flag added to that script for this purpose;
#                            parser/scoring/statistics are otherwise
#                            unchanged)
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_proofwriter_owa_chat_v1.sh sweep llama3 \
#     > proofwriter_owa_chat_v1_sweep_llama3.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_proofwriter_owa_chat_v1.sh sweep qwen2.5 \
#     > proofwriter_owa_chat_v1_sweep_qwen25.log 2>&1 &
#
#   bash run_proofwriter_owa_chat_v1.sh eval llama3   # after that model's sweep finishes
#   bash run_proofwriter_owa_chat_v1.sh eval qwen2.5  # no GPU needed for eval
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_proofwriter_owa_chat_v1.sh sweep [llama3|qwen2.5]" >&2
  echo "       run_proofwriter_owa_chat_v1.sh eval  [llama3|qwen2.5]" >&2
  exit 1
fi
STAGE="$1"
MODEL="${2:-llama3}"

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
PW_DIR="$WORK_DIR/proofwriter_owa"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR}"

# a wrong PY exits 127 before anything runs and the nohup log looks empty
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

EXEMPLAR_FILE="$PW_DIR/exemplar_unknown_v2.json"
[[ -f "$EXEMPLAR_FILE" ]] || {
  echo "[FATAL] $EXEMPLAR_FILE not found." >&2; exit 1; }

MANIFEST_BLIND="$BENCH/manifest_blind.json"
MANIFEST_GOLD="$BENCH/manifest_gold.json"

# Per-model config: SAME mask/band/alpha-set/protocol as that model's own
# bare formal sweep -- deliberately not re-derived here.
case "$MODEL" in
  llama3)
    MODEL_DIR_HF="meta-llama/Llama-3.1-8B-Instruct"
    SIZE="8B"
    MASK="$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
    SWEEP_CONFIGS="neg6-11-20 neg4-11-20 0-11-20 4-11-20"
    LAYERS_SUFFIX="11_20"
    PROTOCOL="proofwriter-owa-chat-v1"
    OUT_DIR="$OUT_ROOT/llama3/proofwriter_owa"
    ;;
  qwen2.5)
    MODEL_DIR_HF="Qwen/Qwen2.5-7B-Instruct"
    SIZE="7B"
    MASK="$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy"
    SWEEP_CONFIGS="neg6-16-22 0-16-22 6-16-22 8-16-22"
    LAYERS_SUFFIX="16_22"
    PROTOCOL="proofwriter-owa-chat-v11"
    OUT_DIR="$OUT_ROOT/qwen2.5/proofwriter_owa"
    ;;
  *)
    echo "[FATAL] unknown model '$MODEL' (llama3 | qwen2.5)" >&2
    exit 1
    ;;
esac

CHAT_A0="$OUT_DIR/formal_chat_v1_mdf_0/proofwriter_owa_${SIZE}_${LAYERS_SUFFIX}.json"
CHAT_AP="$OUT_DIR/formal_chat_v1_mdf_4/proofwriter_owa_${SIZE}_${LAYERS_SUFFIX}.json"
CHAT_AN="$OUT_DIR/formal_chat_v1_mdf_neg6/proofwriter_owa_${SIZE}_${LAYERS_SUFFIX}.json"
if [[ "$MODEL" == "llama3" ]]; then
  CHAT_AN2="$OUT_DIR/formal_chat_v1_mdf_neg4/proofwriter_owa_${SIZE}_${LAYERS_SUFFIX}.json"
  CELL_FILES=("$CHAT_A0" "$CHAT_AN2" "$CHAT_AN" "$CHAT_AP")
else
  CHAT_AP2="$OUT_DIR/formal_chat_v1_mdf_6/proofwriter_owa_${SIZE}_${LAYERS_SUFFIX}.json"
  CELL_FILES=("$CHAT_A0" "$CHAT_AN" "$CHAT_AP2" "$CHAT_AP")
fi
MODEL_TAG="${MODEL/./}"
CHAT_EVAL_OUT="$PW_DIR/results/formal_sweep_chat_v1_${MODEL_TAG}.json"

case "$STAGE" in
  sweep)
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card" >&2
      echo "        (this model's whole 4-point alpha curve must stay on" >&2
      echo "        one machine; an unpinned run risks mixing device" >&2
      echo "        differences into the alpha effect)." >&2
      exit 1
    fi
    [[ -f "$MANIFEST_BLIND" ]] || {
      echo "[FATAL] $MANIFEST_BLIND not found; run the v0/v1" >&2
      echo "        'validate-data' stage of run_proofwriter_owa.sh first" >&2
      echo "        (it builds the full 300-item manifest_blind.json /" >&2
      echo "        manifest_gold.json this launcher reuses unchanged)." >&2
      exit 1; }
    echo "[proofwriter-owa-chat] $MODEL: FORMAL 4-point sweep on the full"
    echo "  300-item manifest, under the HF chat-template interface"
    echo "  condition (protocol $PROTOCOL). max_new_tokens is FROZEN at"
    echo "  1024 (imported from get_answer_proofwriter_owa.py's"
    echo "  MAX_NEW_TOKENS_FROZEN, never a separately hardcoded literal)."
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa_chat.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR_HF" \
      --manifest "$MANIFEST_BLIND" --mask_path "$MASK" \
      --configs $SWEEP_CONFIGS \
      --out_dir "$OUT_DIR" \
      --exemplar_file "$EXEMPLAR_FILE" \
      --batch_size 8
    echo "[proofwriter-owa-chat] $MODEL sweep done. Alpha=0 file:"
    echo "    $CHAT_A0"
    echo
    echo "[proofwriter-owa-chat] next:"
    echo "    bash run_proofwriter_owa_chat_v1.sh eval $MODEL"
    ;;

  eval)
    for f in "${CELL_FILES[@]}"; do
      [[ -f "$f" ]] || {
        echo "[FATAL] $f not found; run 'sweep $MODEL' first (all four" >&2
        echo "        alpha come from ONE 'sweep' call -- --configs" >&2
        echo "        already lists all four)." >&2
        exit 1; }
    done
    if [[ -f "$CHAT_EVAL_OUT" ]]; then
      echo "[FATAL] $CHAT_EVAL_OUT exists; refusing to overwrite." >&2
      echo "        (eval_proofwriter_owa.py itself also refuses --out" >&2
      echo "        overwrite; this check just fails earlier/clearer.)" >&2
      exit 1
    fi
    cd "$WORK_DIR"
    echo "[proofwriter-owa-chat] scoring $MODEL's 4-point chat-interface"
    echo "  sweep (protocol $PROTOCOL; Holm(m=3), $MODEL's own alpha=0 as"
    echo "  baseline; no comparison to the bare sweep or to the other"
    echo "  model is computed here -- that is a separate, descriptive"
    echo "  side-by-side table, not a paired significance test)."
    "$PY" proofwriter_owa/eval_proofwriter_owa.py \
      --protocol "$PROTOCOL" \
      --gold "$MANIFEST_GOLD" \
      --generations "${CELL_FILES[@]}" \
      --out "$CHAT_EVAL_OUT"
    echo
    echo "[proofwriter-owa-chat] wrote $CHAT_EVAL_OUT"
    echo "  Read results.$MODEL.cells.<alpha> for accuracy (FIRST-answer,"
    echo "  main) / sensitivity_last_answer_accuracy / answered_only_accuracy"
    echo "  (diagnostic, never compare across cells/protocols/models) /"
    echo "  no_answer_rate / multiple_marker_rate /"
    echo "  first_last_disagreement_rate / loop_rate / truncation_rate, and"
    echo "  results.$MODEL.workpoint for the McNemar/Holm-based verdict"
    echo "  (computed on the overall, no-answer-included accuracy)."
    ;;

  *)
    echo "[FATAL] unknown stage '$STAGE' (sweep|eval)" >&2
    exit 1
    ;;
esac
