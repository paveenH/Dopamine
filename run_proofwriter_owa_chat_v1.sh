#!/usr/bin/env bash
# ProofWriter OWA, Llama3.1-8B ONLY: FORMAL four-point steering sweep under
# the HF chat-template interface condition. Protocol `proofwriter-owa-chat-v1`.
#
# INDEPENDENT INTERFACE-CONDITION EXPERIMENT -- does NOT redefine the
# ProofWriter-OWA workpoint (bare-string sweep, COMPLETE + CLOSED 2026-09-05,
# proofwriter_owa/results/formal_sweep_v2.json) and does NOT overwrite or
# reuse any bare v2 result. The 30-item alpha=0 diagnostic (chat_v2_mdf_0,
# see diag_chat_template.py) showed HF chat-template wrapping sharply
# reduces llama3's no_answer_rate/loop_rate/truncation_rate vs bare-string
# at alpha=0; this sweep asks whether steering (alpha in {-6,-4,0,+4}) then
# produces a stable, Holm-significant accuracy/submission-rate change once
# that interface confound is controlled for.
#
# Held IDENTICAL to the bare formal sweep: the SAME 300-item manifest and
# order, the SAME v2 1-shot Unknown exemplar, the SAME prompt body and
# "#### <Label>" marker convention, the SAME mask file (band [11,20)), the
# SAME alpha set {-6,-4,0,+4}, greedy/temperature=0, max_new_tokens=1024,
# batch_size=8, prefill-only tail=1. The ONLY variable held across all four
# alpha is the prompt wrapping (bare-string vs apply_chat_template).
#
# Llama3.1-8B ONLY. No Qwen, no other benchmark, no steering-search beyond
# the frozen four-point set.
#
# TWO stages:
#   sweep    run all four alpha (one call, all four via --configs), on ONE
#            GPU (bf16 greedy is not byte-reproducible across cards, and
#            this curve's four cells must be directly comparable)
#   eval     score the four cells (the ONLY script that reads gold), via
#            eval_proofwriter_owa.py --protocol proofwriter-owa-chat-v1
#            (the one flag added to that script for this purpose; parser/
#            scoring/statistics are otherwise unchanged)
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_proofwriter_owa_chat_v1.sh sweep \
#     > proofwriter_owa_chat_v1_sweep.log 2>&1 &
#
#   bash run_proofwriter_owa_chat_v1.sh eval   # after sweep finishes, no GPU needed
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_proofwriter_owa_chat_v1.sh sweep" >&2
  echo "       run_proofwriter_owa_chat_v1.sh eval" >&2
  exit 1
fi
STAGE="$1"

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

# SAME mask and SAME alpha set as the bare formal sweep (llama3 row of
# run_proofwriter_owa_v2_preflight.sh) -- deliberately not re-derived here.
LLAMA_MASK="$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy"
LLAMA_SWEEP_CONFIGS="neg6-11-20 neg4-11-20 0-11-20 4-11-20"
MANIFEST_BLIND="$BENCH/manifest_blind.json"
MANIFEST_GOLD="$BENCH/manifest_gold.json"

LLAMA_OUT_DIR="$OUT_ROOT/llama3/proofwriter_owa"
CHAT_A0="$LLAMA_OUT_DIR/formal_chat_v1_mdf_0/proofwriter_owa_8B_11_20.json"
CHAT_AN4="$LLAMA_OUT_DIR/formal_chat_v1_mdf_neg4/proofwriter_owa_8B_11_20.json"
CHAT_AN6="$LLAMA_OUT_DIR/formal_chat_v1_mdf_neg6/proofwriter_owa_8B_11_20.json"
CHAT_AP4="$LLAMA_OUT_DIR/formal_chat_v1_mdf_4/proofwriter_owa_8B_11_20.json"
CHAT_EVAL_OUT="$PW_DIR/results/formal_sweep_chat_v1.json"

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
    echo "[proofwriter-owa-chat-v1] llama3: FORMAL 4-point sweep on the"
    echo "  full 300-item manifest, under the HF chat-template interface"
    echo "  condition. max_new_tokens is FROZEN at 1024 (imported from"
    echo "  get_answer_proofwriter_owa.py's MAX_NEW_TOKENS_FROZEN, never a"
    echo "  separately hardcoded literal)."
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa_chat.py \
      --model_dir meta-llama/Llama-3.1-8B-Instruct \
      --manifest "$MANIFEST_BLIND" --mask_path "$LLAMA_MASK" \
      --configs $LLAMA_SWEEP_CONFIGS \
      --out_dir "$LLAMA_OUT_DIR" \
      --exemplar_file "$EXEMPLAR_FILE" \
      --batch_size 8
    echo "[proofwriter-owa-chat-v1] sweep done. Alpha=0 file:"
    echo "    $CHAT_A0"
    echo
    echo "[proofwriter-owa-chat-v1] next:"
    echo "    bash run_proofwriter_owa_chat_v1.sh eval"
    ;;

  eval)
    for f in "$CHAT_A0" "$CHAT_AN4" "$CHAT_AN6" "$CHAT_AP4"; do
      [[ -f "$f" ]] || {
        echo "[FATAL] $f not found; run 'sweep' first (all four alpha" >&2
        echo "        come from ONE 'sweep' call -- --configs already" >&2
        echo "        lists all four)." >&2
        exit 1; }
    done
    if [[ -f "$CHAT_EVAL_OUT" ]]; then
      echo "[FATAL] $CHAT_EVAL_OUT exists; refusing to overwrite." >&2
      echo "        (eval_proofwriter_owa.py itself also refuses --out" >&2
      echo "        overwrite; this check just fails earlier/clearer.)" >&2
      exit 1
    fi
    cd "$WORK_DIR"
    echo "[proofwriter-owa-chat-v1] scoring the 4-point chat-interface sweep"
    echo "  (Holm(m=3), llama3's own alpha=0 as baseline; no comparison to"
    echo "  the bare sweep is computed here -- that is a separate,"
    echo "  descriptive side-by-side table, not a paired significance test)."
    "$PY" proofwriter_owa/eval_proofwriter_owa.py \
      --protocol proofwriter-owa-chat-v1 \
      --gold "$MANIFEST_GOLD" \
      --generations "$CHAT_A0" "$CHAT_AN4" "$CHAT_AN6" "$CHAT_AP4" \
      --out "$CHAT_EVAL_OUT"
    echo
    echo "[proofwriter-owa-chat-v1] wrote $CHAT_EVAL_OUT"
    echo "  Read results.llama3.cells.<alpha> for accuracy (FIRST-answer,"
    echo "  main) / sensitivity_last_answer_accuracy / answered_only_accuracy"
    echo "  (diagnostic, never compare across cells/protocols) /"
    echo "  no_answer_rate / multiple_marker_rate /"
    echo "  first_last_disagreement_rate / loop_rate / truncation_rate, and"
    echo "  results.llama3.workpoint for the McNemar/Holm-based verdict"
    echo "  (computed on the overall, no-answer-included accuracy)."
    ;;

  *)
    echo "[FATAL] unknown stage '$STAGE' (sweep|eval)" >&2
    exit 1
    ;;
esac
