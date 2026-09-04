#!/usr/bin/env bash
# ProofWriter OWA multi-hop -- task-specific workpoint exploration
# (protocol `proofwriter-owa-v0`). NOT a GSM8K fixed-workpoint transfer test:
# this task searches its OWN 4-point dose set.
#
#   llama3   layers [11,20)  alpha in {-6,-4,0,+4}
#   qwen2.5  layers [16,22)  alpha in {-6,0,+6,+8}
#
# Seven EXPLICIT stages, each fail-closed and each requiring the previous
# stage's output to already exist on disk. There is NO automatic progression
# from one stage to the next -- in particular pilot -> llama-sweep/qwen-sweep
# is always a manual step.
#
#   bash run_proofwriter_owa.sh validate-data
#   bash run_proofwriter_owa.sh preflight   llama3|qwen2.5
#   bash run_proofwriter_owa.sh pilot       llama3|qwen2.5
#   bash run_proofwriter_owa.sh canary      llama3|qwen2.5
#   bash run_proofwriter_owa.sh llama-sweep
#   bash run_proofwriter_owa.sh qwen-sweep
#   bash run_proofwriter_owa.sh analyze
#
# ONE MODEL PER CARD; a model's whole alpha curve (preflight/pilot/canary/
# sweep) stays on ONE machine, matching the project-wide bf16-reproducibility
# convention. The two MODELS may run on two cards -- they are never compared
# per item at the accuracy level (each model's workpoint is its own).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_proofwriter_owa.sh validate-data|preflight|pilot|canary|llama-sweep|qwen-sweep|analyze [llama3|qwen2.5] [device_tag]" >&2
  echo "  'canary' additionally takes a THIRD positional arg, device_tag, e.g.:" >&2
  echo "    run_proofwriter_owa.sh canary llama3 card0" >&2
  echo "    run_proofwriter_owa.sh canary llama3 card1" >&2
  echo "  Each canary run on a DIFFERENT physical GPU needs its own device_tag" >&2
  echo "  so the two runs write to different output paths and can be diffed --" >&2
  echo "  a fixed tag would make the second run silently skip (existing-file" >&2
  echo "  guard) rather than producing a second, comparable cell." >&2
  exit 1
fi
STAGE="$1"; MODEL="${2:-}"; DEVICE_TAG="${3:-}"

PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark/proofwriter_owa}"
PW_DIR="$WORK_DIR/proofwriter_owa"
ARCHIVE_DIR="${ARCHIVE_DIR:-$BASE_DIR/proofwriter_owa_raw}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR}"     # per-model output goes under $OUT_ROOT/$MODEL/proofwriter_owa

# a wrong PY exits 127 before anything runs and the nohup log looks empty
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

model_cfg() {
  case "$1" in
    llama3)
      SIZE=8B
      MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
      MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
      BAND=11_20
      A0=0-11-20; AN4=neg4-11-20; AN6=neg6-11-20; AP4=4-11-20
      SWEEP_CONFIGS="$AN6 $AN4 $A0 $AP4"
      ;;
    qwen2.5)
      SIZE=7B
      MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
      MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
      BAND=16_22
      A0=0-16-22; AN6=neg6-16-22; AP6=6-16-22; AP8=8-16-22
      SWEEP_CONFIGS="$AN6 $A0 $AP6 $AP8"
      ;;
    *) echo "[FATAL] unknown model '$1' (llama3 | qwen2.5)" >&2; exit 1 ;;
  esac
}

need_model_arg() {
  if [[ -z "$MODEL" ]]; then
    echo "[FATAL] stage '$STAGE' requires a model argument: llama3 or qwen2.5" >&2
    exit 1
  fi
}

need_gpu_pinned() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[FATAL] CUDA_VISIBLE_DEVICES must be set to exactly one card." >&2
    echo "        This model's whole alpha curve (preflight/pilot/canary/sweep)" >&2
    echo "        must stay on one machine; an unpinned run risks mixing device" >&2
    echo "        differences into the alpha effect." >&2
    exit 1
  fi
  if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "[FATAL] one card only; got '$CUDA_VISIBLE_DEVICES'." >&2; exit 1
  fi
}

check_common_inputs() {
  [[ -f "$MASK" ]] || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
  [[ -f "$MANIFEST" ]] || {
    echo "[FATAL] blind manifest not found: $MANIFEST" >&2
    echo "        run stage validate-data first." >&2; exit 1; }
  if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
    echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2
    exit 1
  fi
}

case "$STAGE" in

  # ---------------------------------------------------------------- 1
  validate-data)
    # Network-required download + offline-safe report + frozen manifest
    # build, all in one stage so a human reviews the report before any
    # generation cell can reference the manifest.
    echo "[proofwriter-owa] validate-data: download (if needed) + report + manifest"
    mkdir -p "$ARCHIVE_DIR" "$BENCH"
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/data_proofwriter_owa.py --download "$ARCHIVE_DIR"
    "$PY" proofwriter_owa/data_proofwriter_owa.py \
      --archive_dir "$ARCHIVE_DIR" --report
    echo ""
    echo "[proofwriter-owa] review the report above (theory/question/label/QDep"
    echo "  counts, missing/duplicate/unparseable gold) BEFORE building the"
    echo "  manifest -- no sample is ever chosen by model output, but a data"
    echo "  problem should be caught here, not after a GPU run."
    read -r -p "Proceed to build the frozen 300-item manifest? [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
      echo "[proofwriter-owa] stopped before manifest build."
      exit 0
    fi
    "$PY" proofwriter_owa/data_proofwriter_owa.py \
      --archive_dir "$ARCHIVE_DIR" --out_dir "$BENCH" --build_manifest
    ;;

  # ---------------------------------------------------------------- 2
  preflight)
    need_model_arg
    model_cfg "$MODEL"
    need_gpu_pinned
    MANIFEST="$BENCH/manifest_blind.json"
    GOLD="$BENCH/manifest_gold.json"
    check_common_inputs
    PREFLIGHT_FILE="$BENCH/preflight_blind_${MODEL}.json"
    PREFLIGHT_GOLD="$BENCH/preflight_gold_${MODEL}.json"
    if [[ ! -f "$PREFLIGHT_FILE" ]]; then
      echo "[proofwriter-owa] building the 30-item preflight subset (15 D3 + 15 D5,"
      echo "  all three labels covered) from the frozen manifest..."
      cd "$WORK_DIR"
      "$PY" proofwriter_owa/build_subset.py \
        --gold "$GOLD" --blind "$MANIFEST" --mode preflight \
        --out_blind "$PREFLIGHT_FILE" --out_gold "$PREFLIGHT_GOLD"
    fi
    echo "[proofwriter-owa] preflight: alpha=0 canary on $MODEL (30 items)"
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --manifest "$PREFLIGHT_FILE" --mask_path "$MASK" \
      --configs "$A0" --out_dir "$OUT_ROOT/$MODEL/proofwriter_owa" --tag preflight
    echo "[proofwriter-owa] preflight alpha=0 done. steering_fires must be 0"
    echo "  (checked by the generator itself; a mismatch already stopped the run)."
    echo "  Next: run 'canary' for a non-zero-alpha fire-count check on the"
    echo "  same 30 items, then review truncation before deciding the budget."
    ;;

  # ---------------------------------------------------------------- 3
  canary)
    need_model_arg
    model_cfg "$MODEL"
    need_gpu_pinned
    if [[ -z "$DEVICE_TAG" ]]; then
      echo "[FATAL] canary requires a THIRD positional arg, device_tag, e.g.:" >&2
      echo "        run_proofwriter_owa.sh canary $MODEL card0" >&2
      echo "        A fixed --tag would make a second canary run (on a" >&2
      echo "        different GPU) silently skip via the existing-file guard" >&2
      echo "        instead of producing a genuinely comparable second cell." >&2
      exit 1
    fi
    MANIFEST="$BENCH/manifest_blind.json"
    check_common_inputs
    PREFLIGHT_FILE="$BENCH/preflight_blind_${MODEL}.json"
    [[ -f "$PREFLIGHT_FILE" ]] || {
      echo "[FATAL] $PREFLIGHT_FILE missing; run stage 'preflight' first." >&2
      exit 1; }
    case "$MODEL" in
      llama3)  CANARY_ALPHA="$AN6"; CANARY_ALPHA_TAG="mdf_neg6" ;;
      qwen2.5) CANARY_ALPHA="$AP8"; CANARY_ALPHA_TAG="mdf_8" ;;
    esac
    # --tag includes device_tag so canary runs on DIFFERENT physical GPUs
    # write to DIFFERENT paths (canary_<device_tag>_mdf_.../...) and can
    # actually be diffed -- a fixed "preflight" tag here previously meant a
    # second run always hit get_answer_proofwriter_owa.py's existing-file
    # skip and produced nothing new for the second card. CANARY_ALPHA_TAG
    # is hand-matched to get_answer_proofwriter_owa.py's own
    # `tag = f"mdf_{alpha}".replace("-", "neg")` naming (alpha=-6 -> mdf_neg6,
    # alpha=8 -> mdf_8) since only these two fixed alphas are ever used here.
    CANARY_TAG="canary_${DEVICE_TAG}"
    echo "[proofwriter-owa] canary: non-zero alpha ($CANARY_ALPHA) fire-count"
    echo "  check on the same 30-item preflight set. device_tag=$DEVICE_TAG"
    echo "  steering_fires must equal L * n_samples * 1 exactly (checked by"
    echo "  the generator; a mismatch stops the run)."
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --manifest "$PREFLIGHT_FILE" --mask_path "$MASK" \
      --configs "$CANARY_ALPHA" --out_dir "$OUT_ROOT/$MODEL/proofwriter_owa" \
      --tag "$CANARY_TAG"
    CANARY_OUT="$OUT_ROOT/$MODEL/proofwriter_owa/${CANARY_TAG}_${CANARY_ALPHA_TAG}/proofwriter_owa_${SIZE}_${BAND}.json"
    echo "[proofwriter-owa] canary done: $CANARY_OUT"
    echo "[proofwriter-owa] If this model's alpha curve will ever span more"
    echo "  than one GPU, re-run this SAME command with a DIFFERENT"
    echo "  device_tag on the other card, then run the real outcome-level"
    echo "  comparator (not just a suggestion to 'diff the two outputs'):"
    echo "    $PY proofwriter_owa/compare_canary.py \\"
    echo "      --gold $BENCH/preflight_gold_${MODEL}.json \\"
    echo "      --cell <card0 output> <card1 output> [...]"
    echo "  See PREREG_PROOFWRITER_OWA.md S7 for the pass/fail criterion this"
    echo "  compares against."
    ;;

  # ---------------------------------------------------------------- 4
  pilot)
    need_model_arg
    model_cfg "$MODEL"
    need_gpu_pinned
    MANIFEST="$BENCH/manifest_blind.json"
    GOLD="$BENCH/manifest_gold.json"
    check_common_inputs
    PILOT_FILE="$BENCH/pilot_blind_${MODEL}.json"
    PILOT_GOLD="$BENCH/pilot_gold_${MODEL}.json"
    if [[ ! -f "$PILOT_FILE" ]]; then
      cd "$WORK_DIR"
      "$PY" proofwriter_owa/build_subset.py \
        --gold "$GOLD" --blind "$MANIFEST" --mode pilot \
        --out_blind "$PILOT_FILE" --out_gold "$PILOT_GOLD"
    fi
    echo "[proofwriter-owa] pilot: alpha=0 ONLY, 150 items (D3=75, D5=75)."
    echo "  This stage ONLY REPORTS. It must not be used to revise the prompt,"
    echo "  sample, alpha set, or the scoring metric."
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --manifest "$PILOT_FILE" --mask_path "$MASK" \
      --configs "$A0" --out_dir "$OUT_ROOT/$MODEL/proofwriter_owa" --tag pilot
    echo "[proofwriter-owa] pilot generation done. Score it (read-only on gold):"
    echo "  $PY proofwriter_owa/eval_proofwriter_owa.py \\"
    echo "    --gold $PILOT_GOLD \\"
    echo "    --generations $OUT_ROOT/$MODEL/proofwriter_owa/pilot_mdf_0/proofwriter_owa_${SIZE}_${BAND}.json \\"
    echo "    --out proofwriter_owa/results/pilot_${MODEL}.json"
    echo "  A human reviews this before 'llama-sweep' / 'qwen-sweep' is launched."
    echo "  NOTE: eval_proofwriter_owa.py requires an alpha=0 cell to compute"
    echo "  McNemar contrasts; the pilot is alpha=0-only, so a pilot-only run"
    echo "  through eval_proofwriter_owa.py reports 'cells'/summarize() output"
    echo "  (overall/D3/D5/per-label accuracy) but no mcnemar_vs_alpha0 pairs --"
    echo "  that is expected: the pilot has nothing to contrast against yet."
    ;;

  # ---------------------------------------------------------------- 5 & 6
  llama-sweep|qwen-sweep)
    if [[ "$STAGE" == "llama-sweep" ]]; then MODEL=llama3; else MODEL=qwen2.5; fi
    model_cfg "$MODEL"
    need_gpu_pinned
    MANIFEST="$BENCH/manifest_blind.json"
    check_common_inputs
    A0FILE="$OUT_ROOT/$MODEL/proofwriter_owa/mdf_0/proofwriter_owa_${SIZE}_${BAND}.json"
    echo "[proofwriter-owa] $STAGE: FORMAL 4-point sweep on the full 300-item"
    echo "  manifest. This stage is launched manually AFTER the pilot has been"
    echo "  reviewed -- it does not run automatically from 'pilot'."
    echo "  max_new_tokens defaults to 768; pass MAX_NEW_TOKENS=1024 if the"
    echo "  preflight/pilot truncation rate exceeded 1-2% (see PREREG S6)."
    cd "$WORK_DIR"
    "$PY" proofwriter_owa/get_answer_proofwriter_owa.py \
      --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
      --manifest "$MANIFEST" --mask_path "$MASK" \
      --configs $SWEEP_CONFIGS --out_dir "$OUT_ROOT/$MODEL/proofwriter_owa" \
      --max_new_tokens "${MAX_NEW_TOKENS:-768}"
    echo "[proofwriter-owa] $STAGE done."
    echo "[proofwriter-owa] Alpha=0 file (baseline for every McNemar contrast): $A0FILE"
    ;;

  # ---------------------------------------------------------------- 7
  analyze)
    echo "[proofwriter-owa] analyze: scoring + commitment extractor + stats."
    echo "  Requires BOTH models' formal sweeps to be complete for a full report"
    echo "  (each model's Holm(m=3) family is judged independently, so a"
    echo "  single-model run still scores, just without the other model's row)."
    # Explicit arrays, NOT brace expansion stored in a string variable: bash
    # only performs `{...}` brace expansion on the LITERAL source text of a
    # command at parse time, never on the already-expanded VALUE of a
    # variable. `LLAMA_GLOB="...mdf_{0,neg4,neg6,4}/..."` followed by
    # `for f in $LLAMA_GLOB` iterates over ONE literal string still containing
    # the unexpanded "{0,neg4,neg6,4}" text -- not four paths -- so `[[ -f "$f" ]]`
    # always failed and GEN_FILES was always empty.
    LLAMA_FILES=(
      "$OUT_ROOT/llama3/proofwriter_owa/mdf_0/proofwriter_owa_8B_11_20.json"
      "$OUT_ROOT/llama3/proofwriter_owa/mdf_neg4/proofwriter_owa_8B_11_20.json"
      "$OUT_ROOT/llama3/proofwriter_owa/mdf_neg6/proofwriter_owa_8B_11_20.json"
      "$OUT_ROOT/llama3/proofwriter_owa/mdf_4/proofwriter_owa_8B_11_20.json"
    )
    QWEN_FILES=(
      "$OUT_ROOT/qwen2.5/proofwriter_owa/mdf_0/proofwriter_owa_7B_16_22.json"
      "$OUT_ROOT/qwen2.5/proofwriter_owa/mdf_neg6/proofwriter_owa_7B_16_22.json"
      "$OUT_ROOT/qwen2.5/proofwriter_owa/mdf_6/proofwriter_owa_7B_16_22.json"
      "$OUT_ROOT/qwen2.5/proofwriter_owa/mdf_8/proofwriter_owa_7B_16_22.json"
    )
    echo "[proofwriter-owa] expected generation files:"
    for f in "${LLAMA_FILES[@]}" "${QWEN_FILES[@]}"; do echo "  $f"; done
    cd "$WORK_DIR"
    GEN_FILES=()
    for f in "${LLAMA_FILES[@]}" "${QWEN_FILES[@]}"; do
      [[ -f "$f" ]] && GEN_FILES+=("$f")
    done
    if [[ ${#GEN_FILES[@]} -eq 0 ]]; then
      echo "[FATAL] no generation files found; run llama-sweep and/or qwen-sweep first." >&2
      exit 1
    fi
    echo "[proofwriter-owa] found ${#GEN_FILES[@]}/${#LLAMA_FILES[@]}+${#QWEN_FILES[@]} generation files"
    OUT_FILE="${OUT_FILE:-proofwriter_owa/results/proofwriter_owa_evaluation.json}"
    "$PY" proofwriter_owa/eval_proofwriter_owa.py \
      --gold "$BENCH/manifest_gold.json" \
      --generations "${GEN_FILES[@]}" \
      --out "$OUT_FILE"
    ;;

  *)
    echo "[FATAL] unknown stage '$STAGE'." >&2
    echo "  stages: validate-data preflight pilot canary llama-sweep qwen-sweep analyze" >&2
    exit 1
    ;;
esac
