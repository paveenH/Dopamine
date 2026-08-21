#!/usr/bin/env bash
# ============================================================================
# Qwen2.5-7B-Instruct — AdaptiveThinking SIGNAL collection (Phase 1 replication)
#
# Separate launcher on purpose: run_track_hidden_states.sh drives the FROZEN
# Llama phase1b_eot matrix and must not be edited.
#
# ---------------------------------------------------------------------------
# HARD CONSTRAINT — ONE CURVE, ONE GPU.
#   bf16 greedy is NOT byte-reproducible across GPUs (different cuBLAS
#   accumulation order -> logit ties flip -> the whole CoT chain diverges; one
#   Llama +4 re-run on another box gave 205/300 text mismatches). Every readout
#   here is PAIRED PER QUESTION against that curve's own alpha=0, so splitting a
#   curve across cards mixes the machine difference into the alpha effect, and
#   it cannot be separated afterwards -- summary CSVs carry no device field.
#   Therefore each STEP below is one self-contained curve carrying its OWN
#   alpha=0, and must run start-to-finish on a single card.
# ---------------------------------------------------------------------------
#
# GPU plan (3x RTX 4090 available; persona deferred by decision 2026-08-21):
#   card A : NOCOT  — all 11 alphas -8..+12 incl. 0   (main dose curve)
#   card B : COT    — {0, +6}                          (self-contained 2x2 half)
#   card C : idle / persona later
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_track_dopamine_signal_qwen25.sh NOCOT > nocot.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 nohup bash run_track_dopamine_signal_qwen25.sh COT   > cot.log   2>&1 &
#
# Protocol (frozen, aligned to the behavioural run run_gsm8k_qwen25.sh):
#   same 300 GSM8K questions, bare-string, greedy, bs=1, max_new_tokens=768,
#   layers [16,22) (L=6), mask nmd_0.5_16_22_7B.npy. Llama's [11,20) does NOT
#   transfer. Cross-model alpha is NOMINAL, not a calibrated common dose.
#
# Signal-behaviour pairing 口径 (settled 2026-08-21):
#   this run stores its OWN generated text + commit marker + correctness; those
#   are what pair per-question with the signal. The frozen behaviour table in
#   AdaDopamine_gsm8k.md §4 stays the PRODUCTION accuracy reference and is never
#   mixed per-question with these samples (different batch).
#
# +10/+12 are collected here so G_prefill's linearity can be checked past the
# point where s_t / commitment / accuracy have already saturated. HDF5 (a later
# step) still only covers representative cells up to +8.
# ============================================================================
set -uo pipefail

STEP="${1:-}"
case "${STEP}" in
  CHECK|SMOKE|NOCOT|COT) ;;
  "") echo "usage: bash $0 {CHECK|SMOKE|NOCOT|COT}"; exit 2 ;;
  *)  echo "unknown step '${STEP}' (want CHECK|SMOKE|NOCOT|COT)"; exit 2 ;;
esac

# ── model / protocol constants (must match run_gsm8k_qwen25.sh) ──
MODEL_NAME="qwen2.5"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SIZE="7B"
HS_PREFIX="qwen2.5"
TYPE="non"
MASK_TYPE="nmd"
PERCENTAGE=0.5
LS=16                       # HF hidden_states index; decoder layers 15..20, L=6
LE=22
TASK="gsm8k"
GSM8K_FILE="benchmark/gsm8k_test_sample.json"
N_SAMPLES=300
MAX_NEW_TOKENS=768          # pinned, NOT the tracker's 512 default
EMA_ALPHA=0.95
ROLE="neutral"
RUN_TAG="qwen25_signal_v1"
SMOKE_TAG="${RUN_TAG}_smoke"
SMOKE_N=8            # real generate_one(), few samples, both alpha paths

DATA="${DATA:-data1}"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
# Server convention: the conda env names its interpreter `python`. `python3.10`
# is the LOCAL analysis-box convention and does NOT exist here -- it exits 127
# before anything runs, which under nohup looks like a job that silently died.
PY="${PY:-python}"

MASK_PATH="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/${MASK_TYPE}_${PERCENTAGE}_${LS}_${LE}_${MODEL_SIZE}.npy"
OUT_DIR="${BASE_DIR}/${MODEL_NAME}/dopamine_signal/${RUN_TAG}"


# alpha lists — each list is ONE curve and includes its own 0
ALPHAS_NOCOT=(-8 -6 -4 -2 0 2 4 6 8 10 12)
ALPHAS_COT=(0 6)

# A curve must not be split across cards. Unset CUDA_VISIBLE_DEVICES means
# device_map="auto" claims EVERY visible card, so two concurrent steps would
# collide and neither curve would be single-card.
require_single_card () {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[x] CUDA_VISIBLE_DEVICES is unset."
    echo "    llms.py loads with device_map=auto and would claim all cards,"
    echo "    colliding with the other curve. Pin exactly one card."
    exit 1
  fi
  if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[x] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names several cards."
    echo "    One curve, one card. Pin exactly one."
    exit 1
  fi
}

# Guards that do not need the server filesystem run FIRST, so a mis-pinned card
# reports the real problem instead of a confusing path error.
if [[ "${STEP}" != "CHECK" ]]; then require_single_card; fi  # SMOKE loads the model too

cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }

if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "[x] interpreter '${PY}' not found. Activate the env, or pass PY=..."
  exit 1
fi
if ! "${PY}" -c "import numpy, torch" >/dev/null 2>&1; then
  echo "[x] '${PY}' cannot import numpy/torch — wrong environment."
  echo "    which: $(command -v "${PY}")"
  exit 1
fi

banner () {
  echo "============================================================"
  echo "Qwen2.5 SIGNAL | step=${STEP} | run_tag=${RUN_TAG}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset -- ALL CARDS!>}"
  echo "band [${LS},${LE}) L=6 | mask ${MASK_PATH}"
  echo "max_new_tokens=${MAX_NEW_TOKENS} | n=${N_SAMPLES} | bs=1 greedy"
  echo "out ${OUT_DIR}"
  echo "============================================================"
}

preflight () {
  echo "[*] pre-flight (signal path) — read-only w.r.t. experiment artifacts"
  case "${MODEL_DIR}" in
    /*) : ;;   # local path: nothing is downloaded
    *)  echo "    note: MODEL_DIR='${MODEL_DIR}' is a Hub id, so the tokenizer"
        echo "          may populate the HF cache. Pass a local model directory"
        echo "          via MODEL_DIR=... to avoid any writes." ;;
  esac
  ${PY} check_signal_qwen.py \
    --model_dir "${MODEL_DIR}" \
    --mask "${MASK_PATH}" \
    --layer_start "${LS}" --layer_end "${LE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --base_dir "${BASE_DIR}" \
    --model "${MODEL_NAME}" \
    --run_tag "${RUN_TAG}" || {
      echo "[x] pre-flight FAILED — not collecting."; exit 1; }
}

# The pre-flight's toy forward drives the hooks on identity layers -- it never
# exercises generate_one(), i.e. real KV cache, real EOS, real prefill/decode
# alternation. So a green pre-flight authorizes a SMOKE, not 300x11. This gate
# makes that ordering structural rather than a thing to remember.
SMOKE_DIR="${BASE_DIR}/${MODEL_NAME}/dopamine_signal/${SMOKE_TAG}"

# Counting files is the weakest possible check: a leftover artifact, a
# half-written file from an interrupted run, or another experiment's output all
# pass it. So the gate verifies the EXACT expected files and reads their
# metadata. And since no automatic check can confirm a HUMAN read the generated
# text, the last step is an explicit SMOKE_APPROVED=1.
require_smoke () {   # $1 = "nocot" | "cot"
  local fam="$1"
  local tag; tag=$( [[ "${fam}" == cot ]] && echo cot || echo nocot )
  echo "[*] verifying ${fam} smoke artifacts in ${SMOKE_DIR}"
  "${PY}" - "${SMOKE_DIR}" "${tag}" "${SMOKE_N}" "${SMOKE_TAG}" "${LS}" "${LE}" \
           "${MAX_NEW_TOKENS}" "${MODEL_SIZE}" "${EMA_ALPHA}" <<'PYEOF'
import json, os, sys
d, tag, n, rtag, ls, le, mnt, size, ema = sys.argv[1:10]
n, ls, le, mnt = int(n), int(ls), int(le), int(mnt)
fails = []
for a, atag in ((0.0, ""), (6.0, "_a6")):
    f = os.path.join(
        d, f"dopamine_signal_gsm8k_{size}_{tag}{atag}_ema{ema}_L{ls}-{le}.json")
    if not os.path.exists(f):
        fails.append(f"missing: {os.path.basename(f)}"); continue
    try:
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
    except Exception as e:                      # truncated / interrupted write
        fails.append(f"unreadable (interrupted run?): {os.path.basename(f)}: {e}")
        continue
    m, data = j.get("meta", {}), j.get("data", [])
    def chk(key, want):
        got = m.get(key)
        if got != want:
            fails.append(f"{os.path.basename(f)}: meta[{key}]={got!r} != {want!r}")
    chk("steer_alpha", a)
    chk("cot", tag == "cot")
    chk("run_tag", rtag)
    chk("max_new_tokens", mnt)
    chk("layer_start", ls)
    chk("layer_end", le)
    chk("n_samples", n)
    if len(data) != n:
        fails.append(f"{os.path.basename(f)}: {len(data)} samples, expected {n}")
    if not any((s.get("generated") or "").strip() for s in data):
        fails.append(f"{os.path.basename(f)}: every generation is empty")
if fails:
    print("[x] smoke verification FAILED:")
    for x in fails:
        print(f"    - {x}")
    sys.exit(1)
print(f"[ok] {tag} smoke artifacts verified (alpha 0 and +6, n={n})")
PYEOF
  if [[ $? -ne 0 ]]; then
    echo "    A green pre-flight only authorizes a smoke: the toy forward never"
    echo "    ran generate_one(). Run:  bash $0 SMOKE"
    exit 1
  fi

  if [[ "${SMOKE_APPROVED:-0}" != "1" ]]; then
    echo
    echo "[x] smoke artifacts exist but have NOT been approved."
    echo "    No automatic check can tell whether a human read the generated"
    echo "    text. Inspect the ${fam} smoke JSONs and confirm:"
    echo "      - text ends naturally (not every sample at ${MAX_NEW_TOKENS} tokens)"
    echo "      - '####' present, commit position sane, accuracy plausible"
    echo "      - signal array length == generated length"
    echo "      - alpha=+6 text DIFFERS from alpha=0 (steering reached generation)"
    echo "    Then re-launch with:  SMOKE_APPROVED=1 ..."
    exit 1
  fi
  echo "[ok] ${fam} smoke verified and approved — formal collection authorized."
}

run_one () {   # $1 = alpha, $2 = "" | "--cot", $3 = n_samples, $4 = run_tag
  local A="$1" COT="$2" NS="${3:-${N_SAMPLES}}" RT="${4:-${RUN_TAG}}"
  local tag; tag="$( [[ -n "${COT}" ]] && echo cot || echo nocot )"
  echo
  echo "---- [${tag}] alpha=${A} ---- $(date '+%F %T')"
  ${PY} track_dopamine_signal.py \
    --task        "${TASK}" \
    --model       "${MODEL_NAME}" \
    --model_dir   "${MODEL_DIR}" \
    --hs          "${HS_PREFIX}" \
    --size        "${MODEL_SIZE}" \
    --type        "${TYPE}" \
    --mask_type   "${MASK_TYPE}" \
    --percentage  "${PERCENTAGE}" \
    --layer_start "${LS}" \
    --layer_end   "${LE}" \
    --ema_alpha   "${EMA_ALPHA}" \
    --test_file   "${GSM8K_FILE}" \
    --n_samples   "${NS}" \
    --alpha       "${A}" \
    --role        "${ROLE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --base_dir    "${BASE_DIR}" \
    --run_tag     "${RT}" \
    ${COT}
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[x] alpha=${A} (${tag}) exited ${rc} — STOPPING this curve."
    echo "    A partially collected curve is not paired data; fix and re-run"
    echo "    the whole curve on this same card."
    exit ${rc}
  fi
}

case "${STEP}" in
  CHECK)
    banner; preflight
    echo; echo "[ok] pre-flight passed. Collection not started (CHECK only)."
    ;;
  SMOKE)
    banner; preflight
    echo; echo "[*] SMOKE: ${SMOKE_N} samples x {No-CoT, CoT} x alpha{0,+6}"
    echo "    Both prompt families are smoked: the CoT prompt and its generated"
    echo "    text are a different object and must not be authorized by a"
    echo "    No-CoT smoke."
    echo "    -> ${SMOKE_DIR}"
    run_one 0 ""      "${SMOKE_N}" "${SMOKE_TAG}"
    run_one 6 ""      "${SMOKE_N}" "${SMOKE_TAG}"
    run_one 0 "--cot" "${SMOKE_N}" "${SMOKE_TAG}"
    run_one 6 "--cot" "${SMOKE_N}" "${SMOKE_TAG}"
    echo
    echo "[ok] smoke complete. READ THE OUTPUT before the formal run:"
    echo "  - generated text is well-formed and ends naturally (not all 768 tok)"
    echo "  - '####' present, commit locator sane, accuracy plausible"
    echo "  - signal arrays have length == generated length"
    echo "  - alpha=+6 differs from alpha=0 (steering reached generation)"
    echo "  - meta carries max_new_tokens=768 / run_tag / steer_alpha"
    ;;
  NOCOT)
    banner; preflight
    echo; echo "[*] No-CoT dose curve: ${ALPHAS_NOCOT[*]}  (11 cells, one card)"
    for A in "${ALPHAS_NOCOT[@]}"; do run_one "${A}" ""; done
    echo; echo "[ok] No-CoT curve complete $(date '+%F %T')"
    ;;
  COT)
    banner; preflight
    echo; echo "[*] CoT cells: ${ALPHAS_COT[*]}  (self-contained, own alpha=0)"
    for A in "${ALPHAS_COT[@]}"; do run_one "${A}" "--cot"; done
    echo; echo "[ok] CoT cells complete $(date '+%F %T')"
    ;;
esac
