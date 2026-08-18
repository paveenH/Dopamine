#!/usr/bin/env bash
# PV11: Controlled Evidence-State Micro-Episodes.
#
#   bash run_bandit_pv11.sh llama3 CHECK   # no GPU, ~10s
#   bash run_bandit_pv11.sh llama3 SMOKE   # 16 balanced states, ~10 min
#   bash run_bandit_pv11.sh llama3 A0      # 160 states, alpha=0
#   bash run_bandit_pv11.sh llama3 GATE    # read M1/M2
#   bash run_bandit_pv11.sh llama3 AM4     # PV11-Acq, 80 states, alpha=-4
#   bash run_bandit_pv11.sh llama3 AP4     # PV11-Acq, 80 states, alpha=+4
#
# HOW AM4/AP4 CAME TO EXIST. This launcher originally stopped at GATE and said
# so: no step ran +-4, because a gate failure CLOSES the protocol rather than
# being re-tuned. The gate then FAILED -- M1 fail, M2 pass -- and the two steps
# below exist only because pv11_amendment_01.json was written afterwards, on
# the alpha=0 data alone and before any steered episode existed. Read that file
# before running them.
#
# The amendment did NOT overturn the gate. It withdrew the Commitment block on
# construct grounds visible at alpha=0 (label and display row are perfectly
# collinear; each cell renders only 4 unique visible prompts, not 20), and it
# carried Acquisition forward as an exploratory follow-up because M2 passed
# under both readings. So:
#
#   * AM4/AP4 are NOT a gate PASS and NOT a competence claim.
#   * No commitment or threshold conclusion may be drawn from them.
#   * The primary contrast rests on 3 positive events at baseline. A null
#     reads "not detected at this low-power baseline", never "RSN does not
#     affect acquisition".
#   * They run ONCE. Afterwards the BAI line closes regardless of outcome --
#     no prompt iteration, no metric redefinition, no further follow-up.
#
# The prohibition this file was built around still stands: PV10-A/B/C ended
# after four representation-layer interventions each raised recognition without
# moving acquisition, and a fifth round of prompt edits aimed at a failing gate
# is the loop PV11 exists to escape. A gate failure is a finding, not a bug.
#
# WHY THERE IS STILL NO --alpha FLAG: the launcher hardcodes the alpha of each
# step, and run_bandit_pv11_episodes.py independently refuses any (block,alpha)
# pair outside {(all,0), (acquisition,-4), (acquisition,+4)}. Editing one does
# not unlock the other. In particular BOTH refuse acquisition alpha=0: that
# cell already exists inside the full A0 file, and collecting a second baseline
# would invite quoting whichever suits.
#
# TWO-GPU LAYOUT: A0 is one sequential cell with nothing to parallelize, but
# AM4 and AP4 are independent and SHOULD be run concurrently, one per card:
#
#   CUDA_VISIBLE_DEVICES=0 bash run_bandit_pv11.sh llama3 AM4
#   CUDA_VISIBLE_DEVICES=1 bash run_bandit_pv11.sh llama3 AP4
#
# CUDA_VISIBLE_DEVICES is what actually separates them: llms.py loads with
# device_map="auto", which claims every VISIBLE device, so without it two jobs
# map across both cards and collide. They already write separate --ans_file
# directories, which is mandatory anyway -- the detail JSON name carries the
# alpha but the resume key would still collide inside one directory.
#
# Note the gate REQUIRES both complete blocks: an acquisition-only file is a
# hard error there by design, not a subset. Analyse these cells with
# analyze_bandit_pv11_acq.py, which is frozen (pv11_acq_analysis_manifest.json).

set -euo pipefail

MODEL="${1:-llama3}"
STEP="${2:-CHECK}"

if [[ "$MODEL" != "llama3" ]]; then
  echo "PV11 is llama3-only: the 'Reason: ' anchor is token 220 and the bare"
  echo "candidate letters are 32-35 on the Llama-3.1 tokenizer. Neither has"
  echo "been audited on Qwen2.5. Adding a model means re-auditing both and"
  echo "re-freezing the state bank."
  exit 1
fi

PY="${PY:-python}"          # server conda env; python3.10 exits 127 there

MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
SIZE="${SIZE:-8B}"
HS="${HS:-llama3}"
TYPE="${TYPE:-non}"
MASK_TYPE="${MASK_TYPE:-nmd}"
PERCENTAGE="${PERCENTAGE:-0.5}"
LAYERS="${LAYERS:-11-20}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/${MODEL}/bandit/pv11}"
A0_DIR="${OUT_ROOT}/pv11_a0"
A0_FILE="${A0_DIR}/bandit_pv11_alpha0.json"
# Smoke output lives under the SAME tree as the formal cells, never /tmp:
# a smoke file in a system temp dir is deleted on reboot and is invisible to
# anyone reading the results tree, so a puzzling formal result cannot be
# checked against the smoke that preceded it. Its own subdirectory, and a
# name that says what it is, so it can never be mistaken for a gateable cell.
SMOKE_DIR="${OUT_ROOT}/smoke"
# PV11-Acq cells. One directory per alpha, always: the pv11 detail JSON is
# named by alpha but the summary/resume bookkeeping is per --ans_file, so two
# alphas sharing a directory would race each other.
ACQ_AM4_DIR="${OUT_ROOT}/pv11_acq_am4"
ACQ_AP4_DIR="${OUT_ROOT}/pv11_acq_ap4"

# Every check is offline and runs BEFORE any GPU time is spent. The bank check
# is first: a bank that does not reproduce from its builder makes everything
# downstream uninterpretable, and finding that out after a multi-hour run is
# the expensive way to learn it.
check () {
  echo "=== PV11 offline checks"
  ${PY} build_pv11_state_bank.py --check
  ${PY} test_pv11_state_bank.py
  ${PY} test_bandit_pv11.py
  ${PY} test_bandit_pv11_episode.py
  ${PY} test_pv11_scope_key.py
  ${PY} test_pv11_acq_alpha_guard.py
  ${PY} analyze_bandit_pv11_gate.py --selftest
  ${PY} test_pv11_gate_end_to_end.py
  echo
  echo "NOTE: if test_bandit_pv11.py printed 'SKIP tokenizer unavailable',"
  echo "the token-220 anchor assertions did NOT run on this machine. Sync the"
  echo "HF cache and re-run CHECK before trusting the anchor."
}

# The pilot is the ONLY step before A0 where a real model sees a real prompt.
# Everything in CHECK is FakeVC or pure string work, so a prompt the model
# cannot follow would pass every offline test and only surface an hour into
# A0. It covers BOTH blocks and is DESIGN-BALANCED rather than the first N
# states: `--limit 6` lands on state_ids {0,1} and two labels, where a
# constant first arm cannot be told apart from a label preference.
# See pilot_subset() in run_bandit_pv11_episodes.py.

case "${STEP}" in
  CHECK)
    check ;;

  # Read the generated TEXT, not only the summary line. What matters is
  # whether the model follows the two-line format, whether it drifts into
  # document/code completion, and whether the first action varies at all --
  # a first action that is constant across cells cannot move either gate rule.
  SMOKE)
    check
    mkdir -p "${SMOKE_DIR}"
    echo
    echo "=== PV11 PILOT  16 design-balanced states  ->  ${SMOKE_DIR}"
    echo "    NOT gateable -- a partial run by construction."
    echo "    Both blocks span all 4 labels, 4 display rows and all 4 cells,"
    echo "    so a constant first action cannot be a label or row preference."
    ${PY} run_bandit_pv11_episodes.py \
      --model_dir "${MODEL_DIR}" --size "${SIZE}" \
      --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
      --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
      --alpha 0 --pilot \
      --base_dir "${BASE_DIR}" \
      --ans_file "${SMOKE_DIR}/pv11_pilot.json"
    echo
    echo "Read the TEXT before deciding on A0:"
    echo "  ${PY} inspect_pv11_smoke.py ${SMOKE_DIR}/pv11_pilot.json" ;;

  A0)
    check
    mkdir -p "${A0_DIR}"
    echo
    echo "=== PV11 alpha=0  160 states  ->  ${A0_FILE}"
    echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: ALL GPUs>}  pid=$$"
    ${PY} run_bandit_pv11_episodes.py \
      --model_dir "${MODEL_DIR}" --size "${SIZE}" \
      --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
      --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
      --alpha 0 \
      --base_dir "${BASE_DIR}" --ans_file "${A0_FILE}"
    echo
    echo "Done. Next: bash run_bandit_pv11.sh ${MODEL} GATE" ;;

  AM4|AP4)
    check
    ${PY} freeze_pv11_acq_analysis.py --check
    if [[ "${STEP}" == "AM4" ]]; then
      ACQ_ALPHA="-4"; ACQ_DIR="${ACQ_AM4_DIR}"
    else
      ACQ_ALPHA="4";  ACQ_DIR="${ACQ_AP4_DIR}"
    fi
    if [[ ! -f "${A0_FILE}" ]]; then
      echo "no alpha=0 baseline at ${A0_FILE}"
      echo "PV11-Acq is paired against it; run A0 first."
      exit 1
    fi
    mkdir -p "${ACQ_DIR}"
    echo
    echo "=== PV11-Acq  alpha=${ACQ_ALPHA}  80 acquisition states  ->  ${ACQ_DIR}"
    echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: ALL GPUs>}  pid=$$"
    echo "    Exploratory follow-up per pv11_amendment_01.json. NOT a gate"
    echo "    PASS. No commitment conclusion. Runs once; then the BAI line"
    echo "    closes regardless of outcome."
    ${PY} run_bandit_pv11_episodes.py \
      --model_dir "${MODEL_DIR}" --size "${SIZE}" \
      --hs "${HS}" --type "${TYPE}" --mask_type "${MASK_TYPE}" \
      --percentage "${PERCENTAGE}" --layers "${LAYERS}" \
      --block acquisition --alpha "${ACQ_ALPHA}" \
      --base_dir "${BASE_DIR}" \
      --ans_file "${ACQ_DIR}/bandit_pv11_alpha${ACQ_ALPHA}.0.json"
    echo
    echo "When BOTH alphas are done, analyse with the FROZEN analyzer:"
    echo "  ${PY} analyze_bandit_pv11_acq.py \\"
    echo "      --a0  ${A0_FILE} \\"
    echo "      --am4 ${ACQ_AM4_DIR}/bandit_pv11_alpha-4.0.json \\"
    echo "      --ap4 ${ACQ_AP4_DIR}/bandit_pv11_alpha4.0.json"
    echo "The alpha=0 cell is read from the ORIGINAL full A0 file in place." ;;

  GATE)
    if [[ ! -f "${A0_FILE}" ]]; then
      echo "no alpha=0 result at ${A0_FILE}"
      echo "run: bash run_bandit_pv11.sh ${MODEL} A0"
      exit 1
    fi
    # The gate exits non-zero on FAIL, which under `set -e` would end the
    # script before the closing note prints. The verdict is the deliverable
    # either way, so capture it and report explicitly.
    set +e
    ${PY} analyze_bandit_pv11_gate.py --result "${A0_FILE}"
    verdict=$?
    set -e
    echo
    if [[ ${verdict} -eq 0 ]]; then
      echo "GATE PASSED on both rules. Note this is NOT the state the"
      echo "AM4/AP4 steps were authorized under: those exist because the"
      echo "gate FAILED and pv11_amendment_01.json withdrew Commitment while"
      echo "carrying Acquisition forward. A full PASS would license a"
      echo "different, wider experiment -- re-read the amendment before"
      echo "assuming AM4/AP4 is the right follow-up."
    else
      echo "GATE FAILED. Per the frozen wording: manipulation-gate failure"
      echo "does not trigger prompt iteration. This controlled interface did"
      echo "not elicit the prerequisite behavior needed for an interpretable"
      echo "steering test."
      echo
      echo "If this is the M1-fail/M2-pass verdict recorded in"
      echo "pv11_amendment_01.json, the authorized follow-up is PV11-Acq:"
      echo "  CUDA_VISIBLE_DEVICES=0 bash run_bandit_pv11.sh ${MODEL} AM4"
      echo "  CUDA_VISIBLE_DEVICES=1 bash run_bandit_pv11.sh ${MODEL} AP4"
      echo "Any OTHER failure pattern closes the protocol; it does not"
      echo "inherit that authorization."
    fi
    exit ${verdict} ;;

  *)
    echo "unknown step: ${STEP}"
    echo "expected: CHECK SMOKE A0 GATE AM4 AP4"
    exit 1 ;;
esac
