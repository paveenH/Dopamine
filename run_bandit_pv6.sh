#!/bin/bash
#
# ============ pv6 F-REFERENCE: server driver ============
# Walks the frozen experiment order (BanditExperiment_LiteratureReview.md §3):
#
#   1  manifest check         gate basis must be reproducible on THIS machine
#   2  smoke                  Easy/Hard x bare/chat x 3 seeds
#   3  protocol checks        engineering correctness ONLY, no behaviour reading
#   4  A0_BARE                Track A, RSN-aligned interface (gate is judged here)
#   5  A0_CHAT                Track A, literature interface comparator
#
# Each phase GATES the next: a failure stops the script rather than letting a
# later phase run on an unverified base. Phases 4-5 additionally require an
# explicit --yes, because they are the expensive runs and the smoke output is
# meant to be READ by a human first.
#
# Usage:
#   bash run_pv6_server.sh check                 # phase 1 only
#   bash run_pv6_server.sh smoke                 # phases 1-3
#   bash run_pv6_server.sh trackA --yes          # phases 4-5 (after smoke)
#   bash run_pv6_server.sh trackA --yes --bare   # phase 4 only
#
# Everything is resumable: re-running a phase skips cells already in the CSV.

set -uo pipefail

MODEL="${MODEL:-llama3}"
BASE_DIR="${BASE_DIR:-/data1/paveen/Dopamine/components}"
PY="${PY:-python3.10}"
LOGDIR="${LOGDIR:-./pv6_logs}"

PHASE="${1:-smoke}"
shift || true
CONFIRM=0
ONLY_BARE=0
ONLY_CHAT=0
for a in "$@"; do
  case "$a" in
    --yes)  CONFIRM=1 ;;
    --bare) ONLY_BARE=1 ;;
    --chat) ONLY_CHAT=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)

case "$MODEL" in
  llama3) OUT_ROOT="${BASE_DIR}/llama3" ;;
  qwen25) OUT_ROOT="${BASE_DIR}/qwen2.5" ;;
  *) echo "unknown MODEL '$MODEL'"; exit 2 ;;
esac

hr () { printf '%.0s=' {1..72}; echo; }
say () { echo; hr; echo "$*"; hr; }
die () { echo; echo "*** STOP: $*"; exit 1; }

# ─────────────────── phase 1: manifest / gate basis ────────────────────────
phase_check () {
  say "PHASE 1  frozen baseline manifest"
  $PY freeze_bandit_baseline.py --check \
      2>&1 | tee "$LOGDIR/manifest_${STAMP}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die \
    "manifest does not reproduce on this machine. The competence gate's
    comparison basis is not reproducible, so nothing downstream is citable.
    Do NOT continue."

  say "PHASE 1b  algorithmic corners (shared-module cross-check)"
  for env in easy hard; do
    $PY run_bandit_algorithmic_baseline.py --reference_environment "$env" \
        2>&1 | tee -a "$LOGDIR/algo_${STAMP}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "algorithmic baseline failed on $env"
  done
  echo
  echo "Expect (must match the manifest exactly):"
  echo "  easy  greedy SuffFail 0.25 KxMinFrac 0.040 | random 0.00 / 0.794"
  echo "  hard  greedy SuffFail 0.35 KxMinFrac 0.053 | random 0.00 / 0.742"
}

# ────────────────────────── phase 2: smoke ─────────────────────────────────
phase_smoke () {
  say "PHASE 2  smoke  (Easy/Hard x bare/chat x 3 seeds)"
  echo "GPU memory is sampled in the background to $LOGDIR/gpumem_${STAMP}.log"
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( while true; do
        echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used \
              --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
        sleep 5
      done ) > "$LOGDIR/gpumem_${STAMP}.log" 2>&1 &
    MEM_PID=$!
    trap 'kill $MEM_PID 2>/dev/null' EXIT
  fi

  local t0=$SECONDS
  bash run_bandit_reference.sh "$MODEL" SMOKE \
      2>&1 | tee "$LOGDIR/smoke_${STAMP}.log"
  local rc=${PIPESTATUS[0]}
  local elapsed=$(( SECONDS - t0 ))
  [ "$rc" -eq 0 ] || die "smoke run failed (exit $rc) — see the log above"

  local eps
  eps=$(grep -c "^  Run " "$LOGDIR/smoke_${STAMP}.log" || echo 0)
  echo
  echo "smoke wall-clock : ${elapsed}s over ${eps} episodes"
  if [ "$eps" -gt 0 ]; then
    local per=$(( elapsed / eps ))
    echo "per episode      : ${per}s"
    echo
    echo "Formal Track A budget (smoke horizons equal the formal ones, so this"
    echo "scales directly). NOTE 40 episodes is ONE INTERFACE, not all of it:"
    echo "  A0_BARE  Easy+Hard x 20 seeds = 40 ep = $(( per * 40 / 60 )) min"
    echo "  A0_CHAT  Easy+Hard x 20 seeds = 40 ep = $(( per * 40 / 60 )) min"
    echo "  FULL Track A = 80 episodes / 8000 rounds = $(( per * 80 / 60 )) min"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "peak GPU memory  : $(sort -k2 -n -r "$LOGDIR/gpumem_${STAMP}.log" \
                               2>/dev/null | head -1)"
  fi
}

# ───────────────────── phase 3: protocol correctness ───────────────────────
phase_verify () {
  say "PHASE 3  protocol checks (engineering only — NOT behaviour)"
  $PY - "$OUT_ROOT" <<'PYEOF'
import glob, json, sys

root = sys.argv[1]
cells = ["pv6_smoke_easy_bare", "pv6_smoke_easy_chat",
         "pv6_smoke_hard_bare", "pv6_smoke_hard_chat"]
fails = []

def bad(msg):
    fails.append(msg)
    return "*** FAIL"

for cell in cells:
    files = sorted(glob.glob(f"{root}/{cell}/mdf_*/bandit_pv6_*.json"))
    print(f"\n=== {cell}")
    if not files:
        print(bad(f"{cell}: no result file"))
        continue
    for f in files:
        d = json.load(open(f))
        runs, env = d["runs"], d["environment"]
        at = runs[0]["attestation"]
        ct, tk = at["candidate_tokenization"], at["round_0"]["tokens"]

        inv = {r["invalid_rate"] for r in runs}
        ok_inv = inv == {0.0}
        ok_bos = tk["double_bos"] is False
        ok_anch = bool(tk["injection_is_anchor_tail"])
        ok_end = bool(tk["prompt_endswith_anchor"])
        ok_k = env["k"] == len(ct["suffixes"])
        ok_marg = all(len(r["choice_margins"]) == env["horizon"] for r in runs)
        ok_score = all(isinstance(v, float)
                       for r in runs for v in r["candidate_scores"][0].values())

        print(f"  env={env['name']} K={env['k']} T={env['horizon']} "
              f"n_runs={len(runs)} chat={runs[0]['use_chat']}")
        print(f"    invalid_rate       {inv}  "
              f"{'OK' if ok_inv else bad(cell+': invalid_rate != 0')}")
        print(f"    double_bos         {tk['double_bos']}  "
              f"{'OK' if ok_bos else bad(cell+': DOUBLE BOS')}")
        print(f"    injection token    {tk['injection_token']!r}  "
              f"{'OK' if ok_anch else bad(cell+': injection not anchor tail')}")
        print(f"    ends at anchor     {tk['prompt_endswith_anchor']}  "
              f"{'OK' if ok_end else bad(cell+': prompt not ending at anchor')}")
        print(f"    scoring_mode       {ct['scoring_mode']} "
              f"(single_token={ct['single_token']})")
        print(f"    tokens/candidate   {ct['n_tokens']}  "
              f"{'OK' if ok_k else bad(cell+': candidate count != K')}")
        print(f"    margins per round  {'OK' if ok_marg else bad(cell+': margin count')}")
        print(f"    raw float scores   {'OK' if ok_score else bad(cell+': scores not float')}")
        print(f"    bank_report        cells {d['bank_report']['n_cross_cells_used']}"
              f"/{d['bank_report']['n_cross_cells_total']}, "
              f"max_repeat={d['bank_report']['max_cell_repeat']}")
        print(f"    protocol/steering  {d['protocol_version']} / "
              f"{d['config']['steering']} @ T={d['config']['temperature']}")

print("\n" + "=" * 72)
if fails:
    print(f"PROTOCOL CHECKS FAILED ({len(fails)}):")
    for m in dict.fromkeys(fails):
        print("  -", m)
    sys.exit(1)
print("ALL PROTOCOL CHECKS PASSED")
PYEOF
  [ $? -eq 0 ] || die "protocol checks failed — fix before Track A"

  say "PHASE 3b  role structure (READ THIS BY EYE)"
  for cell in pv6_smoke_easy_bare pv6_smoke_easy_chat; do
    echo; echo "########## $cell ##########"
    $PY - "$OUT_ROOT/$cell" <<'PYEOF'
import glob, json, sys
files = sorted(glob.glob(sys.argv[1] + "/mdf_*/bandit_pv6_*.json"))
if not files:
    print("  (no file)"); raise SystemExit
r0 = json.load(open(files[0]))["runs"][0]["attestation"]["round_0"]
print("---------- RATIONALE PROMPT ----------"); print(r0["rationale_prompt"])
print("---------- RATIONALE RAW -------------"); print(repr(r0["rationale_raw"]))
print("---------- RATIONALE CLEAN -----------"); print(repr(r0["rationale_clean"]))
print("---------- ACTION PROMPT -------------"); print(r0["action_prompt"])
PYEOF
  done
  cat <<'EOF'

Confirm by eye before continuing:
  - task state is in the USER turn
  - the rationale is in the ASSISTANT turn (chat cell), not fed back as user
  - NO <|eot_id|> between the rationale and the anchor (turn is continued)
  - the action prompt ends EXACTLY at "Choice: Button"
  - the rationale reads as reasoning, and sanitization removed any premature
    "Choice:" line without eating the reasoning itself
EOF
}

# ───────────────────────── phases 4-5: Track A ─────────────────────────────
phase_trackA () {
  if [ "$CONFIRM" -ne 1 ]; then
    die "Track A needs --yes. Run the smoke first and READ its output:
    the protocol checks are automated, but the prompt/role structure and the
    timing/memory numbers are for a human to judge."
  fi

  # The competence gate must be IMPLEMENTED and frozen before the data exists.
  # Pre-registering four rules is not the same as fixing their implementation:
  # window boundaries, strict-vs-non-strict comparison and missing-seed
  # handling are all free parameters that can flip a verdict, so they are
  # settled and tested BEFORE A0_BARE rather than after.
  say "PHASE 3c  competence gate evaluator must be frozen and passing"
  [ -f evaluate_competence_gate.py ] || die \
    "evaluate_competence_gate.py not found. Freeze the gate implementation
    BEFORE running Track A — writing it afterwards reintroduces exactly the
    post-hoc freedom the pre-registration exists to remove."
  $PY evaluate_competence_gate.py --selftest \
      2>&1 | tee "$LOGDIR/gate_selftest_${STAMP}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die \
    "gate evaluator selftest failed — do not run Track A against an
    evaluator whose own fixtures do not pass."
  echo "Gate evaluator frozen and passing. It will NOT be edited after this."
  if [ "$ONLY_CHAT" -ne 1 ]; then
    say "PHASE 4  A0_BARE  (N=20, alpha=0) — the gate is judged on THIS"
    local t0=$SECONDS
    bash run_bandit_reference.sh "$MODEL" A0_BARE \
        2>&1 | tee "$LOGDIR/A0_bare_${STAMP}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "A0_BARE failed"
    echo "A0_BARE wall-clock: $(( SECONDS - t0 ))s"
  fi
  if [ "$ONLY_BARE" -ne 1 ]; then
    say "PHASE 5  A0_CHAT  (N=20, alpha=0) — interface comparator only"
    local t0=$SECONDS
    bash run_bandit_reference.sh "$MODEL" A0_CHAT \
        2>&1 | tee "$LOGDIR/A0_chat_${STAMP}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "A0_CHAT failed"
    echo "A0_CHAT wall-clock: $(( SECONDS - t0 ))s"
  fi
  say "PHASE 6  competence gate verdict (reference-bare ONLY)"
  $PY evaluate_competence_gate.py \
      --result "$OUT_ROOT/pv6_easy_bare" \
      --result "$OUT_ROOT/pv6_hard_bare" \
      --json "$LOGDIR/gate_verdict_${STAMP}.json" \
      2>&1 | tee "$LOGDIR/gate_verdict_${STAMP}.log"

  cat <<'EOF'

Track A done. NEXT STEP IS NOT A RUN.
The verdict above is mechanical: point estimates against the FROZEN manifest,
on reference-bare K=4/K=5 only. The paired bootstrap is uncertainty, not a
tiebreaker, and rule 1's churn clause is flagged for reading, not automated.

The competence anchor is the HARDEST reference-bare condition that passed
(Hard > Easy). If neither passed, the alpha sweep can still run on Hard but
its results are FAILURE-MODE CHARACTERIZATION only — do not write
capability-effect / rescue / improvement. That wording distinction is why B1
is deliberately not wired into any launcher until the verdict exists.
EOF
}

case "$PHASE" in
  check)  phase_check ;;
  smoke)  phase_check; phase_smoke; phase_verify ;;
  verify) phase_verify ;;
  trackA) phase_trackA ;;
  *) echo "usage: bash run_pv6_server.sh {check|smoke|verify|trackA} [--yes] [--bare|--chat]"
     exit 2 ;;
esac

echo
echo "logs -> $LOGDIR"
