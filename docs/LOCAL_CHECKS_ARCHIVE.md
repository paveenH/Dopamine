# Local checks — archived commands (CLOSED lines)

Split out of `CLAUDE.md` on 2026-09-02 to shrink the always-loaded context. This
file is the command reference for lines that are **CLOSED or COMPLETE**; nothing
here is needed to run current work. `CLAUDE.md`'s **Local checks** section keeps
the two general traps, the active lines, and the general FakeVC lessons.

The two running rules still apply here: these scripts signal failure through the
**EXIT CODE** (check `$?`, not the last printed line), and **`timeout` does NOT
exist on this macOS box** — wrapping a check in it yields exit 127 for every
script, which looks exactly like a mass test failure. **Use `python3.10`**; plain
`python3` on the analysis box has no numpy.

---

## Bandit line pv6–pv11 (⛔ CLOSED — do not start new Bandit experiments)

Per the PV11 pre-registered termination rule, five intervention classes each
raised the model's *recognition* of uncertainty without moving *acquisition*.
These checks are kept because later protocols reuse the seed banks, gate rules
and analysis 口径 — not because the line is expected to reopen. Design +
results: `AdaBandit.md` §3–5.

```bash

python3.10 test_bandit_reference.py        # pv6 shared module: envs, tapes, seed
                                           # banks, Greedy tie-break, sanitization,
                                           # metrics, manifest, launcher drift
python3.10 test_bandit_pv6_episode.py      # pv6 protocol via a contract-faithful
                                           # FakeVC: steering semantics, anchor,
                                           # BOS, chat role structure, scoring
python3.10 evaluate_competence_gate.py --selftest   # gate rules vs synthetic policies
python3.10 freeze_bandit_baseline.py --check        # frozen gate basis reproduces
python3.10 run_bandit_algorithmic_baseline.py --reference_environment easy
bash -n run_bandit_pv6.sh && bash -n run_bandit_reference.sh   # shell syntax

# B1 numbers (run from RoleAnswer/, needs synced result dirs, no server):
python3.10 analyze_bandit_pv6_alpha.py --part attest   # injection only, fails closed
python3.10 analyze_bandit_pv6_alpha.py                 # all §3.2.5 tables

# pv7 (uses the REAL Llama-3.1 tokenizer from the local HF cache, no GPU):
python3.10 test_bandit_pv7.py              # token 220, candidates 32-35,
                                           # ID-level concat + string/rstrip/
                                           # double-space negative controls
python3.10 test_bandit_pv7_episode.py      # episode runner via a contract-
                                           # faithful FakeVC: both anchors,
                                           # fixed order, alpha=0 hooks none,
                                           # site counts, parser-never-picks,
                                           # resume-key seed content, and the
                                           # driver/wrapper fail-closed greps
python3.10 freeze_pv7_states.py --check    # frozen 20x6 state bank reproduces
python3.10 eval_pv7_frozen_states.py --dry_run   # prompts + validity, no model
python3.10 eval_pv7_stage2_ablation.py --dry_run --source <p1b.json>
                                           # 1440 Stage-2 prompts + token audit
bash -n run_bandit_pv7.sh                  # shell syntax
# pv7 gate (needs a synced pv7 result dir; frozen RULES, pv7 loader):
python3.10 evaluate_competence_gate_pv7.py --result <dir>/pv7_easy_bare

# pv9 (SUPERSEDED by PV10; checks kept — later protocols reuse its basis):
python3.10 test_bandit_pv9.py              # PV9 protocol invariants: the three
                                           # text layers, stop_reason vocabulary,
                                           # Stage-1-only cue/score/history, the
                                           # easy-vs-neartie resume-key split and
                                           # the PV9-manifest rebind
python3.10 freeze_pv9_baseline.py --check  # PV9 frozen basis reproduces AND
                                           # PV9-Easy == pv6-Easy. Floats carry a
                                           # 1e-9 relative tolerance (cross-platform
                                           # ULP); structure/seeds stay exact, and a
                                           # mismatch now prints which paths differ.
bash -n run_bandit_pv9.sh

# pv10 = the last BAI main line, now CLOSED. No GPU; uses the real Llama-3.1
# tokenizer from the local HF cache for the token-220 anchor assertions.
python3.10 test_bandit_pv10.py            # anchor, orders/counterbalance,
                                          # strict parser + its five invalid
                                          # kinds, fire arithmetic (873 != 900)
python3.10 test_bandit_pv10_episode.py    # episode via a contract-faithful
                                          # FakeVC: three terminations, tau
                                          # null on invalid, alpha=0 registers
                                          # NO hook, infra-vs-model failure
python3.10 test_bandit_pv10a.py           # PV10-A differs in EXACTLY one way
                                          # (no COMMIT until the budget ends);
                                          # terminal prompt byte-identical
python3.10 test_pv10_gate_end_to_end.py   # the evaluator can read what the
                                          # DRIVER actually writes ("runs")
python3.10 test_pv10_stop_parity.py       # v2: alpha=0 and +-4 get the SAME
                                          # stop_strings; marker excluded
                                          # before parsing; regex unrelaxed
                                          # (6 checks verified to FAIL against
                                          # pre-fix code)
python3.10 evaluate_pv10_capability.py --selftest
python3.10 evaluate_pv10_capability.py --check
python3.10 pv10_env_prescreen.py --n_sim 10000 --check
bash -n run_bandit_pv10.sh && bash -n run_bandit_pv10a.sh

# pv10c = PV10-B + competitor cue. Minimal on purpose: only the two
# mechanical failures that would void a whole batch (~3s).
python3.10 test_bandit_pv10c.py           # Reason: still ends on token 220;
                                          # C terminal prompt byte-identical
                                          # to B; C prompt reaches the runner;
                                          # B/C resume keys distinct
bash -n run_bandit_pv10c.sh

# pv11 = Controlled Evidence-State Micro-Episodes. The online PV10-A/B/C line
# is closed; PV11 hands the model a SYNTHETIC evidence state instead of one it
# generated, so "willing to continue" and "which arm to sample" separate at a
# state-matched FIRST ACTION. No GPU; the anchor assertions use the real
# Llama-3.1 tokenizer from the local HF cache and print SKIP (never pass
# silently) when it is absent.
python3.10 build_pv11_state_bank.py --check   # the 160-state bank reproduces
                                              # from its builder, and BOTH
                                              # digests match: canonical (over
                                              # the normalized object) and file
                                              # (over the bytes on disk). They
                                              # are different numbers on
                                              # purpose -- see the manifest.
python3.10 test_pv11_state_bank.py            # counts legality, EXACT 5/5/5/5
                                              # marginals, pairwise crossings
                                              # within max-min <= 1 (exact
                                              # joint orthogonality is NOT
                                              # attainable at n=20 and is not
                                              # claimed)
python3.10 test_bandit_pv11.py                # token 220; withheld fields
                                              # (history / totals / probe
                                              # identity) absent; each cell
                                              # differs from its sibling in
                                              # exactly ONE line; (H+1)*L
                                              # fires, NOT PV10's tau-3
python3.10 test_bandit_pv11_episode.py        # FakeVC: no forced init, first
                                              # action captured BEFORE any
                                              # state update, tape pairing
                                              # across cells and alphas, empty
                                              # generation fails closed, and
                                              # attestation rejects a WRONG
                                              # fire count (a fake that models
                                              # INTENT cannot test ARITHMETIC
                                              # -- verified by mutation)
python3.10 analyze_bandit_pv11_gate.py --selftest   # M1/M2 discriminating
                                                    # power, incl. that ONE
                                                    # complete block alone
                                                    # must FAIL
python3.10 test_pv11_gate_end_to_end.py       # the gate reads what the DRIVER
                                              # actually writes ("runs"), by
                                              # running real episodes and
                                              # serializing them -- the PV10
                                              # loader bug is the precedent
bash -n run_bandit_pv11.sh

# One launcher, and it STOPS at GATE by design:
#   bash run_bandit_pv11.sh llama3 CHECK   # every offline check above
#   bash run_bandit_pv11.sh llama3 SMOKE   # 16 DESIGN-BALANCED states, ~10min
#   bash run_bandit_pv11.sh llama3 A0      # 160 states, ~60min
#   bash run_bandit_pv11.sh llama3 GATE    # M1/M2 verdict, then stop
# There is deliberately no +-4 step: the steered cells are gated on alpha=0,
# and a failing gate CLOSES the protocol rather than being re-tuned. Both the
# launcher (no alpha flag) and the driver (--alpha != 0 exits) enforce this,
# independently, so editing one does not silently unlock the other.
python3.10 inspect_pv11_smoke.py <dir>/pv11_pilot.json   # READ THE TEXT: the
                                              # driver's summary line cannot
                                              # show code-completion drift, a
                                              # constant first action, or a
                                              # false claim about sample size

# pv10 gate + tables (needs a synced pv10 result dir; no GPU):
python3.10 evaluate_pv10_capability.py --result <dir>/pv10_a0
python3.10 analyze_bandit_pv10.py --cells am4=<dir>/pv10_am4 \
    a0=<dir>/pv10_a0 ap4=<dir>/pv10_ap4
# pv10c B-vs-C alpha=0 (run from RoleAnswer/; fails closed on pairing):
python3.10 analyze_bandit_pv10c.py --b <dir>/pv10b_v2_a0 --c <dir>/pv10c_a0
# pv9 gate (needs a synced pv9 result dir; frozen RULES, pv9 loader):
python3.10 evaluate_competence_gate_pv9.py --result <dir>/pv9_easy_bare

# PV9 alpha tables (run from RoleAnswer/, needs synced result dirs, no GPU):
python3.10 analyze_bandit_pv9.py --part validity   # attestation only, fails closed
python3.10 analyze_bandit_pv9.py --part all        # every AdaBandit.md section-4 table
python3.10 analyze_bandit_pv9.py --part primary --no-model   # fast: skips the fits

# pv7 frozen-state diagnostics (the lock-in bank; --dry_run needs no GPU):
python3.10 freeze_pv7_lockin_states.py --check     # 123-state bank reproduces
python3.10 test_pv7_stage1_alpha.py        # stage2-never-steered, fail-closed
                                           # fires, critical_arm from the bank,
                                           # seed-clustered bootstrap unit
python3.10 eval_pv7_stage1_alpha.py --dry_run
python3.10 eval_pv7_history_ablation.py --dry_run
python3.10 eval_pv7_calculator.py --dry_run        # prints the re-ranking BAND
bash -n run_bandit_pv8.sh                          # pv8 launcher syntax
```

Each frozen-state evaluator also takes `--report <json>` to re-print its tables
offline, and asserts its shared baseline arm against the previously stored run
(`--pv7_alpha`, `--history_ablation`) — a mismatch is a hard stop, because a
base that did not reproduce makes every contrast above it uninterpretable.

Individual checks are plain `if` statements inside those scripts, so to run one in
isolation, import the module and call the function directly rather than looking for
a test-selection flag.
