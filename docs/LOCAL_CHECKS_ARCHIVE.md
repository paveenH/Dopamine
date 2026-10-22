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
results: `Bandit.md` §3–5.

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
python3.10 analyze_bandit_pv9.py --part all        # every Bandit.md section-4 table
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

## CLAUDE.md text rewritten 2026-10-22 (check/test steps removed)

Original text, byte-identical, of the rules that prescribed a check/test step.

### §0.0 closing sentences

Evidence: CLAUDE.md's own RR/CC and P4 entries; "Local checks" section's
three "bash -n is NOT sufficient" reminders.

**Before writing a new `${N:?...}` line, grep this rule and the "Local
checks" section's `${1:?` reminders — do not rediscover this from a failed
server run.** After writing one, verify it by actually invoking the script
with real arguments and printing the resolved variable, not just `bash -n`.

### §0.4 bullets

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

### Phase 1 / GSM8K bullets

Evidence: the 2026-05-30 offset bug; check_igt_qwen.py / check_mask_qwen.py docstrings.

- **Cross-model PORT pre-flight: run the matching `check_*.py` BEFORE any sweep.** Five exist, all read-only and none needing a full sweep: `check_mask_qwen.py` (mask only, **no GPU and no model weights** — file/dtype, row count == the model's decoder-layer count, non-zero rows EXACTLY `decoder_layer_range(start,end)`, uniform `top_k` sparsity, plus `--compare` to answer "different band or the same neurons relabelled"), and the four paradigm ports `check_gsm8k_qwen.py` / `check_cgt_seq_qwen.py` / `check_igt_qwen.py` / `check_signal_qwen.py` (tokenizer + chat template + double-BOS, the ACTUAL final prompt token at each injection site, mask/band alignment, observed `steering_fires` vs `L*B*t`, and the α=0 execution path). **The injection token is model-specific and must be READ OUT, never assumed** — e.g. IGT's anchor is deliberately EMPTY (an anchor there suppresses the reasoning span both the learning readout and prefill steering need — the CGT-simple4 failure), so injection lands on the assistant header's last token: Llama-3.1 `id 271 '\n\n'` vs Qwen2.5 `id 198 '\n'`. A wrong band or wrong site does not raise; it produces uninterpretable data hours later.

- **Qwen HS backfill is COLLECTED AND ACCEPTED (2026-08-24). Acceptance runs through `check_hs_qwen25.py`, which is READ-ONLY and must pass before any geometry work.** Seven cells under `hidden_states/gsm8k/qwen25_signal_v1/`, ~33.9 GB, frozen by `ACCEPTANCE_20260824.txt` + `HS_MANIFEST_20260824.txt` (sha256 + size per cell). Three checks: integrity (n=300, `n_stored_layers=7`, band `[16,22)`, `mnt=768`, metadata α/CoT agreeing with the filename, per-sample `decode_hs`≡`x_decode_proj`≡`ema_decode_proj` lengths, `question_idx` a full 0..299 cover), projection reproduction, and per-question agreement vs the lightweight cell. Mutation-tested 16/16 against real defects.

- **Cross-model Qwen2.5 GSM8K** (2026-08-20). `run_gsm8k_qwen25.sh` + `check_gsm8k_qwen.py` — its own launcher for the same reason as Mistral's (Llama's `run_gsm8k.sh` plain/pushy/4-role/182-backfill matrix is Llama-specific and frozen). This is a **direct replication** of the Llama dose curve, not a redesign: same 300-question benchmark, same driver, same templates, neutral / plain / greedy / `max_new_tokens=768` / bs=24. Four Qwen-specific facts, none inherited: layers **16–21** written as the exclusive `16-22` (**L=6**; Llama's `11-20` is L=9 and does NOT transfer), mask `nmd_0.5_16_22_7B.npy`, `size=7B` (28 decoder layers), and Qwen's `<|im_end|>` + LIST `generation_config.eos_token_id` (already unioned by `_build_terminators()` — verify, do not "fix"). Steps: `--check` (technical pre-flight only — model/tokenizer/mask/band/paths/fires; makes **no** judgement about results and has no pass/fail-then-block gate), `--baseline` (α=0 → the FINAL No-CoT `mdf_0`, read its raw output by hand), `--nocot` (the other 8 α; deliberately excludes α=0 so `mdf_0` is never re-run or overwritten), `--cot` (−4/0/+4), `--cot6` (−6/+6, extends that same curve). **Every α of one curve must stay on ONE card** (bf16 greedy is not byte-reproducible across GPUs); No-CoT and CoT write different `--ans_file`s (`answer_mdf_gsm8k` / `_cot`) so the two curves can run on two cards in parallel.

  - **The prefill path DOES count fires** (`_regenerate_prefill_only` increments `_steering_fire_count += B * n`), so `--check` verifies `L × B × tail` (6×B×1) for real. α=0 passes a REAL all-zero matrix (`diff = np.load(mask) * alpha`, unconditional at `get_answer_regenerate_gsm8k.py:129`) — hooks DO register and the zero add DOES execute; fires read 0 only because `_layer_is_steered` is False on an all-zero row. **This differs from `get_action_regenerate_gsm8k.py`, whose α=0 really does take a no-mask `get_logits` path** — do not generalize one to the other. Practical consequence: a missing mask file blocks the α=0 cell too.

- **Qwen band-position probe — PREPARED BUT NEVER RUN (as of 2026-08-21): three launchers, zero artifacts.** All Qwen work so far uses `[16,22)` (L=6), chosen as the onset of the layer-wise Expert/Non-Expert Pearson descent. `run_nmd_qwen25.sh` builds a SECOND, EARLIER mask `[11,18)` (L=7) so the band position is **tested rather than assumed**; it does NOT replace the existing mask. Order is fixed in the launcher headers: build the mask → `check_mask_qwen.py --layer_start 11 --layer_end 18 --compare 16 22` → `run_mmlue_qwen25.sh --new` → `--old`. **MMLU-E is the probe because its E option ("I am not sure") is a wanting readout INDEPENDENT of correctness** — E-ratio should move bidirectionally with α while accuracy holds, so a band that cannot move E-ratio without damaging accuracy is not worth spending IGT/GSM8K time on. Two design points that are easy to lose: **(a) the two bands OVERLAP at decoder layers 15–16**, so any difference is "earlier band vs later band", NOT "two independent neuron sets" — read the per-layer Jaccard from `--compare` before wording any conclusion; **(b) the `[16,22)` arm is RE-RUN rather than cited from stored numbers**, and both arms include α=0, so each band carries its own same-card baseline and an E-ratio shift cannot be a baseline or cross-GPU artifact. Keep both arms on ONE card. `run_igt_qwen25.sh` is the downstream consumer if a band wins.
