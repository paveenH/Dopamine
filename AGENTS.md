# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 0. Operating Playbook

Behavioral guidelines to reduce common LLM coding mistakes. **Tradeoff:** these bias toward caution over speed. For trivial tasks, use judgment.

### 0.1 Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 0.2 Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it. Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 0.3 Surgical Changes
**Touch only what you must. Clean up only your own mess.**
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused; don't remove pre-existing dead code unless asked.
- The test: every changed line should trace directly to the user's request.

### 0.4 Goal-Driven Execution
**Define success criteria. Loop until verified.**
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
- For multi-step tasks, state a brief plan (`step → verify: check`). Strong success criteria let you loop independently; weak criteria ("make it work") require constant clarification.

## Project

**Role-Sensitive Networks (RSN)** — dopaminergic adaptive calibration of LLM reasoning via hidden-state steering. The user-level `~/CLAUDE.md` contains the full theory map and phase plan; this file covers only repo-local conventions and recent (Phase 2 GSM8K) work. The definitive, continuously-updated repo brief is `CLAUDE.md` (its Claude twin) — when this file and `CLAUDE.md` disagree, `CLAUDE.md` wins; keep the two in sync when editing repo conventions.

**Required reading before non-trivial changes:**
- `AdaDopamine_gsm8k.md` — current GSM8K re-run state. Since 2026-05-30, old Phase 1/2 GSM8K numbers are not comparable because the prompt and layer-offset pipeline changed (`<|eot_id|>` terminator fix + symmetrized templates); the eot re-run is the authoritative data.
- `AdaptiveThinking.md` — Phases 1–2, Plans A–H3 (design rationale, failure analyses, decode-time shape control ≠ acc control), Yerkes–Dodson framing, EMA + 1-step-lag physics, and the current three-component (tonic/ramping/phasic) + gain-coordinate (G/Z) signal framework.
- `AdaDopamine.md` / `AdaDopamine_bp.md` — behavioral-validation stage (wanting-proxy suite: Betting, Bandit, CGT, IGT, …); `_bp` holds the raw prior results the curated doc only summarizes.
- Literature & dopamine-LLM mapping live in the `AdaDopamine*.md` family above — the earlier standalone `Dopamine.md` / `Dopamine_EN.md` / `Dopamine2.md` are GONE. `AdaDopamineBehaviour.md` holds the clinical excess-vs-deficit signatures.
- **Parent paper (RSN):** `ACLARR/` in this repo (LaTeX `main.tex`) — *"Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering"* (ACL ARR submission). This project extends its §6.1 "Digital Dopamine" hypothesis into empirical wanting validation. Moved here 2026-07-30 from `~/Downloads/ACL/ACLARR`; older refs to that path (or to `~/Downloads/ACLARR`) are stale.
- `~/CLAUDE.md` — running commands and data-directory map

## Architecture: how a run is wired together

A typical experiment is one of the `get_answer_*.py` / `get_action_*.py` entry-points driven by a sibling `run_*.sh` script. The dataflow is:

1. **Dataset loader** (`data_<benchmark>.py`) → JSON of question dicts (in `benchmark/` on the server, gitignored).
2. **`template.select_templates(suite=…)`** picks the prompt family. Three suites exist:
   - `default` — MMLU-style A/B/C/D answer extraction
   - `vanilla` — neutral phrasing without role
   - `action` — self-reported "reasoning willingness" 0–9
   Each suite has CoT / non-CoT / E-option (with "I am not sure") variants. The chosen template is rendered per-role via `utils.construct_prompt` + `utils.make_characters`.
3. **`llms.VicundaModel`** loads the LM. Three loading paths: `dream` diffusion (`AutoModel`, see `diffusion.py`), Mistral3 multimodal (`Mistral3ForConditionalGeneration`), or default `AutoModelForCausalLM`. All use bf16 + `device_map="auto"`. `_find_decoder_layers()` is the abstraction used by every layer-injection hook.
4. **Steering / closed-loop** is layered on top via forward hooks on the decoder layers identified in step 3. The diff vectors come from `mean/` (per-role mean differences) gated by a mask from `detection/` (NMD / KL / KS / LR / PCA / t-test / XGB selectors over `task_list.py`).
5. **Output**: current GSM8K re-run artifacts are written under `/data1/paveen/Dopamine/components/...`. Older experiment scripts may still point at the historical `/data1/paveen/RolePlaying/components/...` tree.

## Phase 2 closed-loop

`closed_loop_gsm8k.py` + `run_closed_loop_gsm8k.sh` is the active iteration target. The shell script is the source of truth for hyperparameters; key knobs:

- `LAYER_START`/`LAYER_END` — injection layer range (Llama3-8B uses 11–20)
- `EMA_ALPHA` — feedback smoothing (Plans D/E/F all read EMA via `self.ema_alpha`)
- `K1` — primary gain. Meaning is plan-dependent: proportional gain (A/B/C/D/E/F/H2/H3), fixed pulse magnitude (G/H1)
- `K2` — secondary knob. Plan C: spike-damping coefficient. Plan G: dead-zone half-width as fraction of `xp`. Unused by D/E/F/H1/H2/H3
- `FLOOR_RATIO` — multi-purpose target ratio. Tonic floor (A/C), EMA homeostasis target (E), peak target (H1: 1.5, H2: 1.25, H3: 1.35)
- `PLATEAU_END_RATIO` — slope endpoint for B; H2/H3 reuse it as the trapezoid end level (default 0.75·xp)
- `AVG_GEN_LEN` — defines T for B's plateau slope and H2/H3's trapezoid decay endpoint
- `--plan {none,static,A,B,C,D,E,F,G,H1,H2,H3}` selects controller (case-sensitive; see `AdaptiveThinking.md` §3.2)
- H1 / H2 / H3 hardcode their window/segment boundaries inside `_compute_alpha()` — there are no `--h1_window` or trapezoid-segment CLI flags. Edit the constants in `closed_loop_gsm8k.py` if you need to sweep them.

When tweaking a plan, modify the `.sh` not the `.py` — the script is committed and reproducible. Past runs are kept commented (not deleted) in `run_closed_loop_gsm8k.sh` so the full sweep history is recoverable.

## File conventions worth knowing

- `get_answer_*` returns answers (multiple-choice extraction); `get_answer_regenerate_*` does free-form regeneration; `get_action_*` extracts the 0–9 action/willingness scalar.
- `*_lesion.py` zeros target neurons; `*_lesion_complement.py` zeros the complement (sanity control).
- `*_fewshot.py` variants prepend few-shot exemplars; never mix few-shot data with the zero-shot Phase 1 baselines.
- `analysis_*` and `analyze_*` are post-hoc plotting; never call them from training-style scripts.
- `mean/mean_diff.py` is the canonical diff-vector builder — other `mean/mean_*.py` are ablations (consistent / pairs / dice / per-layer).
- `harness.py` + `hf_rsn.py` plug into [lm-evaluation-harness] for benchmark-suite eval.
- **GSM8K/MATH answer extraction is centralized in `utils.py`** (`extract_gsm8k_answer`, `is_correct_gsm8k`, `gsm8k_difficulty`, `extract_math_answer`, `is_correct_math`) — all consumers import from here; do not redefine locally.
- The behavioral-economics / wanting-proxy entry-points (bet / delay / effort / bandit / cgt / cgt_seq / igt / reversal / crt / trait) operationalize "wanting" as overt decisions, not accuracy; each has its own output tree and (mostly) its own `RoleAnswer/analyze_*` parser. See `AdaDopamine.md` §4 before touching one.
- **Betting (§3.1) is FROZEN as of 2026-07-29 — do not re-run it.** All four cells (Llama/Qwen × GPQA/MMLU) were re-collected with `sample_idx` + `acc_explicit_pct`/`ans_fallback_rate`/`orig_rows`, so every paired statistic in `AdaDopamine.md` §3.1 is recomputable from stored data. The re-runs closed a SCHEMA gap, not wrong numbers. Betting has **no fixed seed** (`temperature=1.0`), so any further re-run merely resamples: trends replicate, exact values do not byte-match. Cite the stored tables rather than regenerating them.
- **⚠️ Bandit: all pre-2026-07-28 results are VOID** (best-arm position leakage + a permissive parser made OptFrac indistinguishable from a first-option bias). The published inverted-U / peak +2 is retired, which also breaks the "Bandit +2 / IGT +2 / GSM8K −6" cross-task table. Launchers are now `run_bandit_llama3.sh` / `run_bandit_qwen25.sh` (`run_bandit.sh` deleted). A `seed` field in the run JSON marks post-fix data. See CLAUDE.md's Bandit entry before citing or re-running.
- **Bandit's current main line is pv6 (F-reference), a 2026-08 clean-slate redesign** — `bandit_reference.py` + `bandit_pv6_episode.py`, dispatched by `get_answer_bandit.py --reference_environment {easy,hard,native_floor}` (pv1–pv5 paths byte-unchanged). Nothing in pv6 is comparable to a pv1–pv5 cell. It measures a **capability boundary** (whether α moves it), with a **pre-registered competence gate** frozen in `evaluate_competence_gate.py` and a frozen baseline manifest (`freeze_bandit_baseline.py --check`) — neither may be edited now that data exists. Design doc `BanditExperiment_LiteratureReview.md` §3; full caveats in CLAUDE.md's pv6 entry.
- **pv6 gate verdict (2026-08-04, N=20): Easy-bare PASS, Hard-bare FAIL on rule 1 only.** Easy-bare is the B1 competence anchor; **Hard-bare α is failure-mode characterization only** (capability-effect / rescue / improvement are not available for hard). Hard's failure is exploration COVERAGE, not value integration — 10/20 episodes never tried all 5 arms, but conditional on discovering the best arm its late OptFrac (0.725) matches Easy's (0.739). Its rule-1 gap has a bootstrap CI straddling 0, so write "did not pass the gate", never "significantly worse than Greedy".
- **pv6 steering scope**: `--steering_scope {action,both}`. `both` (α in BOTH the rationale and action pass, once at each pass's last prefill token, prefill-only) is the B1 MAIN experiment; `action` is its mechanism ablation. **α=0 is identical under both scopes** (no hook registered either way), so stored Track A α=0 cells are reused rather than re-run. The scope segment `scbothsv1` is appended to the resume key only for a non-default scope, and `PROTOCOL_VERSION` stays `pv6`. One `--ans_file` per α — the detail JSON name carries neither α nor scope, and the summary CSV is per-dir. `steering_fires` in each record counts **injection SITES** (layer × sequence × position), not hook calls: the `11-20` band steers **9** layers (`decoder_layer_range` is half-open), so a T=100 episode must read rationale 900, action 3600 (K=4) or 4500 (K=5).
- **pv6 B1 result (Easy-bare both-stage ±4, N=20, 2026-08-05): +4 damages PERSISTENCE, not discovery; −4 is behaviourally null but distributionally active.** 96.6% of +4's extra switching is on already-tried arms and 99.6% of its extra non-greedy rounds are post-discovery, while `best_never_tried` stays 0/20 — so this is not "more exploration". No inverted-U: α=0 is already ceiling-adjacent on Easy, leaving −α no headroom. Hard-chat (2026-08-05) fails rule 1 at exactly Hard-bare's 0.450 — chat fixes coverage but collapses convergence on both environments, relocating the failure rather than rescuing it. Tables in `AdaDopamine.md` §3.2.5; frozen parser `RoleAnswer/analyze_bandit_pv6_alpha.py` (fails closed on an unverified injection, pairs by seed, matched-state text comparison, tie-tolerant empirical-best matching the gate).
- **A working "rationale −4 + action +4" would show RSN composes into a phased division-of-labour controller — NOT that a single α is dopamine-like** (the episode uses two opposite directions). The `{0,−4}×{0,+4}` interaction test is deferred: it needs separate `rationale_alpha`/`action_alpha` parameters, and `both(−4)` cannot serve as the `(−4,0)` cell.
- **pv7 (2026-08-05) is the Bandit main line; pv6 stays citable as a boundary finding but its α numbers cannot be pooled with pv7's.** pv7 rebuilds the interface after four MEASURED pv6 defects: rationale truncated at the 64-token cap (74–93% end mid-sentence; only 224/2000 reach `## Step 2`), Stage 2 still carrying "Do not state a final choice yet" above the anchor, option-display drift from the TRIED/UNTRIED split (24/120 frozen slots), and a **label** prior (round-1 ~95% "Button A" while display position spreads {1:10,2:4,3:5,4:1} — so counterbalancing position does not fix it; use label fixed effects). Both pv7 stages end in one ASCII space → **token 220**, the same bottleneck token the RSN mask was extracted at; candidates are bare `A`–`E` (32–36) and **must be concatenated at the ID level** — the string path merges `220+A` back into `' A'`(362), shortens the sequence and slides injection to the colon, silently. Files `bandit_pv7.py` / `test_bandit_pv7.py` / `freeze_pv7_states.py` / `eval_pv7_frozen_states.py`; pv6 files byte-unchanged. Anchor collisions and format problems are OBSERVED, never cleaned (report overall AND collision-free; a high rate disqualifies the prompt version). Frozen bank = 120 slots / 107 unique histories: per-type tables keep 20 slots each, pooled stats dedup on `state_fingerprint` and report n=107. Prompts are chosen on validity/grounding/completion/cost, NEVER on true-best outcome. pv7 needs its own α=0 competence-gate re-run before any α claim. Full detail in CLAUDE.md's pv7 entry.

- **pv7 Stage 1 = P1b, Stage 2 = S1; both frozen on the state bank BEFORE any trajectory, on validity/grounding/completion/cost — never reward.** P1 (39.3% policies naming no button, 17.8% continuing into the next prompt) and P2 (its hint became a policy prior) both failed selection; P1b adds explicit termination + a mandatory named button and no hint → parse rate 60.7%→**100%**. **`native_ends_after_policy=79.4%` is NOT stopping** — 119/120 still hit the token cap, so write "parser-assisted interface succeeded, native stopping failed". Three Stage 1 rewrites moved Button-A coverage only ~6pp, which is what pointed at Stage 2: `eval_pv7_stage2_ablation.py` re-scores the frozen P1b rationales verbatim under `{S0,S1,S_maskpolicy} × 4 OPTIONS rotations` (1440 scorings, ~10 min) and **S1 ("Follow the Policy above and select one button.") cut non-A-policy override 54.3%→1.9%** (McNemar n=428, discordant 178/0, p<1e-4), Button C follow 10.9%→95.7%, rotation-invariant (0/107 vs 14/107), margin +0.83 with entropy −0.10 (mass moved onto the target, not flattened).
- **Label effect ≈ 6× row-position effect** (balanced rotations: label span 4.73 vs row span 0.76; choice share flat across rows). Label and position are **separable, not collinear** — A sits in row 1 in only 54/120 states. The trap: the `scores` dict key order is always A,B,C,D, **not** display order; read `arm_order`. Estimate label FE **controlling for `is_policy_target`**, or target-frequency imbalance reads as prior (S1's raw span 3.08 → **+0.30** for non-target A vs S0's +2.17).
- **pv7 splits α into `rationale_alpha` / `action_alpha`; `steering_scope` is NOT inherited** (it cannot express `(α,0)`, now the primary cell). Decided in advance: S1 leaves Stage 2 ~1.9% room to overrule the Policy, so **pv7's action-only cell is expected near-null and that null means "the executor is robust", not "α does not affect the action"** — pv6's action-vs-both contrast does not transfer.
- **pv7 trajectory line** = `bandit_pv7_episode.py` (standalone; importing it must not pull in `bandit_pv6_episode`) + thin driver `run_bandit_pv7_episodes.py` + `run_bandit_pv7.sh` + `evaluate_competence_gate_pv7.py`. Both stages share ONE fixed OPTIONS order — rotation is ablation-only. Output is `bandit_pv7_*.json` in a pv7-only tree and **never disguised as `bandit_pv6_*.json`**: the frozen gate loader is deliberately blind to pv7, so the wrapper adds a loader and calls the SAME frozen `evaluate()` (rules imported, never copied). `environment.name` stays `reference_easy` — the environment is unchanged, which keeps the frozen seed banks and baselines valid. Records separate Stage 1 `policy_target` from Stage 2 `action`/`action_follows_policy`, so a failure has an address (decision quality vs execution consistency; both diagnostic, neither a gate rule).
- **pv7 fail-closed rules:** `steering_fires` checked per episode against `expected_fires` (Easy L=9/K=4/T=100 → rationale-only `{900,0}`, action-only `{0,3600}`), and **`None` is a FAILURE** — `steering_fire_count` is unconditional (llms.py:814), so `None` means a stale server `llms.py`; sync it, never add an escape flag. The gate wrapper's `all_unsteered` also fails on a missing count. The **resume key hashes seed CONTENT**, not `len(seeds)` (a count lets the smoke bank `6 12 13` and any other 3-seed set resume into each other). Two different non-zero alphas raise. `parse_policy` lives in `bandit_pv7.py` as a **hard dependency, no fallback** — a fail-open parser emits data that looks valid and cannot be analysed. `POLICY_PARSER_VERSION` is metadata only; changing the **extractor** must bump protocol/resume key.
- **pv7 is LLAMA3-ONLY** (token 220 / candidates 32–35 are Llama-3.1 facts). `run_bandit_pv7.sh qwen25` exits 1; adding Qwen means re-auditing both invariants and re-freezing the bank. **Frozen wording for a pv7 gate pass:** competence under a *structured, parser-assisted* interface with Policy-following constrained action — **not** native free generation.

## Local checks

No pytest, no linter. Standalone scripts that exit non-zero on failure; run with `python3.10` (plain `python3` lacks numpy):
`test_bandit_reference.py`, `test_bandit_pv6_episode.py`, `evaluate_competence_gate.py --selftest`, `freeze_bandit_baseline.py --check`, `bash -n run_bandit_pv6.sh`.
pv7 (real Llama-3.1 tokenizer from the local HF cache, still no GPU): `test_bandit_pv7.py`, `test_bandit_pv7_episode.py`, `freeze_pv7_states.py --check`, `eval_pv7_frozen_states.py --dry_run`, `eval_pv7_stage2_ablation.py --dry_run --source <p1b.json>`, `bash -n run_bandit_pv7.sh`; with a synced result dir, `evaluate_competence_gate_pv7.py --result <dir>/pv7_easy_bare`.

## Steering-alignment convention (load-bearing)

- **Default = bare-string; do NOT pass `--use_chat`.** The NMD mask / diff vectors are extracted on bare-string prompts, so steering must inject into the same bare activation distribution. `apply_chat_template` shifts the residual geometry and dilutes steering. The only deliberate `--use_chat` exceptions are the betting scripts (`run_gpqa_bet.sh` / `run_mmlu_bet.sh` + `_running`) and CGT chat modes — treated as a feature there, not a confound. Many `.py` carry a `use_chat` arg their `.sh` never passes: "has the flag" ≠ "enabled" — check the `.sh`.
- **Steering is prefill-only + output-side.** `regenerate(prefill_only=True)` injects `α×mask` at the last prompt token's OUTPUT (`hs[:, -1, :] += diff`); decode is untouched. Trackers (`track_dopamine_signal.py` / `track_hidden_states.py`) inject via `register_forward_hook` on the layer OUTPUT inside the same observation hook (post-injection signal). Never re-introduce an INPUT-side pre-hook for steering — it causes a one-layer mask misalignment (the pre-2026-06-28 bug; all pre-fix α≠0 signal/HS data is layer-misaligned).

## Server / data layout

Current GSM8K re-runs run on `/data1/paveen/Dopamine/` (server). Only code is in git; `components/`, `benchmark/`, `llama3/dopamine/`, H5 hidden states, and JSON answer dumps are not. Older experiments still have hard-coded `WORK_DIR=/data1/paveen/RolePlaying`; migrate them only when re-running that experiment family. The server `benchmark/` tree is FLAT single-files (`mmlu_all.json`, `gpqa_train.json`, …), not per-dataset subdirs.

## Offline analysis workspace

Analysis + plotting live OUTSIDE this repo at `~/Documents/RSNResult/RoleAnswer/` (relocated 2026-07-16 from `~/Downloads/RSNResult/`; not in git). **Run its scripts with `python3.10`** (the bare `python3` there lacks numpy). Authoritative accuracy = `analyze_first_last_acc.py` (GSM8K first-`####` / MATH last-`\boxed{}`, fallback chain; pass `--gsm8k_root llama3/gsm8k`) — NOT the inline `correct_*`/`pred_answer` fields the generation scripts store. `phase1_gain.py` recomputes the signal in fixed G (α-unit) / Z (layer-fair) gain coordinates from the stored per-layer projections without touching the server. Phase-1 signal panels: `analyze_alpha_dose.py` (§4.4 α-dose — linear input → inverted-U slow `s_t` peak −6, tracks acc r=+0.74; confidence co-moves so α is NOT selective wanting) and `analyze_cot_alpha.py` (§4.5 CoT×α=−4 2×2 per-question DiD — headline = time-center structure, not orthogonal levers; separate statistical from practical interaction, `_verdict()` labels are heuristics not tests). `analyze_pt_frequency.py` (+ `pt_freq_between_alpha.py`, run `-u`) is the §4.7 `p_t` amplitude-vs-frequency validation: **amplitude (centered RMS) carries the α effect (pre-commit RMS −6 vs 0 = +0.046, p<.001), ALL frequency metrics are α-null and commit-centered spec-entropy is `####`/repetition-confounded** → `p_t` keeps its phasic-like operational definition (retained): this task supports an amplitude change but did NOT detect stable frequency organization; frequency is a negative control. `analyze_slow_state_behavior.py` is the TODO §2 validation (11 conds × 300 pooled). **Modeling-vs-finding split (AdaptiveThinking.md §2.2):** `s_t` SLOPE = ramping/vigor operationalization (kept), LEVEL = commitment-state operationalization (kept); the test asks only whether THIS task elicits the slope's behavioral association — a null is task-level, NOT a rename of `s_t`. Leakage guards (v2, load-bearing): FIXED `[0,W)` predictor + **at-risk subset (commit_step≥W)** for commit-timing (23% commit before token 20 → their `[0,W)` straddles commit) + frozen train scaler for held-out + fixed-window confidence + α-condition FE. **RESULT (task-level): GSM8K commit-timing did NOT elicit a ramping/vigor slope signal; `s_t` behaves as a slow engagement / commitment-state readout** — at-risk LEVEL→commit_step ρ=+0.38/+0.40 (p<1e-85) but leak-free SLOPE ρ=−0.02/+0.01 (ns). The regression's significant slope β=+10.7 (p=.003) is a **SUPPRESSOR artifact** (corr(level,slope)=−0.48; marginal ρ=−0.02; sign +=steeper→LATER commit = opposite to vigor), NOT vigor evidence. premature/抢答 is RETRACTED → diagnostic only (749/754 commit before token W → time-confounded). The vigor slope definition is retained, left to a vigor-eliciting task. See the CLAUDE.md twin for the full 口径 cautions.

## Environment

```bash
bash setup_env.sh   # creates conda env "roleplaying" (py3.10) + bf16/CUDA stack
```

Mistral3 needs `transformers` from main + `mistral-common>=1.8.6` (already in `setup_env.sh`).

## Editing guidance

- **Layer-indexing offset (read before touching any hook/mask code).** `LAYER_START`/`LAYER_END` follow HF hidden_states semantics (index 0 = embedding, 1..N = decoder-layer outputs). Saved masks drop the embedding row (`detection/nmd.py: mask[1:]`), so saved-index `i` ↔ `decoder_layers[i]` ↔ `hidden_states[i+1]`. Use `utils.mask_slice_for(mask, ls, le)` and `utils.decoder_layer_range(ls, le)` (they encode the `-1` offset) instead of raw `mask[ls:le]` / `range(ls, le)`. Verify on the server with `sanity_mask_indexing.py` before changing layer-indexing code (this is the offset bug fixed 2026-05-30).
- Don't refactor `llms.VicundaModel` loading branches casually — Mistral3, dream-diffusion and CausalLM each rely on slightly different hook surfaces.
- Don't change `template.py` strings in place; add a new variant — Phase 1 baselines are tied to exact prompt wording.
- New plans go in `closed_loop_gsm8k.py` behind a new `--plan` value (added to the `choices=[...]` list and as a branch in `_compute_alpha()`); keep prior plans callable for ablation reproducibility.
- The injection–observation loop has a hard 1-step lag (pre-hook injects `α_t` before forward; post-hook reads `x_t`/`ema_t` after, so `α_{t+1}` is decided from `x_t`). Any new controller must assume `α` based on step-`t` observation only takes effect at step `t+1` — controllers that try to react to fast (per-token) signal changes diverge (see Plan D failure in `AdaptiveThinking.md` §4.3).
- Tracking projection and steering injection share the same `nmd_mask` (sparse, ~0.5% of neurons per layer). This co-design means injecting `+α` directly raises next-step `x_{t+1}` by `α × ‖mask‖²` — do not change one without the other.
- Result file naming includes plan + k1 + k2 + ema_alpha + layer range, so renaming any of these breaks the analysis scripts (`analyze_plan{D,EF,G,H1,H2,H3}.py` filter by exact filename).
