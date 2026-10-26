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
- **No tests, smoke runs or check modes by default (2026-10-22).** Current experiments are simple behavioural runs; the repo's `test_*.py` suite and the launchers' check/smoke modes were removed. Do not add a `test_*.py`, a `--check`/`smoke` step or a preflight script unless the user asks or the change genuinely needs one (e.g. new layer-indexing or hook code). Verify by reading the code path and by fail-closed asserts inside the runner/analyzer (cell count, n, mask rows, `steering_fires`, provenance).
- For multi-step tasks, state a brief plan (`step → verify: check`). Strong success criteria let you loop independently; weak criteria ("make it work") require constant clarification.

## Project

**Role-Sensitive Networks (RSN)** — dopaminergic adaptive calibration of LLM reasoning via hidden-state steering.

For project facts, experimental claims, frozen-protocol rules, and run conventions, read `CLAUDE.md` first — it is the single canonical source and wins on any disagreement. This file holds only Codex-specific execution guidance. When a convention changes, update `CLAUDE.md` and record here only the Codex-specific delta; do not re-state repo facts in both files. `CLAUDE.md` §0.5 (Instruction maintenance) governs how rules are added, amended, and retired in both files.

**NOTE RENAME (2026-09-12):** the `Note/` docs were renamed (content unchanged). Any
pre-2026-09-12 reference to `AdaDopamine*.md` / `AdaptiveThinking*.md` / `Ada*.md` is a
stale name, not a deleted file — **the mapping table is in `CLAUDE.md`** (search "NOTE
RENAME"). The frozen `docs/*.json` artifacts deliberately still use the old names.

**Required reading before non-trivial changes:**
- `ReasoningBare.md` — current GSM8K re-run state. Since 2026-05-30, old Phase 1/2 GSM8K numbers are not comparable because the prompt and layer-offset pipeline changed (`<|eot_id|>` terminator fix + symmetrized templates); the eot re-run is the authoritative data.
- `ThinkingCurve.md` — Phases 1–2, Plans A–H3 (design rationale, failure analyses, decode-time shape control ≠ acc control), Yerkes–Dodson framing, EMA + 1-step-lag physics, and the current three-component (tonic/ramping/phasic) + gain-coordinate (G/Z) signal framework. **§5 applies the §4 one-dimensional chain to Qwen2.5** (entry-gain linearity, behavioural curve, commit-aligned slow state, per-layer response profile, random/orthogonal readout nulls). **Say "cross-model analysis", not "replication"**: the analysis chain transfers and entry linearity + entry–decode decoupling DO replicate, but the behavioural curve does NOT — Llama shows an **asymmetric peaked response** (sharp optimum α=−6, collapse at −8, decline then flattening on the positive side), Qwen a **high-dose plateau** saturating at +8 with no right arm in band. Its tables are kept SEPARATE from §4's on purpose: different mask, band (L=6 vs L=9) and activation scale mean equal α is **not** an equal intervention, so raw α, raw projections and p-values are not cross-model comparable — only the shape of the behavioural state is. Frozen conclusion there is compatible with **scalar gain compression along a fixed RSN profile**; manifold analysis is open.
- `Behaviour.md` / `Dopamine_backup.md` — behavioral-validation stage (wanting-proxy suite: Betting, Bandit, CGT, IGT, …); `Dopamine_backup.md` (formerly `AdaDopamine_bp.md` / `AdaDopamine_old.md`) holds the raw prior results the curated doc only summarizes.
- Literature & dopamine-LLM mapping live in the `Behaviour.md` / `ReasoningBare.md` family above — the earlier standalone `Dopamine.md` / `Dopamine_EN.md` / `Dopamine2.md` are GONE. `AdaDopamineBehaviour.md` (old name, never renamed — already absent before the 2026-09-12 rename) held the clinical excess-vs-deficit signatures.
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

## Phase 2 closed-loop (SHELVED)

Plans A–H3 predate the 2026-05-30 template/mask-offset fixes and must be re-run before citing. `closed_loop_gsm8k.py` and `run_closed_loop_gsm8k.sh` are in `archive/` (`git mv` both back to resume); the launcher is the source of truth for hyperparameters. Knob list: `docs/BEHAVIOUR_SUITE_ARCHIVE.md`.

## File conventions worth knowing

- `get_answer_*` returns answers (multiple-choice extraction); `get_answer_regenerate_*` does free-form regeneration; `get_action_*` extracts the 0–9 action/willingness scalar.
- `*_lesion.py` zeros target neurons; `*_lesion_complement.py` zeros the complement (sanity control).
- `*_fewshot.py` variants prepend few-shot exemplars; never mix few-shot data with the zero-shot Phase 1 baselines.
- `analysis_*` and `analyze_*` are post-hoc plotting; never call them from training-style scripts.
- `mean/mean_diff.py` is the canonical diff-vector builder — other `mean/mean_*.py` are ablations (consistent / pairs / dice / per-layer).
- `harness.py` + `hf_rsn.py` plug into [lm-evaluation-harness] for benchmark-suite eval.
- **Reported accuracy is the OFFLINE recompute, not the inline field.** Headline = `first_acc` from the frozen `RoleAnswer/analyze_*` parsers; inline `correct` / `accuracy` is run-health and provenance only. **Any analysis that scores correctness itself must reuse the offline extractor and ASSERT its total reproduces the published cell count** — the two differ by 1–2 items in every Qwen MATH cell. Worked example `RoleAnswer/qwen_step2/math_strat_frozen.py`; the two known unpatched extractor gaps are in `CLAUDE.md`.
- **Cross-model ports** get their OWN launcher (the Llama ones are frozen) and never inherit Llama's band or injection token; the `check_*_qwen.py` pre-flights were deleted 2026-10-22, so a new port puts fail-closed asserts inside the runner (non-zero mask rows == `decoder_layer_range(start,end)`, final prompt token id read from the real tokenizer, `steering_fires == L*B*t`). Bands, tokens and the "nominal α is not a common dose" rule are in `CLAUDE.md`.
- **GSM8K/MATH answer extraction is centralized in `utils.py`** (`extract_gsm8k_answer`, `is_correct_gsm8k`, `gsm8k_difficulty`, `extract_math_answer`, `is_correct_math`) — all consumers import from here; do not redefine locally.
- The behavioral-economics / wanting-proxy entry-points (bet / delay / effort / bandit / cgt / cgt_seq / igt / reversal / crt / trait) operationalize "wanting" as overt decisions, not accuracy; each has its own output tree and (mostly) its own `RoleAnswer/analyze_*` parser. See `Behaviour.md` §4 before touching one.
- **Betting (§3.1) is FROZEN as of 2026-07-29 — do not re-run it.** All four cells (Llama/Qwen × GPQA/MMLU) were re-collected with `sample_idx` + `acc_explicit_pct`/`ans_fallback_rate`/`orig_rows`, so every paired statistic in `Behaviour.md` §3.1 is recomputable from stored data. The re-runs closed a SCHEMA gap, not wrong numbers. Betting has **no fixed seed** (`temperature=1.0`), so any further re-run merely resamples: trends replicate, exact values do not byte-match. Cite the stored tables rather than regenerating them.
- **Bandit pv1–PV11 is CLOSED (code in `archive/`); do not reopen.** All pre-2026-07-28 Bandit results are VOID. Verdicts and binding wording: `CLAUDE.md` (Behavioral-economics suite); protocol internals: `docs/BANDIT_PV6_PV11_ARCHIVE.md`; results: `Note/Bandit.md`.

## Local checks

No pytest, no linter, and no `test_*.py` suite (deleted 2026-10-22 with the launchers' check/smoke modes; recover from git if ever needed). Verify as in §0.4. Local offline analysis uses `/Users/paveenhuang/.venvs/rsn-analysis/bin/python` (Python 3.12 with numpy/matplotlib); `python3.10` no longer exists on this box. **Server launchers must use `PY="${PY:-python}"`** — the server conda env names its interpreter `python`. `bash -n` does not catch the `${N:?...}` brace bug (`CLAUDE.md` §0.0).

## Steering-alignment convention (load-bearing)

- **Default = bare-string; do NOT pass `--use_chat`.** The NMD mask / diff vectors are extracted on bare-string prompts, so steering must inject into the same bare activation distribution. `apply_chat_template` shifts the residual geometry and dilutes steering. The only deliberate `--use_chat` exceptions are the betting scripts (`run_gpqa_bet.sh` / `run_mmlu_bet.sh` + `_running`) and CGT chat modes — treated as a feature there, not a confound. Many `.py` carry a `use_chat` arg their `.sh` never passes: "has the flag" ≠ "enabled" — check the `.sh`.
- **Stored `ema_decode` is PREFILL-seeded; the project `s_t` is decode-seeded.** Recompute `s_t` from `x_decode` (`phase1_gain.decode_ema`) — never read the stored field directly. The entry value grows with α (≈−5 at α=0, ≈+290 at α=+12), so the contamination is α-dependent and silently corrupts cross-α decode comparisons. Companion rules: locate a commit marker with the tokenizer (not a char ratio), and never present `p_t` as a channel independent of `s_t` — it is a residual of the same 1-D projection.
- **HDF5 collection refuses to overwrite** unless `--allow_overwrite` / `ALLOW_OVERWRITE=1`; `g.attrs["generated"]` is stored untruncated so agreement checks are valid.
- **`‖m_l‖²` comes from the MASK FILE; never regress the per-layer prefill mean on α to recover it.** The co-design identity `Δμ_l == α·‖m_l‖²` holds ONLY for the first steered layer: injection lands on each layer's OUTPUT, which is the next layer's INPUT, so later layers carry propagated residue. Measured on `qwen25_signal_v1` the α-slope reads [3.077, 5.327, 9.545, 14.331, 36.834, 73.923] against a file of [3.077, 3.392, 4.673, 7.925, 21.128, 41.682] — only L15 matches, and the per-sample std grows 0.009→1.540 across the band, which IS the propagation. A mismatch is not evidence of a stale mask. (`Z` cancels `‖m_l‖²`; only `G` is affected.)

- **Steering is prefill-only + output-side.** `regenerate(prefill_only=True)` injects `α×mask` at the last prompt token's OUTPUT (`hs[:, -1, :] += diff`); decode is untouched. Trackers (`track_dopamine_signal.py` / `track_hidden_states.py`) inject via `register_forward_hook` on the layer OUTPUT inside the same observation hook (post-injection signal). Never re-introduce an INPUT-side pre-hook for steering — it causes a one-layer mask misalignment (the pre-2026-06-28 bug; all pre-fix α≠0 signal/HS data is layer-misaligned).

## Server / data layout

Current GSM8K re-runs run on `/data1/paveen/Dopamine/` (server). Only code is in git; `components/`, `benchmark/`, `llama3/dopamine/`, H5 hidden states, and JSON answer dumps are not. Older experiments still have hard-coded `WORK_DIR=/data1/paveen/RolePlaying`; migrate them only when re-running that experiment family. The server `benchmark/` tree is FLAT single-files (`mmlu_all.json`, `gpqa_train.json`, …), not per-dataset subdirs.

## Offline analysis workspace

Analysis + plotting live OUTSIDE this repo at `~/Documents/RSNResult/RoleAnswer/` (relocated 2026-07-16 from `~/Downloads/RSNResult/`; not in git). **Run its scripts with `/Users/paveenhuang/.venvs/rsn-analysis/bin/python`** (2026-10-26; `python3.10` is gone and the system `python3` lacks numpy). Authoritative accuracy = `analyze_first_last_acc.py` (GSM8K first-`####` / MATH first-`\boxed{}`, fallback chain; pass `--gsm8k_root llama3/gsm8k`) — NOT the inline `correct_*`/`pred_answer` fields the generation scripts store. `phase1_gain.py` recomputes the signal in fixed G (α-unit) / Z (layer-fair) gain coordinates from the stored per-layer projections without touching the server. Phase-1 signal panels: `analyze_alpha_dose.py` (§4.4 α-dose — linear input → nonlinear slow `s_t` peaking at −6, tracks acc r=+0.74; **not an inverted-U** — the quadratic peak is −1.9 vs the discrete −6, so say `asymmetric peaked working-point response`; confidence co-moves so α is NOT selective wanting) and `analyze_cot_alpha.py` (§4.5 CoT×α=−4 2×2 per-question DiD — headline = time-center structure, not orthogonal levers; separate statistical from practical interaction, `_verdict()` labels are heuristics not tests). `analyze_pt_frequency.py` (+ `pt_freq_between_alpha.py`, run `-u`) is the §4.7 `p_t` amplitude-vs-frequency validation: **amplitude (centered RMS) carries the α effect (pre-commit RMS −6 vs 0 = +0.046, p<.001), ALL frequency metrics are α-null and commit-centered spec-entropy is `####`/repetition-confounded** → `p_t` keeps its phasic-like operational definition (retained): this task supports an amplitude change but did NOT detect stable frequency organization; frequency is a negative control. `analyze_slow_state_behavior.py` is the TODO §2 validation (11 conds × 300 pooled). **Modeling-vs-finding split (ThinkingCurve.md §2.2):** `s_t` SLOPE = ramping/vigor operationalization (kept), LEVEL = commitment-state operationalization (kept); the test asks only whether THIS task elicits the slope's behavioral association — a null is task-level, NOT a rename of `s_t`. Leakage guards (v2, load-bearing): FIXED `[0,W)` predictor + **at-risk subset (commit_step≥W)** for commit-timing (23% commit before token 20 → their `[0,W)` straddles commit) + frozen train scaler for held-out + fixed-window confidence + α-condition FE. **RESULT (task-level): GSM8K commit-timing did NOT elicit a ramping/vigor slope signal; `s_t` behaves as a slow engagement / commitment-state readout** — at-risk LEVEL→commit_step ρ=+0.38/+0.40 (p<1e-85) but leak-free SLOPE ρ=−0.02/+0.01 (ns). The regression's significant slope β=+10.7 (p=.003) is a **SUPPRESSOR artifact** (corr(level,slope)=−0.48; marginal ρ=−0.02; sign +=steeper→LATER commit = opposite to vigor), NOT vigor evidence. premature/抢答 is RETRACTED → diagnostic only (749/754 commit before token W → time-confounded). The vigor slope definition is retained, left to a vigor-eliciting task. See the CLAUDE.md twin for the full 口径 cautions.

## Environment

```bash
bash setup_env.sh   # creates conda env "roleplaying" (py3.10) + bf16/CUDA stack
```

Mistral3 needs `transformers` from main + `mistral-common>=1.8.6` (already in `setup_env.sh`).

## Editing guidance

- **Layer-indexing offset (read before touching any hook/mask code).** `LAYER_START`/`LAYER_END` follow HF hidden_states semantics (index 0 = embedding, 1..N = decoder-layer outputs). Saved masks drop the embedding row (`detection/nmd.py: mask[1:]`), so saved-index `i` ↔ `decoder_layers[i]` ↔ `hidden_states[i+1]`. Use `utils.mask_slice_for(mask, ls, le)` and `utils.decoder_layer_range(ls, le)` (they encode the `-1` offset) instead of raw `mask[ls:le]` / `range(ls, le)`. Verify on the server with `sanity_mask_indexing.py` before changing layer-indexing code (this is the offset bug fixed 2026-05-30).
- Don't refactor `llms.VicundaModel` loading branches casually — Mistral3, dream-diffusion and CausalLM each rely on slightly different hook surfaces.
- Don't change `template.py` strings in place; add a new variant — Phase 1 baselines are tied to exact prompt wording.
- (Phase 2 is shelved; `closed_loop_gsm8k.py` is in `archive/`.) New plans go in `closed_loop_gsm8k.py` behind a new `--plan` value (added to the `choices=[...]` list and as a branch in `_compute_alpha()`); keep prior plans callable for ablation reproducibility.
- The injection–observation loop has a hard 1-step lag (pre-hook injects `α_t` before forward; post-hook reads `x_t`/`ema_t` after, so `α_{t+1}` is decided from `x_t`). Any new controller must assume `α` based on step-`t` observation only takes effect at step `t+1` — controllers that try to react to fast (per-token) signal changes diverge (see Plan D failure in `ThinkingCurve.md` §4.3).
- Tracking projection and steering injection share the same `nmd_mask` (sparse, ~0.5% of neurons per layer). This co-design means injecting `+α` directly raises next-step `x_{t+1}` by `α × ‖mask‖²` — do not change one without the other.
- Result file naming includes plan + k1 + k2 + ema_alpha + layer range, so renaming any of these breaks the analysis scripts (`analyze_plan{D,EF,G,H1,H2,H3}.py` filter by exact filename).
