# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Role-Sensitive Networks (RSN)** — dopaminergic adaptive calibration of LLM reasoning via hidden-state steering. The user-level `~/CLAUDE.md` contains the full theory map and phase plan; this file covers only repo-local conventions and recent (Phase 2 GSM8K) work.

**Required reading before non-trivial changes:**
- `AdaptativeThinking0529.md` — current state. 2026-05-30 起 GSM8K template 已對稱化、mask offset bug 已修,所有舊 Phase 1/2 數字與當前 pipeline 不可比。Phase 1b 重做為當前優先工作;Phase 2 (Plans A–H3) 結論暫擱置,須重跑
- `AdaptativeThinking.md` — 歷史 Phase 1–2 設計、Plan A–H3 動機 / 失敗分析、Yerkes–Dodson framing、EMA + 1-step-lag physics(設計思路仍有效,但數字本身已過時)
- `Dopamine.md` / `Dopamine_EN.md` / `Dopamine2.md` — literature & mapping
- `AdaDopamineBehaviour.md` — clinical behavioral signatures of dopamine excess (anxiety) vs. deficit (anhedonia / loss of drive / bradykinesia). Grounds the two-arm framing: α positive → over-wanting / anxiety (§2.2); α negative → under-wanting (§2.3). Reference for the behavioral-economics suite below, not a pipeline doc.
- `Ada_Dopamine.md` — **behavioral-validation stage of the four-part research line** (RSN → behavioral dopamine → brain dopamine → thinking curve). The source of truth for the wanting-proxy experiments (§3 MCQ/GSM8K/MATH cross-model tables; §4 = the full ⑤–⑩ experiment catalog: Betting / Bandit / Pressure / Effort-choice, plus the *skipped* PIT / Reversal / Agentic / TRAIT with documented why-skipped reasons). When working on any behavioral-economics entry-point, read the matching §4.x here for the design rationale and prior results. **Note its GSM8K numbers are pre-eot** (peak α=−4) and superseded by `AdaptativeThinking0529.md` (peak α=−6); unify before citing.
- `Ada_Dopamine2.md` — brain-dopamine stage: RSA comparison of RSN Δh direction vs. reward regions (ventral striatum / vmPFC). The mechanism evidence that would upgrade the work from behavioral analogy.
- `AdaptativeThinking0529.md` §4 is the in-flight signal-proxy (Phase 1b); `AdaThink.md` / `AdaLogitsLens.md` are auxiliary trace/logit-lens analyses outside the main line.
- `AGENTS.md` — Codex-facing twin of this file; keep the two in sync when editing repo conventions.
- `~/CLAUDE.md` — running commands and data-directory map

## Architecture: how a run is wired together

A typical experiment is one of the `get_answer_*.py` / `get_action_*.py` entry-points driven by a sibling `run_*.sh` script. The dataflow is:

1. **Dataset loader** (`data_<benchmark>.py`) → JSON of question dicts (in `benchmark/` on the server, gitignored).
2. **`template.select_templates(suite=…)`** picks the prompt family. Three suites exist:
   - `default` — MMLU-style A/B/C/D answer extraction
   - `vanilla` — neutral phrasing without role
   - `action` — self-reported "reasoning willingness" 0–9
   Each suite has CoT / non-CoT / E-option (with "I am not sure") variants. The chosen template is rendered per-role via `utils.construct_prompt` + `utils.make_characters`.
3. **`llms.VicundaModel`** loads the LM. Three loading paths: `dream` diffusion (`AutoModel`, see `diffusion.py`), Mistral3 multimodal (`Mistral3ForConditionalGeneration`), or default `AutoModelForCausalLM`. All use bf16 + `device_map="auto"`. `_find_decoder_layers()` is the abstraction used by every layer-injection hook. **Generation entry points**: `VicundaModel.generate(inputs, batch_size)` for batched no-hook runs (`get_answer_*` uses this; pads with `padding=True`); `VicundaModel.generate_one(prompt, ...)` for bs=1 with caller-managed hooks (`track_hidden_states`, `closed_loop_gsm8k`, `track_dopamine_signal` all use this). **Do not mix**—batched vs bs=1 generation differs by ~2% acc on Llama due to padding artifact.
4. **Steering / closed-loop** is layered on top via forward hooks on the decoder layers identified in step 3. The diff vectors come from `mean/` (per-role mean differences) gated by a mask from `detection/` (NMD / KL / KS / LR / PCA / t-test / XGB selectors over `task_list.py`).
5. **Output**: current GSM8K re-run artifacts are written under `/data1/paveen/Dopamine/components/...`. Older experiment scripts may still point at the historical `/data1/paveen/RolePlaying/components/...` tree.

## Phase 2 closed-loop (current focus)

`closed_loop_gsm8k.py` + `run_closed_loop_gsm8k.sh` is the active iteration target. The shell script is the source of truth for hyperparameters; key knobs:

- `LAYER_START`/`LAYER_END` — injection layer range (Llama3-8B uses 11–20)
- `EMA_ALPHA` — feedback smoothing (Plans D/E/F all read EMA via `self.ema_alpha`)
- `K1` — primary gain. Meaning is plan-dependent: proportional gain (A/B/C/D/E/F/H2), fixed pulse magnitude (G/H1)
- `K2` — secondary knob. Plan C: spike-damping coefficient. Plan G: dead-zone half-width as fraction of `xp`. Unused by D/E/F/H1/H2
- `FLOOR_RATIO` — multi-purpose target ratio. Tonic floor (A/C), EMA homeostasis target (E), peak target (H1: 1.5, H2: 1.25)
- `PLATEAU_END_RATIO` — slope endpoint for B; H2 reuses it as the trapezoid end level (default 0.75·xp)
- `AVG_GEN_LEN` — defines T for B's plateau slope and H2's trapezoid decay endpoint
- `--plan {none,static,A,B,C,D,E,F,G,H1,H2,H3}` selects controller (case-sensitive; see `AdaptativeThinking.md` §3.2)
- H1 / H2 / H3 hardcode their window/segment boundaries inside `_compute_alpha()` — there are no `--h1_window` or trapezoid-segment CLI flags. H2 uses rise=50, plateau_end=200, peak=1.25; H3 uses rise=30, plateau_end=120, peak=1.35. Edit the constants in `closed_loop_gsm8k.py` if you need to sweep them.

When tweaking a plan, modify the `.sh` not the `.py` — the script is committed and reproducible. Past runs are kept commented (not deleted) in `run_closed_loop_gsm8k.sh` so the full sweep history is recoverable.

## Phase 1b: Signal-proxy validation (current in-flight work)

The H2/H3 contradiction (H2 +1% but H3 better-shape-worse-acc) raised the question: **is the NMD-projected signal actually RSN-specific, or any sparse projection would show the same expert-vs-non_expert gap?**

- `track_hidden_states.py` + `run_track_hidden_states.sh` is the data-collection step. It runs **greedy, bs=1** generation under each role and dumps per-sample HDF5 groups: `prefill_hs (P, n_stored, H) fp16`, `decode_hs (T, n_stored, H) fp16`, plus on-the-fly NMD sanity scalars (`x_prefill_proj`, `x_decode_proj`, `ema_decode_proj`). Storage is **selective**: middle layers `[LAYER_START, LAYER_END)` + final layer (= 10 layers for Llama3-8B 11–20), not all 32. HDF5 meta records `final_layer_idx_stored / stored_layer_indices / n_stored_layers` so offline tools can distinguish new selective vs legacy full-32 schemas.
- Paper-aligned roles (current 5-run baseline in `run_track_hidden_states.sh`): `expert` → `"an expert"`, `non_expert` → `"a non expert"`, `primary_teacher` → `"a primary school teacher"`, plus `neutral` (No-CoT and CoT) as controls. The earlier `mathematician`/`non-mathematician` framing was retired — task-technique mismatch on grade-school GSM8K.
- Output dir: `${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}/hs_<task>_<size>_<mode>_<role>_L<s>-<e>.h5` (selective storage ≈ 2–3 GB per role, gzip fp16). `run_track_hidden_states.sh` isolates each collection under `RUN_TAG` and refuses to truncate existing HDF5 unless `ALLOW_OVERWRITE=1`. It drives **7 runs** (expert, non_expert, neutral No-CoT, neutral CoT, primary_teacher, neutral α=+4, neutral α=−4); `START_FROM=N` skips runs `<N`, `START_FROM=8` runs **only** the offline Step 2/3 extraction (`extract_signal_json` / `_remask` / `extract_entropy_confidence`, all directory-level globs over every H5). `SKIP_EXTRACT=1` runs the HS-collection runs and **stops before** Step 2/3 — use it to re-collect a single interrupted run (`START_FROM=7 SKIP_EXTRACT=1 …` re-does only α=−4) without re-extracting JSON for all H5, then extract once with `START_FROM=8`. A completed H5 carries `n_samples_done` / `accuracy` / `stored_layer_indices` in its meta; their absence means the run was interrupted (mode="w" auto-truncates on re-collect, so no manual delete needed).
- **Standalone re-extraction**: `run_extract_signal.sh` is the offline-only Step 2/3 (NMD + random + entropy) split out as its own script, parameterized by `RUN_TAG` (default `phase1b_eot`) — use it to re-extract JSON without any risk of touching the model/HDF5. Prefer it over the legacy `run_extract_all.sh`, which has hardcoded paths and is gsm8k-only. Both call the same three `extract_*` scripts; the difference is parameterization + a pre-flight `sanity_mask_indexing.py` check in the newer one.
- `extract_signal_json.py` exports NMD sanity projections; `extract_signal_json_remask.py` reprojects HDF5 against an arbitrary mask such as `diff_random_*`. Both emit per-sample signal JSON matching the neutral-baseline schema (`x_prefill / x_decode / ema_decode / *_per_layer` + meta + diff_stats). Backward-compat: detects selective HDF5 via `final_layer_idx_stored` and slices middle as `[0, n_middle)`, else falls back to `[layer_start, layer_end)`.
- `extract_entropy_confidence.py` loads `model.norm.weight` + `lm_head.weight` from safetensors (no full 8B load) and computes per-step `entropy / top1_prob / margin / info_gain` from the stored final-layer HS, plus a prefill snapshot. Same backward-compat for stored final-layer index.
- Offline analysis lives in `~/Downloads/RSNResult/RoleAnswer/` — reloads the JSON + masks (`nmd_*` and `diff_random_*` from `${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/`) and compares **late-tonic-ratio gap, AUROC, Cohen's d** plus the multi-metric correlation matrix (`analyze_multi_metric.py`) between expert / non_expert / primary_teacher / neutral.
- `track_dopamine_signal.py` is the fast NMD-only path (no raw HS dump) — use when you only need scalars; use the HDF5 path when you want multi-mask flexibility or layer ablations.
- **Prompt self-documentation**: HDF5 meta, `dopamine_signal_*.json`, `random_signal_*.json`, `metrics_*.json`, and `closed_loop_*.json` all carry `prompt_template` (the raw template string) in their meta as of 2026-05-30. `grep '"prompt_template"' <file>` self-attests which prompt produced the result — use this before comparing numbers across runs.
- **Sanity script**: `sanity_mask_indexing.py` confirms saved mask non-zero rows + their decoder-layer alignment on the server. **Run it before any change to layer-indexing code** — this prevents repeating the offset bug fixed on 2026-05-30.

## Capitulation / Pressure experiments

`get_answer_capitulation.py` runs the two-round capitulation protocol (Round 1: original answer; Round 2: pressure prompt + optional RSN steering). Four shell scripts drive it:

- `run_capitulation.sh` — standard social pressure (soft challenge), all three models
- `run_capitulation_gt.sh` — Round 1 uses gold answer (eliminates guessing noise)
- `run_capitulation_pressure.sh` — authority-challenge pressure prompt, Llama3 only
- `run_capitulation_pressure_own.sh` — pressure from the model's own prior answer

Key CLI flags: `--pressure` switches to the authority-challenge prompt; `--config` encodes `{alpha}-{layer_start}-{layer_end}` (e.g. `4-11-20`). Output lands in `${BASE_DIR}/mmlupro/${MODEL}/answer_cap_mmlupro/cap_{config}/`.

Analysis: `gsm8k/analyze_cap_stratified.py` stratifies capitulation rates by difficulty and task category.

## Behavioral-economics / wanting-proxy suite

A family of entry-points that operationalize "wanting" (incentive salience) as **overt decisions** rather than answer accuracy. The shared hypothesis: α=+4 (expert direction) raises wanting → higher bets, longer deliberation, harder-task choice, more reward-seeking; α=−4 (non-expert) lowers it. All use the same `--configs {alpha}-{ls}-{le}` steering convention (e.g. `"0-11-20 4-11-20 neg4-11-20"`) and the `nmd` mask, so they plug into the same hook surface as the GSM8K work. **`Ada_Dopamine.md` §4 is the design+results doc for this whole suite** — each experiment number ⑤–⑩ there has the neuroscience mapping, prompt, prediction, and (where run) the result table; the `❌ Dropped` ones list exactly why a paradigm is incompatible with inference-time injection (e.g. phasic-DA / RPE needs synaptic plasticity). Read the matching §4.x before changing any of these.

> **Strongest current result (per `Ada_Dopamine.md` §4.6): Confidence Betting** — α=+4 raises mean_bet 52–67% and bet10-rate to ~50% while **accuracy is unchanged** across GPQA (n=646) and MMLU (n=14,042). This is the cleanest wanting–knowing dissociation in the project (non-linguistic choice, large sample, knowing held fixed). When discussing publishability or picking a headline experiment, this + the GSM8K/MATH commitment-dynamics analysis (`AdaptativeThinking0529.md` §2/§3) is the most journal-ready pairing; the main open gap is single-model (Llama-only) — betting should be extended to Qwen3/Mistral (pipeline already exists via the cross-model tables in `Ada_Dopamine.md` §3.0).

- `get_answer_gpqa_bet.py` (`run_gpqa_bet.sh`, also drives `run_mmlu_bet.sh`) — **confidence betting**: model bets 0/2/5/10 points before answering. Mean bet / fraction bet=0 = wanting proxy. Out: `${BASE_DIR}/${MODEL}/gpqa_bet/`.
- `get_answer_gpqa_cot_delay.py` (`run_gpqa_cot_delay.sh`) — **delay discounting**: model decides when to commit (`Final Answer: X`); CoT token length before commit = delay tolerance. Out: `${BASE_DIR}/${MODEL}/gpqa_delay/`.
- `get_action_effort_choice.py` (`run_effort_choice.sh`) — **effort-based choice**: GSM8K (easy) vs. MATH (hard) free selection. Includes an all-layer α=±1 (`1-1-33`) condition alongside the best-layer ±4.
- `get_answer_bandit.py` (`run_bandit.sh`) / `get_answer_textbandit.py` (`run_textbandit.sh`) — **multi-armed bandit**: explore/exploit over `--num_runs`×`--num_rounds` (30×50). Reward-seeking under steering.
- `get_answer_reversal.py` (`run_reversal.sh`) — **reversal learning**: reward contingency flips at `--phase_switch 20` of `--num_rounds 40` (`--reward_prob 0.8`); measures adaptation speed after the switch.
- `get_answer_crt.py` (`run_crt_llama3.sh`) — **Cognitive Reflection Test**: intuitive-vs-reflective answer rate under steering. Out: `answer_crt`.
- `get_answer_trait.py` (`run_trait.sh`) — personality/trait self-report under steering.
- `get_action_regenerate_gsm8k.py` — GSM8K **self-report scalars (0–9)**: `run_action_confidence_gsm8k.sh` (`--suite confidence` → pre-answer confidence) and `run_action_willingness_gsm8k.sh` (default action suite → reasoning willingness). Note `alpha=0` runs use `get_logits` (no mask loaded); other α load `diff_mtx * alpha`. **Both runners now sweep the full −8→+8 dose at layers 11–20** (config list `neg8-11-20 … 8-11-20`; the old `±4 + ±1@1-33` set is kept commented), and both have been migrated to `WORK_DIR=/data1/paveen/Dopamine`. The 0–9 suites (`build_gsm8k_action_suite` / `build_gsm8k_confidence_suite` in `template.py`) are a **standalone prompt family** — question-only, logit-extracted over the "0".."9" tokens, no `####` / no answer generation — so they are NOT affected by the GSM8K answer-extraction symmetrization. Status: **manipulation-check only, superseded by Betting (`Ada_Dopamine.md` §4.6)** — Berridge's wanting is non-conscious, so oral self-report is theoretically the wrong proxy; the data confirm this (willingness −4 moves the *wrong* way, confidence tracks PFC not DA). Keep as negative/control evidence, not a core wanting claim.
- `get_answer_sciworld.py` (`run_sciworld.sh`) — ScienceWorld agentic tasks (all 30) under steering.

These are exploratory and **not** part of the frozen GSM8K dose-response table — they have their own output trees and no `analyze_*` parser in `RoleAnswer/` yet, so analysis is currently ad-hoc per experiment. `run_judge_confidence.sh` is a Slurm batch script (note the `#SBATCH` header) for the cluster, unlike the other interactive `run_*.sh`.

## Offline analysis workspace (`~/Downloads/RSNResult/RoleAnswer/`)

This directory is **not** in the RolePlaying git repo. It is the offline analysis workspace for Phase 1b signal-proxy validation and capitulation analysis. Key scripts there:

- `analyze_dopamine_signal.py`, `analyze_flow_shapes.py`, `analyze_dopamine_spikes.py` — recompute x_t / EMA / x_prefill from HDF5 under arbitrary masks
- `analyze_first_last_acc.py` — **AUTHORITATIVE GSM8K/MATH ACC** (first-`####` / last-`\boxed{}` + fallback + norm; emits 改对/改坏 commitment split). `--gsm8k_root llama3/gsm8k` for the 182 same-machine rerun; `GSM8K_DIRS` already includes the full No-CoT dose sweep (`mdf_±2/±4/±6/±8`), CoT `±4`, and pushy variants.
- `analyze_loop_anxiety.py` — **AUTHORITATIVE loop / "can't-let-go" anxiety classifier** for α-steered GSM8K. Two modes: `--mode loop` (tail-loop gate → 3 mutually-exclusive buckets anxiety/mechanical/neutral_repeat, denom = loop count) and `--mode anxious_repeat` (full-text anchor-repeat, denom = 300: a cue + its next 60 chars must recur ≥2× to count, so one-off logical "however" is excluded). Anxiety = 4 overlapping sub-classes (self_doubt / format_anxiety / persona_reassure / over_precision). **Do not hand-write ad-hoc anxiety regexes** — edit the `ANXIETY_PATTERNS` dict here so every α uses one auditable standard. Key finding: anxiety-rate is U-shaped with trough at α=−6 (= acc peak), robust across all three counting conventions.
- `analyze_cot_metrics.py` — **reusable behavioral metric panel** (the 10-metric union of §2.2 Table1 dose-sweep + §2.5.1 CoT-contrast in `AdaptativeThinking0529.md`). `--table dose` → 9-cell No-CoT sweep → `cot_metrics_dose.csv`; `--table cot` → 6-cell CoT×No-CoT → `cot_metrics_cot.csv`. Imports `analyze_first_last_acc` + `analyze_loop_anxiety` so it shares their extractors; every metric was reverse-engineered to **reproduce the existing Table1 values exactly** (commit_rate / committed_acc / median&mean `####` position / gen_len / n_loop), then extended with `step_ge2` / `stuck`(loop ∧ `=`<2) / `preempt_lead`(首字符即数字) / `preempt_any`(lead-digit ∪ `####`-at-start<2%) / `med_eq`. **Two抢答 detectors must be read together**: `preempt_lead` misses the −α `#### N` first-token lock (first char is `#`), so −8 reads as 4 under lead-digit but 175 under `preempt_any`. `committed_acc` & `####`-position are **anchored metrics — only comparable within one generation regime** (CoT's long Step prefix + post-`\boxed{}` empty `####` shift them; use `preempt_lead`/`preempt_any` across the CoT boundary).
- `analyze_plan{D,EF,G,H1,H2,H3}.py` — closed-loop result parsers (filter by exact filename pattern)
- `calculate_*.py` — aggregate accuracy, ECE, entropy, transition matrices, RLHF role-effect
- `correlation_*.py` — correlate logit gaps with generation accuracy

`rsn_projection.py` (in repo root) computes per-layer RSN projection scores `h_i^(l) · v_RSN^(l)` for mediation analysis. Its `DIFF_PATH` can be swapped between the NMD mask and a random sparse mask to test projection specificity.

## File conventions worth knowing

- `get_answer_*` returns answers (multiple-choice extraction); `get_answer_regenerate_*` does free-form regeneration; `get_action_*` extracts the 0–9 action/willingness/confidence scalar (suite-dependent). The behavioral-economics entry-points (bet/delay/effort/bandit/reversal/crt/trait) are documented under their own section above.
- `*_lesion.py` zeros target neurons; `*_lesion_complement.py` zeros the complement (sanity control).
- `*_fewshot.py` variants prepend few-shot exemplars; never mix few-shot data with the zero-shot Phase 1 baselines.
- `analysis_*` and `analyze_*` are post-hoc plotting; never call them from training-style scripts.
- `mean/mean_diff.py` is the canonical diff-vector builder — other `mean/mean_*.py` are ablations (consistent / pairs / dice / per-layer).
- `harness.py` + `hf_rsn.py` plug into [lm-evaluation-harness] for benchmark-suite eval.
- **GSM8K/MATH answer extraction is centralized in `utils.py`** (`extract_gsm8k_answer`, `is_correct_gsm8k`, `gsm8k_difficulty`, `extract_math_answer`, `is_correct_math`). All 5 consumers (`get_answer_regenerate_gsm8k`, `get_answer_regenerate_math`, `closed_loop_gsm8k`, `track_dopamine_signal`, `track_hidden_states`) import from here — do not redefine locally.

## GSM8K re-run conventions (2026-05-31, current rework)

The Phase-2/1b GSM8K numbers from before 2026-05-31 are **discarded** — diagnosis found the prompts and extraction were noisy. Conventions for the re-run:

- **`<|eot_id|>` terminator fix (load-bearing for comparability).** `llms.VicundaModel._build_terminators()` registers `<|eot_id|>` (id 128009) — and `<|end_of_turn|>` if present — as additional `eos_token_id`s, and all three generate sites pass `eos_token_id=self.terminators`. Llama-3.1-Instruct ends each assistant turn with `<|eot_id|>`, **not** `<|end_of_text|>` (128001); without this the model "wants to stop" but the token isn't a terminator, so decoding runs to `max_new_tokens` and the tail degenerates into a `####N####N…` char-level repetition loop. The fix does **not** remove GSM8K loops (the loop is itself a wanting signal we keep), but it makes natural EOS possible and shifts the numbers. **Pre-eot numbers are not comparable to post-eot numbers.** The current data under analysis is the eot rerun: GSM8K answers in `gsm8k_eot/…`, hidden states under `RUN_TAG=phase1b_eot`. **MATH eot rerun has landed** — `RoleAnswer/llama3/math/math_eot/` holds No-CoT + CoT × α (−4/0/+4) (file `math_8B_11_20[_role].json`, field `generated`, gold in `gold_answer`); `run_math.sh` drives the No-CoT + α=0-CoT matrix and `run_math_cot.sh` is the CoT-supplement split-off. Older `math_2048` MATH numbers are pre-eot and not comparable.
- **`analyze_first_last_acc.py` is the AUTHORITATIVE accuracy source** (in the offline `RoleAnswer/` workspace). Paper-reported ACC is computed there offline — first-`####` (GSM8K) / last-`\boxed{}` (MATH) + fallback chain + normalization — **not** from the inline `correct_*` / `pred_answer` fields the generation scripts store (those are process-state only and easy to mis-read: simple string-compare on `pred_answer` undercounts). It also emits the first−last gap and the 改对/改坏 (fixed/broke) commitment split. Always cite ACC from this script.
- **No "honest" on GSM8K roles.** `honest` belongs only to E-option ("I am not sure") MMLU suites. GSM8K has no E-option, so every role uses the `neg` template (`"Now you are {character}."`), neutral uses `neutral`. `get_answer_regenerate_gsm8k.py` now bypasses `construct_prompt`'s default(+honest) branch and selects neutral/neg directly — this makes it produce **identical prompts to `track_dopamine_signal.py`** (the two were silently diverging: regenerate said "an honest expert", track said "an expert").
- **`####` final-answer directive in all GSM8K templates** (`build_gsm8k_default_suite`, both CoT and No-CoT, all 3 keys). Without it the model never emits `####`, runs to `max_new_tokens` (98–100% of No-CoT samples hit the 512 cap with no commit), and `extract_gsm8k_answer` falls back to "last number in text" — pure noise. The reversal *non_expert (63.7%) > expert (58%) > neutral (53%)* in the old results was an extraction lottery, not a real role effect. Symmetry preserved: No-CoT vs CoT still differ only by the `Let's think step by step.` line.
- **`####` wording matters — use `"Provide your final numeric answer after '####'."`** (the neutral, history-aligned wording). A pushier variant (`"Give your final answer as a single number after '####'."`) induced early-####抢答 (model commits an un-reasoned guess; expert 72% early-####, No-CoT acc collapsed to 34%). The pushy wording is kept conceptually as a future positive-control ablation (prompt wording vs α-steering as two levers on commitment timing). `extract_gsm8k_answer` takes the FIRST `####`, so an early 抢答 locks the wrong answer even when the correct one appears later — another reason to keep wording neutral. **Run matrix:** `run_gsm8k.sh` is now a **same-machine backfill** (2026-06-04, machine 182). WHY: bf16 greedy is **not byte-reproducible across GPUs** (different cuBLAS accumulation → logit ties flip → whole CoT chain diverges; a +4 re-run on a different box gave 205/300 sample-text mismatches), so the whole dose-response curve must live on ONE machine. It re-runs (per wording) the OLD-machine cells — No-CoT α=0 all-4-roles → `mdf_0`, No-CoT α=−4 neutral → `mdf_-4`, CoT α=0 neutral → `mdf_0_cot`; pushy re-runs the full old pushy matrix (α=0 4-roles + ±4 + CoT 0). `WORDING` defaults to `plain pushy`; pushy writes to `_pushy` dirs. `±2/±6/±8`, plain `+4`, CoT `±4` were already collected on 182 (not re-run). **Cross-machine rule: never compare ACC across machines/bs — 184 (HS, bs=1) and 182 (regenerate, bs=24) ACC are not comparable; paper ACC is the single-machine 182 dose table only.** `run_math.sh` has **no** pushy variant — MATH uses the neutral `\boxed{}` directive only.
- **`record_template` mislabels GSM8K role templates.** It logs `templates["default"]` (with honest) for non-neg roles, but the code actually constructs the `neg` prompt (no honest). The `template` field in result JSON is therefore unreliable for GSM8K roles — trust the code path, not the logged string.
- **No-CoT is the main condition**; CoT is the contrast control only.
- **Pass roles as full character strings** to `--roles` (e.g. `"an expert,a non expert,a primary school teacher"`), matching `utils.ROLE_TO_CHARACTER` values, so prompts align across scripts.
- **`get_answer_gsm8k.py` was deleted.** Its pure-baseline role is covered by `get_answer_regenerate_gsm8k.py` with a `0-<s>-<e>` config: `diff = mask*0` is a no-op, so α=0 regenerate == batched `generate`. Run baseline + ±4 steering in one regenerate pass (`configs="0-11-20 4-11-20 neg4-11-20"`) for maximum comparability. NOTE: regenerate steering is **prefill-only** (static push on the last prompt token); decode-time per-step steering is `closed_loop_gsm8k.py --plan static`. Different experiments — don't conflate.
- **Two generation paths to validate**: bs=1 (`track_dopamine_signal.py`, also emits signal) vs batched (`get_answer_regenerate_gsm8k.py` α=0). CLAUDE notes ~2% Llama acc gap from padding; run both to quantify it.
- **Phase-2 controllers need their own bs=1 baseline**: compare `closed_loop_gsm8k.py --plan {static,A,...}` against `closed_loop_gsm8k.py --plan none` with identical generation args. Do not treat the batched regenerate baseline as controller-comparable.

## Offline analysis workspace path

The offline analysis workspace was renamed `RoleAnswer_non/` → **`RoleAnswer/`** (`/Users/paveenhuang/Downloads/RSNResult/RoleAnswer/`). Scripts there (e.g. `analyze_multi_metric.py`) and older doc references that still say `RoleAnswer_non/` are stale — use `RoleAnswer/`. Existing `RoleAnswer/llama3/dopamine/signal/*.json` (dated 5/30) were produced with the **old layer-offset mask** (signal values unreliable; acc is unaffected by the mask) AND the old noisy prompts — they are being re-run, not kept.

## Server / data layout

Current GSM8K re-runs run on `/data1/paveen/Dopamine/` (server). Only code is in git; `components/`, `benchmark/`, `llama3/dopamine/`, H5 hidden states, and JSON answer dumps are not. Older experiments still have hard-coded `WORK_DIR=/data1/paveen/RolePlaying`; migrate them only when re-running that experiment family.

## Environment

```bash
bash setup_env.sh   # creates conda env "roleplaying" (py3.10) + bf16/CUDA stack
```

Mistral3 needs `transformers` from main + `mistral-common>=1.8.6` (already in `setup_env.sh`).

## Editing guidance

- **Layer indexing convention** (踩過大坑,務必先讀): `LAYER_START` / `LAYER_END` follow **HF hidden_states semantics** (index 0 = embedding, 1..N = decoder layer outputs). Saved masks drop the embedding row (`detection/nmd.py: return mask[1:]`), so saved-index `i` ↔ `decoder_layers[i]` ↔ `hidden_states[i+1]`. **Always use `utils.mask_slice_for(mask, ls, le)` and `utils.decoder_layer_range(ls, le)`** instead of raw `mask[ls:le]` / `range(ls, le)` — they encode the `layer_start-1` offset so hook registration and mask slicing stay in sync. The `regenerate` family (`get_answer_regenerate_*.py`) sidesteps the offset by `zip(decoder_layers, mask)` full-length and is the canonical alignment reference. Verify on server with `sanity_mask_indexing.py` before changing any layer-indexing code.
- Don't refactor `llms.VicundaModel` loading branches casually — Mistral3, dream-diffusion and CausalLM each rely on slightly different hook surfaces.
- `template.py` 大部分變體不要原地改(歷史 baseline 綁定);但 GSM8K/MATH 的 `build_gsm8k_default_suite` 和 `build_math_suite` 在 2026-05-30 已修正為**對稱**(No-CoT 與 CoT 唯一差別是 `Let's think step by step.` 一行),vanilla / action / confidence 三個 suite 維持不動。舊不對稱模板跑的數字(含早期 61.7% / 76.0% baseline)與當前 pipeline 不可比。
- New plans go in `closed_loop_gsm8k.py` behind a new `--plan` value (added to the `choices=[...]` list and as a branch in `_compute_alpha()`); keep prior plans callable for ablation reproducibility.
- The injection–observation loop has a hard 1-step lag (pre-hook injects `α_t` before forward; post-hook reads `x_t`/`ema_t` after, so `α_{t+1}` is decided from `x_t`). Any new controller must assume `α` based on step-`t` observation only takes effect at step `t+1` — controllers that try to react to fast (per-token) signal changes diverge (see Plan D failure in `AdaptativeThinking.md` §4.3).
- Tracking projection and steering injection share the same `nmd_mask` (sparse, ~0.5% of neurons per layer). This co-design means injecting `+α` directly raises next-step `x_{t+1}` by `α × ‖mask‖²` — do not change one without the other.
- Result file naming includes plan + k1 + k2 + ema_alpha + layer range, so renaming any of these breaks the analysis scripts (`analyze_plan{D,EF,G,H1,H2,H3}.py` filter by exact filename).
- HS recorder writes `(P, n_stored, H)` and `(T, n_stored, H)` fp16 blocks for **middle layers `[LAYER_START, LAYER_END)` + final layer only** (10 layers for Llama3-8B). The on-the-fly NMD projection in HDF5 is a sanity scalar — analysis recomputes projections offline against any mask. `P` is variable-length per sample, so do not stack across samples without per-sample loops. If you need cross-layer ablation outside the stored set, re-run the tracker with a wider `[LAYER_START, LAYER_END)`.
- `track_hidden_states.py` is **bs=1** by design: forward hooks read last-token HS and append per step, and decode length varies per sample (different EOS). Batching would require per-sample masks + per-sample EOS bookkeeping in the hook — not worth it for 300 samples × 5 runs.
