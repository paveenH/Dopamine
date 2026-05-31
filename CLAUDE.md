# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Role-Sensitive Networks (RSN)** — dopaminergic adaptive calibration of LLM reasoning via hidden-state steering. The user-level `~/CLAUDE.md` contains the full theory map and phase plan; this file covers only repo-local conventions and recent (Phase 2 GSM8K) work.

**Required reading before non-trivial changes:**
- `AdaptativeThinking0529.md` — current state. 2026-05-30 起 GSM8K template 已對稱化、mask offset bug 已修,所有舊 Phase 1/2 數字與當前 pipeline 不可比。Phase 1b 重做為當前優先工作;Phase 2 (Plans A–H3) 結論暫擱置,須重跑
- `AdaptativeThinking.md` — 歷史 Phase 1–2 設計、Plan A–H3 動機 / 失敗分析、Yerkes–Dodson framing、EMA + 1-step-lag physics(設計思路仍有效,但數字本身已過時)
- `Dopamine.md` / `Dopamine_EN.md` / `Dopamine2.md` — literature & mapping
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
5. **Output**: per-role answer logits + optional hidden-state H5 are written under `<hs_prefix>/<task>/...` (server path `/data1/paveen/RolePlaying/components/...`).

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
- Output dir: `${BASE_DIR}/hidden_states/${TASK}/hs_<task>_<size>_<mode>_<role>_L<s>-<e>.h5` (selective storage ≈ 2–3 GB per role, gzip fp16).
- `extract_signal_json.py` reprojects HDF5 against any mask (NMD or `diff_random_*`) and emits per-sample `dopamine_signal_*.json` matching the neutral-baseline schema (`x_prefill / x_decode / ema_decode / *_per_layer` + meta + diff_stats). Backward-compat: detects selective HDF5 via `final_layer_idx_stored` and slices middle as `[0, n_middle)`, else falls back to `[layer_start, layer_end)`.
- `extract_entropy_confidence.py` loads `model.norm.weight` + `lm_head.weight` from safetensors (no full 8B load) and computes per-step `entropy / top1_prob / margin / info_gain` from the stored final-layer HS, plus a prefill snapshot. Same backward-compat for stored final-layer index.
- Offline analysis lives in `~/Downloads/RSNResult/RoleAnswer_non/` — reloads the JSON + masks (`nmd_*` and `diff_random_*` from `${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/`) and compares **late-tonic-ratio gap, AUROC, Cohen's d** plus the multi-metric correlation matrix (`analyze_multi_metric.py`) between expert / non_expert / primary_teacher / neutral.
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

## Offline analysis workspace (`~/Downloads/RSNResult/RoleAnswer_non/`)

This directory is **not** in the RolePlaying git repo. It is the offline analysis workspace for Phase 1b signal-proxy validation and capitulation analysis. Key scripts there:

- `analyze_dopamine_signal.py`, `analyze_flow_shapes.py`, `analyze_dopamine_spikes.py` — recompute x_t / EMA / x_prefill from HDF5 under arbitrary masks
- `analyze_plan{D,EF,G,H1,H2,H3}.py` — closed-loop result parsers (filter by exact filename pattern)
- `calculate_*.py` — aggregate accuracy, ECE, entropy, transition matrices, RLHF role-effect
- `correlation_*.py` — correlate logit gaps with generation accuracy

`rsn_projection.py` (in repo root) computes per-layer RSN projection scores `h_i^(l) · v_RSN^(l)` for mediation analysis. Its `DIFF_PATH` can be swapped between the NMD mask and a random sparse mask to test projection specificity.

## File conventions worth knowing

- `get_answer_*` returns answers (multiple-choice extraction); `get_answer_regenerate_*` does free-form regeneration; `get_action_*` extracts the 0–9 action/willingness scalar.
- `*_lesion.py` zeros target neurons; `*_lesion_complement.py` zeros the complement (sanity control).
- `*_fewshot.py` variants prepend few-shot exemplars; never mix few-shot data with the zero-shot Phase 1 baselines.
- `analysis_*` and `analyze_*` are post-hoc plotting; never call them from training-style scripts.
- `mean/mean_diff.py` is the canonical diff-vector builder — other `mean/mean_*.py` are ablations (consistent / pairs / dice / per-layer).
- `harness.py` + `hf_rsn.py` plug into [lm-evaluation-harness] for benchmark-suite eval.
- **GSM8K/MATH answer extraction is centralized in `utils.py`** (`extract_gsm8k_answer`, `is_correct_gsm8k`, `gsm8k_difficulty`, `extract_math_answer`, `is_correct_math`). All 6 consumers (`get_answer_gsm8k`, `get_answer_regenerate_gsm8k`, `get_answer_math`, `closed_loop_gsm8k`, `track_dopamine_signal`, `track_hidden_states`) import from here — do not redefine locally.

## Server / data layout

Code runs on `/data1/paveen/RolePlaying/` (server). Only code is in git; `components/`, `benchmark/`, `llama3/dopamine/`, H5 hidden states, and JSON answer dumps are not. Hard-coded `WORK_DIR=/data1/paveen/RolePlaying` appears in most `run_*.sh`.

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
