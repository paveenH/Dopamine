# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

**Role-Sensitive Networks (RSN)** — dopaminergic adaptive calibration of LLM reasoning via hidden-state steering. The user-level `~/AGENTS.md` contains the full theory map and phase plan; this file covers only repo-local conventions and recent (Phase 2 GSM8K) work.

**Required reading before non-trivial changes:**
- `AdaptativeThinking.md` — Phases 1–2, Plans A–H3 (full design rationale, failure analyses, integrated conclusion that decode-time shape control is measurable but does not robustly recover CoT-level accuracy), Yerkes–Dodson framing, EMA + 1-step-lag physics
- `Dopamine.md` / `Dopamine_EN.md` / `Dopamine2.md` — literature & mapping
- `~/AGENTS.md` — running commands and data-directory map

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
5. **Output**: per-role answer logits + optional hidden-state H5 are written under `<hs_prefix>/<task>/...` (server path `/data1/paveen/RolePlaying/components/...`).

## Phase 2 closed-loop (current focus)

`closed_loop_gsm8k.py` + `run_closed_loop_gsm8k.sh` is the active iteration target. The shell script is the source of truth for hyperparameters; key knobs:

- `LAYER_START`/`LAYER_END` — injection layer range (Llama3-8B uses 11–20)
- `EMA_ALPHA` — feedback smoothing (Plans D/E/F all read EMA via `self.ema_alpha`)
- `K1` — primary gain. Meaning is plan-dependent: proportional gain (A/B/C/D/E/F/H2/H3), fixed pulse magnitude (G/H1)
- `K2` — secondary knob. Plan C: spike-damping coefficient. Plan G: dead-zone half-width as fraction of `xp`. Unused by D/E/F/H1/H2/H3
- `FLOOR_RATIO` — multi-purpose target ratio. Tonic floor (A/C), EMA homeostasis target (E), peak target (H1: 1.5, H2: 1.25, H3: 1.35)
- `PLATEAU_END_RATIO` — slope endpoint for B; H2/H3 reuse it as the trapezoid end level (default 0.75·xp)
- `AVG_GEN_LEN` — defines T for B's plateau slope and H2/H3's trapezoid decay endpoint
- `--plan {none,static,A,B,C,D,E,F,G,H1,H2,H3}` selects controller (case-sensitive; see `AdaptativeThinking.md` §3.2)
- H1 / H2 / H3 hardcode their window/segment boundaries inside `_compute_alpha()` — there are no `--h1_window` or trapezoid-segment CLI flags. Edit the constants in `closed_loop_gsm8k.py` if you need to sweep them.

When tweaking a plan, modify the `.sh` not the `.py` — the script is committed and reproducible. Past runs are kept commented (not deleted) in `run_closed_loop_gsm8k.sh` so the full sweep history is recoverable.

## File conventions worth knowing

- `get_answer_*` returns answers (multiple-choice extraction); `get_answer_regenerate_*` does free-form regeneration; `get_action_*` extracts the 0–9 action/willingness scalar.
- `*_lesion.py` zeros target neurons; `*_lesion_complement.py` zeros the complement (sanity control).
- `*_fewshot.py` variants prepend few-shot exemplars; never mix few-shot data with the zero-shot Phase 1 baselines.
- `analysis_*` and `analyze_*` are post-hoc plotting; never call them from training-style scripts.
- `mean/mean_diff.py` is the canonical diff-vector builder — other `mean/mean_*.py` are ablations (consistent / pairs / dice / per-layer).
- `harness.py` + `hf_rsn.py` plug into [lm-evaluation-harness] for benchmark-suite eval.

## Server / data layout

Code runs on `/data1/paveen/RolePlaying/` (server). Only code is in git; `components/`, `benchmark/`, `llama3/dopamine/`, H5 hidden states, and JSON answer dumps are not. Hard-coded `WORK_DIR=/data1/paveen/RolePlaying` appears in most `run_*.sh`.

## Environment

```bash
bash setup_env.sh   # creates conda env "roleplaying" (py3.10) + bf16/CUDA stack
```

Mistral3 needs `transformers` from main + `mistral-common>=1.8.6` (already in `setup_env.sh`).

## Editing guidance

- Don't refactor `llms.VicundaModel` loading branches casually — Mistral3, dream-diffusion and CausalLM each rely on slightly different hook surfaces.
- Don't change `template.py` strings in place; add a new variant — Phase 1 baselines are tied to exact prompt wording.
- New plans go in `closed_loop_gsm8k.py` behind a new `--plan` value (added to the `choices=[...]` list and as a branch in `_compute_alpha()`); keep prior plans callable for ablation reproducibility.
- The injection–observation loop has a hard 1-step lag (pre-hook injects `α_t` before forward; post-hook reads `x_t`/`ema_t` after, so `α_{t+1}` is decided from `x_t`). Any new controller must assume `α` based on step-`t` observation only takes effect at step `t+1` — controllers that try to react to fast (per-token) signal changes diverge (see Plan D failure in `AdaptativeThinking.md` §4.3).
- Tracking projection and steering injection share the same `nmd_mask` (sparse, ~0.5% of neurons per layer). This co-design means injecting `+α` directly raises next-step `x_{t+1}` by `α × ‖mask‖²` — do not change one without the other.
- Result file naming includes plan + k1 + k2 + ema_alpha + layer range, so renaming any of these breaks the analysis scripts (`analyze_plan{D,EF,G,H1,H2,H3}.py` filter by exact filename).
