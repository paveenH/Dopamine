# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## 0.5 Instruction maintenance

This file contains executable project rules, not a lab notebook. When adding or changing a non-trivial rule:

1. State the rule's scope, the observed failure or decision that motivated it, and where the evidence lives.
2. Prefer amending an existing rule over adding a near-duplicate.
3. Do not preserve a rule solely because it is old. Retain it only while its stated failure mode, dependency, or frozen-protocol role still applies.
4. For a frozen protocol, record its version/date and the artifact, test, manifest, or result that makes the rule binding.
5. When a rule is obsolete, replace it with a short historical pointer rather than silently deleting or restating it.
6. Keep rationale concise. Detailed history belongs in the linked design/result document (`AdaBandit.md`, `AdaDopamine*.md`, amendment JSONs, `test_*.py`), not here.

**Provenance comments.** Add a `<!-- Why / Evidence / Scope -->` comment only to rules that are easy to misdelete, misedit, or misread — frozen protocols, counter-intuitive results, and interface/run invariants. Not to ordinary coding conventions.

```md
<!-- Why: PV10 parser once accepted malformed outputs and produced analysable-looking but invalid data.
Evidence: pv10_capability_amendment_01.json; test_pv10_gate_end_to_end.py.
Scope: PV10 only. Review if protocol version changes. -->
- PV10 parsing is fail-closed; never substitute a fallback action.
```

Three provenance categories are worth the space in this repo:
- **Frozen rules** — why it cannot change, when it was frozen, which manifest/test enforces it.
- **Counter-intuitive conclusions** — the specific failure that produced the reading, and which protocol it applies to (e.g. "flat allocation is NOT evidence of absent directed exploration" holds because TTTS's near-uniform distribution is marginal).
- **Interface / run invariants** — anchor token, parser, seed pairing, output directory: state what breaks (incomparability or a silent error), not just the requirement.

Markdown/HTML comments are still visible context to an agent — they are not stripped from the prompt. Keep them to two or three lines and push the long history into the linked doc.

## Project

**Role-Sensitive Networks (RSN)** — dopaminergic adaptive calibration of LLM reasoning via hidden-state steering. The user-level `~/CLAUDE.md` contains the full theory map and phase plan; this file covers repo-local conventions and the per-experiment record.

**How to use this file.** It is long because most of it is a per-experiment ledger of frozen results, retracted readings and 口径 traps — things that CANNOT be re-derived from the code and that have each already cost a wrong conclusion. Read the section for the experiment you are touching; do not read it end to end.

| If you are… | Go to |
|---|---|
| running or changing an experiment | that experiment's section (Behavioral-economics suite, GSM8K conventions, Phase 1 signal) |
| adding a benchmark | the standalone-pipeline pattern at the end of the Behavioral-economics suite |
| touching layer indexing / masks / hooks | **Editing guidance** + Phase 1's steering hook-alignment bullet |
| analysing stored results offline | **Offline analysis workspace** (`RoleAnswer/`, `python3.10`, not in git) |
| verifying a change without a GPU | **Local checks** |
| writing up a result | the frozen-wording rules in **Active status** and the relevant experiment section |

<!-- Why: the two result docs read DISJOINT data trees, and the one exception runs the
other way; a reader who assumes "AdaDopamine == benchmark folders" will look for P2A's
training data in llama3/gsm8k/, which lacks x_prefill entirely.
Evidence: audited 2026-08-30 across all backing scripts; AdaDopamine_gsm8k.md Table 5.1a.
Scope: any new analysis script -- declare which tree it reads before writing it. -->
**Which document reads which data tree (audited 2026-08-30).** `AdaptiveThinking.md` reads **only** `{llama3,qwen2.5}/dopamine/` — verified across all 11 backing scripts, none touches a benchmark folder. Within it three DISTINCT batches share a schema and must never be joined per-question: `signal/` (lightweight, projections + text), `signal_hs/` (offline re-projection from H5), `metrics_hs/` (logit family from the final layer). `AdaDopamine_gsm8k.md` reads the per-benchmark trees (`llama3/gsm8k/`, `llama3/math/`, `qwen2.5/{gsm8k,math,math_cot}/`, `*/gsm_hard/`) — **with ONE exception running the other way: §5.2's P2A trains on `llama3/dopamine/signal/`**, because the entry-only and combined control arms need `x_prefill`, which the benchmark tree does not carry. That is the only crossing point, and it is why §5.1 carries the 184-vs-182 batch declaration.

**Two conventions that govern everything below.** (1) A rule with a `<!-- Why / Evidence / Scope -->` comment records a real failure — read it before editing that rule. (2) Frozen wording is frozen: where a phrasing is given in bold as required, use it verbatim rather than paraphrasing, because the paraphrase is usually the overstatement the rule exists to block.

<!-- Why: four sections used to claim "current" simultaneously (Phase 2 closed-loop, Phase 1b, pv9, pv10), so a new session could not tell where work actually is and might reopen a closed line.
Evidence: this section's own history; AdaBandit.md §5 (BAI CLOSED).
Scope: update the date + one-liner whenever the active task changes; never add a second "current". -->
### Active status (2026-08-30)

**CGT-Sequential cross-model replication on Qwen2.5-7B-Instruct — COMPLETE, frozen 2026-08-19 as a PARTIAL replication limited by a narrow usable dose window.** v5 (N=20, layers 16–21) is the citable result; v4 stays frozen as interface/capability-boundary evidence. **Do not write "complete cross-model replication"** — Llama v4's clean range is −4…+6 across nine cells, Qwen v5's is −2…+2, and desc +2 did not pass the gate, so asc and desc do not even carry the same number of usable cells.

<!-- Why: the accept-step curve is monotone and extremely significant in BOTH conditions and will read as a full replication if quoted alone; desc +2's colour judgement is at chance while its behaviour looks strongest.
Evidence: qwen2.5/cgt/seq_{asc,desc}_v5_qwen_full; AdaDopamine.md §3.3 CGT-Sequential 跨模型.
Scope: every Qwen2.5-7B CGT-seq number. -->
**FROZEN WORDING (do not soften; never quote the behavioural half alone):**
- **asc −2/0/+2 = the only fully passing three-point dose result** (`accept_step` 3.010/1.185/1.012, all three paired comparisons p≤7.9e−04, exact Wilcoxon n=20/16).
- **desc −2/0 is citable but is ONE contrast, not a three-point band** (3.508→1.677, p=1.9e−06).
- **desc +2 is an over-steering boundary cell.** Its behaviour (`accept_step` 1.048, `step1_rate` .967, `mean_bet` 94.0%) MUST be reported in the same paragraph as `qdm_red = .5031`. Word it "**close to chance and did not pass the gate**" — the per-run bootstrap 95% CI **[.445, .561]** contains BOTH 0.50 and the 0.55 threshold, so neither "equals chance" nor "meets the gate" is supportable. The gate judges point estimates per the pre-registration → FAIL.
- **DAI: −2 (−12.97 [−19.03, −7.08]) and 0 (+70.21 [+65.93, +74.16]) are citable; +2 (+88.71) is DIAGNOSTIC ONLY** — its desc half sits on a failed cell, so it does NOT support a "complete monotone widening" claim.
- **Mechanism:** commitment latency responds monotonically to α in both conditions, but the Blue-label prior is amplified monotonically by +α (`qdm_label_gap` desc .265→.372→.481). desc fails before asc because its `qdm_red` baseline is lower (.617 vs .811).

**v5 = order balance + neutralised example, and the two may NOT be separated.** v4 failed the α=0 gate (desc `qdm_major_red` **.37**, asc .65 — a label lock, present before any steering) while **Llama3-8B v4 PASSES the same gate** (asc .709/.816, desc .691/.819, invalid .000), so the gate discriminates between models rather than rejecting the paradigm. v4 also fixed blue-first in **FOUR** places (scene sentence, mechanics sentence, worked example, output list + question) with no shuffle anywhere, so label and position were completely collinear and v4 data CANNOT distinguish a Blue-label from a first-option preference. v5 balances the option order per round (strict 4/4 per phase, 32/32 per run, independent seed-derived RNG so `make_box_sequence` is byte-unchanged) and neutralises the example. **The improvement includes BOTH changes; never attribute it to one.** The scene sentence is still blue-first — v5's one known residual cue.

**Attribution result (what v5 was built to answer): the residual bias follows the Blue LABEL, not the first position.** Within-cell position effect (holding `major_color` fixed) is only **+0.065 / +0.071**, entirely inside the major=red cells (both major=blue cells are 1.000, no headroom). **The marginal position effect of +0.29 must NOT be cited** — seeds 0–2 happen to skew `first_option` with `major_color` (61/35 rather than 48/48); over 50 seeds the two RNGs are independent (agreement .510), so that skew is small-sample coincidence, not a coupling bug.

**Dose band was selected by a pre-declared N=5 pilot over −6…+6, all seven cells reported.** ±4/±6 fail for QUALITATIVELY DIFFERENT reasons: +4/+6 keep format intact and keep reading the evidence (`qdm_blue` ≈.99) while `qdm_red` collapses to .269/.106 — the Blue prior amplified past the evidence; −4/−6 are colour-step digit drift (`Color: 7`), invalid ≈.52–.63, whose dose curve is clean and monotone (digit share desc .984/.772/.022/.000). N=5 pilots are descriptive validity checks — do not read significance from them.

**`parse_color` takes the FIRST colour word**, so "restate the odds, then name a colour" (`Color: 20% blue, 80% red, I choose Red: Red`) is read backwards. v5 pilot: 2/4480 rounds (both asc −2), correcting them moved `qdm_red` .7733→.7867 and changed NO gate verdict; **the N=20 formal data scans 0/7586**, so its as-stored numbers need no sensitivity caveat. The driver is frozen and deliberately NOT patched (2/4480 changes nothing; editing it would fork the 口径 of all stored v1–v4 data) — `RoleAnswer/scan_cgt_seq_color_misparse.py` is the sanctioned alternative and reports as-stored beside sensitivity-corrected, never replacing one with the other.

<!-- Why: "same states" reads as a replay of stored pilot rounds; it is not, and a row-by-row comparison against the pilot would be invalid.
Evidence: diag_cgt_seq_color_logits.py SCOPE block; the driver runs temperature=1.0.
Scope: every colour-logit diagnostic claim. -->
**The colour-logit sidecar RE-GENERATES its trajectory at temperature=1.0 — it does NOT replay stored pilot states.** Only the box sequence is shared with a stored run of the same seed (`make_box_sequence` is seed-determined); colour choices and accept steps are resampled, so the chat history diverges from the pilot. What is paired is **(logits, generation) at the same state WITHIN the diagnostic run** — never diagnostic-vs-pilot row by row. Its **primary** contrast is `logP("Blue") − logP("Red")`, the exact casing the prompt demands; all four surface forms are stored and `logsumexp(Blue,blue) − logsumexp(Red,red)` is a case-insensitive sensitivity check. A per-colour `max()` over casings is **wrong and must not be reintroduced**: taken independently per colour it can compare `blue` against `Red` and flip the margin's sign (verified: forms −2.0/−1.5/−1.8/−3.0 give max +0.30 but primary −0.20). On the v4 α=0 desc diagnostic it read ρ(signed evidence, margin) = **+0.736** — the logits DO track the chest counts, so "cannot use probability" was excluded even under v4; the failure was a large Blue-ward baseline offset (red-major mean margin **+1.94**, separation +9.87 nats), which is what v5 addressed.

**Qwen CGT-seq is CLOSED — do not add doses, do not iterate the prompt.** The band is frozen at −2/0/+2. Widening it means re-running the pilot, not editing the launcher.

| Line | Status |
|---|---|
| CGT-seq Qwen replication | **COMPLETE + FROZEN** as a PARTIAL replication (v5, band −2…+2; asc 3 cells pass, desc +2 fails `qdm_red`). v4 stays frozen as boundary evidence. Do not add doses. |
| Bandit / directed exploration (pv1–PV11) | **CLOSED** — boundary evidence, do not reopen (see the Bandit block below) |
| Confidence Betting §3.1 | **FROZEN** 2026-07-29, do not re-run |
| CGT-seq Llama v4 | **FROZEN** — descriptive values final; stats口径 moved to paired 2026-08-19 |
| Phase 2 closed-loop (Plans A–H3) | **SHELVED** — conclusions predate the template/offset fixes, must be re-run |
| Phase 1 signal analysis (ex-"Phase 1b") | **COMPLETE** for §4.1–4.8; name "Phase 1b" is retired |
| Qwen GSM8K / MATH / CoT replication | **COMPLETE** 2026-08-21 — see `AdaDopamine_gsm8k.md` §4 |
| Qwen signal replication (§3.1 lightweight) | **COMPLETE** 2026-08-22 — 13 cells, `x_prefill ~ α` R²=.9998, reproduces the frozen behavioural table |
| Qwen HS backfill (§3.3, 7 cells) | **COLLECTED + ACCEPTED** 2026-08-24 — integrity/projection/agreement all pass; agreement is **1.000 in all seven cells** |
| Qwen commit-aligned analysis | **NOT FROZEN** — a "decode-state saturation" reading was RETRACTED; see the 口径 rules in the Phase-1 section |
| Qwen logit family (§5.5 output decisiveness) | **UNBLOCKED + FROZEN** 2026-08-26 — 7 cells, `logit_family.py` / `logit_family_RESULT.txt`. Reads `PARTIALLY AVAILABLE / SPARSELY SAMPLED`, never "available"; CoT is one readable cell and its transition is **not detected**, never "abolished" |
| Qwen band-position probe `[11,18)` vs `[16,22)` | **PREPARED, NOT RUN** — launchers exist, zero artifacts; see the mask/pre-flight block |
| Manifold pilot | **COMPLETE + CLOSED 2026-08-28.** Full chain: **symmetric injection → layer-wise divergence → last-layer piecewise-scalar geometry**, which still does NOT explain Llama's peak vs Qwen's plateau. Each arm is a 1-D scalar family; the two arms coincide at the first steered layer and separate monotonically with depth, so the fixed cross-arm angle is EMERGENT, not an input property. Qwen's positive arm does not saturate while its behaviour plateaus → entry-state saturation EXCLUDED, difference located downstream. Incremental prediction NOT DETECTED; decode did not extend. **Do not extend** — next line is commit-aligned Z_t/s_t + commitment dynamics |
| P2 commitment prediction + MATH transfer | **COMPLETE + FROZEN** 2026-08-28 — both gates pass; MATH locked transfer selects the true best dose on both models. Retrospective, NOT blind |
| P3 blind validation (GSM-Hard) | **COMPLETE + CLOSED** 2026-08-30 — gold unsealed once; direction + workpoint correct on both models, regret 0.00 pp; calibration did NOT transfer. Main analysis 口径 closed; further work is exploratory |
| P3 supplement (CoT condition transfer) | **COMPLETE + FROZEN** 2026-08-30 — both models matched the locked direction and survive Holm (llama −6 +6.00 pp p=.0039; qwen +8 +13.33 pp p=9.4e−06); interaction NOT DETECTED. Locked prospective on the CONDITION, not a new blind test |
| P4 fixed-workpoint transfer (LogiQA 2.0) | **COMPLETE + FROZEN** 2026-09-01 — **DOUBLE NULL**, neither model survives Holm m=2 (llama −6 −4.33 pp raw p=.0533 p_adj=.107; qwen +8 +1.00 pp p_adj=.801). NOT a blind validation — LogiQA gold is public |
| P4b fixed-workpoint transfer (BBH numeric) | **COMPLETE — DOUBLE NULL** 2026-09-02 (`object_counting`, 6 cells) — neither model survives Holm m=2 (llama −6 −0.80 pp p_adj=1.000; qwen +8 +2.40 pp p_adj=1.000); reverse diagnostic BREAKS the ordering on both. Behaviour DOES move (qwen +8 bare-first-line 54.4→0.0%), so the frozen reading is **GSM8K 行为 signature 部分迁移**, never "mechanism transferred". Stage-0 gate PASSED (.4160/.5520) and the earlycand audit (29/29, α=0 at CEILING .952/1.000) stay as history. `bbh-p4b-v0` + `p4b-amend-01`; second task of the SAME P4 question, not a new phase |
| P4c fixed-workpoint transfer (CRUXEval-O) | **PREPARED, PREFLIGHT PASSED, NOT RUN** 2026-09-03 — `docs/PREREG_P4C_CRUXEVAL.md` (`cruxeval-p4c-v0`) frozen before any cell existed; 8 cells (llama 0/−6/−4/+4, qwen 0/+8/+6/−6), 300 of 800 items. **THIRD task of the SAME P4 question**, not a new phase. Scoring is `ast.literal_eval` + Python object equality, **NOT** the official exec-based pass@1 — model output is never executed. Format-only preflight found and fixed two decoding artifacts (`p4c-amend-01`); two remaining non-parsing payloads were classified as MODEL FORMAT FAILURE and the parser was deliberately NOT widened (`p4c-amend-02`) |
| Workpoint-stability supplement (wps-v0) | **PREPARED, NOT RUN** — 7 neighbour cells, `docs/PREREG_WORKPOINT_STABILITY.md` frozen 2026-09-03 before any cell existed. Asks only whether each reported workpoint is a LOCAL OPTIMUM or the edge of an untested region; **not a dose search**, and no frozen workpoint may be redefined by it. Own family, Holm **m=7**; neighbour comparisons are EXPLORATORY and outside it |
| Paper integration (ACL ARR) | in progress — see `TODO.md` |

<!-- Why: four fixed-workpoint transfer tests are now complete and each is written up
in isolation, so the cumulative picture (2 transfers, 2 nulls) exists in no single place
and a reader can cite one arm without the other.
Evidence: docs/p3_result_20260830.json, p3_supp_result_20260830.json,
p4_logiqa2_evaluation.json, bbh_p4b_object_counting_result.json.
Scope: any claim about how far the GSM8K workpoint transfers. -->
**FIXED-WORKPOINT TRANSFER SCOREBOARD (four tests COMPLETE, a fifth PREPARED but NOT
RUN; α read from the frozen GSM8K record and NEVER re-searched — llama `−6`, qwen
`+8`).** Report the four completed rows together:

| Task | Answer space | llama `−6` | qwen `+8` | Verdict |
|---|---|---|---|---|
| GSM-Hard (P3) | constructed integer | **+6.33 pp** p=.0066 | **+16.33 pp** p=1.4e−08 | both transfer |
| GSM-Hard CoT (P3-supp) | constructed integer | **+6.00 pp** p=.0039 | **+13.33 pp** p=9.4e−06 | both transfer |
| LogiQA 2.0 (P4) | choose among 4 given | −4.33 pp p_adj=.107 | +1.00 pp p_adj=.801 | **DOUBLE NULL** |
| BBH object_counting (P4b) | constructed integer | −0.80 pp p_adj=1.00 | +2.40 pp p_adj=1.00 | **DOUBLE NULL** |
| CRUXEval-O (P4c) | constructed **Python literal** | — | — | **NOT RUN** — do not fill this row from anything but `docs/p4c_cruxeval_evaluation.json` |

**The frozen reading, and it is narrower than the table looks.** The workpoint transfers
across *task difficulty and CoT* on GSM-Hard, and fails on **both** a changed answer space
(LogiQA) **and** an option-free numeric task that restored it (P4b). **P4b is what forbids
the tidy story**: it was built to test whether LogiQA's choice interface caused that null,
and restoring the answer space and `####` submission did NOT restore transfer. So write
**"removing the option interface was not sufficient to restore transfer"** — never "the
options caused the LogiQA null", and never "the workpoint transfers to numeric reasoning".
Separating a choice-interface effect from a reasoning-type effect still needs a within-item
with/without-options contrast, which has NOT been run. **P4c will not supply it either**:
CRUXEval-O is option-free but changes reasoning type AND answer space at once (a constructed
integer becomes an arbitrary Python literal), so it is a third, harder point on the
boundary, not a controlled single-factor manipulation — that limit is frozen in its
pre-registration §1 rather than left to be discovered in its results.

**Do not read the two nulls as one finding.** They differ in shape: LogiQA's llama arm is
directionally NEGATIVE and near-significant raw (p=.0533), while P4b's is flat (−0.80 pp,
p_adj=1.00). **P4b's reverse DIAGNOSTIC is the sharper signal and sits OUTSIDE the Holm
family with an UNADJUSTED p** — llama `+4` reads **−8.80 pp, p_raw=.00094**, i.e. the
wrong-direction dose still hurts on a task where the right-direction dose does nothing.
Qwen's reverse cell is flat (+1.60 pp, p=.672). Cite it as a diagnostic, never as a result.

**GSM8K is not thereby established as the boundary condition** — three of the four tasks
share its `####` interface and two of the four ARE GSM-Hard, so the transfers rest on one
task family. wps-v0 (PREPARED, NOT RUN) asks the prior question of whether the workpoint is
even locally stable; until it runs, "the workpoint" is a single α from one frozen curve.


**Required reading before non-trivial changes:**
- `AdaDopamine_gsm8k.md` — current state. 2026-05-30 起 GSM8K template 已對稱化、mask offset bug 已修,所有舊 Phase 1/2 數字與當前 pipeline 不可比。Phase 1b(現已改稱 Phase 1)的重做**已完成**,見 §4.1–4.8;Phase 2 (Plans A–H3) 結論**暫擱置**,須重跑。**+α 機制表述已改框 (2026-07-24)**：主錨從「焦慮 anxiety / VTA→IPN」改為 **over-wanting → 冲动 impulsivity (抢答) + 认知僵化 / 强迫性反复 compulsivity / perseveration (loop),锚 VTA→NAcc**(理由:+α 数据无回避/freezing,抢答率随 α 单调升 = 急着 commit;不做 mania 类比)。DA→焦虑文献降為次要旁證。**脚本字段名 `analyze_loop_anxiety.py` / `ANXIETY_PATTERNS` / 表格 "Any anxiety" 保留原名**(改名破坏 U 形复现),但按 §2.3「命名说明」读作「强迫性 over-checking」——**勿把 doc 表述改回焦虑框架**。
- `AdaptiveThinking.md` — 歷史 Phase 1–2 設計、Plan A–H3 動機 / 失敗分析、Yerkes–Dodson framing、EMA + 1-step-lag physics(設計思路仍有效,但數字本身已過時)
- Literature & dopamine-LLM mapping now lives in the `AdaDopamine*.md` family below (the earlier standalone `Dopamine.md` / `Dopamine_EN.md` / `Dopamine2.md` are gone). Other on-disk docs not indexed here: `AdaLogitsLens.md` (logit-lens directional analysis), `AdaptiveThinking_bp.md` (pre-eot Phase 1–2 backup).
- `AdaDopamineBehaviour.md` — **NOT on disk as of 2026-08-11** (referenced by older notes; treat citations to it as unverifiable until it reappears). Clinical behavioral signatures of dopamine excess vs. deficit (anhedonia / loss of drive / bradykinesia). Grounds the two-arm framing: α positive → over-wanting; α negative → under-wanting (§2.3). Reference for the behavioral-economics suite below, not a pipeline doc. **NOTE (2026-07-24): the +α arm was reframed in `AdaDopamine_gsm8k.md` from "anxiety" to impulsivity + compulsivity/perseveration (VTA→NAcc) — this Behaviour doc's §2.2 still uses the older anxiety wording; unify the framing before citing it, and see the `AdaDopamine_gsm8k.md` doc-map note above.**
- `AdaDopamine.md` — **behavioral-validation stage of the four-part research line** (RSN → behavioral dopamine → brain dopamine → thinking curve). The source of truth for the wanting-proxy experiments (§3 MCQ/GSM8K/MATH cross-model tables; §4 = the full ⑤–⑩ experiment catalog: Betting / Bandit / Pressure / Effort-choice, plus the *skipped* PIT / Reversal / Agentic / TRAIT with documented why-skipped reasons). When working on any behavioral-economics entry-point, read the matching §4.x here for the design rationale and prior results. **Note its GSM8K numbers are pre-eot** (peak α=−4) and superseded by `AdaDopamine_gsm8k.md` (peak α=−6); unify before citing.
- `AdaDopamine_bp.md` — **the original (first-version) experiment record.** `AdaDopamine.md` is the curated current doc; `AdaDopamine_bp.md` is where the *raw prior results* live that the newer docs don't repeat — notably the full Bandit Assistant-Role-vs-No-Role tables (§4.7), the Agentic/ScienceWorld §4.9 data (success%/penalty% per α), and the complete why-skipped diagnoses for Reversal (§4.8, position-bias + phasic-DA incompatibility), PIT (§4.10), Agentic (§4.9) and TRAIT (§4.11). Read it when you need a prior result that `AdaDopamine.md` only summarizes. Its GSM8K/MATH numbers are pre-eot (see the note on `AdaDopamine.md`).
- **Parent paper (RSN):** `ACLARR/` in this repo (LaTeX `main.tex`; moved 2026-07-30 from `~/Downloads/ACL/ACLARR`, and briefly `~/Downloads/ACLARR` — both are gone, older refs are stale) — *"Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering"* (ACL ARR submission). This whole project extends that paper's metaphorical §6.1 "Digital Dopamine" hypothesis into empirical wanting/brain/thinking-curve validation. The behavioral docs above point back to it as the母 paper.
- `AdaptiveThinking.md` §3.1.6 / Phase 1b is the signal-proxy trajectory state-effect analysis (**complete**; "Phase 1b" is a retired name — see §Offline analysis workspace); `AdaThink.md` is an auxiliary trace analysis outside the main line.
- `AdaBandit.md` — **the Bandit design + literature doc** (renamed 2026-08-11 from `BanditExperiment_LiteratureReview.md`). §1.3 `Prompt Design Constraints` / `PV9 Prompt Modifications` is the design source for the pv9 Stage-1 changes; §2 is the literature review (EVOLvE, greedy-agent papers, Krishnamurthy et al.).
- `PV8_SCAFFOLD_DESIGN.md` — **deleted 2026-08-28** (commit `576cd1c`); recoverable from git history. It held the pv8 scaffold design notes; the Bandit line is CLOSED, and the pv8 facts a later protocol still depends on live in the pv8 bullet above.
- **Three `docs/*_ARCHIVE.md` files hold content SPLIT OUT of this one on 2026-09-02 to shrink the always-loaded context** — `LOCAL_CHECKS_ARCHIVE.md` (Bandit pv6–pv11 check commands), `P_PHASE_ARCHIVE.md` (P2/P3/P3-supp result tables, artifact hashes, runbooks), `BANDIT_PV6_PV11_ARCHIVE.md` (pv6–pv11 protocol internals). Every block in them is **byte-identical** to what stood here, and each was verified to still exist somewhere before the cut. They cover CLOSED or COMPLETE lines only; frozen wording, 口径 traps and anything an ACTIVE line needs stayed in this file. **Do not move an active line's rules into an archive** — the whole point of the split is that archived content is out of context by default.
- `docs/PREREG_P4C_CRUXEVAL.md` + `docs/p4c_amendment_0{1,2}.json` — the P4c protocol and its two amendments. Read before touching `data_cruxeval.py` / `get_answer_cruxeval.py` / `eval_cruxeval.py` / `run_cruxeval.sh`. The load-bearing decisions are recorded there, not in the code: why the official executor is NOT used, why FIRST stays MAIN when LAST would score two preflight items higher, and why a non-parsing payload is sometimes a parser bug and sometimes a result.
- **19 superseded `.sh` launchers + 1 orphaned entry-point were DELETED 2026-09-03 (commit `2e0b950`), recoverable from git history.** Four groups, each dead for a recorded reason: the **pv5 Bandit launchers** (`run_bandit_{alpha_direct,e_protocol,validity}.sh` — line CLOSED, results VOID); **`run_igt_v5.sh`** (v5 is documented as having COLLAPSED reasoning into round-robin, and `run_igt.sh` already takes `--prompt_ver`); the **pre-eot batch** (`run_{all_hidden,benchmark_*,mmlue,regenerate_mmlue,dopamine_signal}.sh`, `verify_gsm8k_plus4.sh` — all predate the 2026-05-30 template symmetrisation, so their numbers are not comparable to anything current); and the **self-report GENERATION variants** (`run_action_*_gen_*.sh`, superseded by Betting per §4.6) together with `get_action_generate_gsm8k.py`, whose only callers they were. **Still live and deliberately kept:** `run_demo.sh` (the Slurm template, like `run_judge_confidence.sh`), `run_extract_a2_only.sh` (subset re-extraction — `run_extract_signal.sh` globs the whole dir and re-runs finished cells), and the non-`_gen_` `run_action_{confidence,willingness}_gsm8k.sh`. Deleting an entry-point is safe in a way deleting a module is not — `.sh` files are never imported — but check for orphaned `.py` afterwards, which is how `get_action_generate_gsm8k.py` was found.
- `AGENTS.md` — Codex-facing **delta file, NOT a mirror**. This file is canonical and wins on any disagreement; AGENTS.md carries only the Codex-specific execution guidance plus a short orientation (playbook, architecture, local checks, editing guidance). It deliberately does NOT restate the experiment ledger, frozen-protocol rules, or 口径 traps — those live here only. **So a large size difference between the two is expected and is not drift.** When a convention changes, edit THIS file; touch AGENTS.md only if the Codex-specific delta itself changed.
- `~/CLAUDE.md` — running commands and data-directory map

## Architecture: how a run is wired together

A typical experiment is one of the `get_answer_*.py` / `get_action_*.py` entry-points driven by a sibling `run_*.sh` script. The dataflow is:

1. **Dataset loader** (`data_<benchmark>.py`) → JSON of question dicts (in `benchmark/` on the server, gitignored).
2. **`template.select_templates(suite=…)`** picks the prompt family. Three suites exist:
   - `default` — MMLU-style A/B/C/D answer extraction
   - `vanilla` — neutral phrasing without role
   - `action` — self-reported "reasoning willingness" 0–9
   Each suite has CoT / non-CoT / E-option (with "I am not sure") variants. The chosen template is rendered per-role via `utils.construct_prompt` + `utils.make_characters`.
3. **`llms.VicundaModel`** loads the LM. Three loading paths: `dream` diffusion (`AutoModel`, see `diffusion.py`), Mistral3 multimodal (`Mistral3ForConditionalGeneration`), or default `AutoModelForCausalLM`. All use bf16 + `device_map="auto"`. `_find_decoder_layers()` is the abstraction used by every layer-injection hook. **Generation entry points**: `VicundaModel.generate(inputs, batch_size)` for batched no-hook runs (`get_answer_*` uses this; pads with `padding=True`); `VicundaModel.generate_one(prompt, ...)` for bs=1 with caller-managed hooks (`track_hidden_states`, `closed_loop_gsm8k`, `track_dopamine_signal` all use this). **Do not mix**—batched vs bs=1 generation differs by ~2% acc on Llama due to padding artifact. `VicundaModel.regenerate(...)` (prefill-only steering path) takes an optional `stop_strings: list[str] = None` — default None keeps every existing caller byte-identical; only on the prefill_only branch, and only when set, does it pass `stop_strings`+`tokenizer` to `model.generate`. **Caveat learned the hard way:** HF `stop_strings` halts on the substring appearing *anywhere* in the output, so if a prompt's format spec literally contains the stop token (CGT's system prompt has `</choice>`), the model echoing it mid-reasoning truncates before the real answer — CGT tried `["</choice>"]` and invalid_rate jumped 0.02→0.11, so it reverted to natural EOS. Only use `stop_strings` when the marker can't appear in the prompt. `regenerate` also takes `prefill_tail_len: int = 1` (inject into the last N prompt tokens instead of only the last; default 1 = byte-identical to every existing caller, prefill_only path only — see the CGT Injection mechanics bullet).
4. **Steering / closed-loop** is layered on top via forward hooks on the decoder layers identified in step 3. The diff vectors come from `mean/` (per-role mean differences) gated by a mask from `detection/` (NMD / KL / KS / LR / PCA / t-test / XGB selectors over `task_list.py`).
5. **Output**: current GSM8K re-run artifacts are written under `/data1/paveen/Dopamine/components/...`. Older experiment scripts may still point at the historical `/data1/paveen/RolePlaying/components/...` tree.

## Phase 2 closed-loop (SHELVED)

> **Status: shelved, not active.** Plans A–H3 ran before the 2026-05-30 GSM8K template symmetrisation and mask-offset fix, so their numbers are not comparable to the current pipeline and must be re-run before being cited. Kept because the controller design and the 1-step-lag physics remain valid.

`closed_loop_gsm8k.py` + `run_closed_loop_gsm8k.sh` is the iteration target **if this line is resumed**. The shell script is the source of truth for hyperparameters; key knobs:

- `LAYER_START`/`LAYER_END` — injection layer range (Llama3-8B uses 11–20)
- `EMA_ALPHA` — feedback smoothing (Plans D/E/F all read EMA via `self.ema_alpha`)
- `K1` — primary gain. Meaning is plan-dependent: proportional gain (A/B/C/D/E/F/H2), fixed pulse magnitude (G/H1)
- `K2` — secondary knob. Plan C: spike-damping coefficient. Plan G: dead-zone half-width as fraction of `xp`. Unused by D/E/F/H1/H2
- `FLOOR_RATIO` — multi-purpose target ratio. Tonic floor (A/C), EMA homeostasis target (E), peak target (H1: 1.5, H2: 1.25)
- `PLATEAU_END_RATIO` — slope endpoint for B; H2 reuses it as the trapezoid end level (default 0.75·xp)
- `AVG_GEN_LEN` — defines T for B's plateau slope and H2's trapezoid decay endpoint
- `--plan {none,static,A,B,C,D,E,F,G,H1,H2,H3}` selects controller (case-sensitive; see `AdaptiveThinking.md` §3.2)
- H1 / H2 / H3 hardcode their window/segment boundaries inside `_compute_alpha()` — there are no `--h1_window` or trapezoid-segment CLI flags. H2 uses rise=50, plateau_end=200, peak=1.25; H3 uses rise=30, plateau_end=120, peak=1.35. Edit the constants in `closed_loop_gsm8k.py` if you need to sweep them.

When tweaking a plan, modify the `.sh` not the `.py` — the script is committed and reproducible. Past runs are kept commented (not deleted) in `run_closed_loop_gsm8k.sh` so the full sweep history is recoverable.

## Phase 1 signal-proxy validation (COMPLETE; "Phase 1b" is a retired name)

> **Status: complete** for §4.1–4.8 of `AdaptiveThinking.md`. The naming is settled as Phase 1 = observation / Phase 2 = control; "Phase 1b" survives only in older notes. The collection tooling below is still the way to gather new signal data.

- **Modeling-vs-finding naming rule (load-bearing):** tonic, ramping/vigor, and phasic-like are operational definitions of `G_prefill`, `s_t` slope, and `p_t`; they are not promoted, downgraded, or renamed by any single task result. Each experiment asks only whether that task/readout exhibits the predicted empirical relationship. A negative or null result is therefore a **task-level finding**, not a rejection of the mathematical construction or its operational name. Keep the modeling definition and the empirical verdict explicitly separate in all analysis and reporting.

The H2/H3 contradiction (H2 +1% but H3 better-shape-worse-acc) raised the question: **is the NMD-projected signal actually RSN-specific, or any sparse projection would show the same expert-vs-non_expert gap?**

- `track_hidden_states.py` + `run_track_hidden_states.sh` is the data-collection step. It runs **greedy, bs=1** generation under each role and dumps per-sample HDF5 groups: `prefill_hs (P, n_stored, H) fp16`, `decode_hs (T, n_stored, H) fp16`, plus on-the-fly NMD sanity scalars (`x_prefill_proj`, `x_decode_proj`, `ema_decode_proj`). Storage is **selective**: middle layers `[LAYER_START, LAYER_END)` + final layer (= 10 layers for Llama3-8B 11–20), not all 32. HDF5 meta records `final_layer_idx_stored / stored_layer_indices / n_stored_layers` so offline tools can distinguish new selective vs legacy full-32 schemas.
- Paper-aligned roles (current 5-run baseline in `run_track_hidden_states.sh`): `expert` → `"an expert"`, `non_expert` → `"a non expert"`, `primary_teacher` → `"a primary school teacher"`, plus `neutral` (No-CoT and CoT) as controls. The earlier `mathematician`/`non-mathematician` framing was retired — task-technique mismatch on grade-school GSM8K.
- Output dir: `${BASE_DIR}/hidden_states/${TASK}/${RUN_TAG}/hs_<task>_<size>_<mode>_<role>_L<s>-<e>.h5` (selective storage ≈ 2–3 GB per role, gzip fp16). `run_track_hidden_states.sh` isolates each collection under `RUN_TAG`; **`ALLOW_OVERWRITE=1` is now wired through to `track_hidden_states.py --allow_overwrite`, which refuses PER FILE** (fixed 2026-08-22 — see the overwrite-guard rule below). It drives **7 runs** (expert, non_expert, neutral No-CoT, neutral CoT, primary_teacher, neutral α=+4, neutral α=−4); `START_FROM=N` skips runs `<N`, `START_FROM=8` runs **only** the offline Step 2/3 extraction (`extract_signal_json` / `_remask` / `extract_entropy_confidence`, all directory-level globs over every H5). `SKIP_EXTRACT=1` runs the HS-collection runs and **stops before** Step 2/3 — use it to re-collect a single interrupted run (`START_FROM=7 SKIP_EXTRACT=1 …` re-does only α=−4) without re-extracting JSON for all H5, then extract once with `START_FROM=8`. A completed H5 carries `n_samples_done` / `accuracy` / `stored_layer_indices` in its meta; their absence means the run was interrupted. **Re-collecting now REFUSES rather than truncating** — pass `ALLOW_OVERWRITE=1` deliberately.
<!-- Why: the launcher ECHOED an ALLOW_OVERWRITE guard that never reached the tracker, and h5py opens mode="w"; a single-cell re-collect (START_FROM=11) silently destroyed hours of GPU time.
Evidence: commit 88da0f8; pre-fix `git show 88da0f8^:run_track_hidden_states.sh` BASE_ARGS carries no OVERWRITE arg, and the directory guard at line 93 only fires at START_FROM=1.
Scope: every HS collection, both models. -->
- **HDF5 collection is fail-closed on overwrite (2026-08-22).** `track_hidden_states.py` refuses to open an existing output path unless `--allow_overwrite` is passed; both launchers forward `ALLOW_OVERWRITE=1`. Before the fix the guard was **directory-level and only active at `START_FROM=1`**, so re-collecting one interrupted cell truncated it silently. **`g.attrs["generated"]` is also no longer capped at 4000 chars** — that text is what an agreement check compares, so a silent cut would make a divergence look like a match (the longest Qwen signal generation is 3895, i.e. the cap was about to bite).
- **Standalone re-extraction**: `run_extract_signal.sh` is the offline-only Step 2/3 (NMD + random + entropy) split out as its own script, parameterized by `RUN_TAG` (default `phase1b_eot`) — use it to re-extract JSON without any risk of touching the model/HDF5. Prefer it over the legacy `run_extract_all.sh`, which has hardcoded paths and is gsm8k-only. Both call the same three `extract_*` scripts; the difference is parameterization + a pre-flight `sanity_mask_indexing.py` check in the newer one.
- `extract_signal_json.py` exports NMD sanity projections; `extract_signal_json_remask.py` reprojects HDF5 against an arbitrary mask such as `diff_random_*`. Both emit per-sample signal JSON matching the neutral-baseline schema (`x_prefill / x_decode / ema_decode / *_per_layer` + meta + diff_stats). Backward-compat: detects selective HDF5 via `final_layer_idx_stored` and slices middle as `[0, n_middle)`, else falls back to `[layer_start, layer_end)`.
- `extract_entropy_confidence.py` loads `model.norm.weight` + `lm_head.weight` from safetensors (no full 8B load) and computes per-step `entropy / top1_prob / margin / info_gain` from the stored final-layer HS, plus a prefill snapshot. Same backward-compat for stored final-layer index.
- **wanting vs confidence are computed on DIFFERENT bases — do not conflate (settled 2026-07-16).** `wanting` (`dopamine_signal_*`, `x_decode`/`x_prefill`) = **middle-layer HS · sparse NMD mask** (≈0.5% of neurons selected). `entropy/top1/margin/info_gain` (`metrics_*`) = **full final-layer HS → RMSNorm → full lm_head → softmax over the whole 128k vocab, NO mask**. So confidence is the model's real, universal next-token output distribution; wanting is a selected-subspace projection. This is why persona (a pure-wanting operation) moves wanting but leaves confidence flat: they are not the same quantity measured two ways.
- **"late-decode convergence" means opposite things for the two axes (2026-07-16).** By Q4 (75–100% of decode) every state's `top1`≈0.99 and `entropy`≈0.05 — confidence is **ceilinged/floored** (math tails are near-deterministic: after `= 4 2` the next token is forced), so the late window has NO dynamic range and CANNOT show a state effect. wanting's late plateau (≈0.18) sits far from any bound, so ITS convergence is real. Therefore "state effects are a launch-phase phenomenon" is a genuine result for wanting but partly a measurement-saturation artifact for confidence — the discriminative window (Q1, top1≈0.85) is the only place confidence has room to differ. When reporting confidence dynamics use quartiles (Q1–Q4), NOT a single `decode μ` (μ averages the large early effect with the saturated tail into a misleading mid value — e.g. CoT wanting μ=+0.54 hides a monotone Q1 +0.91 → Q4 +0.08 decay). CoT confidence also flips sign Q1→Q2 (top1 +0.32 → −0.21): CoT is more decisive at the reasoning-template onset, then LESS decisive mid-computation while No-CoT has already committed — driven by WHAT token is being predicted (sentence-opener vs digit vs live arithmetic), not by who is "more confident".
- Offline analysis lives in `~/Documents/RSNResult/RoleAnswer/` (relocated 2026-07-16 from `~/Downloads/RSNResult/`, whose mount dropped — older doc/memory refs to the Downloads path are stale) — reloads the JSON + masks (`nmd_*` and `diff_random_*` from `${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/`) and compares **late-tonic-ratio gap, AUROC, Cohen's d** plus the multi-metric correlation matrix (`analyze_multi_metric.py`) between expert / non_expert / primary_teacher / neutral.
- `track_dopamine_signal.py` is the fast NMD-only path (no raw HS dump) — use when you only need scalars; use the HDF5 path when you want multi-mask flexibility or layer ablations. **As of 2026-06-28 this is the DESIGNATED MAIN tool for DA-curve analysis** (the slow `track_hidden_states.py` dump-HS + offline-remask route is kept but no longer the primary path). It now takes `--alpha` (default 0): `--alpha 0` = pure observer (byte-identical to legacy runs, no injection registered); `--alpha != 0` = prefill-only **output-side** steering (`hs[:, -1, :] += α·mask[l]`, L>1 only). Need a different mask? re-run with `--mask_type random` instead of dumping HS to remask offline. α-tag in the filename follows `_a{n}`/`_aneg{n}` (omitted at α=0, so steered runs never overwrite the α=0 baseline); meta carries `steer_alpha`/`steer_mode`.
- **Steering hook-alignment (load-bearing, fixed 2026-06-28).** Both `track_dopamine_signal.py` and `track_hidden_states.py` now inject steering via `register_forward_hook` on the layer **OUTPUT** (`hs[:, -1, :] += diff`), inside the SAME hook as the observation/recording and **before** projecting/recording, so (a) the stored HS / projected x_t is the **post-injection** signal, and (b) the layer alignment matches the NMD mask. The mask row for `decoder_layers[l]` is extracted from that layer's OUTPUT (`get_answer_logits.py --save` → `output_hidden_states[l+1]`, then `nmd.py: mask[1:]`), and `llms._regenerate_prefill_only` (the canonical steering path behind `get_answer_regenerate_*`) also injects on the OUTPUT — so steering MUST be output-side. **The pre-2026-06-28 `track_hidden_states.py` used `register_forward_pre_hook` (INPUT-side), adding α·mask to `decoder_layers[l]`'s input ≈ `decoder_layers[l-1]`'s output — a one-layer mask misalignment** (its comment claimed "identical to get_answer_regenerate" but it was not). Consequence: **all pre-2026-06-28 steered (α≠0) signal_eot / HS data is layer-misaligned and must be re-collected** before being used as a steering result; the α=0 baselines are unaffected (no injection hook was registered). When adding a new tracker, copy the output-side single-hook pattern — never re-introduce a pre-hook for steering.
- **Prompt self-documentation**: HDF5 meta, `dopamine_signal_*.json`, `random_signal_*.json`, `metrics_*.json`, and `closed_loop_*.json` all carry `prompt_template` (the raw template string) in their meta as of 2026-05-30. `grep '"prompt_template"' <file>` self-attests which prompt produced the result — use this before comparing numbers across runs.
<!-- Why: a misaligned mask or a wrong injection token still runs and still produces
plausible numbers; both have cost a multi-hour sweep before. These are the cheap gates.
Evidence: the 2026-05-30 offset bug; check_igt_qwen.py / check_mask_qwen.py docstrings.
Scope: any port of an existing paradigm to a new model or band. -->
<!-- Why: this rule already existed but only inside the CLOSED Bandit block, under a bullet
saying its mechanics were retired elsewhere — so it was invisible, and a new server launcher
was written with python3.10 and died at exit 127 before the preflight ran (2026-08-21).
Evidence: run_bandit_pv6.sh / run_gsm8k_qwen25.sh both use `python`.
Scope: EVERY server launcher. -->
- **Server launchers use `PY="${PY:-python}"`, never `python3.10`.** The server conda env names its interpreter `python`; `python3.10` is the LOCAL analysis-box convention (`RoleAnswer/` scripts) and does NOT exist on the server, where it exits **127 before anything runs** — under `nohup` the job dies silently and the log looks empty. After launching ANY background job, `cat` the log immediately rather than assuming it started. A new launcher should also assert the interpreter resolves and can `import numpy, torch`, so a wrong env names itself instead of failing obscurely.
- **Cross-model PORT pre-flight: run the matching `check_*.py` BEFORE any sweep.** Five exist, all read-only and none needing a full sweep: `check_mask_qwen.py` (mask only, **no GPU and no model weights** — file/dtype, row count == the model's decoder-layer count, non-zero rows EXACTLY `decoder_layer_range(start,end)`, uniform `top_k` sparsity, plus `--compare` to answer "different band or the same neurons relabelled"), and the four paradigm ports `check_gsm8k_qwen.py` / `check_cgt_seq_qwen.py` / `check_igt_qwen.py` / `check_signal_qwen.py` (tokenizer + chat template + double-BOS, the ACTUAL final prompt token at each injection site, mask/band alignment, observed `steering_fires` vs `L*B*t`, and the α=0 execution path). **The injection token is model-specific and must be READ OUT, never assumed** — e.g. IGT's anchor is deliberately EMPTY (an anchor there suppresses the reasoning span both the learning readout and prefill steering need — the CGT-simple4 failure), so injection lands on the assistant header's last token: Llama-3.1 `id 271 '\n\n'` vs Qwen2.5 `id 198 '\n'`. A wrong band or wrong site does not raise; it produces uninterpretable data hours later.
<!-- Why: three launchers exist for this probe and NO doc recorded it; a later session
would either re-derive the design or run the arms on different cards and lose the contrast.
Evidence: run_nmd_qwen25.sh / run_mmlue_qwen25.sh headers; no *11_18* artifact exists.
Scope: Qwen band position only. Delete this bullet if the probe is abandoned. -->
<!-- Why: check_gsm8k_qwen.py validates the vc.regenerate path; the tracker registers its
OWN hook and injects inside it, so injection site, fire accounting and the alpha=0 path are
different objects. A toy forward also never reaches generate_one().
Evidence: check_signal_qwen.py docstring; the 2026-08-21 preflight exit-127.
Scope: track_dopamine_signal.py ports. -->
- **Qwen2.5 signal replication (`run_track_dopamine_signal_qwen25.sh`, RUN_TAG `qwen25_signal_v1`, started 2026-08-21).** Ports the Phase-1 signal line to Qwen via the LIGHTWEIGHT path (`track_dopamine_signal.py` — projected scalars + generated text, **no HDF5**). Protocol matches the frozen behavioural run (`run_gsm8k_qwen25.sh`) exactly — same 300 GSM8K questions, bare-string, greedy, bs=1, `max_new_tokens=768`, `ema_alpha=0.95`, `role=neutral` — and differs from Llama's `phase1b_eot` ONLY in the band (`[16,22)` L=6 vs `[11,20)` L=9), which is a per-model mask fact, not a knob.
  - **`check_signal_qwen.py` is a SEPARATE pre-flight and cannot be replaced by `check_gsm8k_qwen.py`** — that one validates the `vc.regenerate` path, while the tracker registers its own forward hook and injects INSIDE it, so injection site, fire accounting and the α=0 execution path are different objects. Beyond the usual mask/band/token checks it drives the REAL hooks on real `nn.Module`s (CPU, no GPU): α=0 leaves the state bit-identical, α≠0 touches only the last prefill token, decode is never injected, exactly L layers fire, and the recorded projection is POST-injection, verified numerically as `Δ == α·mean_l‖mask_l‖²`. Mutation-tested against three real implementation errors (pre-hook/input-side, injection-after-projection, missing prefill-only guard). **`--model_dir` is REQUIRED** unless `--skip_tokenizer` is passed, which downgrades the verdict to `PARTIAL CHECKS PASSED` — the injection token is model-specific and must be read out.
  - **The toy forward feeds each layer an INDEPENDENT clean input, deliberately.** The identity `Δ == α·mean_l‖mask_l‖²` holds only when a layer observes its own injection alone; threading one tensor through identity layers makes layer *l* also see layers *<l*'s edits (a real stack's non-linearities do not accumulate this way), inflating Δ by the cross-layer mask overlap — measured as +0.45% on a random-support mask. Propagation is covered separately by the accumulated-edit check. Do not "fix" this back to serial threading.
  - **ONE CURVE, ONE GPU, and each curve carries its OWN α=0.** Every readout is paired per-question against that curve's α=0, and bf16 greedy is not byte-reproducible across GPUs, so a split curve mixes the machine difference into the α effect irrecoverably (summary CSVs carry no device field). The launcher REFUSES an unset or multi-card `CUDA_VISIBLE_DEVICES`. `NOCOT` = 11 α `−8…+12` incl. 0; `COT` = `{0,+6}`, self-contained. Roughly 40–70 min per cell → NOCOT ≈ 8–13 h.
  - **Signal–behaviour pairing 口径:** per-question pairing uses THIS run's own stored text / commit marker / correctness. The frozen table in `AdaDopamine_gsm8k.md` §4 stays the PRODUCTION accuracy reference and is never mixed per-question with these samples (different batch). `+10/+12` are collected so `G_prefill`'s linearity can be checked past the point where `s_t` / commitment / accuracy have already saturated.
  - **HDF5 is a SEPARATE later step and requires re-running those cells** — the lightweight path discards hidden states after projecting. Plan: after the main curves, re-run only the representative cells (No-CoT `−8/0/+6/+8`, CoT `0/+6`) through `track_hidden_states.py`. Call that a **same-protocol representative re-run, NOT the same trajectory**: same card and seed usually reproduce closely but bf16 can still diverge at a critical token, and one divergence changes the whole chain. So an HDF5 batch carries its OWN signal readouts and must not be cross-cited per-question with the lightweight batch; store question IDs + generated text and record the agreement rate rather than assuming it.

- **Qwen2.5 signal curves are COLLECTED (2026-08-22); the HS backfill is `run_track_hidden_states_qwen25.sh`.** All 13 lightweight cells landed (11 No-CoT α ∈ −8…+12, CoT {0,+6}; 300 each, band 16–22, mnt 768, `run_tag=qwen25_signal_v1`), stored under `qwen2.5/dopamine/signal/` (renamed from `dopaminel/` 2026-08-24; older refs are stale). Manipulation check passes: `x_prefill ~ α` is linear at **R²=0.9998**, slope ≈ mean‖mask‖². The batch independently reproduces the frozen behavioural table (+8 86.00 vs 86.00, +10 88.00 vs 88.33, +12 87.67 vs 88.67) and its mechanism variables (`posN_med`, early-candidate %) — same protocol, different batch, bs=1 vs bs=24, so this is a consistency check and **still not a per-question mix** with the frozen table.
  - **The HS launcher is SEPARATE from the Llama one and must stay so** — `run_track_hidden_states.sh` is frozen at band 11–20 / llama3 mask / `phase1b_eot` with runs 1–7 already collected; parameterising it for Qwen would fork the 口径 of every stored Llama H5. Seven representative cells: No-CoT −8/0/+6/+8/+12, CoT 0/+6. **+12 is load-bearing** — entry gain stays linear through +12 while the commit-aligned decode response is heavily compressed, and a 1-D projection cannot distinguish a state ceiling from a reallocation inside the state space.
  - **HS cells are INDEPENDENT, which is what licenses regrouping them across GPUs.** Each cell's agreement check is against its OWN lightweight cell, and every α contrast is read from the already-collected lightweight batch; no HS cell is ever paired per-question with another HS cell. So one-curve-one-GPU binds only WITHIN a cell (which never splits). `G1|G2|G3` is a token-balanced 3-way split (282k/207k/207k decode tokens, 1.36×) replacing the natural `NOCOT|COT` split's 2.35× imbalance. Measured cost: ~27–42 min/cell on the lightweight path, **24–30 GB** total HDF5 for the seven cells.
  - **A retracted conclusion worth not re-deriving.** A first commit-aligned pass reported "decode-state saturation" after +8. It was withdrawn: four 口径 errors (char-proportional commit mapping, prefill-seeded EMA, a raw-vs-standardized ratio, `p_t` miscalled an independent channel — see the Phase-1 rules above). The surviving claim is **decode-response compression on the delayed-commit subset**, in raw projection units, **not a ceiling**. Note the cohort is selected on the outcome of the manipulation (a ≥20-step pre-commit span exists in 4–8% of samples at α≤+4 but 96–97% at +8…+12), so low-α cells are coverage rows, never a matched contrast. Analysis lives in `RoleAnswer/qwen_signal/commit_aligned.py` (offline, `python3.10`), whose header records all four errors with measured magnitudes.

- **Qwen HS backfill is COLLECTED AND ACCEPTED (2026-08-24). Acceptance runs through `check_hs_qwen25.py`, which is READ-ONLY and must pass before any geometry work.** Seven cells under `hidden_states/gsm8k/qwen25_signal_v1/`, ~33.9 GB, frozen by `ACCEPTANCE_20260824.txt` + `HS_MANIFEST_20260824.txt` (sha256 + size per cell). Three checks: integrity (n=300, `n_stored_layers=7`, band `[16,22)`, `mnt=768`, metadata α/CoT agreeing with the filename, per-sample `decode_hs`≡`x_decode_proj`≡`ema_decode_proj` lengths, `question_idx` a full 0..299 cover), projection reproduction, and per-question agreement vs the lightweight cell. Mutation-tested 16/16 against real defects.
  - **`stored_layer_indices` is `[15,16,17,18,19,20,27]`, NOT `[15..21]`** — the stored set is the middle band plus the model's FINAL layer, whose model-space index is `num_layers-1` (27 for Qwen's 28), not `layer_end-1`. A file storing `[15..21]` passes a layer-COUNT check while carrying no final layer at all, so the check compares indices.
  - **The projection check is a MEAN over layers, and getting this wrong fails every healthy file.** `utils.project_rsn_numpy` is `np.sum(hs*dirs, axis=-1).mean()`; summing instead is off by exactly `n_middle` (6× here). Call the shared helper, never reimplement. Tolerance is relative (`rtol=1e-2`) but it is a **wrong-mask/wrong-band detector, not a precision test** — the tracker casts HS to fp16 BEFORE projecting (`track_hidden_states.py:206`, then 223/227 back to fp32), so the stored states ARE the projected states and a healthy full probe reads **exactly 0.00e+00**. A wrong mask reads 3.5, corrupted HS 38.
  - **AGREEMENT MEASURED 1.000 ON ALL THREE FIELDS IN ALL SEVEN CELLS** (question / generated / correct; per-cell accuracy identical to the lightweight batch). So on this batch the HS re-run and the lightweight batch are the same trajectory, and commit positions / correctness / text diagnostics computed on the lightweight batch transfer directly. **This is an OBSERVED result, not a protocol guarantee** — it is a property of this card, code and length distribution. Keep reporting it as "measured agreement 1.000 on this batch"; do NOT restate the rule as "same-protocol re-run == replay", and keep analysing each cell on its own trajectory.
  - **A truncation probe that flags a round length needs the DECODE STEPS and the tail, not the length alone.** `ROUND_CAPS` dropped 1000 after a verified false positive: `nocot` α=0 sample `0101` is exactly 1000 chars but ends `\boxed{8} ####<|endoftext|>` at decode step 281 of 768 — natural EOS. A real cap shows up as a CLUSTER at one length; a lone sample with an EOS tail and a step count far below `max_new_tokens` is coincidence.

<!-- Why: extract_signal_json.py mirrors the FROZEN Llama out_meta, so the alpha exists only in the filename;
a loader that defaults instead of raising puts a steered cell into the alpha=0 reference slot silently.
Evidence: qwen_signal/hs_layerwise.py alpha_from_name + its batch-provenance guard.
Scope: every consumer of components/qwen2.5/signal_hs. -->
- **The HS-derived signal JSON and the lightweight batch share a SCHEMA AND A NAMING CONVENTION, so they must never share a directory.** `extract_signal_json.py` re-projects the H5 offline (`--h5_dir/--mask_path/--layer_start/--layer_end/--out_dir`, its selective branch keys on `final_layer_idx_stored`, so it works on Qwen unmodified) and writes to `components/qwen2.5/signal_hs`; the lightweight cells stay in `components/signal`. Two consequences that have already bitten: its `out_meta` **drops `steer_alpha`**, so the α lives only in the filename and a consumer must parse it STRICTLY and raise on anything unrecognised; and because both batches look alike, a wrong `--out_dir` silently produces plausible numbers from the wrong batch. `qwen_signal/hs_layerwise.py` guards on the α SET (the HS export has exactly No-CoT {−8,0,+6,+8,+12} and CoT {0,+6}; seeing −6/−4/+10 means it is the lightweight batch). **Local layout note:** `qwen2.5/dopamine/` also holds six loose top-level JSONs byte-identical to `signal_hs/`; they are one level above the lightweight `signal/` and would pass an α-set guard, so always point analysis at an explicit subdirectory.

<!-- Why: today the identity was applied to real per-layer data, "disproved" the mask file, and nearly caused a good
mask to be replaced. The existing toy-forward bullet states the same fact but only for the synthetic check.
Evidence: qwen25_signal_v1 (mu_a8-mu_a0)/8 = [3.077, 5.327, 9.545, 14.331, 36.834, 73.923] vs file
[3.077, 3.392, 4.673, 7.925, 21.128, 41.682]; per-sample std 0.009 -> 1.540 monotone.
Scope: any per-layer use of the co-design identity on REAL trajectories, both models. -->
- **NEVER recover `‖m_l‖²` from real data by regressing the per-layer prefill mean on α — the co-design identity holds ONLY for the FIRST steered layer.** Injection lands on each steered layer's OUTPUT, and that output is the next layer's INPUT, so layer *l* carries its own injection PLUS the propagated residue of layers `<l`. Measured on `qwen25_signal_v1`, `(μ_a8 − μ_a0)/8` per sample:

    | | L15 | L16 | L17 | L18 | L19 | L20 |
    |---|---|---|---|---|---|---|
    | mask file | 3.077 | 3.392 | 4.673 | 7.925 | 21.128 | 41.682 |
    | α-slope | **3.077** | 5.327 | 9.545 | 14.331 | 36.834 | 73.923 |
    | per-sample std | **0.009** | 0.021 | 0.057 | 0.175 | 0.503 | **1.540** |

  L15 — the first steered layer — matches to 1e-4 with near-zero spread; every later layer inflates and the std grows monotonically, and **that rising spread IS the propagation**. This is the same trap the toy-forward bullet above describes, which feeds each layer an INDEPENDENT clean input precisely so the identity holds there; a real stack does not, so it does not transfer. **The mask file is authoritative.** A mismatch is not evidence the mask is stale — check the first steered layer before suspecting the artifact. (`Z = G/sig` cancels `‖m_l‖²` entirely, so this affects only `G`; every stored Z-coordinate result is unchanged either way.)

- **Qwen band-position probe — PREPARED BUT NEVER RUN (as of 2026-08-21): three launchers, zero artifacts.** All Qwen work so far uses `[16,22)` (L=6), chosen as the onset of the layer-wise Expert/Non-Expert Pearson descent. `run_nmd_qwen25.sh` builds a SECOND, EARLIER mask `[11,18)` (L=7) so the band position is **tested rather than assumed**; it does NOT replace the existing mask. Order is fixed in the launcher headers: build the mask → `check_mask_qwen.py --layer_start 11 --layer_end 18 --compare 16 22` → `run_mmlue_qwen25.sh --new` → `--old`. **MMLU-E is the probe because its E option ("I am not sure") is a wanting readout INDEPENDENT of correctness** — E-ratio should move bidirectionally with α while accuracy holds, so a band that cannot move E-ratio without damaging accuracy is not worth spending IGT/GSM8K time on. Two design points that are easy to lose: **(a) the two bands OVERLAP at decoder layers 15–16**, so any difference is "earlier band vs later band", NOT "two independent neuron sets" — read the per-layer Jaccard from `--compare` before wording any conclusion; **(b) the `[16,22)` arm is RE-RUN rather than cited from stored numbers**, and both arms include α=0, so each band carries its own same-card baseline and an E-ratio shift cannot be a baseline or cross-GPU artifact. Keep both arms on ONE card. `run_igt_qwen25.sh` is the downstream consumer if a band wins.
<!-- Why: `ema_decode` in every stored signal JSON is PREFILL-seeded while the project s_t is decode-seeded; the gap grows with alpha, so it silently corrupts any cross-alpha decode readout. This produced a retracted "decode-state saturation" conclusion on 2026-08-22.
Evidence: track_dopamine_signal.py:174 (`self._ema_val = self._prefill_proj`) vs phase1_gain.decode_ema ("seed = decode_z[0], NOT prefill"); measured |stored - decode-seeded| at t=20 is 0.17 at alpha=0 but 97.18 at alpha=+12, still 20.9 at t=50.
Scope: every analysis reading `ema_decode` from a dopamine_signal_*.json. -->
- **NEVER read the stored `ema_decode` as `s_t`. Recompute it from `x_decode`.** The tracker seeds its EMA with `x_prefill`, but the project `s_t` 口径 (`phase1_gain.decode_ema`) seeds at `decode[0]`. With β=0.95 the entry value bleeds into dozens of decode steps, and because that entry value GROWS with α (≈−5 at α=0, ≈+290 at α=+12) **the contamination is α-DEPENDENT** — a cross-α decode comparison therefore mixes an entry-injection residual into the effect. It is worst exactly where commitment analyses look: when commit timing also moves with α (Qwen: median first-`####` step ≈1 at α≤+4 → ≈187 at +12), each cell's pre-commit window sits at a different contamination level. `phase1_gain.py` handles this correctly; ad-hoc scripts are where it goes wrong.
- **Two companion 口径 rules for commit-aligned work.** (a) **Locate the commit marker with the tokenizer, not a character ratio** — chars-per-token is ~1 for digits and ~4 for prose, and the pre-commit span is where that ratio is most skewed; re-encoding just the prefix beats a length ratio but is still not exact (boundary re-tokenization), so prefer offset mapping over the full text. (b) **`p_t = z_t − s_{t-1}` is a residual of the SAME one-dimensional projection that yields `s_t`** — report it as residual amplitude, NEVER as an independent "fast channel" corroborating `s_t`.
<!-- Why: cos=0.987 was first read as strengthening a scalar EXCLUSION; it means the opposite, and CV/F alone cannot separate "same profile, smaller" from redistribution.
Evidence: qwen_signal/hs_layerwise_RESULT.txt section 3 (retraction in place); hs_null_specificity.scale_fit.
Scope: every per-layer response claim on the Qwen HS cells. -->
- **The per-layer response test is the SCALAR-COMPRESSION RESIDUAL, not CV and not an F ratio.** Fit `v_high ≈ k·v_low` across layers and report `k` with `residual = ‖v_high − k·v_low‖²/‖v_high‖²`. Measured on Qwen No-CoT: **k = 0.309**, best scaling explains **97.4% of the response ENERGY**, normalised **squared** residual **2.6%** — whose square root, the residual **norm** ratio, is **16.1%**. **Always say which**: a bare "residual 2.6%" reads as an amplitude difference. Note also that the squared residual and `cos` are the same geometric fact (`resid = 1 − cos²` exactly for the least-squares `k`, verified to 1e-16), so **`k` is the only one of the three that adds independent information** — the amplitude compression. Report `k` alongside the residual, never the residual alone. **Heterogeneous loading and a sign-reversed layer do NOT indicate redistribution**: a constant negative loading at L20 is exactly what one latent scalar projecting through a fixed heterogeneous profile produces. So the frozen wording is "decode response AMPLITUDE is compressed; loading is heterogeneous including an L20 sign reversal; but the two dose steps are near-collinear (cos 0.987), so this **excludes a layer-synchronous uniform ceiling**, shows **no appreciable direction rotation**, and **remains compatible with scalar gain compression along a fixed layer profile**." An earlier reading treated cos≈1 and F=5.44 as *strengthening* a scalar exclusion — that is backwards and is retracted in place. **A classical repeated-measures F p-value is invalid here** — the six layers are re-projections of one hidden state and are strongly dependent, so sphericity fails. Use F descriptively and to ORDER directions, never to say "the interaction is significant". This does not mean inference is impossible: a **question-level permutation test or a cluster bootstrap** (clustering on question) would be legitimate, and is the route if the layer×dose difference ever has to carry a claim rather than describe one.
<!-- Why: the step-5 nulls landed on the branch the pre-registration warned about, and a reader seeing "pctile 0.0%, p=0.182" will otherwise reach for "RSN is special" — the extreme points the un-hoped-for way.
Evidence: qwen_signal/hs_null_specificity_RESULT.txt (VERDICT + ASYMMETRY).
Scope: every Qwen null-remask claim. -->
<!-- Why: "commit timing plateaus" contradicted the section's own table (c_med 110->134->163); the two claims are about different quantities and the absolute one never plateaus.
Evidence: qwen_signal/entry_gain_RESULT.txt section-5.3 addendum.
Scope: every Qwen commit-timing claim. -->
- **"Commit timing plateaus" is TRUE of the normalised position and FALSE of the absolute step — always say which.** Qwen `posN_med` saturates at .828/.839/.854 for +8/+10/+12, but absolute `c_med` keeps rising 110→134→163 because decode length grows again (258.9→279.4). Not a contradiction, two different quantities.
- **§5 is a cross-model ANALYSIS, not a replication.** The §4 chain transfers and **entry linearity + entry–decode decoupling do replicate**, but the **behavioural curve does not**: **Llama shows an asymmetric peaked response** (sharp optimum α=−6, collapse at −8, gradual decline then flattening on the positive side), **Qwen a high-dose plateau** (monotone rise saturating at +8, **no right arm in band**). Write "analysis framework portability + partial mechanism replication", never "the result replicates". **Three mechanism items DO replicate** (entry linearity; `jump/Z_prefill` −0.965…−1.003 entry–decode decoupling; the commit-locked signed `p_t` dip, deepening −0.578→−1.322); **two do NOT** (behavioural curve; `p_t` AMPLITUDE, flat across −8…+12 on Qwen where Llama has a clean α=−6 rise); and **ONE is a THIRD CATEGORY — same-sign but ATTENUATED** (post-commit slow-state release, §5.3.1).
<!-- Why: an earlier reading said Qwen's s_t "does not drop after commit"; that came from
comparing a ±20-window mean against the Llama figure's visual drop over ±50 and inverted
"attenuated" into "absent".
Evidence: qwen_signal/plot_qwen_mainfig.py (reproduces all 33 suite34 s_pre/s_post/release/
n_risk values); Llama reference recomputed on matched windows.
Scope: every cross-model magnitude claim. Never read a magnitude off a figure. -->
- **Post-commit release is SAME-SIGN but ATTENUATED, and must not be filed under either "replicates" or "does not".** On MATCHED `±20` windows (`[c−20,c)` vs `[c,c+20)`), computed by the same code path: Llama α=0 No-CoT **−0.279** (`d_z` −0.785, coverage 73.3%), CoT **−0.436**; Qwen **+8 −0.148 / +10 −0.171 / +12 −0.233** (`d_z` −0.356/−0.416/−0.557). So Qwen's largest (`+12`) is **0.84×** Llama's (`−0.233/−0.279`), with `+8`/`+10` at 0.53×/0.61×; same sign — it falls slowly FROM A HIGH LEVEL rather than releasing the state. Frozen wording: "Qwen also shows a commit-related fast transition and a subsequent slow-state release, but the slow release is attenuated." **Two caveats travel with it, always.** (a) **Cohort**: Qwen's only readable cells are `+8/+10/+12`, and their pre-commit window exists BECAUSE α delayed the commit — the cohort is selected on the manipulation's own outcome, while Llama's α=0 is not; Qwen has no readable α=0 reference (coverage 4.0%, n=12) and adding samples cannot create one. (b) **Low-dose cells (`α≤+6`, coverage 3.0–41.3%, n=9–124) are COVERAGE DIAGNOSTICS ONLY** — their release values scatter −0.112…+0.144 with no order and must never carry a mechanism reading, in particular not "release is positive at low dose". They are printed in §5.3.1's table in italics so the coverage gradient stays visible; italic = do not read.
- **The RETRACTED reading, and the procedure that prevents it recurring.** An earlier version said Qwen's `s_t` "does not drop after commit". That came from comparing Qwen's `±20`-window mean against the **Llama figure's visual drop, which runs to ±50 steps** — not the same quantity — and it inverted "attenuated" into "absent". **Never read a magnitude off a figure**: recompute the reference on matched windows through the same code path. Two 口径 traps found while doing so, both of which silently produce plausible-but-wrong numbers: (i) **the commit locator must be the frozen `commit_aligned.commit_step`** (offset mapping, `####`-only). Porting Llama's `char_to_step` + answer-candidate fallback reads `s_pre`=1.361 at +8 instead of 1.448 — the fallback fires constantly on Qwen, which answers first at low α, silently defining a different commit. (ii) **window means must be per-sample, then averaged** — a column-wise `nanmean` over a NaN-padded centered matrix changes the per-column denominator and drifts off the frozen value. `plot_qwen_mainfig.py` reproduces all 33 `suite34` `s_pre`/`s_post`/`release`/`n_risk` values exactly; that agreement is the acceptance check for any future edit to it.
- **`plot_qwen_mainfig.py` is NOT a recolour of `analyze_cot_mainfig.py`.** Three of the Llama figure's four panels cannot be honestly reproduced: entropy/top1/margin are **BLOCKED** (no `metrics_*.json` under the Qwen tree — absence of a file is not a measurement of zero, so those panels are OMITTED with a printed reason, never drawn as null), and Llama's CoT-vs-No-CoT-at-α=0 contrast is unavailable because Qwen's readable commit-centered cells are the high doses. The dose axis is therefore the primary contrast and CoT is secondary. It repoints the SHARED `phase1_gain` machinery via `configure()` — but `FSUFFIX` (`_ema0.95_L16-22.json`) is a module global that `configure()` does NOT expose, and `load_mask_norm2()` hardcodes the Llama slice `m[10:19]`; Qwen's band `[16,22)` is rows `[15:21]`, asserted at load time because a wrong band does not raise.
<!-- Why: the "same working state, different baseline and approach direction" reading is the
natural thing to write from the two curves, and it is NOT supported -- two of its four
quantities are not comparable in kind, one is missing, one has no reference cell.
Evidence: AdaptiveThinking.md §5.7; Qwen α=0 at-risk n=12/300.
Scope: every Qwen<->Llama working-point sentence. -->
- **The cross-model WORKING-STATE ALIGNMENT is NOT executable, and adding samples will not fix it (§5.7).** Do not write "Qwen +8 and Llama −6 reach the same working state". Of the four quantities: `Z_prefill` is opposite-sign AND standardized per-model, so "is +32.68 comparable to −19.74" has no definition without a model-free scale; `s_t` is a paired Δ on Llama but a LEVEL on a Qwen at-risk cohort, and the matching Qwen α=0 reference **does not exist** (at-risk 12/300); output decisiveness is BLOCKED on Qwen. **Especially do not write "Qwen needs +α, Llama needs −α, so their baselines differ"** — that sentence is built entirely from the raw-α comparison the section forbids. What IS sayable is shape-level: both models show entry-linear → decode-nonlinear decoupling, and both best working points carry commitment moving off the opening plus reduced degenerate tails.
<!-- Why: this family was BLOCKED for months and the block is now lifted, but the replacement is NOT "available" — 7 cells vs the RSN family's 11, and no matched alpha=0 cohort. A reader who drops the caveats will plot the two families as equal evidence.
Evidence: RoleAnswer/qwen_signal/logit_family_RESULT.txt; AdaptiveThinking.md §5.5 Result 2-3.
Scope: every Qwen entropy/top1/margin claim. -->
- **Qwen's logit family is UNBLOCKED (2026-08-26) and reads `PARTIALLY AVAILABLE / SPARSELY SAMPLED` — never "available".** `extract_entropy_confidence.py` produced 7 cells under `qwen2.5/dopamine/metrics_hs/` (No-CoT {−8,0,+6,+8,+12}, CoT {0,+6}); analysis is `RoleAnswer/qwen_signal/logit_family.py`, frozen as `logit_family_RESULT.txt`. **The §4.4 Result 3 question is now ANSWERED on Qwen: α is NOT a selective wanting intervention there either** — it moves the output distribution as well as the RSN state. **But the effect is STAGE-SPECIFIC and that phrasing is load-bearing**: task entry becomes monotonically more decisive with α (entropy 2.099→0.315, top1 .240→.892, R²≈.93–.95 over 5 cells), the commit step itself sharpens further with dose (entropy dip +0.044/+0.058/+0.092 at +6/+8/+12), and the ±20 window AFTER commit is LESS decisive than the window before (Δentropy +0.215/+0.150/+0.140, all p<0.01). **Four caveats travel with every number.** (a) **7 cells vs the RSN family's 11** — report monotone trends only; a peak or inverted-U is `under-sampled / not decidable`, and the two families must never be plotted as equally sampled. (b) **`####` changes the output format AT the alignment point**, so the post-commit entropy rise is a stage-specific transition, NOT wholly a real confidence drop. (c) **The cohort is selected on the manipulation's own outcome** (≥20-step pre-commit span: 4.0% at α=0, 75.7% at +12), so α=0 (n=12) and −8 (n=19) are coverage diagnostics only and there is NO matched α=0 reference. (d) **CoT +6 is the ONLY readable CoT cell and shows no transition (Δtop1 −0.0002, p=.59) — write "a commit-locked confidence transition was NOT DETECTED in the CoT +6 readable cohort", NEVER "CoT abolishes the transition".** **This result is filed under `Qwen Only`, NOT under `null`** (§6 item 22): the α effect is real and stage-dependent, so classifying it as a null would misreport a positive finding as an absence. **`entropy/log(V)` is a VOCABULARY-SIZE normalisation, NOT a model-free axis** — tokenizer granularity, vocabulary composition and distribution shape still differ, so it makes the two models more comparable without making them directly commensurable; cross-model use additionally requires Llama's own commit locator and the same ±20 window, never raw entropy.
- **`logit_family.py` reuses `commit_aligned.commit_step` by IMPORT, and this is the load-bearing decision.** Offset mapping, `####`-only, no answer-candidate fallback — porting Llama's `char_to_step` + fallback reads a DIFFERENT commit on Qwen (the fallback fires constantly because Qwen answers first at low α), silently redefining the event every aligned number is built on. Two further 口径 it enforces: window means are **per-sample then averaged** (a column-wise `nanmean` over a NaN-padded centered matrix changes the per-column denominator and drifts off the frozen values), and `load_metrics` guards the **α SET, not just the cell count** — `metrics_hs` and a lightweight `metrics` tree would share a schema AND a naming convention, so a wrong dir silently yields plausible numbers from the wrong batch. **Fixed decode quartiles are DESCRIPTIVE ONLY and the script labels them so**: α moves the median commit from decode step ≈3 to ≈187, so a fixed quartile samples a different generation phase per cell, and the Q1-up/Q4-down pattern is largely window composition — the same trap `commit_aligned.py`'s header records for `s_mid`.

<!-- Why: the extraction is a one-shot server job whose failures were all silent —
a zero-filled empty trajectory and a missing question_idx both produce a loadable
JSON that cannot be paired or audited afterwards.
Evidence: test_extract_entropy_confidence.py (33 checks, committed and re-runnable).
Scope: every metrics_*.json run, both models. -->
- **`extract_entropy_confidence.py` run + read rules (hardened 2026-08-25).** The script computes the logit family on the **FINAL layer** (via `final_layer_idx_stored`), so the band never enters the maths — it is verified against each H5's meta and used in the filename only.
  - **`--model_dir` must be a LOCAL snapshot.** A bare HF repo id is refused: the old fallback built a full `AutoModelForCausalLM` on CPU to read two tensors, and `config.json` must be on disk anyway.
  - **Output also carries `vocab_size` / `log_vocab_size` and per-sample `entropy_norm_*`** — `entropy/log(V)` cannot be recovered from the JSON without V.
  - **`rms_norm_eps` is read from the checkpoint's `config.json`, never defaulted** (Llama-3.1-8B 1e-5, Qwen2.5-7B 1e-6). Correctness hygiene only: real final-layer states have RMS 1–100, so `mean(x²) ≫ eps` and the two values are numerically indistinguishable (measured: entropy Δ=1.4e-06, identical to the correct-eps run). **It changes no existing or future conclusion — do not cite it as a data-quality fix.**
  - **Output MUST carry `question_idx` (per sample) plus `steer_alpha` / `steer_mode` / `rms_norm_eps` / `source_h5` (meta).** `question_idx` is the only key permitting per-question pairing with the signal JSON, and `steer_alpha` otherwise survives only in the filename — the same trap `extract_signal_json.py` has.
  - **Qwen backfill: exactly 7 H5 cells, band `[16,22)`.** `--h5_dir` is a GLOB, so a wrong cell count silently changes the batch; assert it with `--expect_n_cells 7`.
  - **Fail-closed**: empty decode; metric length ≠ decode steps; non-finite values; hidden-size disagreement (H5 / `norm.weight` / `lm_head.weight`); duplicate `question_idx`, or one not covering `0..n−1` (a GAP like `[0,1,3]` means a question is missing while `n` still looks plausible, silently misaligning any per-question pairing); missing final-layer pointer, or one outside the ACTUAL stored layer axis; truncated cell (`n_samples_done` missing, ≠ group count, or < `n_samples_planned`; `n_samples_planned` missing is itself fatal — it was checked only when present, i.e. fail-OPEN on the condition the check exists for); missing `steer_mode`, or a `steer_mode` contradicting `steer_alpha` (`.get(…, "none")` was fail-OPEN — a steered cell would have been recorded as unsteered). Each previously passed silently.
  - **`test_extract_entropy_confidence.py` is the guard suite** — synthetic H5 + two-tensor fake checkpoint, no GPU/server/real model, ~10s. Each guard is mutation-tested: the test builds the specific defect and asserts rejection, because a guard never shown to fire on a real defect is not a guard. It caught a live `NameError` on first run.
  - **Existing output is never overwritten** without `--allow_overwrite`; all cells are checked before any is written.
  - **`--verify_head` is an OPTIONAL first-port check, NOT a gate** — it loads the full model on CPU (~30 GB fp32 for 7B) to compare hand-computed logits against the model's native `final_norm + lm_head`. Run it once per new model, then drop it.
- **The Qwen logit family will have 7 cells against the RSN family's 11, so a flat dose curve has THREE readings, not two.** Besides replicated / attenuated / different / null, a fifth verdict is **`under-sampled / not decidable`**: No-CoT covers only `{−8,0,+6,+8,+12}`, and absence of monotonicity across five points can be sampling density rather than a null. Never plot the two families as equally sampled, and never promote an under-sampled non-result to a null.
- **Two DIFFERENT `n_risk` definitions coexist in §5 and the values legitimately differ.** `vigor_slope` needs only `commit_step ≥ 20` (one-sided; α=−8 → 25); the `s_pre/s_post` and `p_t` tables need ≥20 steps on BOTH sides of commit (α=−8 → 19). Always cite `n_risk` together with its table.
<!-- Why: the layerwise/null work ran AHEAD of the actual section-3.4 replication, so §5 can be mistaken for a completed multi-metric cross-model result when the logit family is not even extracted.
Evidence: qwen_signal/suite34_{nocot,cot}_RESULT.txt; suite34.py --part logit prints the blocked state.
Scope: any claim about how complete the Qwen replication is. -->
- **The §3.4 suite is the MAIN Qwen line; `hs_layerwise` / `hs_null_specificity` are a high-dose mechanism SUPPLEMENT that ran ahead of it.** `s_pre` is a local LEVEL of `s_t` and is **not** `vigor_slope`; a per-layer profile is **not** `p_t`. `suite34.py` covers entry/boundary_jump/`s_t` trajectory/`vigor_slope`/`p_t`/behaviour; **the logit family (entropy/top1/margin/info_gain/cum-entropy/rolling-var) was BLOCKED and is now UNBLOCKED (2026-08-26)** — `metrics_hs/` holds 7 cells and the analysis lives in `logit_family.py`, not in `suite34.py`, whose `--part logit` still reports the block rather than skipping. Its dose curve is **7 cells against the RSN family's 11** — never plot them as equally sampled.
<!-- Why: 7 of 11 Qwen cells cannot yield a leak-free slope at all, and a reader who does not see the coverage column will read the 4 surviving cells as a dose curve.
Evidence: suite34.py part_slope MIN_COVERAGE gate; qwen_signal/suite34_nocot_RESULT.txt.
Scope: every Qwen vigor_slope number. -->
- **Qwen `vigor_slope` is UNAVAILABLE for α≤+4 and this is a result, not a gap.** Qwen answers first (median commit = decode step 3), so at-risk coverage is 4–9% there and a fixed `[0,20)` slope would measure post-commit release for ~94% of samples — the §4.7 leakage problem, far worse than Llama's 23%. `suite34.py` **refuses** any cell under 50% coverage and prints the coverage instead. Readable cells only: +6 **−0.0211** [−0.0269,−0.0162], +8 −0.0063, +10/+12 CI contains 0 → **relaxation weakening with dose, not vigor**. Do NOT widen the window to fill the blanks.
- **Two further mechanism replications did land (No-CoT and CoT alike).** `boundary_jump/Z_prefill` is **−0.965…−1.003 across all 11 doses** (CoT +6: −0.960) — the same near-complete entry-pulse release Llama shows; and `p_t` carries a **signed event-locked dip exactly at commit** in every cell, deepening with dose (−0.55 at α=0 → −1.32 at +12).
<!-- Why: a first loop proxy read 80-86% and was pure false positive; the number is easy to regenerate wrongly.
Evidence: suite34.is_loop docstring; flagged sample is a real 'Final Final Final...' tail, unflagged ends in clean EOS.
Scope: Qwen loop diagnostics. -->
- **The loop diagnostic must be a DEGENERATE-TAIL test, not "any repeated n-gram".** A 12-char-n-gram-≥3× proxy read **80–86%** at α≤+4 — all false positives, firing on ordinary restatement of the question in generations ending at clean EOS. `is_loop()` instead requires the final 40-char block to recur ≥4× in the whole text, giving 8–14% at low α falling to **2.7%** at +8/+10. Verified by hand on flagged and unflagged samples before citing.
- **STEP 5 RESULT (frozen 2026-08-25): the nulls argue AGAINST redistribution, not for it.** RSN's scalar-compression residual is **LOWER** than every null family's (0.026 vs null medians .341 `diff_random` / .102 `ortho_gauss_same` / .227 `ortho_gauss_off`; percentile 0.0/0.0/20.0%, p .182/.182/.545), and **RSN's CV is LOWER too** (0.917 vs 2.373/3.943/2.936). So the RSN direction behaves **more** like a clean scalar channel than a generic sparse direction. Two consequences. **(a)** Step 4's "CV 0.92 is high" was written with no reference distribution — against one it is at the LOW end, so per-layer heterogeneity is not evidence of anything special. **(b)** Do NOT convert the extreme percentile into "RSN is special": the percentile only says RSN sits at an extreme, and here it points the un-hoped-for way. **The low-SNR caveat applies to only TWO of the three families.** Per-draw `‖v_68‖` medians: RSN 0.907, `diff_random` 0.270, `ortho_gauss_same` **0.752**, `ortho_gauss_off` 0.349. For `diff_random`/`ortho_off` (30–38% of RSN) fitting `k` to near-noise does inflate the residual. But **`ortho_gauss_same` carries 83% of RSN's response magnitude and still shows ~4× its residual (0.102 vs 0.026)**, and it is the **best-matched** null (NMD's own support, weights ⊥ role-diff, norm-matched) — so it cannot be explained away, and the verdict rests mainly on it. The wording stays "argues against" rather than "excludes" for the independent reason that N=10, p floor .182, and the three families are not independent replicates. Redistribution remains unshown and is the manifold pilot's question; step 5 makes **scalar gain compression the hypothesis to beat there**.
<!-- Why: re-projecting RSN-STEERED hidden states onto another direction bounds the READOUT, not the intervention; a percentile of 100% here reads as causal specificity if the distinction is not stated at the point of use.
Evidence: qwen_signal/hs_layerwise_RESULT.txt (BOUNDARY); hs_null_specificity.py module docstring.
Scope: every Qwen null-remask number; the same caveat governs the Llama §4.6 families. -->
- **A remask null is a READOUT-specificity control, NOT a causal control on the steering direction.** `run_null_remask_qwen25.sh` re-projects the seven collected H5 cells onto `diff_random` / `ortho_gauss_same` / `ortho_gauss_off`, which asks whether the per-layer response structure is peculiar to the RSN direction or would appear under any sparse direction. Those states were **generated under RSN steering**, so nothing here can support "the steering direction is causally specific" — that needs random/orthogonal **injection** and a re-collection, a different experiment. The **primary** statistic per direction is the scalar-compression residual above (CV and F are descriptive companions), and its DIRECTION must be reported, not just a percentile: RSN residual **higher** than the nulls means it departs from pure rescaling more than a generic direction does — word it as a **readout-level departure from pure rescaling**, never as geometric redistribution, which needs the manifold analysis (step 6) and is not substituted by a high residual here; RSN residual **lower** means it behaves *more* like a clean scalar channel, which argues **against** reading the per-layer table as redistribution — a real possible outcome that must not be re-labelled as a success. Two further 口径 the analyzer enforces: every direction is evaluated on the **same text-derived cohort rows** and through the **same `response_profile()` code path** (a null computed by a second implementation would confound "the null differs" with "the implementations differ"), and each direction builds its **own α=0 reference and uses its own `‖m_l‖²` read from its mask file**, so a direction with a larger raw projection scale cannot win on scale alone. Each family is N=10, so the two-sided p floor is 0.18 and the three families share the same hidden states, cohort and reference convention — they are **exploratory orderings, not independent replicates**, and "all three agree" is not a low-probability argument.
- **`NMD_VERIFY` is the load-bearing step of the null launcher, not a formality.** It rebuilds the NMD mask from the on-server role-diff files and requires it byte-identical to the frozen `nmd_0.5_16_22_7B.npy` (md5 `59f5695533045b9be34a2510946890e5`). If it differs, the null draws are constructed from a different role-diff basis than the direction the H5 were actually steered with, and every percentile below is meaningless. The launcher also refuses to proceed unless the H5 dir holds exactly 7 cells, because `extract_signal_json_remask.py` **globs** that directory — a wrong count silently gives the null a different cell set than the RSN result.
- **Mixed reference scales invalidate a response ratio.** Comparing a decode change against an entry change requires ONE reference: `Z_prefill` built from α=0 prefill μ/σ and `Z_pre` built from α=0 decode μ/σ are different units, and their ratio means nothing. Either put both in raw projection units, or share the prefill reference the way `phase1_gain.py` does.

- **Sanity script**: `sanity_mask_indexing.py` confirms saved mask non-zero rows + their decoder-layer alignment on the server. **Run it before any change to layer-indexing code** — this prevents repeating the offset bug fixed on 2026-05-30.

## Capitulation / Pressure experiments

**Dropped from the main line (2026-06).** Capitulation was moved out of the wanting/dopamine narrative: it measures PFC-level goal-maintenance under pressure (an indirect DA downstream), not tonic-DA wanting; the "capitulate vs. hold" readout is bidirectionally interpretable; it conflates with instruction-tuning sycophancy; and steering only works on Llama (Qwen≈0%, Mistral reverses). See `AdaDopamine.md` for the rationale. Code/script are kept for reference but not part of the active suite.

`get_answer_capitulation.py` runs the two-round protocol (Round 1: original answer; Round 2: pressure prompt + optional RSN steering); Round 2 reads a single-token argmax via `regenerate_logits` (no generation). Only `run_capitulation.sh` (own answer + soft pressure, all three models) is kept — the three deleted variants (gold-R1, authority-challenge, authority-own) are recoverable by setting `--gold_r1` / `--pressure` on it.

Key CLI flags: `--pressure` switches to the authority-challenge prompt; `--gold_r1` uses the gold label as Round 1; `--configs` encodes `{alpha}-{layer_start}-{layer_end}` triplets (e.g. `0-11-20 4-11-20 neg4-11-20`). Output lands in `${BASE_DIR}/mmlupro/${MODEL}/answer_cap_mmlupro/cap_{alpha}/`.

Analysis: `gsm8k/analyze_cap_stratified.py` stratified capitulation rates by difficulty and task category. **Not on disk (verified 2026-09-03), and there is no `gsm8k/` directory in this repo** — the line is dropped from the main line, so treat this as a historical pointer, not a runnable script.

## Behavioral-economics / wanting-proxy suite

A family of entry-points that operationalize "wanting" (incentive salience) as **overt decisions** rather than answer accuracy. The shared hypothesis: α=+4 (expert direction) raises wanting → higher bets, longer deliberation, harder-task choice, more reward-seeking; α=−4 (non-expert) lowers it. All use the same `--configs {alpha}-{ls}-{le}` steering convention (e.g. `"0-11-20 4-11-20 neg4-11-20"`) and the `nmd` mask, so they plug into the same hook surface as the GSM8K work. **`AdaDopamine.md` §4 is the design+results doc for this whole suite** — each experiment number ⑤–⑩ there has the neuroscience mapping, prompt, prediction, and (where run) the result table; the `❌ Dropped` ones list exactly why a paradigm is incompatible with inference-time injection (e.g. phasic-DA / RPE needs synaptic plasticity). Read the matching §4.x before changing any of these.

> **Strongest current result (per `AdaDopamine.md` §4.6): Confidence Betting** — α=+4 raises mean_bet 52–67% and bet10-rate to ~50% while **accuracy is unchanged** across GPQA (n=646) and MMLU (n=14,042). This is the cleanest wanting–knowing dissociation in the project (non-linguistic choice, large sample, knowing held fixed). When discussing publishability or picking a headline experiment, this + the GSM8K/MATH commitment-dynamics analysis (`AdaDopamine_gsm8k.md` §2/§3) is the most journal-ready pairing; the main open gap is single-model (Llama-only) — betting should be extended to Qwen3/Mistral (pipeline already exists via the cross-model tables in `AdaDopamine.md` §3.0).

- `get_answer_gpqa_bet.py` (`run_gpqa_bet.sh`, also drives `run_mmlu_bet.sh`) — **confidence betting**: model bets 0/2/5/10 points before answering. Mean bet / fraction bet=0 = wanting proxy. Out: `${BASE_DIR}/${MODEL}/gpqa_bet/`. **`run_gpqa_bet.sh` was extended (2026-06-16) to a full −8→+8 scan** (CONFIGS holds the 8 steered cells; orig=α=0 via the orig branch) — betting's readout is mean_bet, a SATURATING-MONOTONE quantity (no overload collapse; wanting rises until the bet=10 ceiling), so the scan confirms monotonicity + locates saturation, giving a second "needs-engagement → positive-α" curve for the motivation-knob argument. (**This originally read "alongside Bandit" — that pairing is withdrawn as of 2026-08-05**: the old Bandit +2 peak came from the voided pre-fix data, and the rebuilt pv6 Bandit shows no peak at all over −4/0/+4, only +α damage to post-discovery persistence. See `AdaDopamine.md` §3.2.5 / §3.4.) **This is the deliberate exception to the "betting no alpha-scan" stance below** — that stance was about *inverted-U peak-finding* (betting is single-step, can't show right-arm overload); a monotone-saturation scan serves a different purpose. GPQA (n=646, already chat) carries the scan; MMLU stays ±4 (large-n dissociation, no curve).
  - **CROSS-MODEL: Qwen2.5-7B-Instruct (2026-07-28), layers 16–21 (`CONFIGS` exclusive end `16-22`, mask `nmd_0.5_16_22_7B.npy`) — NOT Llama's 11–20.** Launchers `run_gpqa_bet_qwen25.sh` / `run_mmlu_bet_qwen25.sh`; `run_gpqa_bet_qwen25_raw.sh` was a `--save_all_raw` diagnostic for +6/+8 (qualitative only — it re-samples at temperature=1.0, so **never cite its paired counts**; the main sweep is the source). Layer choice = onset of the layer-wise Expert/Non-Expert Pearson descent, same criterion as Llama, corroborated by the MMLU-E scan's best cell `4[16,21]`. **Result: the dose-response replicates on the −8…+4 band** (GPQA n=646, ρ=0.455, mean_bet 4.14→7.11) with accuracy unmoved (McNemar Holm p_adj=1.00 in ALL 8 cells, incl. +8 whose raw p=0.039 does not survive Holm) — so the wanting–knowing dissociation is not Llama-specific. **The two models break at OPPOSITE ends: Llama degrades on −α (band −4…+8), Qwen on +α (band −8…+4).** Frame as a model-specific *intervention failure boundary*, NOT "betting can't reach the right arm" — that was a Llama property, not a paradigm property. Qwen's +6 and +8 are DIFFERENT failures: **+6** = real behavioural collapse with intact format (bet5=99.5%, entropy 0.021, invalid 0.0 — a wanting readout degenerated to a constant, so it is NOT dissociation evidence); **+8** = prose overload (invalid 51.6%, commit 85.1%).
    - **AUTHORITATIVE SOURCE = `qwen2.5/bet/gpqa/`, which now holds the FIXED-PARSER re-run** (it landed as `gpqa2/` on 2026-07-29 and was then promoted over the old dir, so the `gpqa2/` name no longer exists). Verify before citing: the fixed data HAS the `acc_explicit_pct`/`n_explicit_answer`/`ans_fallback_rate` columns and `orig_rows` in its JSON, and reads +8 invalid 0.5155; the superseded old-parser run had neither column nor `orig_rows` (the `--skip_orig` baseline-loss bug) and read +8 invalid 0.7663. **The re-run MEASURED +8 invalid at 51.6%, not the 56.2% that was predicted** from decomposing the old 76.6% — so the parser recovered MORE than estimated (76.6→51.6, i.e. 25.0pp was parser artifact, not 21.2pp). Do not cite 56.2%. All other cells are stable across the two runs (mean_bet within ±0.2, invalid unchanged), confirming the fix is a no-op outside +8 and that `temperature=1.0` resampling noise is small at n=646.
    - **`acc_explicit_pct` is what makes +8 readable.** Micro accuracy at +8 drops to 30.5% (−3.4pp) but that is denominator contamination: only 526/646 replies carry an explicit answer. On in-format replies accuracy is **36.3% — the HIGHEST of all 9 cells**, i.e. +8 does not degrade knowing, it degrades *reply form*. Read `acc_explicit_pct` alongside micro whenever `commit_rate_pct < 100`.
  - **Re-run triage: "are the numbers wrong?" and "can the numbers be VERIFIED?" are separate questions (settled 2026-07-29).** A cell can be perfectly valid and still be un-citable because the stored artifact lacks what the reported statistic needs. Before concluding a sweep is finished, check the per-sample CSV header for `sample_idx` (paired tests need it), `acc_explicit_pct`/`ans_fallback_rate` (needed to read any cell with `commit_rate_pct < 100`), and `orig_rows` in the JSON (the paired baseline; lost by `--skip_orig` before the fix). All three are schema, not results — a re-run that adds them will NOT change the conclusions, and expecting it to is a misreading.
  - **Betting is NOT byte-reproducible and this was deliberately NOT fixed.** `get_answer_gpqa_bet.py` runs `temperature=1.0` with **no `manual_seed` anywhere** (unlike the bandit/CGT family, which seed per `(run, round)`). Every re-run resamples: trends replicate, exact values do not. Consequence for doc maintenance — after ANY betting re-run, `AdaDopamine.md` §3.1.1 / §3.1.2 tables must be updated to the new values; a small delta is the expected behaviour, not a regression to investigate. Adding a `--seed` is a small change (it would not disturb existing callers if defaulted to `None`) and is the obvious next step if betting goes into a paper, but as of 2026-07-29 it is **unresolved** — do not assume a stored betting number can be regenerated.
  - **Llama GPQA re-run RESULT (`static_0729`, 2026-07-29): the paired statistics are now recomputable, and the dissociation holds across the whole sweep.** mean_bet 5.20→7.78 at +4 (+2.58/question paired, Wilcoxon p_adj=2e−48), **accuracy McNemar Holm p_adj = 1.00 in ALL 8 cells**. Resampling noise is small (every cell's mean_bet within ±0.2 of `static_0616`), so the trends replicate exactly — cite the re-run, but do not treat the small deltas as a finding. Two structural facts survive: **the negative arm is NOT monotone** (−2/−4 strongly significant, −6 attenuates, −8 is n.s. at p_adj=0.066), so Llama's interpretable under-wanting effect sits at MODERATE doses only; and **`acc_explicit_pct` tracks micro to within 0.5–0.8pp at every α** (commit_rate 96–98%) — unlike Qwen's +8, Llama never breaks format, so both accuracy readings agree.
  - **Statistics convention (load-bearing — mixing these silently corrupts a row):** bet-fraction rates use **all-sample denominators** (every row sums to 100.0%), `mean_bet` uses **valid replies only**. Same questions across α ⇒ **paired tests**: McNemar for accuracy, Wilcoxon for bets — not χ²/MWU. On MMLU (n=14,042) report **Cliff's δ**, never bare p — at that n every real effect has an astronomically small p (Llama +4 Wilcoxon underflows to 0).
  - **MMLU RESULT (frozen 2026-07-29): dissociation holds on both models** — bets move, accuracy does not (McNemar p_adj: Llama 0.241/0.890, Qwen 0.336 both cells; invalid 0.0 everywhere). Llama +4 mean_bet 4.42→7.49 (Δ=+3.06/question, **δ=+0.585 — the largest effect in the whole betting suite**, 62.9% of questions changed their bet); Qwen +4 5.10→5.57 (Δ=+0.46, δ=+0.093, 10.3% changed).
  - **⚠ Do NOT read that gap as "Qwen's wanting effect is ~5× weaker" — compare bet DISTRIBUTIONS first.** Qwen's MMLU baseline is nearly degenerate: **97.9% of questions bet 5** (bet0/bet2 both 0%), so α has almost no headroom and mean_bet can barely move — a baseline ceiling artifact, not a weak intervention. The SAME model on GPQA, where its baseline is dispersed, moves 5.57→7.11 (Δ=+1.53). Llama's MMLU baseline spans bet2/5/10 (26.8/68.3/4.7%), which is why α has room. **`% of questions whose bet changed` is the headroom-robust readout; `mean_bet` alone is not comparable across models/tasks with different baseline dispersion.**
  - **Parser + schema (2026-07-28).** `BET_LEADING_RE = r"^\s*[:：]?\s*(\d+)"` — the optional leading colon recovers Qwen's `': 5\nAnswer: A'`. `parse_output` now returns `(bet, answer, answer_src)` with `answer_src ∈ {explicit, fallback, ""}`, and `summarise()` emits `acc_explicit_pct` / `n_explicit_answer` / `ans_fallback_rate` so accuracy can be read on in-format replies only (guards against fallback-recovered answers inflating a degraded cell). Rows carry `sample_idx` + `raw` (the latter under `--save_all_raw`). **§3.1 IS NOW FROZEN (2026-07-29) — all four cells re-run, do not re-run again.** Authoritative dirs: `llama3/gpqa/bet/static_0729`, `llama3/mmlu/bet/static0729`, `qwen2.5/bet/gpqa`, `qwen2.5/bet/mmlu` (note the Llama MMLU dir has NO underscore before the date). The re-runs closed a schema gap: the pre-2026-07-29 CSVs had **no `sample_idx`**, and §3.1's headline McNemar/Wilcoxon are PAIRED tests keyed on it, so those p-values could not be recomputed from stored data. The parser fix was a genuine no-op everywhere except Qwen GPQA +8 (Llama invalid 0.0000 in 8 of 9 cells; MMLU invalid 0.0 for both models). Generalizable lesson: **"the model was re-run" does NOT imply "all of that model's tasks were re-run"** — the Qwen re-run covered GPQA only, leaving Qwen MMLU on the old schema for a day. Check the header per FILE, not per model. Two fixed bugs: the new summary fields were missing from the CSV `fieldnames` (`ValueError` at the very END of a multi-hour run, after JSON and per-sample CSV were written), and `--skip_orig` loaded `orig_rows` but never carried them into the rewritten JSON, so the paired baseline was lost after one use. **Reproducibility boundary: `temperature=1.0` with no fixed seed — statistical trends reproduce, exact numbers do not byte-match.**
- `get_answer_gpqa_cot_delay.py` (`run_gpqa_cot_delay.sh`) — **delay discounting**: model decides when to commit (`Final Answer: X`); CoT token length before commit = delay tolerance. Out: `${BASE_DIR}/${MODEL}/gpqa_delay/`.
- `get_action_effort_choice.py` (`run_effort_choice.sh`) — **effort-based choice**: GSM8K (easy) vs. MATH (hard) free selection. Includes an all-layer α=±1 (`1-1-33`) condition alongside the best-layer ±4.
<!-- Why: the CLOSED verdict lived only in deep pv10c/pv11 bullets, so a reader skimming the suite could
reopen a line that five intervention classes already failed to move.
Evidence: AdaBandit.md §5 + Protocol Lineage; PV11 pre-registered termination rule.
Scope: the whole Bandit line. Reopening needs a NEW protocol, not a re-run. -->
- **⛔ THE ENTIRE BANDIT LINE (pv1–PV11) IS CLOSED — do not start new Bandit experiments.** Per the PV11 pre-registered termination rule, **five** intervention classes (Stage-1 α, choice history, Beta calculator, competitor cue, controlled evidence state) each raised the model's *recognition* of uncertainty without moving *acquisition*. Bandit's role in the paper is **boundary evidence for directed exploration** — α changes policy stance / commitment / decision sharpness, but is NOT a general exploration controller. Do not re-run, extend seeds, or iterate the prompt: with events this sparse that is chasing noise. Conclusions are frozen in `AdaBandit.md` §5 + the Protocol Lineage table. The detail below is a **historical record**, kept because later protocols reuse its seed banks, gate rules and analysis 口径.
- `get_answer_bandit.py` (`run_bandit_llama3.sh` / `run_bandit_qwen25.sh`) — **multi-armed bandit**: explore/exploit over `--num_runs`×`--num_rounds` (30×50; `--pilot` = 2 runs × 3 α, rounds stay 50). De-roled port of the EVOLvE/BanditBench boutique MAB (`~/Documents/Benchmark/EVOLvE`, Nie et al. ICML 2025): explore/exploit paragraph verbatim, persona stripped, own local `build_prompt` (NOT `utils.build_prompt`), **bare-string — `use_chat` is not implemented at all**, so steering sits on the bare distribution the NMD mask was extracted in.
  - **⚠️ ALL PRE-2026-07-28 BANDIT RESULTS ARE VOID — do not cite them, they cannot be salvaged offline (no raw text was stored).** Two design faults were found and fixed in `get_answer_bandit.py`: **(A) position leakage** — `shuffle_arms()` shuffled arm NAMES then zipped against the descending prob list, so the best arm sat at display position 1 for EVERY seed (verified seeds 0–29); **(B) permissive parser** — `parse_choice()` returned the first arm name appearing anywhere, so a reply that merely echoed the option list scored as a VALID choice of the BEST arm. Together they made OptFrac indistinguishable from a first-option bias. **The published inverted-U (plateau α=0…+6, peak +2, +8 collapse 0.515) rests on this data and is retired**; `AdaDopamine.md` §3.2 and any cross-task table listing Bandit's working point must be re-derived, not merely re-labelled. Post-fix: `shuffle_arms` shuffles name→prob and display order independently (`position_of_best(seed)` exposed for the counterbalance check); `parse_choice` is valid only on EXACTLY one match, returning `n_matched` so "said nothing" (0) and "restated the menu" (>1) separate offline; per-round `raws`/`valid_flags`/`n_matched` and `seed`/`best_position`/`arm_order` are now stored (**the `seed` field's presence is the pre/post-fix marker** — the 15 legacy dirs lack it).
  - **pv1–pv5 operational detail is RETIRED to `AdaBandit.md` §1.4 (the four inherited constraints + pv5's frozen verdict, moved there verbatim) and the Protocol Lineage table (results).** What survives here is only what a later protocol still depends on. **All pre-2026-07-28 results are VOID** (see the bullet above). The two design faults and their fixes live in `get_answer_bandit.py`'s `PROTOCOL_VERSION` comment. Four facts that later protocols inherit: (a) the **counterbalanced seed set `0 3 4 9 37`** (best arm at display positions 2,4,5,3,1, five distinct best names) — the earlier 2-seed pilot accidentally shared BOTH name and position, which is bad luck at n=2, **not** a `shuffle_arms` defect, so do not "fix" the shuffle; (b) `parse_choice_exact` is **strict on purpose** and its `n_matched` separates three failure signatures — `(invalid,0)` nothing recognisable, `(invalid,1)` **named an arm but wrapped it in prose**, `(invalid,>1)` restated the menu; (c) `fallback_rng` is a continuous stream seeded `1_000_000 + seed`, so cells with different invalid counts draw different fallback arms at the same round — reward draws stay paired but ITT `opt_frac` absorbs a little fallback luck (**unfixed**; stratify absolute OptFrac by `best_position`); (d) the resume key needs an `iface` segment or reusing one `ans_file` across changed interface flags silently returns the old row. **pv5's frozen verdict:** E-direct's failure is an **early-outcome-dependent greedy confirmation loop** — whichever arm earns the FIRST reward=1 becomes a self-reinforcing incumbent; given equalized trial counts the model tracks the empirical-best set correctly (adherence ≈0.93), so the gap is a missing uncertainty bonus, not integration failure. E-CoT's marginal value is **UNESTABLISHED**, and invalid rounds may **not** be offline-patched (a differing fallback pull changes every downstream state).
  - **pv6–pv11 protocol internals moved to `docs/BANDIT_PV6_PV11_ARCHIVE.md`
    (2026-09-02)** — ~140 lines of environment/gate/steering-scope/prompt-version
    detail for a CLOSED line, byte-identical to what stood here. The surviving
    summary: **pv6 Easy-bare PASSED the competence gate** and its B1 α=±4 showed
    **+4 damages PERSISTENCE, not discovery** (96.6% of extra switching lands on
    already-tried arms) while −4 is behaviourally null but distributionally active;
    **pv7 FAILED on a tie** (one-shot-zero lock-in) and **pv9 PASSED** all four
    rules, with α moving policy WORDING and COMMITMENT but never information
    seeking; **PV10 (BAI reframe) and PV11 are CLOSED**, and `analyze_bandit_pv10c`'s
    `next_sample_targets_competitor` = .774 is DEPRECATED as a CONSTRUCT FAILURE and
    must not be cited. Frozen wording for any pv7/pv9 gate pass: competence under a
    **structured, parser-assisted, cue-scaffolded interface with Policy-following
    constrained action — NOT native free generation**.
  - `get_answer_textbandit.py`/`run_textbandit.sh` are **RETIRED** (kept for reference): TextBandit was faithfully reproduced but its paper has a load-bearing design flaw — few-shot teaches 5 different best-arms, contradicting the fixed Machine-2-optimal structure. Raw failure record in `AdaDopamine_bp.md` only.
- `get_answer_cgt.py` (`run_cgt.sh`) — **CGT-SIMULTANEOUS = transparent-odds single-shot betting probe. STRICTLY NOT a Cambridge Gambling Task** (it drops the ascending/descending betting stage, CGT's defining manipulation per Rogers 1999 / CANTAB) and is **RETIRED from the main line** — kept as the confidence-confound control for §3.1 Betting (odds are stated, so a bet shift cannot be explained by "more confident"). Port of the Near-Optimal repo (`~/Downloads/Benchmark/Near-Optimal/cambridge_gambling_task`): 8 phases × 8 rounds, 8 box ratios, 5 bet tiers, simultaneous 10-choice, ×2 payoff. Reuses the bandit α-hook (`vc.regenerate`, bs=1, per-round rebuild). Out `${BASE_DIR}/${MODEL}/answer_cgt/`. Metric panel `RoleAnswer/analyze_cgt.py` (3 layers: WANTING / KNOWING [flat = dissociation, NOT null] / DIAGNOSTIC). **Full prompt-evolution history and result tables: `AdaDopamine.md` §3.3.**
  - **EIGHT prompt modes exist behind flags** (faithful / `--simple_prompt` / `--simple2` … `--simple5`); **always check `config.simple*` in a detail JSON before comparing numbers.** `--simple5` is the main line. faithful + `--simple_prompt` run BARE-STRING; **`--simple2` and later REQUIRE `--use_chat`** (CLI-guarded, mutually exclusive) — chat is a PREREQUISITE here, not a dilution risk: bare modes were qdm≈0.50 (near-random; rational play → ~1.0), chat lifted qdm@9:1 to 0.99.
  - **The settled result: betting-side α-NULL, but a clean `delib_tok` dose-response** (+α → longer pre-commit reasoning → MORE conservative bets). Framing matters — this is a **mediation** result, not "wanting reduces to confidence": transparent odds clamp the confidence mediator, so the wanting push has no expression channel. Pair it with CGT-seq (wanting fully expresses as delay aversion) → RSN moves an upstream wanting latent; whether it shows depends on the task offering a channel.
  - **Four dead ends worth not repeating** (each cost a full sweep): the 0–9 `Choice:` GRID was a noisy intent→digit translation layer (simple3 removed it); stronger reward wording ruled OUT "under-engaged wanting" as the null's cause (simple3b); an `Answer: ` anchor placing injection ON the decision STILL gave α-null (simple3c — so the null is not injection-locality); and **dropping the reasoning instruction collapses the probe** (simple4 kept "EXACTLY one line" but lost "think briefly" → 98% zero-reasoning replies, median 28 chars, nothing for prefill steering to land in). simple5 = simple3c's skeleton + only the genuine fixes, reasoning instruction KEPT.
  - **`--inject_turn_len 4` (tail-4) is OVER-STEERING, not signal-boosting** — it broke qdm (ρ=−0.81) and degraded `delib_tok` into U-shaped noise. **tail=1 is the validated strength; widen the ANCHOR (content reframe), never the tail.**
- **`get_answer_cgt_seq.py` (`run_cgt_seq.sh`) — CGT-SEQUENTIAL (delay aversion).** The faithful CGT: the bet is revealed ONE TIER AT A TIME (`--presentation asc` 5→25→50→75→95 or `desc` 95→75→50→25→5) and the model Accepts (lock, round ends) or Waits. **PRIMARY readout = ACCEPT STEP, not bet amount** (bet amount is presentation-confounded: desc anchors high, asc low); `delay_aversion_index = mean_bet_desc − mean_bet_asc`, so BOTH conditions must be run (separate `--ans_file`). Reuses simple5's prompt skeleton + the `get_answer_cgt` α-hook. **Results, tables and interpretation live in `AdaDopamine.md` §3.3 — cite there, not here.** Operational facts that are NOT recoverable from that doc:
  - **Generator fixes (2026-08-19), all schema/identity — no stored numbers change.** (a) **`forced_lock` is now recorded**: "Wait at every tier" and "Accept at the last tier" both yield `accept_step == n_tiers`, so without the flag they are indistinguishable and the analyzer's `forced_lock_rate` read a constant 0 — legacy data cannot be repaired offline. (b) The generator's `early_stop_rate` (step==1) was **renamed `accept_step1_rate`** to match the analyzer field of that name; the analyzer's own `early_stop_rate` is a DIFFERENT quantity (`step <= n_tiers//2`) and is unchanged — do not "unify" them, the doc tables key on the analyzer口径. (c) `prompt_template` metadata now calls `build_seq_system_prompt()` instead of duplicating its selector: the old copy tested `== "v2"` while the builder tests `in ("v2","v3","v4")`, so **every stored v3/v4 run records the V1 template — that field is wrong on pre-fix v3/v4 data**, and the prompt-self-attestation convention silently failed there. (d) The resume key gained an `iface` segment (prompt_ver/anchor/chat/tail/model/mask/num_runs) + CSV column; it was previously only `(alpha, start, end, presentation)`, so reusing one `--ans_file` across a different prompt or MODEL returned the old row — the same failure pv6 fixed with its own `iface`. Rows lacking the column are reconstructed as the legacy interface, so existing sweeps still resume.
  - **Qwen2.5 port (2026-08-19). `--check` PASSED on the server; the validity pilot is PENDING.** `run_cgt_seq_qwen25.sh {--check|--pilot_asc|--pilot_desc|--asc|--desc}` + `check_cgt_seq_qwen.py`. Deliberately a SEPARATE launcher — `run_cgt_seq.sh`'s v1/v2b/v3/v4 x asc/desc mode matrix is Llama-specific and its results are frozen. Qwen uses **layers 16-21** (`--configs` exclusive end `16-22`, mask `nmd_0.5_16_22_7B.npy`), same band as the betting port; Llama's 11-20 does NOT transfer. `--check` is a **read-only pre-flight** that must pass before any sweep: it reads out (1) the chat template and a double-BOS guard, (2) the ACTUAL final prompt token at both the colour step (`Color: ` anchor) and the bet step (**no anchor** by design), since prefill-only steering lands there and the token differs from Llama's, (3) mask rows vs `decoder_layer_range` alignment, (4) observed `steering_fires` against `L*B*t` plus an assertion that alpha=0 fires **0**. **alpha=0 here is NOT the pv6/PV10 'no hook at all' case** — `run_episode`'s single `gen` closure always calls `vc.regenerate(diff_matrices=diff_mtx)` and the driver builds `diff_mtx = list(raw_mask * alpha)`, so at alpha=0 it passes a REAL all-zero matrix (and `regenerate` rejects `None`, llms.py:880, so there is no no-diff branch). Hooks DO register and the zero add DOES execute; fires read 0 only because `_layer_is_steered` (llms.py:947) is False on an all-zero row. The check therefore exercises `regenerate` with a zeroed mask — checking `vc.generate` would verify a path the experiment never takes. **Do not 'unify' the driver with pv6**: that would change the Llama protocol whose results are frozen. CGT-seq REQUIRES `--use_chat` (bare gives qdm~0.50), so the injection site is whatever that template ends with — do not assume Llama's. The pilot gate (alpha=0/+-4, 5 runs, both presentations) is in the script header: low alpha=0 invalid, qdm clearly >0.50, `mean_accept_step` moving with alpha, near-zero empty/colour-confusion. **Expect a Qwen-specific usable band** — betting broke at OPPOSITE ends on the two models (Llama on -a, Qwen on +a), so an asymmetric band is the prior, not a bug.
  - **Qwen `--check` MEASURED VALUES (2026-08-19, Qwen2.5-7B-Instruct).** Recorded because they cost GPU time and are the facts a Llama-trained intuition gets wrong. Tokenizer `Qwen2Tokenizer`, vocab 151665, **`bos_token_id` is `None`** (so Llama's double-BOS hazard cannot occur here). Mask `(28, 3584)`, model has **28 decoder layers**, band `16-22` → `decoder_layer_range` gives `[15..20]`, **L=6** (Llama's `11-20` is L=9) — non-zero mask rows matched the hooked band exactly. **Injection tokens differ by step**: colour step ends on `id=220` (`' '`, the `Color: ` anchor — the same decision-bottleneck token PV7/PV10 anchor on), bet step ends on `id=198` (`'\n'`, the assistant header, because the bet step deliberately has NO anchor). `steering_fires` read **6 = L·B·t** at both steps under α=4, and **0** on the α=0 all-zero-diff path. Generations were well-formed (`'Blue'` / `'Accept'`). **Do not read `'\n'` as a weak injection site a priori** — it still carries the full 417-token context's hidden state; whether it is weak is answered by the pilot's `accept_step` response, and if that is flat the first hypothesis is the anchor, NOT a bigger α or a wider tail (tail-widening is over-steering, see the Injection-mechanics bullet).
  - Batch wrappers: `run_cgt_seq_v2_verify.sh` (all 4 v2 verify cells) / `run_cgt_seq_v2b_full.sh` (v2b asc+desc full sweep). The betting family also has `run_gpqa_bet_running.sh` / `run_mmlu_bet_running.sh` (running-score variants, chat).
  - **Prompt versions `--prompt_ver {v1,v2,v3,v4,v5}` + `--anchor {default,answer,none}`. v4 = PAPER MAIN LINE (Llama); v5 = the Qwen line (see the Active-status block)** (direction-only hint, no "Wait" word); v2b = v2+no-anchor, the earlier main line; v3 = per-tier hint restating "Wait", kept only as the impulsivity-vs-rule-ignorance control (its extra "Wait" token inflates −α color-stage confusion). v1 ≡ commit `e55b132`, byte-identical.
  - **`v5` = the Qwen label/position de-confounding prompt (2026-08-19). Its α=0 gate PASSED and the −2/0/+2 sweep is DONE and frozen — see the Active-status block for the citable band and its limits; do not add doses.** v4 fixed the colour order in FOUR places — the scene sentence, the mechanics sentence, the worked example (`9 blue, 1 red → blue 90%`), and the output list + question wording — with no shuffle anywhere, so label and display position were COMPLETELY COLLINEAR and a blue preference could not be attributed. v5 changes exactly two things: the option order is **balanced per round** and used identically in the output list and the question wording, and the worked example is **neutralised** (`7 of the 10 chests are one colour and 3 are the other → 70%/30%`) rather than swapped (which just moves the anchor to red) or deleted (which removes the share=probability rule the gate is testing). **Balance is STRICT by construction, not in expectation**: `make_order_sequence` permutes a half/half block per phase, so every run is exactly 32/32 and each `first_option × major_color` cell stays populated — random assignment drifts and can starve a stratum at n=64. It uses its OWN rng seeded off `seed`, so `make_box_sequence(seed)` is byte-identical to v4 and the two versions face the same chests. Records carry `first_option`/`chose_first`, so attribution is a table read (preference following the colour = label prior; following position = position prior). `ORDER_VERSION` enters the resume-key `iface` for v5 only, so stored v1–v4 keys are unchanged. **v5 metadata stores `prompt_template_first_blue` + `prompt_template_first_red` + `order_version`, NOT a single `prompt_template`** — the prompt changes every round, so one string would misattest the run. **One known unbalanced cue remains**: the scene sentence still names blue first (it introduces the colour words and is not an option presentation) — check it first if the attribution table shows unexplained residual bias.
  - **`diag_cgt_seq_color_logits.py` = the READ-ONLY colour-logit sidecar** (α=0 diagnostic, not an experiment). Scores Blue/Red candidate log-probs at each colour step to separate "evidence is read but generation is constant" from "candidate preference is internal". It imports every prompt builder, constant and parser from the frozen driver rather than copying them, refuses to overwrite its `--out`, and fails closed on prompt-ID mismatch. **It RE-GENERATES its trajectory at temperature=1.0 — it does NOT replay stored pilot states**; only the box sequence is shared with a stored run of the same seed, so pairing is `(logits, generation)` WITHIN the diagnostic run, never diagnostic-vs-pilot row by row. Primary contrast is `logP("Blue") − logP("Red")` (the exact casing the prompt demands); all four surface forms are stored plus a `logsumexp` case-insensitive sensitivity check. **A per-colour `max()` over casings is wrong and must not be reintroduced** — taken independently per colour it can compare `blue` against `Red` and flip the margin's sign. Candidates MUST be concatenated at the ID level: on the real Qwen tokenizer, string-level concat merges `220 + "Blue"` into `" Blue"`, shortening the sequence by one token and silently sliding the injection site off the anchor.
  - **α=0 here is NOT the pv6/PV10 "no hook at all" case.** The driver builds `diff_mtx = list(raw_mask * alpha)` **unconditionally**, so α=0 passes a REAL all-zero matrix; `regenerate` rejects `None` (llms.py:880) because CGT-seq has no no-diff branch. Hooks still register and the zero add still executes — `steering_fires` reads 0 only because `_layer_is_steered` is False on an all-zero row. Any new CGT-seq tool must do the same, or it both crashes and runs on a different execution path than the driver.
  - **The bet step must have NO anchor** — opposite to simultaneous simple5. `--anchor answer` (v2a) COLLAPSES the task (α=0 accept_step≈1.1: the model answers `Answer: Accept` at the first tier, leaving no Wait decision for α to move) and emits empty payloads at ±6. Only raw inspection catches this; the summary metrics look plausible. The colour step keeps `Color: `.
  - **⚠️ CONFIRMED REGRESSION in commit `08567b9` — never re-add.** It introduced a `Decision: ` anchor + `max_new_tokens=8` + a strict `^\s*(?:Decision:\s*)?(Accept|Wait)` parser as a code-review "improvement" that was never validated. It forces `valid=False` on any bet reply not starting exactly with Accept/Wait and truncates generation, collapsing `invalid_rate` from a stable 0.00 to 0.12–0.36 **even at α=0** (deterministic, not sampling noise). `forced_lock` arrived in the same commit. The validated build is `e55b132`: no bet anchor, full `max_new_tokens=64`, loose full-text parse. **A server checkout of `08567b9` reproduces the regression even though the PROMPT greps byte-identical** — the regression is in the anchor/cap/parser, so diffing prompts will not find it.
  - **Usable α band = −4…+6**; ±8 and asc −6 are over-steer (excluded from fits). The two failure modes are qualitatively different — **−8 = 垮/沉默** (empty generations, colour stage emits nothing) vs **+8 = 散/抢答** (non-empty but malformed). `final_score` shows an inverted-U but std ≫ mean at n=20 → report qualitatively only, never as a significant bidirectional peak.
  - **`final_score = sum(phase_end_scores)`** (the prompt says "sum across all phases"); `mean_phase_score` is kept separately. An earlier build used `mean` — check which one a stored JSON used before comparing.
- **Injection mechanics (`llms.py`):** `vc.regenerate(prefill_only=True)` injects `α×mask` only at the prompt's LAST token during prefill (`hs[:, -1, :] += diff`), decode untouched. New `prefill_tail_len` param (default 1 = byte-identical to all existing callers — GSM8K/bandit/betting unaffected) lets `--inject_turn` widen to the last N tokens (`hs[:, -n:, :]`, clamped to L-1). The `Answer: ` anchor (simple3c/4/5) and the `Color: ` anchor (cgt_sequential colour step) are the preferred fix over tail-widening — they reframe the *content* at the last token rather than smearing α over control tokens. (NOTE: cgt_sequential's bet step deliberately uses NO anchor — the `Decision: ` anchor added in `08567b9` is a regression, see the cgt_sequential bullet above.) **Lesson (2026-06-17): tail-widening is OVER-STEERING, not signal-boosting** — simple5's `--inject_turn_len 4` (4× effective α onto the decision-anchor segment, header-safe) broke qdm (knowing collapsed, ρ=−0.81) and degraded the delib_tok signal into U-shaped noise. **tail=1 is the validated strength; only the anchor (content reframe), not tail-widening, is safe.**
- **Run matrix / mechanics (all modes):** verify flags α=0/±6 5 runs (`--verify`/`verify2`/`verify3c`/`verify4` → matching `_verify` dirs); full flags = −8→+8 9-α × 20 runs (`--simple2`/`--simple3`/`--simple3c`/`--simple4` → `answer_cgt_<mode>/`). max_new_tokens=200 (chat modes), `--save_all_raw` ON. Reuses bandit α-hook (bs=1, per-round rebuild); runs are independent repeats (seed=run_idx); server `summary_*.csv` is throwaway, stats recomputed locally from each run's `records` (`valid==True` filter). `--pilot` = α=0/2-run near-random check → `answer_cgt_pilot/`. **`I_LC` = loss-chasing** (mean bet_frac INCREASE after a loss; >0=chasing) — only behaviourally valid once simple4 adds explicit feedback.
- **`get_answer_igt.py` (`run_igt.sh`) — IGT (Iowa Gambling Task, Bechara 1994).** 100 trials, 4 decks, trial-by-trial feedback so the model LEARNS to avoid disadvantageous decks — the long-term-learning axis CGT/Bandit lack. Deck schedule = classic Bechara (= Near-Optimal `igt_configs.py`; **100 trials, not 80** — an older note said 80). A/B = +100 / net −250 per 10 (B = RARE huge 1250 penalty, the deck impulsive subjects chase); C/D = +50 / net +250. Multi-turn chat reusing CGT-seq's `build_chat_messages2` + α-hook; **history accumulates across all 100 trials, NEVER reset** (one continuous learning curve; ~12K tokens by trial 100). Deck order rotates per run; records store the DE-ROTATED true deck. **Format: NOT the repo XML** (that caused CGT-faithful's 85% off-task) — CGT-simple5 style NL reasoning + a final `Chest: N`, parser takes the LAST match. **Full v1→v6 prompt chain + result tables: `AdaDopamine.md` §IGT.** Operational essentials:
  - **`--prompt_ver v6b` (invitation, DEFAULT) = the NATURAL/unforced main line; `v4` (command "First reason…") = an EXTERNAL-FORCE control.** Settled stance: **v6b is the result, v4 EXPLAINS it, v4 does NOT falsify it** — do not re-frame IGT as a flat negative. v6b gives a clean **+2 inverted-U**; once v4 force-supplies the deliberation, value/risk/RPE readouts all return to n.s., which LOCALISES the +α overshoot to an **engagement** drop rather than a change in value computation.
  - **RETIRED v4-only interpretation (provenance, 2026-06-25):** before v6b existed, IGT was labelled a channel-mismatch / boundary condition because v4's forced-reasoning results were weak and unstable. v6b superseded that verdict: IGT is a working-point experiment, and v4 now explains why externally fixing the deliberation span suppresses the visible effect. The surviving limitation is narrower: inference-time α does not implement synaptic plasticity or establish a biological phasic-RPE mechanism, so do not translate the IGT result into such a claim.
  - **The reasoning cue is load-bearing.** v3 (terminal `<Chest:N>` tag) and v5 (no cue) both COLLAPSED reasoning into bare mechanical round-robin — the CGT-simple4 failure reproduced twice. Formula: **`+α visible behavior = exploration drive × engagement`**; `delib_tok` is the one readout stable across v4/v6b, so it is the main readout.
  - **`net_score` alone MISLEADS — always read the EXPLOIT-vs-CYCLE diagnostics** (`last50_entropy` / `max_run_len`). −α can degenerate into mechanical round-robin (DACBDACB), which raises net by ARITHMETIC (cycling all 4 puts ~50% on C/D), not learning. `net_score` also confounds risk preference with LEARNING capacity — use the learning-independent layers for DA claims. `lose_shift_rate`/`post_bigloss_switch` SATURATE ~0.95 → read `ws_ls_asymmetry`, not absolutes.
  - **Metric hierarchy (moved from the curated results doc, 2026-08-20):** first validate learning with `net_block1..5`, `learn_slope=block5−block1`, `last50_net`, entropy/run length and history-grounded text; then read local punishment with `return_to_B_{3,5}_rate` and `big_penalty_exposure`; use `ws_ls_asymmetry=win_stay−lose_shift` rather than either saturated absolute rate. `p_adv=P(C+D)` and `net_score=2p_adv−1` are the SAME outcome on different scales; `high_reward_deck_pull=p_disadv/p_adv` is another transform, not independent evidence. Win/loss is defined by single-trial `payoff=reward−penalty` and only valid→valid consecutive pairs enter transition metrics. `invalid_rate`, `parse_fail_rate` and `premature_stop_rate` are task-control diagnostics, never risk-preference evidence.
  - Run `bash run_igt.sh --verify` FIRST (**delete the server dir first — CSV resume otherwise skips every cell**), then `--full`.
  - **Why the v-chain exists (don't re-derive):** the +4 premature-stop ("the game has ended at Round N") was SEEDED BY THE FORMAT DIRECTIVE's END/FINAL/CONCLUDE words, which +α amplifies (118/500 such declarations at +4 vs 0/500 at −4) — v3+ drop those words. `v6a` ("Think step by step") is the stable alternative to v6b; v4's one-sentence command COMPRESSES reasoning (v6a `delib_tok` is 3–4× v4's). **⚠️ CROSS-TASK TABLE BROKEN:** the "Bandit +2 / IGT +2 / GSM8K −6" working-point drift is void — Bandit's +2 came from position-leaked data (see the Bandit entry), leaving only IGT +2 vs GSM8K −6. Do not cite the three-task table; re-derive Bandit from a fixed-code run first. The surviving framing: each task's natural optimum α tracks its wanting demand.
  - **Qwen2.5-7B v6b CROSS-MODEL RESULT (2026-08-20; layers 16–21, `CONFIGS=16-22`, N=20/cell): baseline competence PASS, reasoning-channel replication, descriptive learning trend not Holm-confirmed.** The negative and positive arms were run separately with their own α=0 on the same card as that arm. Both α=0 cells pass all six frozen baseline rules. They may be pooled only for a DESCRIPTIVE 40-run baseline (`net_score=.147`, `p_adv=.573`); because both copies use seeds 0–19, the batch comparison is paired by seed (pos−neg Δnet=+.0996, Wilcoxon p=.207), so write "no systematic batch difference detected", never "proved no GPU effect". Every α effect remains paired against ITS OWN ARM'S α=0. The full panel MUST retain every α cell and non-significant metric: significance limits claim strength, not data visibility. Learning shows local elevations at −4 (`net_score=.248`) and +6 (`.241`); the strongest paired contrast is −4 vs neg-0, Δnet=+.151, raw p=.031 → Holm p_adj=.250, while the other adjusted comparisons are n.s. `p_adv=(net_score+1)/2`, so it is the SAME outcome on another scale and must not be counted independently. In contrast, explicit deliberation collapses: positive-arm `delib_tok` 24.66/4.96/0.00/2.05 at α=0/+2/+4/+6; +4 is 2,000/2,000 zero-deliberation trials. +8 is excluded (`invalid=.876`; fallback makes its apparent block-5 learning uninterpretable). Frozen wording: **Qwen can learn IGT at baseline; α robustly compresses expressed deliberation, while deck learning shows local improvements but no stable Holm-confirmed dose effect.** This is not a pointwise replication of Llama's +2 performance peak.
  - **IGT paired-statistics correction (2026-08-20):** `seed=run_idx` gives matched deck order across α, so `analyze_igt.py` now uses paired Wilcoxon, not MWU. The earlier hand-written "exact" test was ALSO wrong: it sign-flipped raw magnitudes `|d|`, whereas the signed-rank null flips RANKS. SciPy is authoritative (−4 vs neg-0 p=.0312; exact ranked check .0305; retired value .079). Preserve alignment by run index and drop a pair only when EITHER side is NaN; never filter the two cells independently. Holm family = the eight non-zero Qwen α cells. The analyzer's KW/Spearman remain descriptive across-cell summaries and are unaffected by the paired-test change.
  - **Cross-task behavioral framing (2026-08-20):** RETIRE "one α-wanting axis" and the IGT `+2` / GSM8K `−6` contrast as an established optimal-α law. The defensible cross-task observation is narrower: α repeatedly moves engagement, commitment timing, policy sharpness or strategy expression, but construct-valid behavior and outcomes are task/model/interface dependent. Betting and Llama CGT are positive bridges; Qwen CGT is a narrow-window partial replication; IGT peaks are model-internal descriptive patterns; Bandit is boundary evidence; HaluEval is challenge-threshold evidence; GSM8K is a reasoning-task control, not an independent wanting assay. A unified latent wanting axis remains a HYPOTHESIS, not a result.
- `get_answer_reversal.py` (`run_reversal.sh`) — **reversal learning**: reward contingency flips at `--phase_switch 20` of `--num_rounds 40` (`--reward_prob 0.8`); measures adaptation speed after the switch.
- `get_answer_crt.py` (`run_crt_llama3.sh`) — **Cognitive Reflection Test**: intuitive-vs-reflective answer rate under steering. Out: `answer_crt`.
- `get_answer_trait.py` (`run_trait.sh`) — personality/trait self-report under steering.
- `get_action_regenerate_gsm8k.py` — GSM8K **self-report scalars (0–9)**: `run_action_confidence_gsm8k.sh` (`--suite confidence` → pre-answer confidence) and `run_action_willingness_gsm8k.sh` (default action suite → reasoning willingness). Note `alpha=0` runs use `get_logits` (no mask loaded); other α load `diff_mtx * alpha`. **Both runners now sweep the full −8→+8 dose at layers 11–20** (config list `neg8-11-20 … 8-11-20`; the old `±4 + ±1@1-33` set is kept commented), and both have been migrated to `WORK_DIR=/data1/paveen/Dopamine`. The 0–9 suites (`build_gsm8k_action_suite` / `build_gsm8k_confidence_suite` in `template.py`) are a **standalone prompt family** — question-only, logit-extracted over the "0".."9" tokens, no `####` / no answer generation — so they are NOT affected by the GSM8K answer-extraction symmetrization. Status: **manipulation-check only, superseded by Betting (`AdaDopamine.md` §4.6)** — Berridge's wanting is non-conscious, so oral self-report is theoretically the wrong proxy; the data confirm this (willingness −4 moves the *wrong* way, confidence tracks PFC not DA). Keep as negative/control evidence, not a core wanting claim.
- `get_answer_sciworld.py` (`run_sciworld.sh`) — ScienceWorld agentic tasks. **TASK-DIFFICULTY VERDICT: too hard for 8B to be a positive RSN assay — a paper-level fact, not a harness bug.** Wang et al. 2022 report an 11B T5 at 0.08/100 beaten by a 1.5M-param DRRN at 0.17; the benchmark punishes static reasoning. Our Llama3-8B α=0: **0 of 30 tasks ever solved**, median partial ~3/100, nearly every episode hits the 50-step cap. Baseline is floored ⇒ RSN has NO upward room. The only live signal is **+α triggering catastrophic −100 terminal actions** (0% at α=0/−4 → 9.3% at +4, all crashes on the FINAL step) — the same Yerkes-Dodson right arm as GSM8K 抢答 / CGT-seq early-accept, but a **single-sided floor-effect destruction signal**, not a dose-response. Keep as an appendix boundary point at most; do NOT pursue as "RSN improves agentic ability". **ALFWorld/NetHack share the same floor + OOM + capability-binds-wanting trap — do not substitute them.**
  - `--prompt_style {legacy,action_line,action_number}`; `parse_action` is **number-path-FIRST** for bare digits (a lone "7" must not go through substring match, which silently mapped `Action: 7`→"look around") and takes the FIRST match so a looped `Action: 7 Action: 7…` collapses to one decision.
  - **The "You are a science experiment agent." persona was REMOVED** from `legacy`/`action_line` (same de-roling as Bandit No-Role), so **`legacy` is NO LONGER byte-identical to the existing `sciworld/mdf_*` results — do not compare ACC across that boundary.**
  - **OOM on a 24G card:** bf16 8B ≈16GB leaves ~7GB; `regenerate` is per-step (KV cache not cumulative) so the peak is the single longest-prompt prefill (logits `[L×128k×fp32]` dominates), which rolling chat blows up. Knobs: `--obs_char_limit` (default 600), `--history_window` (default 8). `--save_trace` dumps per-step obs/raw/parsed action — without it the −100 crash actions are unrecoverable.

- `get_answer_regenerate_socialiqa.py` (`run_socialiqa.sh`, loader `data_socialiqa.py`) — **SocialIQA 3-choice (A/B/C) accuracy under steering.** Built 2026-06-25 as a *fully standalone* pipeline (does NOT import `template.py` or touch any existing loader/runner — own bare-string 3-choice prompt + `Answer: ` anchor + `vc.regenerate` generation + `parse_letter`). Reuses the existing Llama3 nmd mask 11-20; bare-string greedy (temp=0, max_new_tokens=8). **RESULT (α=0/+4/−4, validation n=1954): a KNOWING-AXIS NULL — acc 71.75/71.95/72.31% (−4>0>+4, spread 0.56pp ≪ binomial SE ~1pp = n.s.), invalid 0%, no pred-distribution drift.** Exactly the predicted outcome: SocialIQA's only observable is answer correctness (no independent wanting channel), so RSN α (a motivation knob) can't move it — same "knowing pinned" pattern as §1 MCQ / §2 MMLU. **Value = a negative/control data point for the wanting–knowing dissociation** (α moves GSM8K commitment / betting size / CGT accept-timing but NOT a social-commonsense MCQ's correctness), NOT a positive wanting experiment. Do not pursue further; if a non-accuracy readout is ever wanted, the only angle is an E-option abstention version (effort-willingness, §2.2 line). **Data source: `lighteval/siqa` (pure Parquet); the original `social_i_qa` / `allenai/social_i_qa` are loading-script datasets that newer `datasets` REJECTS ("Dataset scripts are no longer supported") — use the Parquet mirror.** Output `${BASE_DIR}/llama3/answer_socialiqa/mdf_{α}/socialiqa_8B_answers_11_20.json`; dataset JSON lives flat at `${BASE_DIR}/benchmark/socialiqa.json` (the server's `benchmark/` tree is FLAT single-files: `mmlu_all.json`, `gpqa_train.json`, … — NOT per-dataset subdirs).
- `get_answer_regenerate_halueval.py` (`run_halueval.sh`, loader `data_halueval.py`) — **HaluEval-QA = SIGNED-BIAS engagement probe, NOT a hallucination-accuracy test.** The loader expands N source rows into a `right_answer` item (gold=No) + a `hallucinated_answer` item (gold=Yes) → **600 BALANCED items**; that balance is what makes a SIGNED bias readable instead of one accuracy number. **Headline metric is `FNR` (credulity: hallucinations let pass) and `acceptance_rate`, never accuracy.** Standalone pattern, bare greedy, nmd 11-20. **9-α table + text diagnostics: `AdaDopamine.md` §3.5.** Key operational points:
  - **The result is a real monotone dose-response on `yes_rate` (−8→+4), and it is ENGAGEMENT, not discrimination — proven, not argued:** +α raises recall AND FPR *together*, which a genuine accuracy gain would not do. Both rising = a global response-bias shift. **Wording (settled): "+α lowers the THRESHOLD for challenging", NOT "braver/敢于质疑"** — there is no authority to defy here, and −α is lazy-default-accept, not cowardice.
  - **Usable band ≈ −8…+4; +6/+8 over-steer.** The failure modes mirror CGT-seq: **+8 = 散** (reasoning-first swallows the verdict — `startYN%` collapses 99→50.7% as the Yes/No gets pushed out of the 64-token window) vs **−8 = 垮** (form intact, content collapses to a blanket 98.7% "No"). `bare%`=0 and `expl%`≈flat at every α, so +α does not ADD explanation, it moves the argument BEFORE the verdict.
  - Greedy ⇒ deterministic: re-running any α reproduces byte-identically, so **do NOT re-sample**. Data source `pminervini/HaluEval` config `qa` (Parquet). No `analyze_*` parser; recompute offline from `raw_neutral` + `is_hallucination`.
- `get_answer_regenerate_truthfulqa_gen.py` (`run_truthfulqa_gen.sh`, loader `data_truthfulqa_gen.py`) — **TruthfulQA-Generation = open-ended over-generation probe.** Free generation (bare, greedy, 128 tok), so it avoids both SIQA's accuracy-only null and HaluEval's Yes/No default-action trap: there is no binary default to coast on, so α expresses on *what / how much / how boldly* the model asserts. n=100, α=0/±4. **Results: `AdaDopamine.md`.** Key points:
  - **TRUTH AXIS = NULL** (hand-labelled: no monotone, +4 not more truthful) — same knowing-axis null as SocialIQA. **The real signal is the LABEL-FREE generation-form layer** (needs no T/H judgment, hence most reliable): `deg%` (degenerate loop / self-made MC-grid / benchmark-completion) drops monotone 22→20→9, `hedge%`↓, and on the deg-removed subset +α still writes longer. **Cite as over-generation/engagement evidence, NOT as "+α more/less hallucinate".**
  - **Total char_len is CONFOUNDED — remove `deg` first.** −4's higher deg% (long loops) DEPRESSES its total, so "+α longer" is only clean after removal (deg deflates −4; it does not inflate it).
  - **Prompt is deliberately NEUTRAL** (`"Answer the following question."`): an earlier `"...truthfully and concisely."` was rejected because 'truthfully' damps the +α false-commit being observed and 'concisely' damps the length readout. **Auto-scoring by string-match against the reference lists FAILED** (TQA refs are paraphrases; ~80% fell into dispute) — T/H needs semantic judgment, which is why this is hand-read. Data source `truthfulqa/truthful_qa` config `generation`.

These are exploratory and **not** part of the frozen GSM8K dose-response table — they have their own output trees and (except CGT, which now has `RoleAnswer/analyze_cgt.py`) no `analyze_*` parser in `RoleAnswer/` yet, so analysis is mostly ad-hoc per experiment. `run_judge_confidence.sh` is a Slurm batch script (note the `#SBATCH` header) for the cluster, unlike the other interactive `run_*.sh`. **Standalone-pipeline pattern (SocialIQA / HaluEval, 2026-06-25): when adding a new benchmark, prefer a self-contained `data_<x>.py` + `get_answer_regenerate_<x>.py` + `run_<x>.sh` triple that does NOT import `template.py` or reuse `get_answer_regenerate_logits.py`'s `TASKS` loop — that keeps the frozen MMLU/GSM8K pipelines byte-stable while reusing only the shared `vc.regenerate` α-hook + `utils.parse_configs` + nmd mask. New script-based HF datasets are increasingly rejected by current `datasets`; reach for a pure-Parquet mirror (`lighteval/siqa`, `pminervini/HaluEval`, `truthfulqa/truthful_qa`).** **Benchmark-selection criterion (settled this session): only worth a positive wanting experiment if it has a readout channel INDEPENDENT of answer-correctness. SIQA (pure accuracy) → doomed flat; HaluEval (Yes/No has a low-effort default) → engagement bias not hallucination; TruthfulQA-Gen (free generation, no binary default) → cleanest over-generation channel. Ask "where can wanting express besides being right?" before building.**

### chat-template alignment (steering-validity caveat, 2026-06)

**Default convention: do NOT pass `--use_chat`.** The NMD mask / diff vectors are extracted on **bare-string** prompts (`run_hidden_mmlu.sh` → `get_answer_logits.py` with `--use_chat` OFF; `run_mean_diff.sh`, `run_nmd.sh` likewise), so steering must inject into the same bare activation distribution the diff vector was measured in. `tokenizer.apply_chat_template` prepends `<|start_header_id|>system…` control tokens and shifts the residual-stream geometry away from that point, which can **dilute or mis-align** the steering. The GSM8K main line runs bare. **"Has the flag" ≠ "enabled" — many `.py` accept `use_chat` but their `.sh` never passes it (= bare in practice); check the `.sh`.**
- **Two TESTED exceptions, both deliberate:** (1) **Betting** — a bare re-run COLLAPSED the §3.1 effect (mean_bet flat across α), so all four betting scripts were reverted to chat. Stance (settled): chat here is a **FEATURE, not a flaw** — showing the dose-response survives the harder, mask-divergent chat distribution is a STRONGER generalization claim. Do NOT re-migrate betting to bare, and do NOT frame it as an "un-validated premise" in the paper. (2) **CGT** — bare modes were qdm≈0.50 (near-random); chat is a PREREQUISITE for the task to work at all.
- **CGT also proves chat is not the α-null's cause:** `--simple3c` adds a betting-style `Answer: ` anchor so prefill steering lands ON the decision token rather than a chat control token, and the α-null SURVIVED — same `regenerate` path that gives GSM8K a −6 peak and betting a +52% lift. So CGT's null is neither chat-dilution nor injection-locality.
- When in doubt on a NEW experiment, run a bare-vs-chat decision sweep (`0/±4`, 2–3 runs) to quantify the dilution before trusting a chat result.

## Offline analysis workspace (`~/Documents/RSNResult/RoleAnswer/`)

This directory is **not** in the RolePlaying git repo. It is the offline analysis workspace for Phase 1 (observation) signal-proxy validation and capitulation analysis. **Location note: relocated 2026-07-16 to `~/Documents/RSNResult/RoleAnswer/`** (the old `~/Downloads/RSNResult/` mount dropped). **Phase naming (settled 2026-07-16): Phase 1 = observation (α=0 signal characterization + steered α-dose signal), Phase 2 = control (closed-loop, currently shelved pending Phase 1 wrap-up). The older "Phase 1b" name is retired — just Phase 1 / Phase 2.** The current Phase 1 signal set lives under `llama3/dopamine/signal/` and holds three file classes per condition — `dopamine_signal_*` (NMD-mask wanting projection), `random_signal_*` (random-mask RSN-specificity control), `metrics_*` (entropy/top1/margin/info_gain) — across the α-dose (`_a4/_a6/_a8/_aneg4/_aneg6/_aneg8`, α=0 untagged; ±2 backfill pending), the four No-CoT roles (`_expert/_non_expert/_primary_teacher/` + neutral), and CoT (α=0, −4). **Steered files carry inline `accuracy`** (184/bs=1; role/α comparable same-machine, but never配 onto the 182 dose table — see the 184-vs-182 bullet). **Steering is prefill-only + output-side**: `track_dopamine_signal.py` injects `α×mask` into the last prompt token's OUTPUT inside the observation hook *before* projecting, so `x_prefill` is the POST-injection point (`x_prefill(α) ≈ x_prefill(0) + α·‖mask‖²` by the co-design identity), while `x_decode` steps are NOT re-injected — they are the natural aftermath of that prefill perturbation. Key scripts there:

- `plot_phase1_state.py` — **Phase 1 trajectory plotter** (α=0 state contrasts). Three figure types × 5 metrics (wanting/entropy/top1/margin/info_gain): `plot_overlay` (two state curves), `plot_diffs` (a−b difference + bootstrap band), `plot_allroles` (four No-CoT roles overlaid). Prefill is drawn as point 0 on an integer index axis (0 = last prompt token, 1..100 = decode); EMA-smoothed by default, `--raw` for unsmoothed (writes `_raw` suffix). Reads `dopamine_signal_*` (wanting = `x_decode`) + `metrics_*`. Outputs `llama3/dopamine/plots_eot/phase1_{state,diff,roles}_<metric>.png`. Run with `python3.10`.
- **Gain-coordinate (V1) Phase 1 signal analysis — the CURRENT main line for §4 of `AdaptiveThinking.md`** (supersedes the V0 `x_t`/prefill-seeded-EMA scalar scripts). All read `dopamine_signal_*` + `metrics_*` from `llama3/dopamine/signal/`, `python3.10`:
  - `phase1_gain.py` — **core V1 recompute**. Reads the stored per-layer projections (`x_prefill_per_layer` / `x_decode_per_layer`) and re-expresses the signal in fixed gain coordinates: `μ_l^ref`/`σ_l^ref` from neutral-α0-No-CoT prefill; `g=(r−μ)/‖m_l‖²` (α-unit, **`G`**, for dose linearity / boundary_jump / calibration = H1) and `z=g/σ_l^ref` (layer-fair, **`Z`**, main trajectory = H2/H3); decode-only EMA `s_t` (seed = decode[0], NOT prefill), `p_t = Z_t − s_{t-1}`, `boundary_jump`. Emits Fig1 layer heatmap / Fig2 Z-trajectory / Fig3 p_t + aggregate table. **No server / HDF5 access needed.** Other §4.1 scripts import its machinery.
  - `analyze_wrong_right_gain.py` — §4.1 correct-vs-wrong: per-sample G/Z/boundary_jump/Z_early/Z_late/vigor_slope/p_early_std, Cohen's d + AUROC, + length-normalized Z-trajectory & p_t bands. Splits by the **authoritative `correct` field** (inline `pred_answer` is unreliable — 89/40 mismatches vs `correct`; but `sum(correct)/n` == `meta.accuracy`, so `correct` itself is trustworthy).
  - `analyze_wrong_right_suite.py` — §4.1 full §3.4 multi-metric suite (RSN side Z/s_t/p_t from `dopamine_signal_*`; logit side entropy/top1/margin/info_gain/cum_entropy_reduction/rolling_conf_variance from `metrics_*`), correct/wrong bands. **RSN and logit are different bases — plotted in separate figures, never overlaid.** top1/margin saturate (~0.99) late-decode → read entropy/rolling_var not top1.
  - `analyze_wrong_right_commit.py` — §4.1 **commit-timing controls (the load-bearing result).** Tokenizes `generated` to map first-`####`(∪ first answer-candidate) char pos → decode step. **Test 2**: commit step correct-vs-wrong (result: correct commits LATER, wrong抢答 earlier — so "correct reaches low-wanting faster" is NOT "finished earlier"). **Test 1 + commit-aligned suite**: re-aligns every metric to each sample's own commit step (±40); shows commit=0 is a 7-metric resonance node (entropy↑/conf↓/phasic spike-then-crash = phasic commitment), wanting (`s_t`) splits correct/wrong BEFORE commit (real level diff, not timing), confidence splits only AT commit → wanting leads confidence in encoding correctness. Open guard: the commit-前 `s_t` gap still needs within-difficulty replication to rule out difficulty co-variation.
- **§4.2 CoT vs No-CoT cluster (V1 gain coords, PAIRED — same 300 GSM8K Q, index-aligned; all `python3.10`, all read `llama3/dopamine/signal/`, figures → `plots_gain/`):**
  - `audit_cot_nocot.py` — **§4.2 Step 1 signal-data audit** (run FIRST). Checks same-300 + order + signal↔metrics alignment + trajectory health + definition provenance + records (never filters on) acc. **Load-bearing finding: 97% of trajectories hit the 767 max_new_tokens cap, and the tail after the 2nd `####` is a degenerate `#### N …` loop (loop-onset ≈17% No-CoT / ≈29% CoT), NOT reasoning.** Any late-window / `Z_late` / full-slope readout is a LOOP measurement unless the span is truncated — this caveat governs the whole section.
  - `analyze_cot_step2_tonic.py` — **§4.2 Step 2 task-entry tonic**, paired `d_z` + Wilcoxon on `G_prefill`/`Z_prefill`/`boundary_jump_G`. Result: CoT lifts task-entry wanting (`Z_prefill` d_z=0.89) — the effect is present before decode. NOTE the paired `d_z`=0.89 vs the old §4.2-table **un-paired** prefill (+0.03 ns) is the same data — pairing removes between-question variance.
  - `analyze_cot_step3_slow.py` — **§4.2 Step 2/Step 3 slow-decode dynamics** (three-span, paired). THREE spans, never conflate: `precommit` (decode→1st `####`, **pure reasoning = MAIN**), `clean`/pre-repeat (→2nd `####`, incl. post-commit text — NOT pure reasoning), `full` (0→767, loop-dominated = **pollution diagnostic only**). The **level-vs-shape verdict** = `plot_step3_shape` (baseline-centered `s̃_t = s_t − s_0` + paired CoT−No-CoT band) on precommit: centered curves do NOT coincide → **level-dominant, NOT pure level shift** (CoT relaxes less in pre-commit). `full`'s `relax_mag −0.49***` / reversed `slope_full` is a loop artifact that VANISHES in precommit — do NOT read it as "CoT relaxation stronger", and it does NOT transfer to §4.1 correct/wrong (a separate comparison axis). Figures: `fig42_step2_shape_test.png`, `fig42_step3_slow_st.png` (pre-repeat/full loop-pollution), `fig42_step3_commit_st.png` (commit-aligned level + slope). (Script/figure basenames still say `step3`; the section is Step 2 — internal titles renamed, basenames kept to avoid breaking refs.)
  - `analyze_cot_5panel.py` — **§4.2 unified event-centered + lifecycle figure layer (CURRENT plotting main line; 2026-07-21).** One 5-panel figure PER signal (`fig42_5panel_{s_t,p_t,entropy,top1}.png`), 2×3 grid: **Row 1 event-centered `[-50,+50]`** = (A) commit1-centered ∣ (B) commit2-centered `[C2-valid subset]`; **Row 2 lifecycle norm 0–100%** = (C) end-at-C1 (clean reasoning) ∣ (D) end-at-C2 (incl. C1, C2-valid) ∣ (E) full (incl. both, loop tail). Signals paired by index across the gain-coord JSON (`s_t`/`p_t` via `phase1_gain.four_part`) and the metrics JSON (`entropy`/`top1`); confidence is a DIFFERENT basis → its own figure, never overlaid on wanting. Bands = unpaired bootstrap 95% CI (paired significance stays in the Step 2/3/4 tables). **Load-bearing results this layer surfaced (not visible in the segment-mean tables):** (1) **commit1 = a multi-signal phasic node** — `s_t` collapses in PARALLEL after commit (CoT = baseline SHIFT, not event reshape), `p_t` has a sharp event-locked negative dip (CoT deeper: at-commit No-CoT −0.66 / CoT −1.05), entropy spikes + top1 dips (No-CoT more extreme = more "惊险" commit). This is the **event alignment Step 3 declared missing** — so the `neg_peak` "double-confound, record-only" caveat can be upgraded to "commit-aligned confirms a real event-locked dip (magnitude still partly EMA-lag modulated)". (2) **commit2 is ALSO an event-locked dip** (`p_t` −1.6, even deeper), not just loop noise. (3) **loop contamination enters between C2→full** — Row-2 (E) `s_t` crashes to a −1.3~−1.5 dead plateau after ~40%, top1/entropy saturate to ~0.99/~0.05 (the "can't read full-decode fixed-percentile confidence" caveat, made visible). C1≈242/247, C2≈221/204 (each condition's own 2nd-#### subset — a loop-prone selection). NESTED-SPAN rule: C1/C2/full are cumulative, NOT three independent tests. Superseded scratch scripts from the same session (`analyze_cot_figA_event.py` / `analyze_cot_event_all.py` / `analyze_cot_figB_lifecycle.py` / `probe_commit1_centered.py`) are folded into this one — prefer `analyze_cot_5panel.py`. Step 4/5 scalar analyses stay in `analyze_cot_step4_confidence.py` (pre-commit Q1–Q4 quartile table **+ the three-stage pre_commit/post_commit/loop_tail confidence cut** — mirrors wanting's stage boundaries so both axes share spans; entropy/top1/margin effect is confined to pre_commit and dies at saturation) / `analyze_cot_step5_timeaxes.py` (length-norm vs absolute sanity check).
  - `analyze_cot_mainfig.py` — **§4.2 MAIN paper figure (2×2, `fig42_main_2x2.png`).** Panels A/B/C = C1-centered `s_t` / `p_t` / entropy (bootstrap 95% CI band); **Panel D = paired-`d_z` forest** over pre-commit [s_t mean, p_t abs_mean, p_t std, entropy, top1, margin], colored by basis (green=wanting / orange=fast-residual / purple=confidence). Panel D is load-bearing because `p_t`'s real result is DISPERSION (`abs_mean`/`std`), which the mean curves in A/B cannot show. Imports the stage-feature machinery from `analyze_cot_step3_slow`. C1-centered → main text; C2-centered / lifecycle / top1 / margin → Supplement.
  - `analyze_persona.py` + `analyze_persona_supp.py` — **§4.3 Persona (Expert vs Non-Expert ONLY) — same pipeline as §4.2.** `analyze_persona.py` = the 2×3 main figure (`fig43_persona_main.png`): A/B/C C1-centered s_t/p_t/entropy, **D = pseudo-event control** (align to a RANDOM interior step in `[10, c1-10]`; `set_ylim(A)` so it shares Panel A's scale — no cliff at random ⇒ A's transition is `####`-specific), E = forest, F = text caption. `analyze_persona_supp.py` = two supplements: `fig43_supp_commit2_suite.png` (**commit2 aligned, following §4.1's `event_aligned_suite` exactly** — 2×4 7-metric [Z/s_t/p_t/entropy/top1/margin/info_gain], half=40, plain nanmean, imports `_align` from `analyze_wrong_right_commit`) + `fig43_supp_full.png` (full-decode length-normalized, loop-tail DIAGNOSTIC). **Result pattern (conservative, settled):** task-entry (prefill) gain huge (`G_prefill` d_z≈+2.8, `Z_prefill` +3.2) but does NOT persist as a full-pre-commit offset (`s_t` d_z≈−0.10, ns); confidence shifts weakly OPPOSITE (Expert slightly LESS decisive). Frame this as **"temporal and directional separation"**, NOT a clean dissociation, because persona is a prompt manipulation rather than causal α evidence and the NMD mask itself derives from the Expert−Non-Expert contrast, making task-entry gain primarily a cross-task manipulation check. The earlier valid-sample concern is now RESOLVED: paired windows on the common-valid subset (n=194; Expert analyzable 219/300, Non-Expert 236/300) confirm a localized reversal near C1 (`[−20,0] s_t mean` d_z=−0.278, p=.0013), faster Non-Expert post-commit release (`[0,+20] s_t slope` d_z=+0.291, p=.0004), and stronger release residual (`p_t abs_mean` d_z=−0.248, p=.0016); all three survive BH-FDR. This is a localized temporal redistribution, not a persistent decode-level shift and not proof that RSN dynamics mediate the 68% vs 58% accuracy difference. **Commit definition (both §4.1 & these §4.3 C1 figures):** first `####` ∪ first bare answer-candidate — so a sample that submits an answer WITHOUT `####` IS counted as commit (`commit_char` fallback in `analyze_wrong_right_commit.py`); only commit2 is `####`-only (needs a 2nd marker), which is why its subset is smaller/loop-prone.
- ~~`analyze_dopamine_signal.py`, `analyze_flow_shapes.py`, `analyze_dopamine_spikes.py`~~ — **deleted 2026-07-20** (pre-eot V0, superseded by gain coords; backup in `_archive_20260720/`). The V0 `signal_eot`-family scripts (`analyze_state_curves.py` / `analyze_signal_features.py` / `analyze_metrics_curves.py` / `analyze_signal_eot.py`) were **archived to `_v0_signal_eot/`** (they read the now-removed `signal_eot/`; kept for recompute until §4.2–4.4 migrate to V1). Retired data dirs `signal_old/` `signal_oold/` `plot_old/` `plots_eot_old/` removed.
- `analyze_bandit_pv9.py` — **the FROZEN analyzer for the PV9 six-cell sweep (Easy × NearTie × α∈{−4,0,+4})**; every number in `AdaBandit.md` §4 has exactly one definition here. `--part {validity,discovery,directed,random,utilization,persistence,text,outcome,primary,did,all}`, `--env {easy,neartie,both}`, `--no-model` to skip the slow bootstrap refits. **`attest()` runs for EVERY part and fails closed** (derives each cell's expected `steering_fires` from that cell's own config, and checks config / mask-SHA256 / reward-tape / arm_order equality across cells) — a single-part call is the normal way to pull one table and must not bypass it. The five frozen interpretive limits print at the top of every run. Conventions it enforces: the unit of inference is always the **seed (n=20)**, paired within an environment; pooled round counts are descriptive and are never fed to a test as independent samples; ties are tie-tolerant everywhere (matching `evaluate_competence_gate.py:115`) and structural Beta(1,1) ties are reported as a **BAND** (unique-max lower bound .. tie-inclusive upper bound); bimodal outcomes (`late_opt_frac` is strictly 0/1 in both environments) are headlined by lock-correct counts + exact McNemar with the mean as description only; Holm is applied within pre-frozen metric families, not per table. `python3.10`; a full `--part all` run takes ~30–40 min (240 bootstrap model refits).
- `analyze_first_last_acc.py` — **AUTHORITATIVE GSM8K/MATH ACC** (first-`####` / first-`\boxed{}` + fallback + norm; emits 改对/改坏 commitment split). `--gsm8k_root llama3/gsm8k` for the 182 same-machine rerun; `GSM8K_DIRS` already includes the full No-CoT dose sweep (`mdf_±2/±4/±6/±8`), CoT `±4`, and pushy variants.
- `analyze_loop_anxiety.py` — **AUTHORITATIVE loop / "can't-let-go" anxiety classifier** for α-steered GSM8K. Two modes: `--mode loop` (tail-loop gate → 3 mutually-exclusive buckets anxiety/mechanical/neutral_repeat, denom = loop count) and `--mode anxious_repeat` (full-text anchor-repeat, denom = 300: a cue + its next 60 chars must recur ≥2× to count, so one-off logical "however" is excluded). Anxiety = 4 overlapping sub-classes (self_doubt / format_anxiety / persona_reassure / over_precision). **Do not hand-write ad-hoc anxiety regexes** — edit the `ANXIETY_PATTERNS` dict here so every α uses one auditable standard. Key finding: anxiety-rate is U-shaped with trough at α=−6 (= acc peak), robust across all three counting conventions. **`--task math` (via `set_task`) swaps ONLY `format_anxiety`** — GSM8K's `####` cues become MATH's `\boxed{}` / `in the form`, while `self_doubt` / `persona_reassure` / `over_precision` transfer byte-unchanged. That is what makes the §2.3 and §3.4 subtype counts directly comparable across the two tasks, unlike `analyze_cot_metrics.py --task math`, whose re-anchored position metrics are NOT. It must be called before `analyze_file`; forgetting it silently scores MATH with the GSM8K format regex, which reads a plausible near-zero rather than raising. "Any" is the DEDUPLICATED union of the four subtypes, so it is ≤ their sum.
  - **`--mode persona` is a THIRD mode with its own frozen detectors** (§2.1 / §3.2 identity-confirmation loop), denominator = all 300. `IDENTITY_RE` matches `I am [not] [a/an] <ROLE_NOUNS>` or `As a/an … <ROLE_NOUNS>` where `ROLE_NOUNS = expert|teacher|tutor|genius|master|professional|student|non[- ]?expert|math(ematician)?|whiz|human|regular person|assistant`; `heavy` = ≥5 hits in one sample; `DENY_MATH_RE` = an explicit disclaimer of math authority. **`soft_deny` is counted ONLY inside identity samples and must be read BESIDE `deny_math`, never instead of it** — it catches the functional self-diminishment the literal deny misses (`just a student` / `not sure` / `can make mistakes` / `a computer program`). That pairing IS the §2.1↔§3.2 result: on GSM8K expert reads deny_math 0 / soft_deny 0 (genuine self-credentialing), while on MATH the literal deny_math 6 hides that **13 of 15** identity samples collapse softly — the SAME persona flipping credential→collapse with task difficulty. **`DENY_MATH_RE` was deliberately left unchanged when `soft_deny` was added (2026-06-05)** so every published GSM8K §2.1 number keeps reproducing; a new readout gets a new column rather than a widened old one. Its examples come from `--dump_examples`, which selects them mechanically (the first `heavy` samples in index order) — §2.1.1's raw quotes are therefore script-selected, not hand-picked, which is the claim that section rests on. It reads `gsm8k/mdf_0` and `gsm8k/mdf_0_pushy` (the pushy-wording arm, field `generated_<role>`); question numbers in the doc are 0-based sample indices.
- `analyze_cot_metrics.py` — **reusable behavioral metric panel** (the 10-metric union of §2.2 Table1 dose-sweep + §2.5.1 CoT-contrast in `AdaDopamine_gsm8k.md`). `--table dose` → 9-cell No-CoT sweep → `cot_metrics_dose.csv`; `--table cot` → 6-cell CoT×No-CoT → `cot_metrics_cot.csv`. **`--task math` re-anchors two metrics onto `\boxed{}`** (→ `cot_metrics_dose_math.csv` / `cot_metrics_cot_math.csv`): commit position reads the first `\boxed{}` instead of the first `####`, and `preempt_lead` becomes "`\boxed{}` is the first token" rather than "first char is a digit". So MATH and GSM8K position/抢答 VALUES are not comparable — only their trend directions are; §3.3 also has just three doses (−4/0/+4) against GSM8K's nine. Imports `analyze_first_last_acc` + `analyze_loop_anxiety` so it shares their extractors; every metric was reverse-engineered to **reproduce the existing Table1 values exactly** (commit_rate / committed_acc / median&mean `####` position / gen_len / n_loop), then extended with `step_ge2` / `stuck`(loop ∧ `=`<2) / `preempt_lead`(首字符即数字) / `preempt_any`(lead-digit ∪ `####`-at-start<2%) / `med_eq`. **Two抢答 detectors must be read together**: `preempt_lead` misses the −α `#### N` first-token lock (first char is `#`), so −8 reads as 4 under lead-digit but 175 under `preempt_any`. `committed_acc` & `####`-position are **anchored metrics — only comparable within one generation regime** (CoT's long Step prefix + post-`\boxed{}` empty `####` shift them; use `preempt_lead`/`preempt_any` across the CoT boundary). Exact definitions: `####` position = `first ####-or-bare-#### char offset / total chars × 100`, medianed **over the samples that HAVE a `####`** (so its denominator is `commit_rate`'s numerator, not 300); the mean version is a right-skew diagnostic — `mean > median` means a late-submission tail. `gen_len` is in CHARACTERS, not tokens. `early-####` in `preempt_any` means within the first 2% of the text. `loop samples` is `analyze_loop_anxiety.py --mode loop`'s `n_loop`, and its ~74–88% baseline with no clean α trend is why the raw loop rate must NOT be read as perseveration (§2.3).
- `analyze_cgt.py` — **AUTHORITATIVE CGT (simultaneous) metric panel.** `--dir llama3/cgt/<mode>_verify` → a 3-LAYER table + stats (KW / Spearman ρ vs α / MWU): WANTING [I_LC, mean_bet, max_bet_rate, bet@{60,70,80,90} odds bins, `delib_tok` = commit-pre reasoning tokens via the Llama tokenizer, risk_taking] / KNOWING [qdm, qdm_blue/red_major, risk_adj_slope, bet@90−60, I_EC — these flat = wanting–knowing dissociation, NOT a null] / DIAGNOSTIC [invalid_rate, color_bias, frac_no_reason]. The `delib_tok` extractor counts tokens BEFORE the committed `Color:` line (Answer: anchor + reasoning); needs the HF tokenizer (falls back to whitespace if unavailable). This is what surfaced the simple5 result (betting null, delib_tok dose-response).
- `analyze_cgt_seq.py` — **AUTHORITATIVE CGT-SEQUENTIAL (delay-aversion) metric panel.** `--asc llama3/cgt/seq_asc_v2b --desc llama3/cgt/seq_desc_v2b` → 4-LAYER per-condition table + KW/Spearman/**paired Wilcoxon** + cross-condition DAI with a **paired bootstrap 95% CI**. Layers: **PRIMARY** (mean_accept_step / `valid_only_accept_step` [= same value, explicit "trend ≠ invalid-survivor" proof] / early_stop_rate / accept_step1_rate / `max_bet_first_stop` [desc-only impulsivity disambiguator] / forced_lock_rate); **PERFORMANCE** (`final_score` / `mean_phase_score` — inverted-U, desc peaks +4 / asc peaks −4, but report qualitatively only); **BETS** (mean_bet / max_bet_rate / risk_taking — within-condition only); **KNOWING/DIAGNOSTIC** (qdm / invalid_rate / `empty_gen_rate` [−α 垮/沉默] / `color_confusion_rate` [Wait/Accept leak or empty color step] / color_blank_rate / `exact_bet_rate` / `dirty_bet_rate` / `dirty_valid_rate`). **TWO over-steer gates** (auto-flagged, EXCLUDE from fit): `invalid≥0.15` (−α empty/collapse) OR `dirty_valid≥0.15` — the latter catches the +α failure invalid_rate MISSES: at +8 the parser still extracts Accept/Wait but ~44% of valid replies are a paragraph / "Wait Accept" / probability restatement, not a bare one-word decision. `_bet_is_clean` judges EACH tier reply separately (a legit Wait→Wait→Accept chain is clean — do NOT concatenate across tiers). Display labels disambiguate the two "early" metrics: `early(step<=2)` (=early_stop_rate) vs `step1_accept` (=accept_step1_rate); the doc table's "asc/desc early" column = `step1_accept`. DAI section reports BOTH `DAI(bet)=mean_bet_desc−mean_bet_asc` (canonical CANTAB index, use this) and `DAI(step)` (timing-pure but ceiling-confounded on +α — do NOT report). **PAIRING IS LOAD-BEARING (fixed 2026-08-19).** `get_answer_cgt_seq.py` uses `seed=run_idx`, so run *i* faces a byte-identical chest sequence in EVERY α cell, and the same index pairs asc with desc (verified per-round on the stored v4 data). The unit of inference is therefore the **run (n=20, paired)** — same house convention as PV9/PV10. The pre-2026-08-19 build used an UNPAIRED Mann-Whitney; **its p-values are void.** Re-running flipped five cells (desc `mean_accept_step`/`mean_bet` at −8, desc `final_score` at +6/+8 → n.s.; asc `qdm` at −2 → sig) and left every clean-range primary result significant. The two −8 flips are a CORRECTED FALSE POSITIVE, not lost power: −8 has `invalid≈0.99`, so only **7 of 20** runs yield a value (asc: 0), and MWU was comparing 7 survivors against 20 — `_paired` now reports that `n` explicitly. `_paired` returns `(p, median Δ, n_pairs)` and drops NaN pairs. Reads the `e55b132`/v2b/v3/v4 JSON schema (pre-2026-08-19 runs have NO `forced_lock` field → treated as not-forced, which is why `forced_lock_rate` reads a constant 0 on all legacy data; empty-records cells emit all-NaN so the full sweep's dead −8 cells don't crash it). **Table口径 = valid-only** (`valid==True` filter); a prior doc table mistakenly INCLUDED invalid-fallback rounds (random accept_step/color), which made −8 look like a fake early-commit signal (step≈2, qdm≈0.527 = pure fallback noise) — always recompute valid-only.
  <!-- Why: an aggregate QDM hides a label prior — Qwen v4 α=0 desc reads a moderate .65 that is actually 1.000 (blue) / 0.306 (red), i.e. a locked label, not partial probability use.
  Evidence: analyze_cgt_seq.py qdm_major_* / asym_gradient; AdaDopamine.md §3.3 QDM 成分分解.
  Scope: every CGT-seq QDM claim, both models. -->
  - **QDM MUST be read as TWO subgroups, never as the aggregate alone (2026-08-19).** `qdm_major_blue` / `qdm_major_red` / `qdm_label_gap` split accuracy by which colour was the majority, and `asym_gradient` (= qdm@asym-8 − qdm@asym-2) measures whether choice tracks the jar at all. **`asym_gradient` averages WITHIN each major_color first, then weights the two labels equally**, so an unequal blue/red composition at one asymmetry level cannot masquerade as a gradient. Both models carry a label prior in OPPOSITE directions — Llama favours red (gap .11–.19 across the clean range, both subgroups .58–.84), Qwen favours blue (α=0 desc: 1.000 / 0.306). Llama's clean range shows **no detected systematic α-dependence** in either component (label gap ρ(α)=−.09 asc / +.04 desc; asym gradient ρ=+.19 / +.12) — n.s., which is NOT an equivalence proof. Llama's +8 QDM drop is `major_red` collapsing ALONE (.82→.59, paired p<.001) while `major_blue` is unmoved (p=.198/.294).
  - **These four are DIAGNOSTIC ONLY and deliberately NOT in the over-steer gate.** The gate stays frozen at `invalid≥0.15 OR dirty_valid≥0.15` so every published Llama v4 number keeps reproducing — verified byte-for-byte after the metrics landed. A formal gate using them belongs to Qwen v5 / a later protocol version and must never be applied retroactively to v4. **Qwen v5's α=0 prompt-selection gate (pre-registered): invalid < 0.05; `asym_gradient` ≥ 0.15; and BOTH `qdm_major_blue` ≥ 0.55 AND `qdm_major_red` ≥ 0.55** — a subgroup reaching 1.000 is not itself lock-in; lock-in is the OTHER subgroup at/below chance, so a range cap would wrongly penalise genuine perfect performance.
- `analyze_igt.py` — **AUTHORITATIVE IGT metric panel.** `--dir llama3/igt/<ver>` → multi-layer table + KW/Spearman/**paired Wilcoxon**, 口径 aligned with `analyze_cgt_seq` (valid-only, over-steer gate). Runs pair by index (`seed=run_idx`); a pair is dropped only if either side is NaN. Layers: TEMPORAL PROXIES (`delib_tok` = the pre-`Chest:` decision-time proxy, the main readout) / PERFORMANCE+LEARNING (net_score, learn_slope, `avoidance_rate` vs `parse_fail_rate`) / LEARNING CURVE (net_block1..5) / CHOICE / LOCAL PUNISHMENT SENSITIVITY / **REWARD-PUNISHMENT ASYMMETRY** (`ws_ls_asymmetry` = the Frank-2004 RPE readout) / OPERANT-STRATEGY (entropy, max_run_len, cycle_score) / TEXT DIAGNOSTICS. Over-steer gate: invalid>0.20 OR cycle-signature OR cycle_score≥0.80 OR premature_stop>0.15 (the last symmetrically catches the −α avoidance collapse that an invalid average dilutes); α=0 is never flagged. win/lose by single-trial payoff, only valid→valid CONSECUTIVE pairs.
  - **TWO load-bearing cautions.** (1) **`net_score` alone misleads** — it rises for BOTH +α real exploit AND −α fake round-robin (read the EXPLOIT-vs-CYCLE diagnostics alongside), and it confounds risk preference with LEARNING capacity, so DA claims must use the learning-independent layers. (2) **`lose_shift_rate` / `post_bigloss_switch_rate` SATURATE ~0.95–1.0** because baseline switch_rate is already 0.64–0.80 — read `ws_ls_asymmetry` (the difference cancels the baseline), never the absolute rates.
- `analyze_rsn_specificity.py` — **RSN DIRECTION-SPECIFICITY (§4.6, CLOSED).** Asks whether the §4 gain-coord signal is NMD-specific or any sparse direction would show it. **Offline RE-PROJECTION ONLY** — causal steering is already validated in the RSN paper; this re-projects the SAME stored HDF5 against different masks. Building a new null needs the HDF5 (**server-side**); the local box has only projection-output JSON, so **you cannot add a draw locally**. `python3.10`.
  - **METRIC CORRECTION (load-bearing): use SIGNED commit-aligned temporal metrics, NOT unsigned `|d_z|` / `p_t abs_mean`.** The earlier unsigned framing ("decode s_t/p_t not specific") was a **FALSE NEGATIVE** — unsigned aggregation averages away the commit-locked sign structure. Frozen primaries = `s_pre_mean` (s_t on `[-40,0)`) + `p_post_mean` (p_t on `[0,+10]`).
  - **Each mask uses its OWN reference** (μ/σ from its own neutral file, its own `‖m_l‖²`) so a larger raw projection scale cannot fake-win. `‖m‖²` CANCELS in the decode Z-coordinate, so the diff_random ¼-norm gap affects only `G_prefill`, not decode readouts.
  - **Result:** task-entry `G_prefill` is NMD-specific but = co-design/manipulation-check; the **commitment-locked signed `s_t`/`p_t` temporal organization is the strongest specificity evidence** (NMD at pctile 0/100% in all 30 primary cells across three null families, null medians ≈0). Read the three families as a **control matrix, NOT a clean factorial** — the off-support cell is load-bearing, and together they point to a specific COMBINATION of top-|diff| support AND role-diff-aligned weights, not to either alone.
  - **Do NOT argue low chance from "3 families agree"** — they share the same hidden states/conditions/baseline/metrics, so they are NOT independent replicates (each N=10–11, p floor 0.083–0.091; exploratory ordering only). §4.6 needs no more same-type seeds; the next step is cross-task / cross-model / causal control.
  - **BUG (fixed 2026-07-26) worth knowing:** `leave_one_layer_report` must reset ALL THREE globals via `pg.configure(sig_dir=…, fprefix=…, mask_path=…)`. Passing `mask_path` alone leaves `SIG_DIR`/`FPREFIX` pointing at the previous null draw, so LOO silently reads the wrong signal JSON (symptom: LOO numbers change between `--null_family` runs, effect collapses to ~0). The signed-temporal + LOO-RMS tables were never affected — they pass mask/dir explicitly.
- `analyze_alpha_dose.py` — **§4.4 α-STEERING DOSE-RESPONSE panel** (`AdaptiveThinking.md` §4.4). Single file, `--part {validity,slow,fast,confidence,integrated,mainfig,all}`; 9 α cells (−8…+8, incl. ±2). `python3.10`, reuses `phase1_gain.py` (V1 G/Z coords) + `analyze_wrong_right_commit.py` (C1 align) + `analyze_cot_step3_slow.py` (three-stage). **口径:** reference μ/σ fixed = neutral α=0 No-CoT prefill; the dose main curve uses each α's full n=300, but every event window is **each-α-vs-α=0 paired common-valid** (NOT the global 9-cell intersection — extreme α would shrink n); acc = inline **184** (each α's own `correct`, never配 onto the 182 table). **Result: LINEAR input → NONLINEAR output** — task-entry `G_prefill ~ α` is linear (R²=0.999) but the decode-internal slow `s_t` peaks at **−6** (d_z=+0.58***), matching the behavioral acc peak (−6); slow `s_t` tracks acc (r=+0.74) while the linear prefill gain does NOT (r=−0.37). Confidence (top1) ALSO peaks at −6 (d_z=+0.64) ⇒ **α is NOT selective wanting** — wanting+confidence co-move (like CoT §4.2, unlike the clean persona dissociation §4.3). Figures `fig44_{validity,slow,fast,confidence,integrated,dose_main}.png` → `plots_gain/`.
<!-- Why: "inverted-U" was the wording here until 2026-08-25 and contradicts §4.4's own fit — the quadratic peak is −1.9 while the discrete optimum is −6, so the curve is not the smooth symmetric shape the term implies.
Evidence: AdaptiveThinking.md §4.4 Result 4 (quadratic R²=0.352 vs linear 0.147, fitted peak ≈−1.9).
Scope: every Llama α-dose shape claim. Qwen's own shape is a high-dose plateau — a different thing again. -->
  - **The −4…+2 band is the `Intermediate calibration range`, NOT an "adaptive range".** "Adaptive" reads as a value judgment the data contradict: accuracy across that band is **monotonically decreasing** (74.3 → 68.3 → 60.0 → 55.3) and the discrete optimum **−6 lies outside it**. The band names where the STATE changes smoothly, not where the model performs well.
  - **Do NOT call the Llama α curve an "inverted-U".** The quadratic fit beats linear (R²=0.352 vs 0.147) but its peak sits at **α≈−1.9** while the discrete optimum is **−6**, so the data do not support a smooth symmetric inverted-U. The frozen wording is **`asymmetric peaked working-point response`**: sharp optimum at −6, collapse at −8, gradual decline then flattening on the positive side.
- `analyze_cot_alpha.py` — **§4.5 CoT × α=−4 SIGNAL-INTERACTION panel** (`AdaptiveThinking.md` §4.5). 2×2 factorial `{No-CoT,CoT}×{α=0,−4}` (the signal side has CoT only at −4/0 — there is no CoT dose), same 300 questions index-paired. `--part {entry,slow,fast,confidence,mainfig,all}` (verified against `PARTS` in the script; §4.5's body used to list an incomplete set, which is one reason the option list lives here and not there). Interaction = per-question **DiD** then Wilcoxon, not a comparison of 4 means. `python3.10`.
  - **Headline = TIME-CENTER STRUCTURE, not "orthogonal levers":** Task-entry α≫CoT, Decode CoT≫α, Confidence approximately additive — built on the significant main-effect dominance flip, NOT on any DiD verdict.
  - **Three 口径 cautions.** (a) Statistical vs practical interaction are separate — the task-entry DiD is *** but is only ~0.4% of α's main effect ⇒ *approximately* additive; do NOT downgrade *** to ns because the magnitude is small. (b) **"one significant + one ns simple effect" ≠ the two differ significantly** — the early-window α effect turns ns under CoT but its DiD is p=.14, so report an **attenuation trend**, never a formal redundancy/saturation verdict (accept-null fallacy). (c) `_verdict()`'s labels are a magnitude-ratio heuristic, NOT a significance test — read the DiD Wilcoxon p, not the label.
  - cot_−4 is the highest-`s_t` / highest-acc cell but is treated as **co-occurrence only** (§4.4 already showed signal-high ≠ behavior-good — no mediation claim). Single-dose limit: cannot infer CoT's dose-response or whether CoT shifts §4.4's working point (do not call that curve an inverted-U — see the §4.4 entry above).
- `analyze_slow_state_behavior.py` — **TODO §2 Slow-State Behavioral Validation** (`AdaptiveThinking.md` §4.7). Pools 11 conditions × 300; features (`s_t`) and behaviors (commit step, gen_len, premature, loop, eos_fail) come from the SAME signal-JSON object, index-aligned. `--part descriptive|regression|heldout|plot|all`; `python3.10 -u`.
  - **Modeling-vs-finding split (mirrors AdaptiveThinking.md §2.2 命名約定):** the `s_t` SLOPE remains the operationalization of the ramping/vigor hypothesis. This script asks only whether THIS task elicits the predicted association — **a null here is a task-level finding, NOT a construct downgrade or rename.**
  - **LEAKAGE GUARD is the whole point.** A slope over `[0,c1)` is mechanically coupled to c1, so the PRIMARY predictor is a **fixed early window `[0,W)`** that never sees the commit. Three further 口径 fixes the v1 numbers lacked: **landmark filtering** (23.4% of samples commit before token 20, so the analysis runs on the at-risk subset `commit_step≥W` with target `commit_excess`), a **frozen train scaler** applied to test (v1 standardized separately → invalid held-out R²), and **α-condition fixed effects**.
  - **Result:** GSM8K did NOT elicit a ramping/vigor slope signal; here `s_t` behaves as a **slow engagement / commitment-state readout** — LEVEL→commit_step ρ≈+0.38–0.40 (p<1e-85) but leak-free SLOPE→commit_step is ns. The regression's significant slope β is a **SUPPRESSOR artifact, not vigor**: corr(level,slope)=−0.48, marginal slope↔excess null, and the sign is POSITIVE (steeper→LATER commit) = opposite to vigor.
  - **`premature`/抢答 is RETRACTED as slope evidence → DIAGNOSTIC only** (749/754 premature samples commit before W, so the window overlaps post-commit release = time-order confound). `eos_fail` R²≈0.98 ≡ the length cap → control only.
- `analyze_rsn_specificity.py` — **RSN DIRECTION-SPECIFICITY null** (TODO §1; `AdaptiveThinking.md` §4.6, preliminary result landed 2026-07-24). Asks: is the §4 gain-coord signal NMD-specific, or would any sparse direction show the same state effects? **Offline re-projection ONLY** — causal steering is already validated in the RSN paper (AdaDopamine.md §2); this re-projects the SAME stored HDF5 HS against different masks (needs the HDF5 → mask + remask step is SERVER-side; the local box only has the projection-output JSON, so you CANNOT add a draw locally). **Each mask uses its OWN reference** (μ/σ from its own neutral file + its own `‖m_l‖²`) so a bigger raw projection scale can't fake-win. `‖m‖²` CANCELS in the decode Z-coordinate → the diff_random ¼-norm gap does NOT affect decode readouts; it only affects `G_prefill`. **METRIC CORRECTION (load-bearing, 2026-07-24): use SIGNED commit-aligned temporal metrics, NOT unsigned `|d_z|`/`p_t abs_mean/std`.** The earlier unsigned framing ("decode s_t/p_t not specific") was a FALSE NEGATIVE — unsigned aggregation averages away the commit-locked sign structure. Frozen primaries = `s_pre_mean` (s_t on `[-40,0)`) + `p_post_mean` (p_t on `[0,+10]`), signed raw Z contrast (no d_z), reported as (NMD signed / null median / signed percentile / median-centered two-sided p), plus **leave-one-layer-out** (踢 any of L11–19, effect keeps sign = **not driven by any SINGLE layer**; leave-two/three-out was never run and there is no per-layer null side, so this may NOT be upgraded to "not few-layer-driven") and **LOO-centroid RMS** curve distance (s_t/p_t separately; RMS is amplitude+shape, so read the `centroid_std` flat-guard before trusting shape-corr). K-gate = `max(30,⌈0.15·n⌉)`, SHARED across NMD+null (identical text → identical per-offset counts, asserted). **RESULT (THREE null families, NMD remains extreme in all, 2026-07-26):** two-layer conclusion — task-entry `G_prefill` NMD-specific but = co-design/manipulation-check; **commitment-locked signed `s_t`/`p_t` temporal organization = the STRONGEST current NMD direction-specificity evidence.** All three families — ① `diff_random` (support-selection, N=11) + ② `ortho_gauss_same`/`ortho_gauss_off` (generic-direction, per-layer Gaussian ⊥ dense Δ_l norm-matched, N=10 each) — leave NMD at **pctile 0/100% in all 30 primary cells, null median all ≈0** (null central tendency ≈0 and no sampled direction reproduces NMD's window-mean signed effect — NOT "the null curve is flat"; some nulls, esp `p_t`, still show commit-region wiggle, just with sign-averaging≈0), sign flips α−/CoT (s_pre↑ p_post↓) vs α+/Expert (mirror), LOO-RMS 90–100%, leave-one-layer-out 10/10 no-flip. **Read the three as a control matrix (NOT a clean factorial): the off-support cell is load-bearing** (positions moved OUT of NMD + weights ⊥ role-diff, NMD still owns the extreme, null median≈0 = off direction's window-mean has no signal) → with ① (random support fails) and `ortho_gauss_same` (NMD support alone fails), the evidence points to a **specific COMBINATION of top-|diff| support AND role-diff-aligned weights** — NOT "support unimportant" or "weights unimportant" but their matching. Natural CoT/Persona match injected α → not a co-design artifact. `run_generic_null.sh` builds ② on the server (guards assert `|cos(m_l,Δ_l)|<1e-5`); analyze via `--null_family {diff_random,ortho_gauss_same,ortho_gauss_off} --null_root …` (frozen metrics unchanged; default diff_random unchanged). See `GENERIC_NULL_RUNBOOK.md` (**not on disk as of 2026-08-11**; the `run_generic_null.sh` guards are the live source). **§4.6 is CLOSED — no sign-shuffle / more seeds needed.** The one un-run decomposition is a **same-support sign-shuffle** (does the sign↔position correspondence matter). It is NOT known to be uninformative: the role-diff sign COUNTS on NMD support are near-balanced (180 neurons, **+86/−94**, imbalance 0.044), but that only says the ± counts are symmetric, not that the correspondence is irrelevant — hence lower-priority, not settled. The orthogonal nulls already norm-match and remove the role-diff weight direction entirely, which is why it is not required for the frozen conclusion. Exploratory ordering only (each family N=10–11, p floor 0.083–0.091; the three families share the same hidden states/conditions/baseline/metrics so they are NOT independent replicates — do NOT argue low chance from "3 families agree"); next step = cross-task / cross-model / causal random-direction control, not more same-type seeds. Only Llama3-8B, offline. `python3.10`. **BUG FIXED 2026-07-26: `leave_one_layer_report` must reset ALL THREE globals** — `pg.configure(sig_dir=SIG_DIR, fprefix=nmd_prefix, mask_path=nmd_mask)`; `configure(mask_path=…)` alone leaves `SIG_DIR`/`FPREFIX` pointing at the last null draw's dir, so LOO silently read the null's signal JSON (symptom: LOO numbers differed between `--null_family` runs, effect collapsed to ~0). The signed-temporal + LOO-RMS tables were NEVER affected (they pass mask/dir explicitly to `commit_aligned_contrast_multi`). **`phase1_gain.configure(sig_dir, fprefix, mask_path)`** repoints the shared V1 machinery at a different mask; §4.1–4.5 never call it and are unaffected.
<!-- Why: one sentence can hit `final answer`, `\boxed{}` and `####` at once, so an unmerged second marker looks like a second answer that never happened.
Evidence: AdaptiveThinking.md §4.8 Declaration marker 段; plot_sample_traj.py merge rule.
Scope: any per-sample trajectory plot marking answer declarations. -->
- **Answer-declaration markers within 25 characters are MERGED into one declaration** (`plot_sample_traj.py`); a dotted line means a **second DISTINCT** declaration. Even then it is a repetition/revision proxy, **never a verified loop onset**. §4.2's aggregate analysis sidesteps this by using the literal 1st/2nd `####`, but its **C2 is a second-answer-marker boundary**, not a loop onset.
- `analyze_pt_frequency.py` (+ lean `pt_freq_between_alpha.py`) — **§4.4 Result 3–4 `p_t` amplitude-vs-frequency validation** (`AdaptiveThinking.md` §4.4; the case figures it accompanies are §4.8). **Relocated 2026-08-25** from §4.7, which is now Slow-State Behavioral Validation — older refs to "§4.7 amplitude/frequency" are stale. Answers ONE question: is `p_t`'s post-commit/loop change a commitment-related fast temporal *reorganization* or just repetition/token-pattern? α=−6/0/+6, all 300, two window schemes — commit-centered `[c1−40,c1)` vs `[c1,c1+40)`, and stage-based reasoning `[0,c1)` vs a **strict repeated-ngram tail proxy** (earliest 12-char n-gram recurring ≥3× — a repetition-tail approximation, NOT a verified loop onset). Per-segment readouts: centered RMS (DC-removed residual amplitude), zero-crossing rate, Welch dominant freq, spectral centroid, normalized spectral entropy; plus confound corr of each Δfreq metric with Δ{repetition,digit,hash}. `pt_freq_between_alpha.py` is the fast between-α add-on (skips the slow `step_to_char` confound pass — `analyze_pt_frequency` block-buffers to a file, so run it with `python3.10 -u`). **RESULT (the settled §4.4 Result 3–4 dissociation): amplitude is the real signal, frequency is not.** Only **centered RMS** stably separates reasoning from post-answer/loop and carries the α effect — pre-commit RMS α=−6 vs 0 = **+0.046, p<.001, paired n=200** (α=+6≈null); ALL frequency metrics are α-null in both directions. Commit-centered spec-entropy↓ is confounded with `####`/repetition (ρ≈−0.55); the stage-based subset (n=24–42) shows flat zcr but is NOT a confound-free negative (detector selects on repetition + range restriction). So `p_t` **keeps its phasic-like operational definition** (retained, not downgraded): this task supports an amplitude/dispersion change but did NOT detect stable frequency organization, and no biological-phasic-dopamine correspondence is established — frequency is a negative control, amplitude/dispersion is the validated readout. (Modeling-vs-finding split per AdaptiveThinking.md §2.2 命名約定: a null is a task-level finding, not a rename of `p_t`.) These scripts live in `RoleAnswer/` (offline), `python3.10`. The 9 case figures §4.8 reads are `RoleAnswer/llama3/dopamine/plots_gain/sample_traj3_q{010,080,092,140,152,189,225,251,284}.png` (one per question, each overlaying α=−6/0/+6 `s_t` + `p_t` + the generated text). They are a **qualitative sanity check only** — not pre-registered, not randomly sampled, no effect size.
- `analyze_slow_state_behavior.py` — **TODO §2 Slow-State Behavioral Validation** (`AdaptiveThinking.md` §4.7; settled 2026-07-28). **Modeling-vs-finding split (load-bearing, mirrors AdaptiveThinking.md §2.2 命名約定):** `s_t` SLOPE is the *operationalization* of the ramping/vigor hypothesis (kept), `s_t` LEVEL of the slow-engagement/commitment-state hypothesis (kept); this script asks ONLY whether **this task (GSM8K commit-timing / premature) elicits the slope's predicted behavioral association** — a null here is a **task-level finding, NOT a construct downgrade / rename** of `s_t`. Pools 11 conditions (No-CoT dose −8…+8 incl. ±2, CoT α=0/−4) × 300 = 3300 samples; features (`s_t` via `phase1_gain.four_part`) AND behaviors (commit step, gen_len, premature=lead-digit∪early-####, answer oscillation, repeated-ngram loop, post-commit tokens, eos_fail) come from the SAME signal-JSON object, index-aligned; entropy/top1 controls from the sibling `metrics_*.json`. **LEAKAGE GUARD is the whole point:** a slope over `[0,c1)` is mechanically coupled to c1, so the PRIMARY predictor is a **fixed early window `[0,W)`** (W=20/40) that never sees the commit; commit-aligned `pre_slope`/`pre_level` are descriptive-only. **FOUR leakage/口径 fixes (load-bearing, added in the 2nd pass — the v1 numbers were leaky):** (1) **landmark filtering** — 23.4% (754/3229) of samples commit before token 20, so for them `[0,W)` straddles commit and its slope carries post-commit release; the commit-timing analysis is therefore on the **at-risk subset (commit_step≥W)** only, target = `commit_excess = commit_step−W`; (2) **held-out freezes the train scaler** onto test (`_design` returns `(…, scaler)`; v1 standardized train/test separately → invalid held-out R²); (3) **confidence controls use the fixed `[0,W)`** window (not the old length-proportional first 20%); (4) **α-condition fixed effects** (dummies vs a0) so a cross-α pooled association can't masquerade as within-sample. Three layers: descriptive Spearman (at-risk + within-regime), item-level nested regression (`_ols_cluster` cluster-robust SE by qid), question-level 70/30 held-out on the at-risk subset. `--part descriptive|regression|heldout|plot|all`; caches the re-tokenized table to `/tmp/slow_state_rows_*.pkl` (keyed on beta+windows); `python3.10 -u`. **RESULT (task-level): this task did NOT elicit a ramping/vigor slope signal; `s_t` here behaves as a slow engagement / commitment-state readout** — at-risk LEVEL→commit_step ρ=**+0.379/+0.400** (p<1e-85) but leak-free SLOPE→commit_step ρ=**−0.020/+0.012 (ns)**. Regression: +level R² 0.235→0.259 (β=+19.6, p=7e-7); +slope→0.267 shows a significant slope β=+10.7 (p=.003) **BUT it's a SUPPRESSOR artifact, NOT vigor** — corr(level,slope)=−0.48, slope↔excess marginal ρ=−0.02 (null), partial(|level) r=+0.20, and the sign is +（steeper→LATER commit）= OPPOSITE to vigor; level β jumps +19.6→+25.9 when slope enters (textbook suppression). Held-out (frozen scaler): +level test R²=0.249, +slope 0.254 (weak, same reversed sign). **premature/抢答 is RETRACTED as slope's signal → DIAGNOSTIC only:** 749/754 premature samples commit before token W, so `[0,W)` slope overlaps post-commit release (time-order-confounded), not independent prediction. `eos_fail` R²≈0.98 (≡ length-cap) → control only. **The ramping/vigor slope definition is retained**; naming `s_t` slope "vigor" at the construct level is left to a vigor-eliciting task (effort/betting/agentic) — GSM8K (grade-school arithmetic, no progressive reward-approach structure) does not falsify it. Fig `plots_gain/fig_slow_state_behavior.png` (A/B at-risk level/slope→commit_step decile lines, one sloped one flat; C/D premature boxplots = DIAGNOSTIC, time-confounded). Offline, `python3.10`.
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
- `detection/nmd.py` builds masks (`nmd` / `random` / `diff_random` / `sparse_fv` / `ortho_gauss_same` / `ortho_gauss_off`). **Gotcha:** `--logits` controls the OUTPUT directory suffix (`mask/{model}_{type}_logits` vs `mask/{model}_{type}`), NOT whether logits data is read (that only matters without `--base_dir`); the existing NMD/diff_random masks live in `…_logits/`, so mask-generation runs MUST pass `--logits` or the file lands in the wrong dir. For `random`/`diff_random` a `--seed ≠ 42` appends `_seed{s}` to the filename (seed 42 stays unsuffixed for back-compat); the two `ortho_gauss_*` types ALWAYS seed-tag (no seed=42 baseline). `run_random_null.sh` sweeps seeds for the §4.6 diff_random (support-selection) null; `run_generic_null.sh` sweeps the two `ortho_gauss_*` (generic-direction) nulls. **`ortho_gauss_*` = per-layer Gaussian EXACTLY ⊥ dense role-diff Δ_l + norm-matched to the NMD row** (float64; built-in guards assert per-layer ⊥, norm-match, exact `top_k` nonzeros, and correct support relation, aborting on failure — a clean run IS the validation). `d_sub` MUST come from the dense Δ (not the sparse NMD mask), especially off-support.
- `harness.py` + `hf_rsn.py` plug into [lm-evaluation-harness] for benchmark-suite eval.
- **GSM8K/MATH answer extraction is centralized in `utils.py`** (`extract_gsm8k_answer`, `is_correct_gsm8k`, `gsm8k_difficulty`, `extract_math_answer`, `is_correct_math`). All 5 consumers (`get_answer_regenerate_gsm8k`, `get_answer_regenerate_math`, `closed_loop_gsm8k`, `track_dopamine_signal`, `track_hidden_states`) import from here — do not redefine locally.

## GSM8K re-run conventions (2026-05-31, current rework)

The Phase-2/1b GSM8K numbers from before 2026-05-31 are **discarded** — diagnosis found the prompts and extraction were noisy. Conventions for the re-run:

- **`<|eot_id|>` terminator fix (load-bearing for comparability).** `llms.VicundaModel._build_terminators()` registers `<|eot_id|>` (id 128009) — and `<|end_of_turn|>` if present — as additional `eos_token_id`s, and all three generate sites pass `eos_token_id=self.terminators`. Llama-3.1-Instruct ends each assistant turn with `<|eot_id|>`, **not** `<|end_of_text|>` (128001); without this the model "wants to stop" but the token isn't a terminator, so decoding runs to `max_new_tokens` and the tail degenerates into a `####N####N…` char-level repetition loop. The fix does **not** remove GSM8K loops (the loop is itself a wanting signal we keep), but it makes natural EOS possible and shifts the numbers. **Pre-eot numbers are not comparable to post-eot numbers.** The current data under analysis is the eot rerun: GSM8K answers in `gsm8k_eot/…`, hidden states under `RUN_TAG=phase1b_eot`. **MATH eot rerun has landed, and as of 2026-09-01 the MATH tree is CONSOLIDATED TO ONE VERSION at `RoleAnswer/llama3/math/`** (file `math_8B_11_20[_role].json`, field `generated`, gold in `gold_answer`); `run_math.sh` drives the No-CoT + α=0-CoT matrix itself (its step [2] passes `--cot`). **`run_math_cot.sh` no longer exists** — deleted 2026-06-09 in `0316de0`; older refs to it are stale, and the CoT ±4/−6 supplement cells were driven by `run_math_llama3_wp.sh` / `run_math_cot_llama3_wp.sh`.
<!-- Why: three parallel MATH trees coexisted with DIFFERENT token budgets and
overlapping cell names, so a wrong root silently produced plausible numbers from
the wrong batch; math_2048 also looked like the main line while being a SUBSET.
Evidence: md5 of all 8 math_2048 JSON identical to math_eot; math_1024 mdf_0 acc
39.67 vs 37.00. Scope: all Llama MATH reporting. -->
- **Llama MATH is ONE tree at `RoleAnswer/llama3/math/` (consolidated 2026-09-01) and every cell is `max_new_tokens=2048`, `bs=8`.** `math_eot/` was flattened up into `math/`; `math_2048/`, `math_1024/` and `math_oold/` were DELETED. `math_2048` was verified byte-identical to `math_eot` on all 8 shared JSON before deletion — it was a SUBSET (it lacked `mdf_neg4_cot` / `mdf_4_cot`, so §3.5's CoT ±4 table could never have been computed from it) and was never the main line despite the name. `math_1024` was a different budget (α=0 reads 39.67 vs 37.00) and `math_oold` older still; both are gone, so a budget-comparison would need a fresh run. `analyze_first_last_acc.py`'s default MATH root is now `llama3/math` with a fallback to the old nested `math_eot/` — pass `--math_root` only to point at a non-default tree.
  - **Current cells (7 neutral + 3 roles): No-CoT `mdf_{neg6,neg4,0,4}`, CoT `mdf_{neg4,0,4}_cot`, plus `expert`/`non_expert`/`math_expert` at α=0 No-CoT.** `MATH_DIRS` in `analyze_first_last_acc.py` carries `mdf_neg6` and `mdf_neg6_cot`; a cell absent from that dict is SILENTLY SKIPPED, not flagged — the table just comes back one row short. **The same trap bit GSM8K on 2026-09-02**: `GSM8K_DIRS` had no `mdf_-6_cot`, so a freshly collected cell simply did not appear. Register a new cell in the matching dict IN THE SAME COMMIT that collects it, and check the row count, not just that the script exited 0.
  - **The budget is 2048/bs=8 and the data is NOT truncation-bound — the two facts are separate and both matter.** Measured char-length distributions show only 1–4 samples within 2% of each cell's max, with no cluster, so MATH length and position readouts (`median boxed pos`, `gen_len`) are natural lengths. Contrast GSM-Hard Llama, which is 91–96% cap-hit at 768 and whose length readouts carry a ceiling caveat. Judge a cap by a CLUSTER at one length, never by a lone long sample (the same rule that removed 2000 from Llama's `ROUND_CAPS` and 1000 from Qwen's). **Keep new MATH cells at 2048/bs=8 anyway** — changing the budget forks the 口径 against the existing seven cells, which is exactly why `math_1024` existed.
  - **MATH is a COMPLETE 4×2 neutral matrix as of 2026-09-01** (`mdf_neg6` and `mdf_neg6_cot` both collected). Both curves are monotone `−6 > −4 > 0 > +4`: No-CoT 43.33 / 40.00 / 36.67 / 33.00, CoT **49.00** / 45.00 / 42.00 / 38.67. **`α=−6 CoT` 49.00 is the highest accuracy in the matrix.** −6 is the workpoint read from the frozen GSM8K record, **NOT re-searched** — it turns out to also be the observed best of the four points, which is a result, not the selection rule.
  - **FOUR statistical families, each with its own Holm denominator. They may NEVER be pooled:** (1) **P4 No-CoT fixed-workpoint, CROSS-MODEL Holm m=2** — llama −6 +6.67 pp raw p=.0245 **p_adj=.0489**, qwen +8 +2.67 pp p=.3581 p_adj=.3581; (2) **within-Llama No-CoT dose family, Holm m=3** — −6 +6.67 pp **p_adj=.0734**, −4 +3.33 .3697, +4 −3.67 .3697; (3) **within-Llama CoT dose family, Holm m=3** — −6 **+7.00 pp p_adj=.0225** (the one that survives), −4 +3.00 .5057, +4 −3.33 .5057; (4) **CoT gain across the four α, Holm m=4** — +5.67/+5.00/+5.33/+5.67, **all p_adj=.1718, none survives**. So "CoT lifts every dose by ~5 pp" is a consistent EFFECT SIZE, not a corrected significant result.
  - **The interaction is DESCRIPTIVE / EXPLORATORY and was NOT frozen before the −6 CoT cell was read.** `[CoT(α)−CoT(0)] − [NoCoT(α)−NoCoT(0)]` = +0.33 / −0.33 / +0.33 pp at −6/−4/+4, CI [−6.67,+7.67] / [−6.67,+6.00] / [−6.67,+7.33], computed by question-level joint paired bootstrap (never by subtracting four aggregate accuracies). **All three contain 0 → "steering 效应随 CoT 变化未检出"**, and the ±7 pp width means this is **NOT** equivalence evidence — do not use the interaction null to argue mechanistic independence, additivity, or that CoT and steering are the same lever.
  - **`corrected` / `corrupted` must satisfy `first_correct − last_correct = corrupted − corrected` in every cell — check it before writing a number.** A summary reading `first 49.0 / last 48.0 / corrected 0 / corrupted 0` is internally impossible; the true `mdf_neg6_cot` values are **147/144, corrected 0, corrupted 3**. Verified against the frozen extractor per item in all seven neutral cells.
  - **Equal discordant counts across two comparisons are coincidence, not a shared subset.** Table `3.5.1`'s `−6` and `+4` CoT-gain rows are both `40/23`: the SAME 300 items and the same index space, but different α conditions, and the discordant subsets overlap only 7/40 (0→1) and 3/23 (1→0). Never write "different question sets".
  - **Two limits travel with every −6 number.** The stored `mdf_{0,4,neg4}` and `mdf_{0,4,neg4}_cot` cells' physical GPU is unrecoverable (`summary_math_*.csv` has no device field), so **both** `−6 vs 0` and `−6_cot vs 0_cot` are **CROSS-RUN pairings**; and the No-CoT and CoT halves share the same 300 problems, so they are **not two independent task replications**.
  - **The behavioural panels ARE recomputed at −6 (2026-09-01) — every MATH table is now a full 4×2.** Two offline scripts had MATH hardcoded to three doses AND still pointed at the pre-consolidation `math_eot/` root: `analyze_cot_metrics.py` (`DOSE`/`COT` cell lists + `root=`) and `analyze_loop_anxiety.py` (its `dose` list + two `root=` defaults). Both fixed; `.bak` kept. So `−6` is computed by the SAME script, definitions and denominator as the other cells — never hand-filled. Rerun with `analyze_cot_metrics.py --task math --table {dose,cot}` and `analyze_loop_anxiety.py --task math --mode {loop,anxious_repeat}`.
  - **Adding −6 makes `commit_rate` a monotone column (92.7 / 87.3 / 86.0 / 85.3) — on `−4/0/+4` alone it reads flat.** Same for `≥2 Step` (239 at −6 vs 131–177 elsewhere) and `median gen_len` (3804 → 5719). **Two rows are NOT monotone and must be labelled so**: `premature (either)` is 16/13/29/26 (−6 above −4; at 13–29 per 300 a 3-item difference carries no direction — GSM8K's range is 195–232), and full-text compulsive repetition is 42/40/68/82, i.e. the negative arm has hit a floor and −6 does not push it lower.
  - **CoT shortens generation only at `α ≥ 0`.** 5557→5141 and 5719→4123, but `−6` is 3804→3820 and `−4` 4798→3864 — the low-α output is already short, so "CoT is shorter" must not be stated unconditionally.
  - **Difficulty stratification at −6 is L1 86 / L2 65 / L3 58 / L4 31 / L5 20 %.** `−6` is the per-level best in four of five levels; **L2 is the exception** (65% vs −4's 73%). Cells are 11–40 items, so this is small-sample noise, NOT a dose reversal at that difficulty — do not read a per-level inference off it.
  - **`commit_rate` / gap / `corrected` / `corrupted` are output-submission stability readouts and may not be cited as "premature commitment is lowest"** — that needs candidate-formation timing (`early_candidate`, pre-marker content), not marker bookkeeping. **MATH's boxed position is likewise a NEGATIVE CONTROL** (§4.3: 1.0× dynamic range across nine α, because `\boxed{}` sits at the end by LaTeX convention), so its monotone shift with α is a writing-position fact, not commitment delay.

- **`analyze_first_last_acc.py` is the AUTHORITATIVE accuracy source** (in the offline `RoleAnswer/` workspace). Paper-reported ACC is computed there offline — first-`####` (GSM8K) / **first**-`\boxed{}` (MATH) + fallback chain + normalization — **not** from the inline `correct_*` / `pred_answer` fields the generation scripts store (those are process-state only and easy to mis-read: simple string-compare on `pred_answer` undercounts). It also emits the first−last gap and the 改对/改坏 (fixed/broke) commitment split. Always cite ACC from this script.
- **No "honest" on GSM8K roles.** `honest` belongs only to E-option ("I am not sure") MMLU suites. GSM8K has no E-option, so every role uses the `neg` template (`"Now you are {character}."`), neutral uses `neutral`. `get_answer_regenerate_gsm8k.py` now bypasses `construct_prompt`'s default(+honest) branch and selects neutral/neg directly — this makes it produce **identical prompts to `track_dopamine_signal.py`** (the two were silently diverging: regenerate said "an honest expert", track said "an expert").
- **`####` final-answer directive in all GSM8K templates** (`build_gsm8k_default_suite`, both CoT and No-CoT, all 3 keys). Without it the model never emits `####`, runs to `max_new_tokens` (98–100% of No-CoT samples hit the 512 cap with no commit), and `extract_gsm8k_answer` falls back to "last number in text" — pure noise. The reversal *non_expert (63.7%) > expert (58%) > neutral (53%)* in the old results was an extraction lottery, not a real role effect. Symmetry preserved: No-CoT vs CoT still differ only by the `Let's think step by step.` line.
- **`####` wording matters — use `"Provide your final numeric answer after '####'."`** (the neutral, history-aligned wording). A pushier variant (`"Give your final answer as a single number after '####'."`) induced early-####抢答 (model commits an un-reasoned guess; expert 72% early-####, No-CoT acc collapsed to 34%). The pushy wording is kept conceptually as a future positive-control ablation (prompt wording vs α-steering as two levers on commitment timing). `extract_gsm8k_answer` takes the FIRST `####`, so an early 抢答 locks the wrong answer even when the correct one appears later — another reason to keep wording neutral. **Run matrix:** `run_gsm8k.sh` is now a **same-machine backfill** (2026-06-04, machine 182). WHY: bf16 greedy is **not byte-reproducible across GPUs** (different cuBLAS accumulation → logit ties flip → whole CoT chain diverges; a +4 re-run on a different box gave 205/300 sample-text mismatches), so the whole dose-response curve must live on ONE machine. It re-runs (per wording) the OLD-machine cells — No-CoT α=0 all-4-roles → `mdf_0`, No-CoT α=−4 neutral → `mdf_-4`, CoT α=0 neutral → `mdf_0_cot`; pushy re-runs the full old pushy matrix (α=0 4-roles + ±4 + CoT 0). `WORDING` defaults to `plain pushy`; pushy writes to `_pushy` dirs. `±2/±6/±8`, plain `+4`, CoT `±4` were already collected on 182 (not re-run). **Cross-machine rule: never compare ACC across machines/bs — 184 (HS, bs=1) and 182 (regenerate, bs=24) ACC are not comparable; paper ACC is the single-machine 182 dose table only.** `run_math.sh` has **no** pushy variant — MATH uses the neutral `\boxed{}` directive only. **184-vs-182 divergence anatomy (re-verified 2026-06-28).** The signal_eot (184, track, bs=1) vs 182 (regenerate, bs=24) ACC gap has THREE mixed-in variables, never isolated: (1) cross-machine bf16 (the dominant, random one), (2) bs=1 vs bs=24 padding (~2%), (3) — newly identified this session — input-vs-output steering injection on the OLD track code (now fixed; see the hook-alignment bullet). The gap's FINGERPRINT is **scatter, not a monotone |α| ramp**: steady-state cells (−6/−4/0/+4/+8) agree to ≤2.6% (α=0 is exactly 60.0/60.0), while −8 and +6 each diverge ~9–10% with +4/+8 fine — that random "hits some cells, spares the neighbour" pattern is the bf16 tie-flip signature, NOT the systematic offset a fixed injection-position difference would produce. So **read signal_eot's own inline accuracy (= authoritative first-#### recompute, they match) when plotting 184 signal curves; cite the 182 first-#### table for paper ACC; never配 184 acc onto a 182 curve.** **What the two batches DO share is the dose SHAPE** — both are the same asymmetric peaked response with the discrete optimum at **α=−6** — which is why 184 can carry same-batch signal–behaviour alignment at all: the alignment rides on the shape, never on the absolute level. This is the one sentence `AdaptiveThinking.md` §4.0 keeps; the anatomy above stays here. signal_eot acc is recomputed correctly via `RoleAnswer/analyze_first_last_acc.py --gsm8k_root llama3/gsm8k` (default root is the stale `llama3/gsm8k_new` → "no files found"; always pass `--gsm8k_root llama3/gsm8k`).
- **Cross-model Mistral GSM8K** (2026-06-09). `run_gsm8k_mistral.sh` is the Mistral-7B-Instruct-v0.3 dose curve — its own script (do **not** edit Llama's `run_gsm8k.sh`, whose plain/pushy/"already-on-182" matrix is Llama-specific). `MODEL_NAME=HS_PREFIX=mistral` (→ mask dir `mistral_non_logits`), layers **14–22** (Mistral middle band, where its NMD mask was computed — steering layers MUST equal the mask's layers), full `-8→+8` sweep, neutral-only, plain wording. Mistral-7B-v0.3 loads as plain `AutoModelForCausalLM` (NOT the Mistral3 multimodal branch). **No-CoT collapsed to ~14–19% acc across ALL α** — diagnosed as an **extraction floor, not a steering null and not pure incapacity**: gold appears in the generated text in ~51% of mdf_0 samples (the true ceiling), but Mistral ignores the `####` directive (~65–70% of samples never emit `####`, preferring `N bolts\nExplanation:…` free-form or — at α=−8 only — degenerating into MATH-style `\boxed{}`), so the first-`####`+fallback chain recovers only 19%. The one clean signal in No-CoT is **commit% rising with +α** (+6=32.7%, +8=41.3%, vs ~11% mid-band: strong +α shortens output and forces `####` — same "early-commit/抢答" direction as Llama). Hence the switch to **CoT-only** for Mistral (CoT makes it reason + close properly, lifting commit% off the floor). Authoritative ACC: `analyze_first_last_acc_mistral.py` (in `RoleAnswer/`, reuses Llama's `first_last_stats`/`all_hash`/`fallback_gsm8k`; `--cot` reads `mistral/gsm8k_cot/`). Do **not** hand-write a Mistral-only extractor unless reporting No-CoT cross-model ACC explicitly — CoT is expected to dissolve the loss, and forking extraction risks Llama comparability (`utils.py` is the shared source).
- **Cross-model Qwen2.5 GSM8K** (2026-08-20). `run_gsm8k_qwen25.sh` + `check_gsm8k_qwen.py` — its own launcher for the same reason as Mistral's (Llama's `run_gsm8k.sh` plain/pushy/4-role/182-backfill matrix is Llama-specific and frozen). This is a **direct replication** of the Llama dose curve, not a redesign: same 300-question benchmark, same driver, same templates, neutral / plain / greedy / `max_new_tokens=768` / bs=24. Four Qwen-specific facts, none inherited: layers **16–21** written as the exclusive `16-22` (**L=6**; Llama's `11-20` is L=9 and does NOT transfer), mask `nmd_0.5_16_22_7B.npy`, `size=7B` (28 decoder layers), and Qwen's `<|im_end|>` + LIST `generation_config.eos_token_id` (already unioned by `_build_terminators()` — verify, do not "fix"). Steps: `--check` (technical pre-flight only — model/tokenizer/mask/band/paths/fires; makes **no** judgement about results and has no pass/fail-then-block gate), `--baseline` (α=0 → the FINAL No-CoT `mdf_0`, read its raw output by hand), `--nocot` (the other 8 α; deliberately excludes α=0 so `mdf_0` is never re-run or overwritten), `--cot` (−4/0/+4), `--cot6` (−6/+6, extends that same curve). **Every α of one curve must stay on ONE card** (bf16 greedy is not byte-reproducible across GPUs); No-CoT and CoT write different `--ans_file`s (`answer_mdf_gsm8k` / `_cot`) so the two curves can run on two cards in parallel.
  - **Injection lands on token `220` = `' '`** — the trailing space of the `Answer: ` anchor, identical for No-CoT and CoT. Measured on the real tokenizer, not assumed.
  - **The prefill path DOES count fires** (`_regenerate_prefill_only` increments `_steering_fire_count += B * n`), so `--check` verifies `L × B × tail` (6×B×1) for real. α=0 passes a REAL all-zero matrix (`diff = np.load(mask) * alpha`, unconditional at `get_answer_regenerate_gsm8k.py:129`) — hooks DO register and the zero add DOES execute; fires read 0 only because `_layer_is_steered` is False on an all-zero row. **This differs from `get_action_regenerate_gsm8k.py`, whose α=0 really does take a no-mask `get_logits` path** — do not generalize one to the other. Practical consequence: a missing mask file blocks the α=0 cell too.
  - **α=0 RESULT (N=300, `first_acc` = 68.0%, `last_acc` = 73.33%): Qwen's first−last gap is NEGATIVE (−5.33), the OPPOSITE SIGN to Llama's (+4.7 at α=0).** 改对 **21** / 改坏 **5** (Llama α=0: 0 / 14). Llama's multi-marker samples are over-wanting→loop damage, so taking the first `####` is the forgiving choice; Qwen instead answers BEFORE reasoning (median normalized first-`####` position **0.0031**, 85.3% inside the first 1% of the generation) and then often **corrects itself** in prose (`#### 70 seems incorrect … The correct answer is 140. #### 140`). 口径 stays frozen regardless: **`first_acc` is the MAIN readout** (identical to Llama production extraction), `last_acc` is the sensitivity / self-correction readout, and gap + 改对/改坏 are a cross-model BEHAVIOURAL difference — never a substitute metric.
  - **`multi_marker` is NOT homogeneous — read `max_n_markers` beside it.** Qwen α=0's 95 multi-marker samples mix genuine two-marker self-correction with Llama-style **answer-candidate oscillation** (one sample carries 77 markers, tail cycling `#### 109 / #### 55 …`). A negative gap therefore does not imply "all self-correction".
  - **Two commit counts, deliberately distinct — do not conflate.** `commit_pct` (81.0 at α=0) counts a PARSEABLE `#### <number>` via `all_hash()` and is the Llama-comparable one; bare-substring `####` presence is **100.0**. The 57-sample difference is Qwen writing `70\n####\n\n` — marker emitted, no number after it. Both numbers are correct; `analyze_first_last_acc_qwen.py` prints them side by side under separate names.
  - **This is NOT a Mistral-style extraction floor, so do NOT switch to CoT-only.** gold-in-text 87.0% vs `first_acc` 68.0% (19pp) with **0/300 missing markers**; Mistral was 51% vs 19% with 65–70% never emitting `####`. `gold_in_text` is a PERMISSIVE DIAGNOSTIC only (upward-biased, matches incidental digits) — its sole job is that discriminator, never an accuracy.
  - Authoritative ACC: **`analyze_first_last_acc_qwen.py`** (in `RoleAnswer/`, not in git). It **imports** `analyze_first_last_acc`'s `all_hash` / `norm_gsm8k` / `fallback_gsm8k` / `first_last_stats` rather than reimplementing them, so Qwen numbers are produced by the same code as the Llama table. `--nocot_dir` / `--cot_dir` are configurable because the server writes `answer_mdf_gsm8k[_cot]` while synced local copies are usually `gsm8k[_cot]`.
  - **Qwen accuracy / marker 口径 (frozen 2026-08-20).** `first_acc` is the MAIN cross-model readout; `last_acc` is sensitivity only and MUST include the shared fallback chain. The authoritative α=0→+8 `last_acc` is **73.33→80.33 (+7.00pp)**. The once-reported strict-marker-only 58→78 is invalid as accuracy evidence: its extra 15.33pp is entirely the no-parseable-marker count falling from 57 to 9. Keep `commit_pct` (parseable `#### <number>`) separate from bare `####` presence; at α=0 they are 81.0% and 100.0%, respectively. `gold_in_text` is a permissive diagnostic, never accuracy or proof that the model formed a correct solution.
  - **Qwen CoT completion modes.** `--cot` (−4/0/+4), `--cot6` (−6/+6), `--cot-rest` (the six missing from the original triple), and `--cot-rest4` (four missing if `--cot6` landed) preserve the existing `answer_mdf_gsm8k_cot` tree. Because the original triple's physical GPU cannot be recovered from its summary CSV, **`--cot-full` is the authoritative option**: it runs all nine α sequentially on one pinned card into the clean `answer_mdf_gsm8k_cot9` tree and leaves the old cells untouched as a consistency check. Do not split one curve across cards.
  - **Frozen detector transfer.** `RoleAnswer/analyze_loop_anxiety_qwen.py` imports the existing `analyze_loop_anxiety` regex / gate / bucket objects unchanged (`set_task()` is not called; source sha256 was checked before/after). Do not adapt its regexes inside the comparable result. One column is a confirmed wording false negative: frozen `I made a mistake/error` gives zero in all Qwen cells, while Qwen externalizes the same review as `the provided solution/answer ... incorrect`; therefore `n_made_mistake` is NOT cross-model comparable, though the broader `self_doubt` bucket still fires through other cues. The blind audit is 9×20 rows with opaque `cell_id`, shuffled rows and a separate key; it audits wording drift only and MUST NOT be used to estimate phenotype prevalence or merged with detector rates.
  - **Cross-model α is nominal, not a calibrated common dose.** Llama and Qwen use different masks, layer counts (L=9 vs L=6) and activation scales, so an equal numeric α is not an equal effective intervention. Compare models on behavioral commitment state (commit position, candidate stability, task-boundary retention, post-commit revision), not by placing raw α on one universal biological-dose axis. Full results and concise conclusions live in `AdaDopamine_gsm8k.md` §4.
  - **Qwen MATH + high-dose are SEPARATE launchers, and their generation budgets differ on purpose.** `run_math_qwen25.sh` inherits the **MATH** budget (`max_new_tokens=2048`, `bs=8`), NOT the GSM8K port's 768/24 — "Qwen config equals GSM8K" means only the Qwen-specific facts (model, band `16-22`/L=6, mask `nmd_0.5_16_22_7B.npy`, prefill-only steering); reusing 768 would truncate MATH solutions and manufacture an extraction floor that destroys comparability with Llama MATH. Steps `--baseline` / `--nocot` / `--full`; **`--full` exits 2 when `mdf_0` already exists** (it includes α=0, so running it after `--baseline` would regenerate a finished cell — the intended follow-up is `--nocot`, which excludes α=0). `run_gsm8k_qwen25_highdose.sh` runs `0/+6/+8/+10/+12` into its OWN `answer_mdf_gsm8k_highdose` dir and deliberately RE-RUNS 0/+6/+8 rather than reusing the frozen cells: the sequence must be internally paired on one card, and the frozen cells' device provenance is unrecoverable (`summary_*.csv` records model/size/alpha/layers/task/role/acc/suite/cot — **no device field**). Two-digit α parses correctly (`utils.parse_configs` → `mdf_10`/`mdf_12`, no collision with `mdf_1`/`mdf_2`; verified).
  <!-- Why: the earlier form of this bullet concluded from a terminal boxed position that MATH
  cannot test ordering; raw-text reading refuted it (the early commitment is an UNBOXED bare
  candidate, which the boxed-position probe cannot see).
  Evidence: qwen2.5/math first-line candidate rate 87.0%->3.3% monotone over alpha -8..+8.
  Scope: Qwen MATH; re-check before reusing the probe on any new task. -->
  - **A terminal `\boxed{}` does NOT mean "no ordering to flip" — CORRECTION (2026-08-21).** The
    boxed-position probe reads `posN_med` 0.96–0.98 in all nine Qwen cells (first `\boxed{}` in the
    last 2–4% of the generation), and an earlier version of this bullet concluded from that that
    MATH could not test the GSM8K commitment-ordering mechanism. **That conclusion is withdrawn.**
    Reading the raw text shows the early commitment is an **unboxed bare answer candidate on the
    first line**, which the boxed-position probe is structurally blind to. The probe measures only
    where the *marker* sits, not where the model first commits. So the correct reading is: the
    boxed MARKER has no ordering to flip; the model's answer CANDIDATE does. Run a candidate
    detector on the raw text before concluding a task lacks an ordering channel.
  <!-- Why: CLAUDE.md and analyze_first_last_acc.py's docstring both said MATH production takes
  the LAST boxed; utils.extract_math_answer() takes the FIRST, and its own comment says why
  (tail loops pollute the last). Reporting last would fold an alpha-VARYING pollution into the
  dose effect -- Qwen a=-8 has nbox_max=131 and reads 4pp lower on last than on first.
  Evidence: three-way replay inline == utils_prod exactly in all 9 cells; |utils - first_acc|
  <= 0.67pp vs |utils - last_acc| 0.33-4.33pp.
  Scope: all MATH reporting, both models. -->
  - **MATH口径 is `first_acc` MAIN / `last_acc` SENSITIVITY — the opposite of what this file said
    before 2026-08-21.** `utils.extract_math_answer()` takes the **FIRST** `\boxed{}` (its comment:
    "tail loops pollute the last"), and `utils.extract_boxed(which=…)` is not a free choice — `first`
    is correct for model PREDICTIONS, `last` for the dataset GOLD. Inline `correct` reproduces
    `utils` exactly in all nine Qwen cells, so **`first_acc` is the column sharing a definition with
    production**; `last_acc` stays reported as a tail-pollution sensitivity, never as the headline.
    GSM8K is unaffected (`first_acc` was already MAIN there, for the independent reason that
    `extract_gsm8k_answer` takes the first `####`).
  <!-- Why: both gaps make offline first_acc read LOWER than the model's actual answer,
  so a future extractor pass could "fix" them and shift every frozen MATH number at once.
  Evidence: backup/qwen_step2/math_strat_frozen.py docstring; idx 253 disagrees in 9/9 cells.
  Scope: MATH extraction, both models. Do not patch without re-freezing. -->
  - **Two pre-existing MATH extractor gaps, located 2026-08-21, deliberately NOT patched.** (a) **`norm_math` does not normalize a leading zero** — gold `.35625` vs the model's `0.35625` reads as wrong. This ONE item (`idx 253`) disagrees with inline `correct` in **all nine Qwen cells** and is 9 of the 12 total disagreements. (b) **An empty `\boxed{}` can occupy the first-marker slot** — one generation writes `\boxed{}` while *discussing* the format ("...should be boxed as follows") and only then boxes the answer three times, so `all_boxed` returns `['', '6', '6', '6']` and the first-口径 takes the empty string. Empty first markers are 3/2700 library-wide. **Consequence for any analysis that scores MATH correctness itself: reuse `analyze_first_last_acc`'s extractor/normalizer/fallback and assert your total reproduces the published cell count** — inline `correct` and offline `first_acc` differ by 1–2 items in EVERY Qwen MATH cell (offline always lower), which is enough to make a stratification's group weights disagree with the section it sits in. `backup/qwen_step2/math_strat_frozen.py` is the worked example. Patching either gap would move every frozen MATH and GSM8K number, so it needs its own re-freeze, not a drive-by fix.
  - **The Yerkes-Dodson right arm is established by ACCURACY FALLING, not by contamination rising.** Qwen `+8` already carries 60.3% document-continuation contamination (`You are an AI assistant…`, vs 17.3% at α=0) and it is **not** the source of the accuracy gain — the clean subset at +8 reads 83.97% vs α=0's clean 70.40%. So rising contamination alone must never be reported as overload evidence; that is exactly the prediction `run_gsm8k_qwen25_highdose.sh` exists to falsify.
  <!-- Why: the launcher pre-registered "accuracy must peak and fall"; it did not, and a
  later reader seeing 60% contamination at +8 will otherwise re-derive the retired
  overload reading. Evidence: backup/qwen_step2/gsm8k_extended_curve.txt + highdose_overload.py.
  Scope: Qwen GSM8K only -- MATH DOES fall at +8. -->
  - **HIGH-DOSE RESULT (2026-08-21): the prediction was FALSIFIED — GSM8K has NO right arm in −8…+12; the curve SATURATES.** `first_acc` +6 78.00 / +8 86.00 / +10 88.33 / +12 88.67, with every high-dose pair n.s. (`+8→+10` p=.248, `+8→+12` p=.268, `+10→+12` p=1.000, exploratory uncorrected) while +10/+12 are both p_adj<1e−4 vs α=0. **The escape hatch is closed too**: contamination PLATEAUS at +8 (17.7→44.7→60.3→58.7→59.7%) rather than climbing, empty generation is 0.0% at every cell, truncation ~1%, and the clean-subset accuracy RISES (70.85→78.31→84.03→87.90→84.30) — so the flat total is a real plateau, not an average over a healthy and a broken half. Extreme repetition keeps falling (`####`-count p99 61→7). Mechanism variables are saturated by +10 (early candidate 2.7/3.7%, posN .79/.80, commit% 98.3), so more dose measures something else, not this curve. **Consequence for wording: the inverted-U is a MATH property (it falls 5pp at +8), NOT a general Qwen property** — write "the working point and usable band are task-dependent", never "there is a common right arm". Same monotone-saturating shape as betting's `mean_bet`. `+10/+12` come from a different batch than the frozen 0…+8 cells (cross-run, not strictly paired); at temperature=0 the drift is small — this run re-read +8 at exactly 86.00, matching the frozen cell.
  <!-- Why: posN is the PRIMARY commitment readout on GSM8K (251x range) and a NEGATIVE
  CONTROL on MATH (1.0x); reading MATH's flat posN as "no ordering change" is the error
  that once retired the whole MATH ordering line.
  Evidence: AdaDopamine_gsm8k.md Table 4.3b; RoleAnswer/backup/qwen_step2/commit_columns.py.
  Scope: every commitment claim, both tasks. -->
  - **Commit-position `posN` and `early-candidate%` are NOT interchangeable, and which one is the readout FLIPS between tasks.** Nine-alpha dynamic range: GSM8K `posN` .003→.754 (**251×**), GSM8K-CoT .003→.825 (**275×**) — it IS the commit, so it is the main readout there. MATH `posN` .960–.978 (**1.0×**, span .018), MATH-CoT .964–.973 — `\boxed{}` sits at the end by LaTeX convention, so its position is fixed by writing convention, not by when the model commits. **On MATH, `posN` is a NEGATIVE CONTROL and `early-candidate%` (85.0→10.3) is the commitment readout**; `first_acc ≈ last_acc` says only that later markers rarely flip correctness and must NEVER be read as "reasons before answering". Recompute with `RoleAnswer/backup/qwen_step2/commit_columns.py`.
  - **MATH-CoT RESULT (2026-08-21, nine alphas, `--cot-full`, one card): the peak does NOT move — it stays at `+6` on both curves.** The pre-registered "CoT moves the peak to +4" is REJECTED (+4 is one of two negative-ΔCoT points). But per the pre-registered falsifiable form (right-arm position / compression point / candidate-suppression threshold — all three static ⇒ "raises the baseline only"), **two of three moved**, so it is not baseline-only: the suppression threshold shifted RIGHT (CoT keeps 9–12pp more early candidates at +2/+4/+6, converging only at +8) and compression is SHALLOWER (pre-commit chars at +8: CoT 1002 vs No-CoT 828); the right arm did not move (both +8). **Two Holm families, declared separately, never pooled** — Family A (CoT vs its own α=0, m=8): all eight p_adj=1.0000; Family B (CoT vs No-CoT per α, m=9): none survive, largest is −4 at +7.00pp, p=.0065, **p_adj=.0581**. Point estimates and the full curve are retained; failing correction limits claim strength, it does not delete the pattern. **The flattening is HEADROOM, not mechanism loss** — CoT's α=0 is already 63.00 against a shared 66–68 peak, and early-candidate still collapses 71.3→10.3 under CoT, so **never write "CoT eliminates the RSN effect"**. Say "可提升空间被明显压缩", NOT "接近能力上限" (no experiment approached the latter). Right-arm attribution, consistent across both curves: the Level-5 fall at +8 accompanies **pre-commit reasoning shortfall** (lost items' pre-commit chars 1398→886) and NOT post-commit rewriting (post-commit chars stay 27–34, truncation ~0%). Wording boundary: the overall +6→+8 fall is **DESCRIPTIVE only** (66.00→64.00); the clearer right arm is on **Level 5**, so do not write "CoT establishes a right arm across the whole curve". Recompute with `RoleAnswer/backup/qwen_step2/math_cot_curve.py` (nine-α table) + `math_cot_stats.py` (the two Holm families; it `exec`s the curve script, so keep them side by side) + `math_cot_rightarm.py` (Level-5 attribution) + `math_cot_boxed_tail.py` (the §4.3.2 tail). All import the extractor from `analyze_first_last_acc` rather than reimplementing it — an independent copy is how the inline/offline口径 gap opened on MATH.
  <!-- Why: precision 1.000 makes it tempting to treat the frozen rate as exact; the 3 FN are
  all one dose-dependent shape, so the high-alpha MAGNITUDE is understated even though the
  direction holds. Evidence: backup/qwen_step2/earlycand_audit_RESULT.txt.
  Scope: every MATH early-candidate number. GSM8K is unaffected -- verified, not assumed. -->
  - **BLIND AUDIT of `earlycand-v1` (180 rows, α/task hidden): precision 1.000, recall 0.976, agreement .983 — FP 0/180.** So the flagged set carries no false-positive contamination. **All three FN are `math/+8` and share one shape: the answer IS stated first but inside a LONG prose sentence** (first lines 107/114/120 chars vs `MAX_LINE_CHARS=60`) — rule (1) behaving as specified, not a bug, but **dose-dependent**: first-line >60 chars AND already containing `\boxed` is 0.3% at α=0, 6.7% at +6, **14.7% at +8** (MATH-CoT 0.3/4.3/11.0%). Median first-line length of that group at math/+8 is 232 chars (65–556), so **no plausible cap recovers them** (cap=80 is already in `--sensitivity`; wider would swallow genuine reasoning openings). **Reported as a BOUND, NOT retuned** — the published curves stay as the frozen definition produced them. Consequence: frozen rates are a **LOWER bound** on answer-first behaviour with the gap growing in α; direction unchanged (67.3→10.3 frozen, 67.7→25.0 including this shape) but the high-α **magnitude is understated**, so **never read "+8 suppresses to 10.3%" as "89.7% reason first"**. **GSM8K is NOT affected, and the check that shows this inverts a naive reading**: the same crude probe flags 17.3%/27.3% of GSM8K at +6/+8, but splitting by WHERE the answer sits gives **0 front-loaded cases in every GSM8K cell** — those lines are reasoning ending in `… = 17 stickers. ####17`, so counting them would invert their meaning. `####` is terminal by construction and cannot be front-loaded.
  - **`freeze_qwen_artifacts.py --check` iterates a HARDCODED alpha list, so a cell missing from it is INVISIBLE, not flagged.** `mdf_10`/`mdf_12` landed in the same `gsm8k/` tree and `--check` kept printing a clean `OK: 27 artifacts` beside two unfrozen cells. `ALPHAS_GSM_HIGH` now covers them (29 artifacts); the re-freeze was verified purely additive — all 27 prior hashes byte-identical. When a run extends an existing tree rather than creating its own, extend the freeze list in the same commit.
- **`record_template` mislabels GSM8K role templates.** It logs `templates["default"]` (with honest) for non-neg roles, but the code actually constructs the `neg` prompt (no honest). The `template` field in result JSON is therefore unreliable for GSM8K roles — trust the code path, not the logged string.
  <!-- Why: every other curve peaks at -6, so a reader will assume GSM8K CoT does too and
  cite the wrong workpoint; the fallback is that -6_cot looks fine on every format diagnostic.
  Evidence: RoleAnswer/wps_region.py (near-optimal regions); AdaDopamine_gsm8k.md 1.2 / 2.3.
  Scope: Llama GSM8K CoT only. -->
- **GSM8K CoT peaks at `−4`, NOT at the `−6` workpoint — the ONLY condition where the two disagree (collected 2026-09-02).** Offline `first_acc`: `−4_cot` **85.0** > `−6_cot` **75.3** > `0_cot` 69.0 > `+4_cot` 59.7. Within the Llama CoT dose family (Holm **m=3**, never pooled with the No-CoT family or P4's cross-model m=2): `−4` +16.00 pp **p_adj=4.0e−07**, `+4` −9.33 pp p_adj=.0133, and **`−6` +6.33 pp p_adj=.0503 — not significant**. Direct `−4 vs −6` is +9.67 pp, discordant 42/13, p=1.1e−04.
  - **The reversal is real, not a parser artifact.** The four-cell paired DiD `[(CoT−4)−(CoT−6)] − [(NoCoT−4)−(NoCoT−6)]` is **+14.67 pp**, bootstrap 95% CI **[+7.67, +21.67]**, sign-flip permutation **p<1e−4**; the `last_acc` sensitivity is same-signed (+13.00 pp, [+5.67, +20.33], p=.0005). Every format diagnostic of `−6_cot` sits inside its three siblings' range (no-marker 62.3%, degenerate tail 95.0%, median 2128 chars), so nothing is broken — the accuracy is simply lower.
  - **The mechanism candidate is premature LOCK-IN, not miscomputation, and it stays post-treatment evidence.** `−6_cot`'s answer-first rate is **40.7%** (46/113 committed) against 15.8/19.5/0.0% at `−4`/`0`/`+4`, and **29 of the 42 items `−4` gets right and `−6` gets wrong (69%) open with a wrong `####` and then derive the gold later in the same generation** — production takes the FIRST `####`, so the correct answer is present but unscored. Consistent with the same signature P3-supp recorded at GSM-Hard CoT `−6` (58.3%), where accuracy nonetheless ROSE +6.00 pp — so a high answer-first rate is not by itself a degradation marker.
  - **Do NOT generalise this into "CoT needs a smaller |α|".** Evidence is mixed: Llama GSM8K supports it, **Llama MATH does not** (both conditions peak at `−6`), Qwen GSM8K CoT is weakly consistent (`+6` vs `+8` p=.371, not separated). Frozen wording: *CoT may compress the effective dose range or move the best workpoint toward a smaller absolute dose, but this depends on model and task and is not a uniform rule.* `|α|` is comparable only within one model, mask and band.
  - **Report a near-optimal REGION, not an argmax** — `RoleAnswer/wps_region.py` recomputes all six curves through the frozen extractor chain and asserts that 10 published values reproduce exactly (it caught two live defects while being written: a missing `fallback_math`, and a glob that scored `mdf_0/math_8B_11_20_expert.json` as if it were neutral). Measured regions: Llama GSM8K No-CoT {−6,−4}, **GSM8K CoT {−4}**, MATH No-CoT {−6,−4}, MATH CoT {−6,−4}, Qwen GSM8K No-CoT {+8}, Qwen GSM8K CoT {+6,+8}. **Four of six "optima" are intervals, not points.**
  - Pairing is CROSS-RUN (the stored CoT cells' device is unrecoverable), same caveat as the MATH `−6` cell.
- **No-CoT is the main condition**; CoT is the contrast control only.
- **Pass roles as full character strings** to `--roles` (e.g. `"an expert,a non expert,a primary school teacher"`), matching `utils.ROLE_TO_CHARACTER` values, so prompts align across scripts.
- **`get_answer_gsm8k.py` was deleted.** Its pure-baseline role is covered by `get_answer_regenerate_gsm8k.py` with a `0-<s>-<e>` config: `diff = mask*0` is a no-op, so α=0 regenerate == batched `generate`. Run baseline + ±4 steering in one regenerate pass (`configs="0-11-20 4-11-20 neg4-11-20"`) for maximum comparability. NOTE: regenerate steering is **prefill-only** (static push on the last prompt token); decode-time per-step steering is `closed_loop_gsm8k.py --plan static`. Different experiments — don't conflate.
- **Two generation paths to validate**: bs=1 (`track_dopamine_signal.py`, also emits signal) vs batched (`get_answer_regenerate_gsm8k.py` α=0). CLAUDE notes ~2% Llama acc gap from padding; run both to quantify it.
- **Phase-2 controllers need their own bs=1 baseline**: compare `closed_loop_gsm8k.py --plan {static,A,...}` against `closed_loop_gsm8k.py --plan none` with identical generation args. Do not treat the batched regenerate baseline as controller-comparable.

## Offline analysis workspace path

The offline analysis workspace was renamed `RoleAnswer_non/` → **`RoleAnswer/`**, now at **`/Users/paveenhuang/Documents/RSNResult/RoleAnswer/`** (relocated 2026-07-16 from `~/Downloads/RSNResult/`). Scripts there (e.g. `analyze_multi_metric.py`) and older doc references that still say `RoleAnswer_non/` or the Downloads path are stale — use the Documents path. The **superseded** `signal_eot/` tree (dated 5/30, old layer-offset mask + noisy prompts) has been replaced by the current **`llama3/dopamine/signal/`** set: output-side prefill steering, correct layer alignment, α-dose + roles + CoT, three file classes (see the workspace section above). Cite `signal/`, not `signal_eot/`.

## Server / data layout

Current GSM8K re-runs run on `/data1/paveen/Dopamine/` (server). Only code is in git; `components/`, `benchmark/`, `llama3/dopamine/`, H5 hidden states, and JSON answer dumps are not. Older experiments still have hard-coded `WORK_DIR=/data1/paveen/RolePlaying`; migrate them only when re-running that experiment family.

<!-- Why: a launcher guard that required the stored comparison cell to be present on the
server refused a correct run twice; the server had been cleared, as it always is.
Evidence: run_wps_llama3.sh / run_wps_gsm_hard.sh headers; run_gsm8k_cot_llama3_wp.sh 53f39e9.
Scope: every launcher guard that tests whether a cell already exists. -->
**THE SERVER IS SCRATCH SPACE — a finished cell is downloaded to the local `RoleAnswer/` tree and may be gone from the server at any time.** Runs move between machines, so the server holds whatever the current job needs and nothing more. Two consequences for launcher design: **(a) never gate a run on the presence of the stored comparison cell** — pair OFFLINE, in `RoleAnswer/`, not on the server (an overwrite guard on the launcher's OWN output dir is still correct and should stay fail-closed); **(b) a stored cell's physical GPU is unrecoverable** — `summary_*.csv` records model/size/alpha/layers/task/role/acc/suite/cot and **no device field** — so any new-vs-stored contrast is a **cross-run pairing** no matter which device the new cell uses, and pinning one card cannot convert it into a same-card pairing. Record the device with the result instead of constraining it.

<!-- Prospective project-wide device policy, requested 2026-09-03. This changes
launcher policy, not the provenance of already-completed experiments. -->
**GPU assignment is NOT a project-wide experimental constraint for future runs.** Different cells in the same curve or paired comparison may run on different physical GPUs or machines; launchers must not require a particular GPU, require cells to share one GPU, or reject a run solely because `CUDA_VISIBLE_DEVICES` is unset or lists multiple devices. Record `CUDA_VISIBLE_DEVICES`, host, and whether the model was single-device or sharded as provenance. Pairing means alignment by the frozen item identity/order, **not** hardware identity. Because bf16 greedy output may vary across hardware, report a new-vs-stored comparison as cross-run and include this limitation; do not present unrestricted hardware placement as byte-level reproducibility. Earlier sections that say an already-completed experiment used one card remain historical provenance and are not retroactively rewritten.

## Environment

```bash
bash setup_env.sh   # creates conda env "roleplaying" (py3.10) + bf16/CUDA stack
```

Mistral3 needs `transformers` from main + `mistral-common>=1.8.6` (already in `setup_env.sh`).

## Manifold pilot (Llama CLOSED; Qwen last-prefill is the only open step)

> **`AdaManifold.md` is the results document — full tables, interpretation and claim boundaries live there and are NOT duplicated here.** This section is the implementation, provenance and runbook: what was run, what guards it, and which failures are already paid for.
>
> **Scientific verdict (one paragraph).** Llama last-prefill geometry shows the negative arm is **approximately a one-dimensional scalar family** (cos(−8,−6)=0.989, residual 2.1%, `k`=1.379 matching the independent magnitude ratio 1.394) and the positive arm is **partially anti-aligned with a substantial orthogonal residual** (cos≈−0.66, `cos²`≈44%). So steering is piecewise, not one global scalar gain, and **α=−6 reaches a working region while α=−8 overshoots along the same axis**. **No stable incremental behavioural predictive value was detected**, and **last-prefill geometry did not stably extend to commit-aligned decode**. The line is therefore CLOSED as last-prefill explanatory geometry. Llama is not being extended further — no TLE, no UMAP, no per-layer sweep, no extra doses, no more prediction work.

**§1 ACCEPTED (2026-08-26).** `check_hs_llama.py` (+ `test_check_hs_llama.py`, 27 mutation-tested guards) accepted the four primary cells at **full probe (`--n_probe 0`, not sampled)**: No-CoT α = −8 / −6 / 0 / +6 under `phase1b_eot`, all n=300, `stored_layer_indices=[10..18, 31]`, band `[11,20)`.

- **Projection reads exactly 0.00e+00 in all four cells** — the tracker casts HS to fp16 BEFORE projecting, so the stored states ARE the projected states. The `rtol=1e-2` tolerance is a **wrong-mask / wrong-band detector, not a precision test** (a wrong mask reads 3.5, corrupted HS 38).
- **Agreement measured 1.000 on all three fields in all four cells.** Frozen wording: "**本批次实测 1.000**". Do NOT restate it as "same-protocol re-run == replay" — it is an OBSERVED property of this batch, not a protocol guarantee.
- **acc 79.67 / 60.00 / 51.67 / 40.67 (α = −6 / 0 / +6 / −8) reproduces the −6 peak SAME-BATCH.** This is the 184 bs=1 batch, so per the 184-vs-182 rule it may **never** be mixed per-question with the 182 dose table.
- **A verified round-cap FALSE POSITIVE removed `2000` from `ROUND_CAPS`.** Two cells each had exactly ONE 2000-char sample; both ran 767/768 decode steps and both tails were degenerate repetition loops. A real cap makes a **CLUSTER**; judge a flagged length by decode steps and the tail, never by the length alone. Same reasoning that removed 1000 on Qwen.

**§2 FROZEN — the split lives in THIS repo, not in `RoleAnswer/`.** `manifold/split_manifest.{py,json}` + `test_split_manifest.py` (21 mutation tests). 60/20/20 **by QUESTION**, realised **185/55/60**, digest `64af9b38…`.

<!-- Why: manifold_fit.py REFUSES to run without the manifest, so a required input of versioned server code cannot live in an unversioned tree; and hash() is process-salted, which would silently re-roll the split every run.
Evidence: manifold/test_split_manifest.py sections [2] and [6]; split_manifest.py module docstring.
Scope: the whole manifold line, both models. -->
- **Assignment is `sha256(salt:question_idx)`, never `hash()` or an RNG shuffle.** `hash()` on str is **process-salted** in Python 3 — a different split every run, silently. The test carries that failure mode as an explicit control.
- **Split by QUESTION, effective n = 300 questions, never tokens.** Tokens within a question are strongly correlated, so a token split leaks: "held-out" tokens sit in a trajectory the basis was fit on.
- **Counts are 185/55/60, NOT 180/60/60, and must not be re-balanced.** Thresholds are on the hash value; sorting-and-slicing to exact counts would make each assignment depend on the whole set again, so extending n would re-shuffle existing questions. Verified: growing n 100→300 moves nobody.
- **ONE split shared by every cell** (all α, CoT, roles). Per-cell splits would confound the dose effect with question difficulty — the failure that made PV10-A's cross-α accuracy uninterpretable.

**§3 RUN AND COMPLETE (2026-08-26).** `manifold_fit.py` (+ `test_manifold_fit.py`, 74 mutation tests) + `run_manifold_pilot.sh`. Server-side, CPU-only, READ-ONLY on the H5.

- **Phases are option B, frozen:** `prefill` = **last-prefill state ONLY** (the one strictly α-matched position, so the only phase where a displacement claim is licensed); `pre_commit` = `[c−20, c)`; `post_commit` = `[c, c+20)`; `decode_all` = option-A sensitivity.
- **A sample committing before token 20 is KEPT on its actual tokens, never dropped.** Dropping `c < 20` would systematically delete FAST commitment — the behaviour α moves (Llama α=0 already commits before token 20 in ~23% of samples, and that fraction is itself α-dependent). at-risk filtering belongs to speed/curvature analyses needing a fixed landmark, NEVER to the PCA training set. A no-commit sample is absent from the aligned phases, its coverage is **reported per cell**, and it still enters `decode_all`.
- **Per-question equal weighting, and the basis is fit on α=0 TRAIN only.** Without it a 20-token trajectory outvotes a 3-token one 7:1 and the "natural manifold" becomes the natural manifold OF THE SLOW SAMPLES — which correlates with α. Every cell is centered by the **same α=0 `mu`**; centering a steered cell on its own mean subtracts the displacement under test.
- **`decode_all` reduces each question to its ROW MEAN before fitting** (unbounded rows: up to 768/question). Consequence that must be read per phase: it is fit on ~n_questions points, so **a flat spectrum THERE is the reduction removing within-question variation, NOT an unstable manifold.** Recorded in `basis_meta.json` as `reduction_note`, not left to folklore.
- **PCA is an exact partial solve, and the threading fix is the load-bearing half.** The server's OpenBLAS inverts on high core counts: one 3700×3700 float32 `eigh` measured **956.72s on default threads vs 8.20s pinned to one** (117×), because `eigh` spins on lock contention rather than parallelising. With 18 such matrices the fit phase ran ~4.8h with NO intermediate output; pinned it is **65s for all 36 bases**. `manifold_fit.py` sets the thread env vars **before `import numpy`** (anywhere later is a no-op). This is an OpenBLAS behaviour on many-core machines, **NOT** the data, the method, or this box — a higher-core machine is worse, not better. On top of that, `scipy.linalg.eigh(subset_by_index=…)` solves for `k_max+10` eigenpairs (exact, not randomized): top-20 becomes the basis, the extra 10 are diagnostic tail only. `total_var` moves to the exact Frobenius identity `‖A‖_F²/nq` because a partial solve returns no full spectrum and `total_var` is the denominator of the explained ratio. Verified equivalent to the previous full-eigh path: `total_var` rel.err **1.2e-16**, top-20 eigenvalues agree to **1.8e-15**.
- **Rank comes from ROWS and the numerical spectrum, NOT from question count.** An earlier `n_questions−1` cap was wrong and silently discarded real directions: 2 questions × 5 tokens support **4** directions; 2 × 1 token support 1.
- **The commit locator is COPIED from `analyze_wrong_right_commit.py`, deliberately not improved.** Its `>=` boundary can land one token early, but every published Llama commit-aligned number uses that definition, so changing it here would silently redefine the event the phases are built on. The test pins **equivalence** (patterns, flags, and behaviour on cases spanning both branches), not correctness. It is **not** Qwen's `####`-only locator.
- **Four fail-closed guards, each added after being found missing:** `--split_manifest` is REQUIRED (no default → no silent fit on all questions); the manifest is **structurally validated** (mutual exclusion, full 0..n−1 cover, counts, non-empty train) and its **question-text digest is RECOMPUTED FROM THE α=0 H5** by `question_idx` order — §2's digest is worthless unless a consumer enforces it; `--base_cell` is verified `steer_alpha == 0` from H5 metadata, not from the filename; and `basis.npz` / `basis_meta.json` are overwrite-guarded alongside the per-cell JSON (a stale basis beside fresh coordinates still loads and still looks reasonable — the worst failure mode here).
- **Export carries per-token `coord_t` (bounded phases) and `re_by_k` (k=1..k_max).** A phase MEAN destroys token order, and speed / curvature / turning are defined on the ordered sequence; NRE at k=20 alone cannot be re-expanded offline, so validation could not otherwise choose k. `decode_all` exports summary stats only, so **option-A sensitivity is a LEVEL comparison, not a trajectory-shape one**.
- **Cheap guards run BEFORE the tokenizer load.** Twice during construction the H5/manifest checks sat behind it, so a wrong `--model_dir` hid them behind an unrelated failure.
- **NO GPU.** No forward pass anywhere: the H5 hold precomputed states and `AutoTokenizer` loads no weights. Do **not** set `CUDA_VISIBLE_DEVICES` — unlike signal/HS collection, nothing here generates model output, so cross-machine reproducibility does not apply and it can run beside GPU jobs.
- **Startup must print all three lines; if any is missing, stop rather than let it run:** `[split] … (structure verified)`, `[split] question-text digest matches the H5 (300 questions)`, `[fit] basis from nocot TRAIN (185 questions), k_max=20`.

**§3 RESULTS — full tables in `AdaManifold.md` §3.** Runbook facts only: 36 bases in 65s; `stored_layer_indices=[10..18,31]`, `n_middle=9`; fit-phase rows prefill/`decode_all` m=185, `pre_commit` m=2940 (nq=**150**), `post_commit` m=3576 (nq=183) — `pre_commit`'s nq is lower because a sample committing at decode step 0 has a post window but no pre window.

<!-- Why: 45% explained at k=20 in dim=4096 reads as "high-dimensional noise" until it is put beside a matched null, where the honest comparator is 2%, not 100%. Reading the absolute number alone once produced a wrong "the manifold premise is weakened" call in-session.
Evidence: AdaManifold.md 3.2; 20 isotropic draws matched on m/nq/dim and per-question weighting.
Scope: every manifold spectrum claim, both models. -->
- **A MATCHED isotropic null is mandatory beside every spectrum, and it must match `m`, `nq`, `dim` AND the per-question weighting, through the same Gram path.** Otherwise low-rank structure cannot be told from the sampling necessity of `m ≪ dim`. **Frozen wording: `low-rank spectral concentration relative to a matched isotropic null`** — NOT "low-dimensional manifold" (PCA shows linear low-rank only), and `k=20` is an analysis cap, **NEVER an intrinsic dimension**. **Ratios compare WITHIN a phase only**: prefill has `m = nq = 185` so its null alone already reaches 14.8%, which is why prefill's 3.4× and the commit windows' ~20× are not commensurable. The spectrum is a slow tail with no elbow, so k is FIXED at 20 with k=5/10/20 sensitivity rather than "chosen" — val NRE falls monotonically in k and cannot produce an extremum.

<!-- Why: displacement MAGNITUDE is monotone in alpha while DIRECTION is not, so a single dose-ordered reading of one number silently merges two independent axes. A residual-ENERGY difference was used in-session as if it bounded the normal part; it bounds nothing (the cross term -2<n_a,n_0> is missing) and those figures were withdrawn.
Evidence: AdaManifold.md 3.3; manifold_prefill_exact.py, validated against a synthetic 25% split.
Scope: every prefill displacement claim. -->
- **PREFILL DISPLACEMENT is computed in AMBIENT space, and the exported scalars cannot substitute.** `energy − ‖coord‖²` is a residual ENERGY, so differencing two cells drops the cross term `−2⟨n_a,n_0⟩`: it is not an upper bound, not a lower bound, not a bound. A residual NORM difference would at least lower-bound the normal part, and is still not the wanted quantity. `d = h(α) − h(0)` at decoder 18, paired by `question_idx`, `f_k = ‖W_k d‖²/‖d‖²`, primary = **energy-POOLED** ratio (a per-question ratio would weight a near-zero displacement as heavily as a large one). **The random reference is mandatory beside the numbers**: an isotropic displacement puts `20/4096 = 0.488%` of its energy in any 20-D subspace, so the observed values are 20–43× random and all three doses ARE strongly aligned — without it 9.8% misreads as "barely aligned". **Frozen wording: energy INSIDE / OUTSIDE the α=0 top-k PCA subspace — never "off-manifold"**, since k=20 spans only ~50% of α=0 variance.

<!-- Why: an equal INSIDE ratio was read in-session as "same direction" and a differing one as "scalar gain excluded"; neither follows. Two displacements can fill the same top-k subspace equally and point different ways, and refuting scalar gain needs the scalar model FITTED, not one of its corollaries.
Evidence: AdaManifold.md 3.4; manifold_prefill_direction.py, validated on synthetic pure-scalar and orthogonal cases.
Scope: every cross-dose direction claim. -->
- **DIRECTION conclusions come ONLY from the cosine / scalar fit, never from the inside ratio.** Least-squares `d_a ≈ k·d_b` with a per-question sign fraction (a pooled cosine can hide a mixture of aligned and anti-aligned questions). **`resid ≡ 1 − cos²` exactly at the LS `k`** (verified 1.1e-16), so **`k` is the only independent number of the three** and must be reported beside the residual. **The inside ratio and the cosine are two AGREEING observations, not the same fact** — showing they are one requires testing whether +6's orthogonal component sits outside top-k.

<!-- Why: the plan's own gate ("beyond s_t/Z_t or it is demoted") is easy to answer with whichever metric happens to move; the frozen three-of-three rule and the exhausted-test provenance are what keep the verdict honest.
Evidence: AdaManifold.md 3.5-3.6; RoleAnswer/manifold/PREREG_incremental.md, PREREG_negative_arm_confirm.md.
Scope: the manifold pilot's status in the paper. -->
- **INCREMENTAL PREDICTION: NOT DETECTED.** Two outcomes with DIFFERENT admissible baselines, and mixing them is the trap: for **commit position** the baseline is **pre-generation only** (`Z_prefill`, prefill confidence) because commitment behaviour IS the outcome; for **correctness** commitment IS admissible, as a predictor. Frozen rule: a dose counts as improved only if BOTH members of its metric pair move the right way, and "stable" needs ALL THREE doses to agree. Commit position came out **mixed** (both negative doses improve on R²/MAE/ρ, +6 worsens on all three); correctness is **inconsistent across metrics and doses** — **do NOT write "all three worsened", −6's AUC improves .386→.497**. **Three caveats travel with every number**: TEST was spent on round 1 and is exhausted (round 2 was designed after seeing it, so it is **post-hoc** — using a test set once is not the error, re-tuning against it afterwards would be); n=60, so sampling error exceeds the differences; and the wording is **"no stable incremental predictive power detected"**, never "disproved".

<!-- Why: passing a directional criterion on a model whose R2 is NEGATIVE is easy to over-read as "geometry predicts commitment"; it improved from worse-than-the-mean to less-bad.
Evidence: AdaManifold.md 3.6; RoleAnswer/manifold/PREREG_negative_arm_confirm.md and its addendum.
Scope: every CoT confirmation claim. -->
- **CoT CONDITIONAL CONFIRMATION: H1's negative half PASSED its pre-set directional criterion, with weak absolute predictive power.** All three of R²/MAE/ρ moved the predicted way at CoT α=−4, but **R² stays NEGATIVE after adding geometry** and ρ stays small, so the frozen reading is **"a reproducible weak directional signal, not strong predictive evidence"**. Correctness on the same cell got clearly WORSE. **`manifold_fit.py --reuse_basis` is what makes this a confirmation rather than a new model** — CoT is projected onto the EXISTING No-CoT α=0 basis and never refit; the flag verifies split version / band / k_max agree and skips rewriting `basis.npz` + `basis_meta.json` (the meta write originally sat outside the branch and would have clobbered the frozen artifact while reporting success). Scope limits frozen BEFORE the run: CoT has only α=0/−4, so **only the negative half of H1 is testable** — the positive half must not be reported as confirmed nor quietly dropped; **α=−4 is not among the doses H1 was derived from** (−8/−6), so this is a generalisation to an unmeasured dose; and the numbers describe where CoT states sit relative to the **No-CoT** manifold.

<!-- Why: "decode contradicts prefill" overstates it -- pre-commit coverage moves with alpha, post-commit sits after ####, and steering happens at last-prefill, so divergence there is expected rather than contradictory.
Evidence: AdaManifold.md 3.7; RoleAnswer/manifold/PREREG_decode_minimal.md.
Scope: the decode half of the manifold line, now closed. -->
- **MINIMAL DECODE: the frozen three-part rule FAILED, and the line closed on it.** Rule (frozen before running): (a) −6 and −8 consistent with each other, (b) +6 stably separated, (c) visible in BOTH pre_commit and post_commit. None held. **Frozen wording: "last-prefill geometry did not stably extend to commit-aligned decode"** — NOT that the manifold was falsified, and **NOT "the doses point in opposite directions"** (NRE is a magnitude ratio, so a high and a low value are not opposite directions). Why divergence is EXPECTED rather than contradictory: pre-commit coverage moves with α, post-commit sits after `####` where states reflect answer format, and steering is injected at last-prefill — so texts fork immediately after. `pre_commit` n is also strongly imbalanced (−8: 28 vs −6: 55), a selection effect on top of everything else.

<!-- Why: slot 6 on Qwen reads the stored FINAL layer (decoder 27) legally and silently, producing band-looking geometry from the wrong layer; Llama's slot-8 default is a band layer, so a Qwen run inheriting it is not merely wrong but wrong in a way that used to raise nothing.
Evidence: guard + 10-case mutation test in manifold_prefill_{exact,direction}.py.
Scope: every Qwen prefill run. Re-check if the stored layer set changes. -->
**Qwen prefill analysis MUST pass `--slot 5` explicitly, and `--cells` too.** The band's middle slots are `[0, layer_end-layer_start)` — Llama `[11,20)` gives 0..8 (default 8 = decoder 18), Qwen `[16,22)` gives 0..5 (**5 = decoder 20**). The H5 stores the model's FINAL layer AFTER the middle band, so **Qwen slot 6 is decoder 27** and used to read legally. Both prefill scripts now REFUSE a slot outside the middle band (`decoder_layer = layer_start - 1 + slot` still labels it correctly). `--cells` must also be explicit: every default names `nocot_aneg6`, which Qwen does not have. `manifold_fit.py`'s `basis_meta.json` writes `commit_locator` from `args.commit_locator` (it was hardcoded to the Llama convention, so a Qwen basis would have carried a metadata lie; the Llama branch is byte-identical to the old string, so re-running Llama reproduces its meta exactly).

<!-- Why: the pre-registered compression prediction was FALSIFIED, and a reader who remembers only the hypothesis will re-derive it; the exclusion is the result.
Evidence: AdaManifold.md 3.8; prefill_{exact,direction}_qwen.json; PREREG_qwen_prefill.md failure conditions.
Scope: every cross-model manifold claim. -->
**Qwen last-prefill is RUN AND FROZEN (2026-08-27) — the pre-registered compression account was FALSIFIED, and the result is EXCLUSIONARY.** Qwen's positive arm IS a single near-1-D axis (adjacent cos .965–.983, 100% same-signed, `k_ls` within .012–.017 of the independent magnitude ratio), but **`‖d‖/|α|` is constant at 14.85 / 15.09 / 15.11 across +6/+8/+12 — no saturation** — while its accuracy has already plateaued (86.00/88.33/88.67, high-dose pairs n.s.). Per the prereg's own failure condition, `magnitude keeps growing at +12 → compression account NOT supported`. **So the behavioural plateau is NOT entry-state saturation**, and last-prefill geometry does not explain the Llama-vs-Qwen difference: both models show smooth, collinear, dose-linear entry displacement yet peak-vs-plateau behaviour. Frozen framing: **entry gain is similar, the dose–response function differs** — the difference lives in commitment / decode dynamics, consistent with §3.7's decode null. **Two limits travel with every citation.** (a) **Qwen's No-CoT negative arm has only `−8`, one point**, so "is Qwen's negative arm single-axis" is NOT answerable — write "Llama's two arms and Qwen's positive arm are single-axis linear; Qwen's negative arm is not decidable", never "all four arms". (b) **Inside ratios are not comparable across models** — each uses its own basis, band (L=6 vs L=9) and isotropic reference (`20/3584 = 0.558%` for Qwen vs `20/4096 = 0.488%`); Qwen's positive arm is 8.4–10.1× its own random reference, Llama's 20×, and those multiples describe each model against itself only.

<!-- Why: the fixed cross-arm angle reads as a property of the intervention until you see that the two arms are anti-parallel at the injection point; a reader who skips this will attribute the asymmetry to the mask rather than to propagation.
Evidence: AdaManifold.md §3.9; xlayer/direction_{llama_s0..s8,qwen_s0..s5}.json; PREREG_cross_layer.md.
Scope: every cross-arm angle claim, both models. -->
<!-- Why: the layer-matched Llama>Qwen alignment gap is real but the UNPAIRED ranges OVERLAP,
so "Llama is higher everywhere" is false and is the natural misreading of the primary-layer table.
Evidence: AdaManifold.md 3.10 + S4; recomputed from the frozen xlayer/exact_*.json, no new data.
Scope: every inside-ratio cross-model claim. -->
**Inside-ratio k sensitivity is RUN AND FROZEN (2026-08-28) — the ordering is stable, and the manifold line is CLOSED with it.** Recomputed offline from the already-frozen `xlayer/exact_*.json` (no model, no server): at k=5/10/20 and across all 15 layer slots, three orderings hold with **no reversal** — Llama negative arm > Llama positive arm; Llama > Qwen layer-matched; and Qwen's four doses are nearly dose-INDEPENDENT (4.7–6.6% at k=20, versus Llama's monotone 21.4→9.8%), which is what pure scaling along one axis predicts. **Enrichment `inside/(k/H)` is mandatory beside every ratio** — `H` differs (4096 vs 3584), so raw inside ratios are not cross-model comparable. Layer-matched, Llama is **2.4–4.8×** Qwen at every k. **But the UNPAIRED per-model ranges OVERLAP** (k=20: Llama 6.1–55.9×, Qwen 4.7–18.3× — Llama's low layers sit BELOW Qwen's high layers), so the only supportable wording is "**in a layer-matched comparison** Llama's entry displacement aligns more strongly with top-PCA natural directions", never "Llama is higher everywhere". **The gap is NOT off-manifold evidence and NOT an explanation of peak-vs-plateau** (both models' displacements are linear and single-axis; k=20 spans only ~half the α=0 variance) — do not investigate further. Qwen's first steered layer reads an identical 18.3 at all four doses, which is analytic: the displacement there IS `α·mask`, so direction and inside ratio are α-independent.

**The cross-model summary figure is `fig_crossmodel_summary.png`, built from stored artifacts only (2026-08-28).** Four panels: accuracy change vs each model's own α=0; normalized `‖d‖`; within-arm `|cos|`; geometry–behaviour decoupling. Two plotting 口径 are load-bearing and were each wrong in a first draft: **panel B plots `|α|` with the two arms SEPARATED** — `‖d‖` is linear in `|α|`, so plotting it against signed α produces a V shape that reads as non-monotone when the underlying quantity is not; and **panel D's points are UNCONNECTED**, because ordering by geometric magnitude does not order behaviour (Llama −6 and −8 sit at normalized 1.00 and 1.39 with cos 0.989 — nearly the same axis — yet +19.7pp and −19.3pp). Llama normalizes by `‖d₋₆‖`, Qwen by `‖d₊₈‖`; each model keeps its own α=0 basis and band, and α stays nominal. Accuracy comes from the frozen tables (Llama `acc_nine_point.json`, the 184 bs=1 batch; Qwen `AdaDopamine_gsm8k.md` Table 4.1a), so the panel is **not** a per-question join and the 184-vs-182 rule still applies.

<!-- Why: commit-state was first coded as a BOOLEAN ("no parseable '#### <num>'") and read a
flat ~41% at every Llama alpha, merging degenerate loops (repeated commits) with genuine
no-marker cases -- an inversion of meaning, not a rounding issue.
Evidence: AdaptiveThinking.md 5.8.0; thinking_curve/extract_metrics.py docstring.
Scope: every commitment readout on GSM8K, both models. -->
## P2 commitment prediction + cross-task workpoint transfer (COMPLETE + FROZEN 2026-08-28)

> **`AdaDopamine_gsm8k.md` §5.1–5.4 is the results document** — 口径, tables, evidence level
> and boundaries live there. This section is provenance, hashes and reproduction.

Offline in `RoleAnswer/p2/` (**not in this git repo** — a server `git pull` will not
fetch it), `python3.10`, no server, no GPU. The protocol copy IS in this repo at
`docs/PREREG_P2.md`.

### Frozen prediction and evaluation specification

- **Data coverage.** P2 uses stored outputs only; no new inference was run. GSM8K uses
  the lightweight No-CoT signal JSONs (Llama 9 alpha cells; Qwen 11 alpha cells), all of
  which carry `x_prefill`. MATH uses the existing No-CoT outputs (Qwen 9 cells spanning
  `−8…+8`; Llama only `−4/0/+4`) and does not carry `x_prefill`. Every CoT cell is
  excluded. Consequently Qwen supports direction, full-curve workpoint and regret
  evaluation, whereas Llama supports local direction only.

- **Primary transferable feature set.** The commitment-only predictor is text-based:
  `early_candidate`, three commit-state dummy columns, `posN`, and `posN_observed`.
  Commit state is encoded as the four mutually exclusive categories `committed /
  marker_unparsed_nonloop / loop / no_marker`, with `committed` as the reference level.
  Frozen extractors (`all_hash`, `all_boxed`, `norm_gsm8k`, `fallback_gsm8k`, and
  `has_early_candidate`) are imported rather than reimplemented. `x_prefill`/entry gain
  is available only for the GSM8K entry-only and combined supplementary comparisons;
  it is not part of the transferable MATH predictor.

- **MATH output-format adapter.** GSM8K `####` and MATH `\\boxed{}` implement the same
  operational semantics: final-answer marker, first parseable commit position, and
  repeated submission. MATH reuses the frozen balanced-brace `all_boxed` extractor; an
  empty `\\boxed{}` is an unparseable marker, not a parseable commit. This adapter was
  frozen before MATH accuracy was read and was not adjusted in response to the result.

- **Question-grouped cross-validation.** P2 uses a deterministic five-fold manifest
  derived from the question hash (realised fold sizes `56/68/70/55/51`). Every dose of
  one question stays in the same fold, and the manifest is shared across both models.
  All dose rows enter training, but **raw alpha is never a predictor**. Missing `posN`
  is filled with the observed training-fold median, while `posN_observed` preserves its
  missingness information; imputation and standardisation are fitted on the training
  fold only. The inference unit is the question, and all confidence intervals and model
  contrasts use question-cluster bootstrap.

- **Accuracy convention.** Offline `first_acc` is the only MAIN MATH outcome and
  `last_acc` is sensitivity only. Any scoring code must reuse the frozen offline
  extractor and reproduce the published cell total before analysis. Inline `correct`
  remains run-health/provenance metadata and is never substituted for the MAIN outcome.

### P2A / P2B frozen result details

**Moved to `docs/P_PHASE_ARCHIVE.md` (2026-09-02)** — the OOF AUROC / transfer
tables and their CIs. Narrative version: `AdaDopamine_gsm8k.md` §5.1–5.4. The
frozen wording and 口径 traps those numbers must be reported under stay below;
the headline is that commitment-only prediction PASSED its gate on both models
(Llama `.687`, Qwen `.749`), entry gain added **no detectable incremental
predictive gain**, and P2B is **retrospective locked selection, NOT blind
validation**.

<!-- Why: the protocol lives outside git, so the repo hash does not pin its content;
a reader citing "frozen at 75d738c" would be citing nothing.
Evidence: docs/PREREG_P2.md commit 37713c8; p2/p2_freeze_manifest.json.
Scope: every P2 provenance claim. -->
- **The protocol is pinned by a GIT COMMIT, not by the repo hash at freeze time.**
  `PREREG_P2.md` lives in `RoleAnswer/`, so `75d738c` (the repo hash recorded inside the
  document) does NOT pin its content. Two remedies were applied before any modelling:
  `p2/p2_freeze_manifest.json` records every artifact's SHA256, and a verbatim copy was
  committed to `docs/PREREG_P2.md` (commit `37713c8`, protocol sha256 `8fddcf12…`).
  Cite the commit.

- **Order of operations is the result.** Protocol frozen → audit → P2A fitted → gates →
  predictors frozen → **`p2b_predictions.json` frozen (`4e52b079…`)** → only THEN MATH
  accuracy read. `run_p2b_predict.py` loads MATH with `want_label=False` and asserts no
  label reached a row; `run_p2b_eval.py` **refuses to start** unless the prediction file
  already exists. Reading accuracy first would destroy the locked character of the test,
  and no later re-freeze can repair it.

- **The label firewall is structural, not a convention.** `build_features` receives a
  STRING, never the sample dict, so `gold_answer / gold_solution / correct / pred_answer /
  level / type` are **unreachable** rather than merely unused; a source-level assert also
  rejects any mention of them in the function body. `level` is forbidden as a feature
  because it is a MATH-only difficulty label with no GSM8K counterpart — using it would
  make the predictor untransferable.

- **The four-value commit-state is PREDICTOR ENCODING ONLY and does not revise P1.**
  P1's three values stay `committed / marker_unparsed / no_marker`; the predictor splits
  the middle into `marker_unparsed_nonloop + loop` purely because `is_loop` is a strict
  subset of `marker_unparsed`, so a three-way one-hot plus a `loop` column is structurally
  collinear. The audit asserts the split re-sums to P1 exactly (Llama GSM8K α=0 →
  177/66/57, loop 52) — that reproduction is the acceptance check for any edit here.

<!-- Why: two of the six frozen features are near-dead on MATH, so a reader crediting the
transfer to "commitment features" broadly would overstate which channel carried it.
Evidence: PREREG_P2.md §13.2, recorded BEFORE any model was fitted.
Scope: every P2B citation. -->
- **The transfer channel is NARROWER than the six features, and this was recorded before
  fitting.** `\boxed{}` does not produce the degenerate repeated tail `####` does, so
  `cs_loop` and `cs_marker_unparsed_nonloop` are near-zero-variance on MATH (Llama 0/0 of
  900, Qwen 0/1 of 2700) and their coefficients are inert there. The signal that actually
  transfers is `early_candidate + posN + posN_observed + cs_no_marker`. Word P2B as
  **answer-formation and submission-timing transfer**, never as GSM8K's degenerate-loop
  behaviour transferring. The features were NOT dropped after seeing this — altering the
  set post-hoc is the freedom the protocol exists to prevent.

- **`posN` medians differ ~5× across models (Llama .1478, Qwen .7652)** — consistent with
  P1's Llama-commits-early vs Qwen-answers-first, and the concrete reason the two models
  share no absolute threshold and are frozen as separate artifacts.

- **The gate is `AUROC 95% CI lower bound > 0.5`, NOT "CI does not contain 0.5".** A CI
  lying entirely below 0.5 satisfies the latter and must fail. Amended as `p2-v1.1` before
  any model was fitted. Models are gated INDEPENDENTLY: one failing does not block the
  other. Both passed (Llama .6561, Qwen .7098).

- **P2B's primary readout is ORDERING, not calibration.** The predictors are fitted on
  GSM8K, and MATH differs in base rate, marker convention and length, so absolute
  probabilities drift (Qwen predicts .83–.88 against a true .54–.68). **A calibration gap
  on MATH is expected and is not a failure**; reporting the predicted curve as an estimate
  of MATH accuracy misreads what was frozen. Predictions are never rescaled or recalibrated
  against MATH accuracy — `plot_p2.py` panel A shows the raw gap deliberately.

- **Offline `first_acc` reads 1–2 items BELOW inline in EVERY MATH cell** — the two known
  unpatched extractor gaps (leading-zero `idx 253`; empty first `\boxed{}`). Uniform across
  cells, so it cannot move an ordering. Recorded, never patched in P2 (patching needs its
  own re-freeze, per the MATH extractor rules above).

- **The per-α AUROC breakdown is EXPLORATORY** (Llama .59–.71, Qwen .50–.80; α=0 alone
  .677/.748). Not preregistered; run to exclude the artifact that the predictor merely
  separates doses. It does not enter the gate.

- **Llama P2B is LOCAL DIRECTION ONLY** (3 doses `−4/0/+4`). Its `ρ=+1.000` is a necessity
  of three points and **must not be cited beside Qwen's `ρ=+0.962`**. The limitation is
  stored as a field in `p2b_predictions.json`, not left to prose.

- **Artifacts + SHA256 and the reproduction runbook moved to
  `docs/P_PHASE_ARCHIVE.md` (2026-09-02);** the authoritative copy of the hashes
  is `p2/p2_freeze_manifest.json`. `run_p2b_predict.py` and `build_p2_folds.py`
  refuse to overwrite a frozen file — re-running the chain requires deliberately
  deleting it, so a later fit cannot masquerade as the locked prediction.

Both `run_p2b_predict.py` and `build_p2_folds.py` refuse to overwrite an existing frozen
file: re-running the pipeline end-to-end requires deliberately deleting them, which is the
point — a silent regeneration would let a later fit masquerade as the locked prediction.

**§4 Qwen behavioural recompute scripts live in `RoleAnswer/backup/qwen_step2/`, NOT `RoleAnswer/qwen_step2/` (relocated; older refs are stale).** `commit_columns.py` (§4.1/§4.3 early-candidate + posN + pre-commit chars), `highdose.py` / `highdose_overload.py` + `gsm8k_extended_curve.txt` (§4.1 +10/+12), `math_cot_curve.py` / `math_cot_stats.py` / `math_cot_rightarm.py` / `math_cot_boxed_tail.py` (§4.3.1–4.3.2), `math_strat_frozen.py` (MATH extractor-gap worked example), `earlycand_audit_RESULT.txt` (§4.3.3 blind audit). The frozen detector they import, `early_candidate_detector.py`, stayed at the `RoleAnswer/` top level and did NOT move — so a stale `qwen_step2/` path fails on the SCRIPT, not on the definition. **The move BROKE five of the eight scripts and they were repaired (2026-08-30): `commit_columns.py` / `highdose_overload.py` / `math_strat_frozen.py` computed their import root as `__file__.parent.parent`, which resolved to `RoleAnswer/` before the move and to `RoleAnswer/backup/` after; `math_cot_curve.py` / `math_cot_stats.py` used `sys.path.insert(0, ".")` and so depended on the launch directory.** All five now derive the root from `__file__` and resolve imports from any cwd. **`math_cot_curve.py` still holds `ROOT = Path("qwen2.5")`, so the MATH-CoT scripts MUST be run from `RoleAnswer/`** — that one is a data root, not an import path, and was left relative rather than hardcoded to this machine. Acceptance after the repair: `commit_columns.py` reproduces Tables 4.1a/4.2 exactly and `math_cot_stats.py` reproduces both Holm families (Family A all p_adj=1.0000; Family B largest −4 at +7.00pp, p=.0065, p_adj=.0581).

**§5 Qwen offline script index (moved out of `AdaptiveThinking.md` §5.8, 2026-08-28 — this file is the single source of truth for implementation).** `RoleAnswer/qwen_signal/`, `python3.10`: `suite34.py` (§5.2–5.5 main line), `commit_aligned.py` (§5.3, §5.6.1), `hs_layerwise.py` (§5.6.2), `hs_null_specificity.py` (§5.6.3), `logit_family.py` (§5.5 Result 2–3), **`plot_section5.py` (all §5 figures)**, `plot_qwen_mainfig.py` (§5.3.1 commit-centered main figure). Frozen records beside them: `entry_gain_RESULT.txt`, `suite34_nocot_RESULT.txt`, **`suite34_cot_RESULT.txt`**, **`commit_aligned_v3_RESULT.txt`**, `hs_layerwise_RESULT.txt`, `hs_null_specificity_RESULT.txt`, `logit_family_RESULT.txt`. Server-side: `check_hs_qwen25.py` (H5 acceptance), `run_null_remask_qwen25.sh` (null remask). Figures live in `qwen2.5/dopamine/plots_gain/`, deliberately SEPARATE from `llama3/dopamine/plots_gain/` — the two models' values are not comparable and a shared directory is where cross-model mixing starts.

<!-- Why: the two指标 are not interchangeable and one of them does not exist on
Llama's nine-dose curve; filling that cell with premature(either) would swap a n.s.
rho=-.300 for a significant one under the same column header.
Evidence: AdaptiveThinking.md 5.8.1 footnote; early_candidate_detector.py (frozen 2026-08-21).
Scope: every cross-model commitment-timing claim. -->
- **The cross-model commitment-timing correlation is `acc ~ posN` rho=+.941 (Llama, 9 doses, 182) and +.863 (Qwen, 11 doses); `acc ~ early-cand%` is -.804 on Qwen and DOES NOT EXIST on Llama's nine-dose curve** (that table predates the frozen detector and carries `premature (either)`, a different definition reading rho=-.300, n.s.). Leave the cell empty rather than substituting. **Frozen wording is "transition out of a premature-commitment regime", NEVER "later is always better"** — Qwen's `posN` keeps rising past +8 (.754->.802) while accuracy flattens (+8.00/+2.33/+0.34 pp), and Qwen's whole rank correlation is carried by ONE threshold flip (`posN` is constant at .003 for every alpha<=+4). `early-candidate%` is the MAIN indicator (defined on all samples); `posN` is SUPPORTING (defined only on committed samples, whose share moves 79%->98% across the Qwen curve). The two come from one generation and are NOT independent evidence, and both are outcomes of alpha, so stratifying on them is post-treatment.

<!-- Why: the 5.8.1 nomk% column once read 1.0-16.7% because the classifier keyed on
marker COUNT (>=4) and fell through, misfiling 469 samples that carry 1-3 markers; a
reader who assumes the published table was always right will not re-check a rebuild.
Evidence: AdaptiveThinking.md 5.8.1; thinking_curve/extract_metrics.py.
Scope: every commit-state number, both models. -->
- **The four §5.8.1 readouts, frozen 2026-08-21, ONE definition shared by GSM8K and MATH.** `early-cand%`: the FIRST NON-EMPTY LINE is (1) <=60 chars stripped, (2) contains a number token, (3) is NOT a numbered reasoning opening (`1. To find …`), (4) is NOT a bare heading (`Step 1:` / `Solution:`) — i.e. an answer-shaped bare number written before any derivation; sensitivity at 40/60/80 is reported alongside and the conclusion does not depend on the choice. `posN`: char start of the FIRST PARSEABLE `#### <number>` over total generated chars, **defined only on `committed` samples**; measured in CHARACTERS, never tokens (a token 口径 needs a tokenizer, and the column is OMITTED rather than estimated when absent). `unparsed%`: `####` present but the answer unparseable (= `marker_unparsed`), with `loop%` its >=4-marker degenerate-resubmission subset. `nomk%`: no `####` anywhere. **Qwen's `nomk%` is 0.0 in all 11 cells** — it always emits `####`, so its unparseable samples are all `marker_unparsed` (`70\n####\n\n`: marker emitted, no number after), the same fact as §4's two commit counts. **A previous version reported a spurious 1.0–16.7% there** because the classifier keyed on marker COUNT and fell through to `no_marker`, misfiling 469 samples across all 20 cells (Qwen 337, Llama 132); the published table is the corrected one, and `extract_metrics.py` asserts the three states sum to 100 at runtime.

<!-- Why: posN and cand_pos are computed on DIFFERENT denominators, and reading the
gap as "the locator sees through the placeholder" is wrong -- it returns offset 4-6 on
those samples, i.e. it treats the marker's number AS the candidate.
Evidence: AdaDopamine_gsm8k.md Table 5.6d; p3/precandidate_reasoning.py denominators block.
Scope: every answer-formation-timing claim. -->
- **Answer-formation timing has THREE denominators and mixing them produces a wrong reading.** All samples (300: accuracy, coverage) / committed subset (`posN`, answer-first — Table 5.6c) / candidate-covered subset (`cand_pos`, `pre_*`, reason-first — Table 5.6d). On GSM-Hard CoT α=−6 the SAME 91 answer-first samples are **58.3% of committed** but **34.7% of candidate-covered**, which is the whole reason `posN` median reads .0000 while `cand_pos` reads .1012. **The locator does NOT skip a leading `#### N`** — it returns offset 4–6 on all 91. A "placeholder" reading was asserted once and is RETRACTED. **`reason_before_answer` is the MAIN indicator** (boolean, denominator-robust; four paired McNemar all p<1e−8), `pre_chars` second, `cand_pos` auxiliary — all three from one generation, NOT independent evidence. **`cand_pos` is a LOWER BOUND**: its locator accepts the RHS of `=`, so it can fire on an intermediate result; the bias is constant across α so paired contrasts stay readable. **These are NEW exploratory readouts and do NOT modify P2's frozen `early_candidate` feature** (one of the six in `p2_predictor_*.json`) or any frozen artifact — `p3/precandidate_reasoning.py` is read-only and GSM-Hard accuracy comes from the frozen evaluation files, since those generations carry no gold. **Frozen wording: "accuracy gain travels with a change in answer-formation timing", never "reasoning first causes the gain"** — the metrics are outcomes of α, so stratifying on them is post-treatment. GSM-Hard CoT is a MIXED regime: the distribution shifts later AND a substantial committed subset still emits the marker first; do not write that the earlier contradiction is resolved.

**P1 cross-model Thinking Curve is RUN (2026-08-28): the behavioural difference is located in COMMITMENT TRANSFER, not entry gain.** Written up as `AdaptiveThinking.md` **§5.8** (renumbered from §5.9 when §5.7/§5.8 were merged into `5.7 Cross-Model Synthesis and Evidence Boundaries`). Offline in `RoleAnswer/thinking_curve/` (`extract_metrics.py`, `curves.py`, `fig_p1_commitment.png`), `python3.10`, no server — frozen extractors (`all_hash` / `norm_gsm8k` / `fallback_gsm8k` / `has_early_candidate`) are IMPORTED, never reimplemented. **Pairing is BY ORDER and that is verified, not assumed**: all 20 cells of both models hold the same 300 questions in the same order (so the comparison is item-matched across models too), asserted per cell at load with a fail-closed check.
  <!-- Why: the bucket test keyed on marker COUNT (>=4) and fell through to no_marker, so 469 of 6000 samples
  carrying 1-3 markers were filed as "never committed" -- the opposite of what they are; Qwen's published
  nomk% of 1.0-16.7% was entirely this artifact and is really 0.0 in all 11 cells.
  Evidence: thinking_curve/extract_metrics.py; AdaptiveThinking.md 5.8.0/5.8.1.
  Scope: every commit-state number, both models. -->
  - **`early-candidate` and `posN` are frozen operational definitions; the paper states them, this file states the implementation.** `early_candidate` comes from `early_candidate_detector.has_early_candidate` (frozen 2026-08-21, ONE definition for GSM8K and MATH, `MAX_LINE_CHARS=60`, sensitivity at 40/60/80 reported alongside) — **import it, never re-derive a first-line rule**, which is exactly the per-task divergence the freeze replaced. `posN` = char start of the FIRST PARSEABLE `#### <number>` divided by total generated chars, defined **only on `committed` samples**, so its denominator is a dose-dependent subset and it must be cited beside `cmt%`. Stratifying any outcome on `early_candidate` is POST-TREATMENT stratification (the flag is itself an outcome of α) — consistent-with evidence, never mediation.
  - **`commit_state` is a THREE-WAY EXHAUSTIVE partition keyed on marker PRESENCE — `committed` / `marker_unparsed` / `no_marker`** (Llama α=0: **177 / 66 / 57 = 300**). `is_loop` (≥4 markers) is a DESCRIPTIVE SUB-FLAG of `marker_unparsed` (52 of the 66), **not a fourth bucket**; counting the retired name `loop_commit` reads a silent 0.0 for every cell. A boolean version read a suspiciously flat ~41% at EVERY Llama α because it merged the degenerate-loop samples (tail writes `#### . 36#### . 36…`, so the regex misses the number — the model committed REPEATEDLY) with genuine no-marker samples; a later three-value version still tested `>= 4` and fell through, misfiling the 14 samples with 1–3 markers. **`curves.py` asserts the three sum to 100 at runtime.** **Qwen's `no_marker` is 0.0 in ALL 11 cells** — it always emits `####`, and its unparseable samples are all `marker_unparsed` (`70\n####\n\n`: marker emitted, no number after), the same fact as the GSM8K two-commit-count rule above. **Production applies the fallback chain to all of them and 63/123 still score correct, so commit-state is a BEHAVIOURAL readout and NEVER an accuracy proxy** — surface format failure is not the same as being wrong.
  - **The dose axis for cross-model comparison is `z`, standardized WITHIN each model** (`(x̄_prefill(α) − x̄_prefill(0))/SD_{α=0}`). Frozen wording: "compare the transfer curves on each model's own standardized entry coordinate" — **never** "the two models received an equal intervention".
  - **Result.** Llama's early-candidate rate rises on BOTH arms (−8 77.0%, +8 69.3%, α=0 47.7%) with its MINIMUM at the accuracy peak −6 (19.3%), and at −8 `posN` collapses 0.298→0.043 with **95.7% of the generation after the commit** — premature lock-in, not shorter thinking. Qwen's negative arm does not move (95–97%) and only the high positive doses collapse it 96%→5%, after which the mechanism variables SATURATE at +8 (commit 98.0%, `n_markers` 2.1), which is why +10/+12 add nothing.
  - **Explanatory R² is CURVE-LEVEL DESCRIPTION, n=9/11 — not mediation and not a causal decomposition.** entry gain alone explains Llama's curve at .136 versus early-candidate at .945 (Qwen .898/.923). The robustness comes from the ITEM level instead: the early-candidate → error association holds in all 20 cells and survives question AND dose fixed effects nearly undiminished (Llama β=+0.297, cluster-robust SE .028, t=10.8; Qwen +0.200, .036, t=5.5), so difficulty confounding is excluded — but it stays an observational association.
  - **Premature commitment is NOT over-confidence — a negative result worth keeping.** On Llama's same-batch `metrics_*`, early-candidate samples are not more decisive (Δtop1 −0.058…+0.019), and among them the WRONG ones are not more decisive than the correct ones (−0.085…−0.001, the opposite direction). Word it as a commitment-TIMING failure, never a confidence-calibration failure. Qwen cannot be tested symmetrically (`metrics_hs` is 7 cells and a different batch — `PARTIALLY AVAILABLE / SPARSELY SAMPLED` per §5.5).

**Cross-layer sensitivity is RUN AND FROZEN (2026-08-28); the manifold line is CLOSED.** 15 layer slots (Llama decoder 10–18 primary 18, Qwen 15–20 primary 20), each on ITS OWN α=0 basis, `PREREG_cross_layer.md`. **Primary-layer recomputation matched the frozen artifacts EXACTLY (max diff 0 on every metric)** — that regression is the first thing to check if a loop is ever re-run. Three results: (a) `‖d‖ = β·|α|` holds at EVERY layer (Llama R² .985–1.000; Qwen `‖d‖/|α|` constant within each layer) while magnitude grows ~21× (Llama) / ~8.6× (Qwen) across the band — the propagation the per-layer `‖m_l‖²` rule already describes; (b) **the two arms are anti-parallel at the FIRST steered layer and separate monotonically with depth** (Qwen dec15 `|cos| = 0.99996` with `k_ls` exactly the dose ratio — analytic, since the displacement there IS `α·mask`, so it doubles as a correctness check — reaching −0.754 at dec20; Llama −0.977 → −0.675); (c) within-arm cosine stays ≥0.97 at every layer. **So the "two axes at a fixed angle" of §3.4/§3.8 is a LAST-LAYER EMERGENT structure, not a property of the injection** — the mask is symmetric, the asymmetry is built by layer-wise propagation. **Llama's first layer needs the SNR rule, not a direction reading**: its `|cos|` is 0.888 at ‖d‖=0.31 rising to 0.977 at ‖d‖=1.18, i.e. fp16 quantisation at small displacement. The cross-arm separation itself passes all three SNR preconditions (displacement large and linear, cos stable across splits to ≤0.005, monotone in depth), so it is a real layer-wise effect. Figure `fig_xlayer_crossarm.png`; complete per-slot magnitude and cosine tables are in AdaManifold.md §3.9, while per-slot PCA inside ratios remain in `xlayer/exact_*.json`.

**Qwen prefill runbook (frozen).** `--slot 5` = decoder 20 = the LAST band layer for `[16,22)`; storage is `[15..20, 27]`, so **slot 6 is the final layer 27** and both prefill scripts now refuse a slot outside the middle band. `--cells` must be explicit (every default names `nocot_aneg6`, which Qwen lacks): fit `nocot,nocot_aneg8,nocot_a6,nocot_a8,nocot_a12`, prefill `nocot_a6,nocot_a8,nocot_a12,nocot_aneg8`. Reference axis was fixed at `d_+8` (behavioural plateau onset) BEFORE the run. Outputs: `components/qwen2.5/{manifold/qwen25_signal_v1/basis.npz, prefill_exact_qwen.json, prefill_direction_qwen.json}`, synced to `RoleAnswer/qwen2.5/dopamine/manifold/`.

**The original pre-registration, for provenance.** `RoleAnswer/manifold/PREREG_qwen_prefill.md` fixes the scope before any Qwen projection: last-prefill only, Qwen's OWN α=0 basis / band `[16,22)` (L=6, not Llama's 9) / `--commit_locator qwen`, three questions (does the positive arm share one direction; does displacement magnitude saturate; inside ratio against each model's own subspace), and failure conditions written in advance. **`--commit_locator` exists because Llama's locator (first `####` else first answer candidate) silently redefines the event on Qwen**, which answers first at low α so the fallback fires constantly; it is irrelevant to prefill (no commit is used) but a later Qwen decode run would have inherited it. Default stays `llama`, so every stored Llama result is unaffected. **Ruled out in advance**: comparing raw α across models (different masks, L=9 vs L=6, different activation scales — an equal α is not an equal intervention), comparing PC axes or hidden-state values, and reading either arm as a change in biological dopamine. Remaining No-CoT doses (±2/±4/+8) are **continuity checks only** — same questions, same basis — never an independent validation set.

**Where the artifacts live.** Source H5 `components/hidden_states/gsm8k/phase1b_eot/`; server output `components/llama3/manifold/phase1b_eot/` (basis + the four No-CoT cells) and `components/llama3/manifold/phase1b_eot_cot/` (the CoT cells, **same basis reused via `--reuse_basis`** — that dir deliberately holds NO `basis.npz`, which is how you verify the transfer actually happened). Synced local copies sit under `RoleAnswer/llama3/dopamine/manifold/` (`basis.npz`, `basis_meta.json`, `manifold_*.json`, `prefill_exact.json`, `prefill_direction.json`); the offline analyses (`incremental.py`, `incremental2.py`, `confirm_cot.py`, `decode_minimal.py`) and the four pre-registrations live in `RoleAnswer/manifold/`, which is **not in this git repo** — a server `git pull` will not fetch them.

**Known limitation to state explicitly at §5, not to discover there:** this batch has **no independent second α=0 cell**, so α=0 manifold stability can only be estimated by train/val subsampling / bootstrap, never by cross-validating two independent α=0 batches. That weakens the "the manifold is stable" premise, and its failure is a pre-registered **stop condition**.

## P3 blind cross-task validation on GSM-Hard (COMPLETE + UNSEALED 2026-08-30)

> **`docs/PREREG_P3.md` is the protocol** (frozen `p3-v1`, tag `p3-prereg-v1`,
> commit `1c0b865`). This section is provenance and the run order. Amendments are
> `docs/p3_amendment_0{1..4}.json`, each ADDITIVE — none overwrites its predecessor.

<!-- Why: the blind property is the only thing separating P3 from P2, it cannot be
restored once lost, and nothing in the code can enforce "a human did not look".
Evidence: docs/PREREG_P3.md section 0; the eligibility audit in section 1.
Scope: all of P3. Read before touching anything under gsm_hard_p3. -->
- **P3's entire value is that GSM-Hard accuracy has never been viewed. Reading
  `components/benchmark/gsm_hard_p3_gold.SEALED.json` before `p3_predictions.json`
  is frozen destroys that PERMANENTLY** — no later re-freeze repairs it, and the
  result degrades to another retrospective test like P2. That includes an editor
  preview or a `head`. Eligibility was audited before download (no data, no loader,
  no artifacts anywhere on the machine) and recorded in the protocol.

- **The loader writes TWO files and that split IS the firewall.**
  `data_gsm_hard.py` → `gsm_hard_p3_questions.json` (question + `sample_id`, meta
  declares `contains_labels: false`) and `gsm_hard_p3_gold.SEALED.json`. Generation
  reads only the former. A single file would leave the label one attribute access
  from every script; P2 established that "unused" is far weaker than "unreachable".
  Both live in **`components/benchmark/`** (the server's flat benchmark tree), NOT
  `benchmark/`.

- **`get_answer_gsm_hard_blind.py` is a FORK of `get_answer_regenerate_gsm8k.py`, not
  a reuse.** That script calls `is_correct_gsm8k` per sample and prints accuracy
  (lines 83–96), so running it would compute GSM-Hard accuracy during generation.
  Prompt construction and every `vc.regenerate` argument are identical to the frozen
  GSM8K path — so commitment features keep the meaning they had in P2 — while
  correctness/gold/`pred_answer` are removed entirely.

- **Generation params match the frozen GSM8K main line, deliberately: `max_new_tokens
  =768`, `temperature=0.0`, `bs=24`.** The P2 predictor was fitted on GSM8K output
  produced this way and `posN` / `early_candidate` are sensitive to length and
  sampling. **Do NOT borrow MATH's 2048/bs=8** — GSM-Hard keeps GSM8K's structure and
  only enlarges the numbers.

- **Doses are fixed in advance and are NOT re-searched:** Llama `−8/−6/−4/0/+4`
  (band `11-20`), Qwen `−4/0/+4/+6/+8` (band `16-22`). **Qwen keeps a negative probe
  by design** — with every candidate on the positive side, "correctly predicted the
  positive direction" could not have been wrong and would not be a test.

- **One MODEL's five doses must share one card; the two models may run on two.**
  bf16 greedy is not byte-reproducible across GPUs (a measured re-run differed on
  205/300 samples), so a split dose curve mixes the device difference into the α
  effect. The models are never compared per-question, so they parallelise freely.

- **Sample = 300 by frozen hash, never dataset order** (`sha256(salt:question_text)`,
  dedup on exact text first). Digest `48cc7635…`, revision pinned to
  `960448f73503112d4226baeb8eb41d3fb5ae2506`. `--revision` is type-checked to a full
  40-hex SHA: `required=True` alone is not a pin, since `--revision main` still
  follows upstream.

<!-- Why: auditing a float64 precision failure BY CALLING float() is self-defeating,
and the string-form test passed while the real input shape would have been missed.
Evidence: docs/p3_amendment_02.json; test_p3_label_firewall.py section 4b1.
Scope: the 2^53 audit only. Measured 0 of 300 at the pinned revision. -->
- **The `2^53` audit must never route through `float`, and a float64 gold in the
  unsafe range is a HARD STOP.** `float('9007199254740993') == 2**53`, so a
  float-based audit misses the first value that matters; `Decimal` parses the literal
  exactly. Separately, HF stores `target` as float64, so `datasets` hands back a
  python float whose `+1` is already destroyed upstream — unrecoverable, hence
  `assert_float_safe` refuses rather than under-reporting. **Measured at the pinned
  revision: 0 of 300 exceed `2^53`, so the ORIGINAL extractor is used and
  `norm_exact` is NOT enabled.** `p3/p3_bigint_audit.py` verified `norm_exact` against
  6600 stored GSM8K predictions with **0 verdict changes**.

- **Two near-optimal sets exist and must never be conflated.** The *predicted*
  region comes from the frozen predicted-score rule; the *observed* set is the doses
  whose paired per-question accuracy difference from the empirical best is
  indistinguishable. **The verdict is whether the predicted workpoint lands in the
  OBSERVED set** — closeness in predicted score is not sufficient.

- **No α=0 accuracy threshold gates readability.** A low baseline does not mean the
  curve is unreadable. Instead: a paired omnibus over each model's five doses, Holm
  across the two models; if no dose difference is detected the result is **"no
  readable dose curve"** and direction/workpoint are **not evaluable** — neither a
  success nor a failure.

- **Locked run order:** freeze protocol → download + schema/`2^53` audit → format
  preflight (`--preflight`, 5 samples, format ONLY) → generate all ten cells with NO
  accuracy → extract commitment features, apply the frozen P2 predictors, **freeze
  `p3_predictions.json` + SHA256** → only then unseal gold, once. **If the preflight
  format violates the frozen protocol the response is a HARD STOP** — not a redesigned
  prompt or extractor.

**Reproduction runbook (download, preflight, the ten cells) moved to
`docs/P_PHASE_ARCHIVE.md` (2026-09-02).** The trap it carries still applies to any
server launch: `cat` the log immediately — a wrong `PY` exits 127 before anything
runs and the `nohup` log looks empty.

### RESULT (gold unsealed 2026-08-30, `p3-result-unsealed`)

> **`AdaDopamine_gsm8k.md` §5.5 is the results document.** This block is provenance,
> hashes, and the 口径 traps.

<!-- Why: the five-point table and the fixed-workpoint table share one set of
per-question correctness, and llama's a=-6 vs a=0 contrast literally appears in
both -- reporting them as two independent confirmations doubles one result.
Evidence: docs/p3_result_20260830.json; p3/p3_evaluation.json.
Scope: every P3 citation. -->
- **The two readouts are NOT independent evidence.** Both are computed from the same
  per-question `first_acc`, and Llama's `α=−6 vs α=0` contrast appears in each. The
  fixed-workpoint table earns its place by testing a DIFFERENT proposition (can an
  already-established workpoint be transferred unchanged), not by replicating the first.

- **Five-point ordering.** Llama `.1100/.2433/.2400/.1800/.1700` (α = −8/−6/−4/0/+4);
  Qwen `.3433/.3400/.3467/.4033/.5033` (α = −4/0/+4/+6/+8). Both curves readable
  (paired McNemar over all 10 dose pairs, Holm within model; min p 2.29e−07 / 1.35e−07).
  Direction correct on both. Selected = observed best on both, **regret 0.00 pp**.
  ρ = +1.000 / +0.600.

- **Fixed-workpoint transfer** (`p3-amend-05`, α read from the frozen GSM8K record and
  NOT from any GSM-Hard predicted score): Llama α=−6 `.2433` vs `.1800` = **+6.33 pp**,
  discordant 32/13, exact McNemar p=.0066; Qwen α=+8 `.5033` vs `.3400` = **+16.33 pp**,
  63/14, p=1.41e−08. One test per model, never pooled with the five-point family.

<!-- Why: "llama identified -6" is the natural but wrong summary -- the predicted
scores differ by 6e-05 and the OBSERVED near-optimal set is also {-6,-4}, so nothing
in the data distinguishes them.
Evidence: p3_amendment_05.json correction_1_llama_margin.
Scope: every llama P3 selection claim. -->
- **Llama did NOT distinguish −6 from −4 — write "selected a workpoint inside the
  observed near-optimal set {−6,−4}, empirical regret 0".** Predicted scores differ by
  `5.94e−05` (−6 IS strictly greater, so the tie-break branch never fired — an earlier
  note saying "chosen by tie-break" is retracted), and the observed near-optimal set is
  likewise `{−6,−4}` (.2433 vs .2400). Only **Qwen's** selection is sharp: predicted
  margin `0.137`, observed near-optimal singleton `{+8}`.

- **Qwen's ρ is +0.600, not perfect, and must not be rounded up.** The predictor put
  −4 above 0 while the observed order is reversed (.3433 vs .3400 — one question). The
  inversion is inside noise; the number is still +0.600.

- **Calibration did NOT transfer, and this is not a failure.** Predicted `.55–.86`
  against observed `.11–.50` — an absolute over-estimate consistent with GSM-Hard's
  change in task difficulty. Ordering was declared the primary readout before unsealing;
  **no prediction was rescaled or recalibrated.**

- **`predicted_direction_from_alpha0` is the sign of the argmax over ALL FIVE doses**
  expressed relative to α=0 — a property of the five-point ordering, NOT a prediction
  made from the α=0 cell alone. Selecting direction from α=0 alone is the open gap, and
  is a separate protocol.

- **Order of operations, each step tagged or hashed.** protocol `p3-prereg-v1` (`1c0b865`)
  → download + sealed gold (`2^53` audit **0/300**, so the original extractor was used and
  `norm_exact` stayed OFF) → ten cells with no accuracy code path → **predictions frozen**
  `9ad21ee`, sha256 `8f0c514ae4776441` → **evaluator frozen** `p3-evaluator-frozen`
  (`850ef29`) → gold unsealed ONCE (`p3-result-unsealed`, `ed55753`). The evaluator that
  ran was verified **byte-identical** to the tagged one before the result was accepted.

- **Artifacts.** gold `a464d591abe659dc`, predictions `8f0c514ae4776441`, evaluation
  `0e3511c74ba89115`, evaluator `2ca6de427d2a8ceb`, unseal log `0ba5510226ef30e8`,
  questions `48cc763545d2ee23`, GSM-Hard revision `960448f73503112d4226baeb8eb41d3fb5ae2506`.
  Full record: `docs/p3_result_20260830.json`; the ten generation SHA256 are in
  `docs/p3_amendment_05.json`.

- **`run_p3_predict.py` copies `apply_frozen`/`near_optimal` VERBATIM from
  `p2/run_p2b_predict.py` and re-verifies them character-for-character at every run.**
  That P2 script cannot be imported (its module body refuses to run once
  `p2b_predictions.json` exists), and a first attempt that sliced the functions out of the
  source text at runtime silently extracted the WRONG region — `inspect.getsource` showed
  the module docstring instead. Mutation-tested: perturbing the sigmoid makes the guard
  fail closed.

- **`run_p3_eval.py` fails closed** on a missing `p3_predictions.json` (the ordering
  guard), a cell whose `questions_sha256` differs from gold, a missing/extra/duplicate
  `sample_id`, a missing cell file, or an existing output. Alignment is **by `sample_id`,
  never by row position** (asserted by reversing a cell's row order). `first_acc` is MAIN
  via the frozen offline extractor; `last_acc` is sensitivity only.

- **`test_p3_eval.py`: 35/35 on synthetic fixtures, no real gold.** Statistics are checked
  against hand-computed values (`McNemar(10,0) = 2/1024`), every guard is mutation-tested,
  and — the load-bearing one — a deliberately WRONG prediction fixture confirms the
  evaluator reports `direction_correct=False`, large regret and negative ρ rather than
  passing a bad result silently.

<!-- Why: Qwen's posN rises again at +8, which reads as breaking the monotone
fall; it survives a fixed denominator, so the tempting "denominator effect"
dismissal is wrong -- the actual split is absolute position vs generation length.
Evidence: AdaDopamine_gsm8k.md §5.5; p3/commit_panel_gsm_hard.py.
Scope: every GSM-Hard posN claim. -->
- **Commitment panel (`p3/commit_panel_gsm_hard.py`, `AdaDopamine_gsm8k.md` §5.5) is
  EXPLORATORY and permanently so** — the ten No-CoT cells were unsealed first. It imports
  P1's `per_sample` rather than reimplementing the three-way partition. Both models peak
  where `early_candidate` is lowest (Llama −6: 28.7%; Qwen +8: 6.0% against +6's 58.7%);
  Llama −8 is extreme premature lock-in (`posN` 0.0000, **100%** of the generation after
  the commit, acc .110); the failure modes differ (Llama loops and drops the marker, Qwen
  never loops and instead emits an unparseable `####`). **Qwen's `posN` rebound at +8
  (0.597→0.777) is NOT a denominator effect** — it survives on the 153 questions committed
  in all five cells (0.566→0.758); the decomposition is `commit_char` rising 274→436 while
  `gen_chars` keeps shrinking 959→831, i.e. the same normalized-vs-absolute split §5.5
  already records. Llama's common-committed subset is n=35 and selection-biased by −8, so
  only the full-sample table is reported.

<!-- Why: llama's GSM-Hard generations hit the 768 cap in 91-96% of samples in EVERY
cell, so its chars-median and post-commit share describe truncated text; and an exact
==768 test undercounts the cap population by ~12pp because the text is re-tokenized.
Evidence: docs/p3_supp_amendment_01.json; AdaDopamine_gsm8k.md §5.5.
Scope: every llama GSM-Hard length or post-commit number, No-CoT and CoT alike. -->
- **LLAMA GSM-HARD IS TRUNCATION-BOUND: 91-96% cap-hit in all five cells, median decode
  length exactly 768** (Qwen 13-23%). So Llama's `chars med` (~2100) and `post-commit%`
  are measured on truncated generations and may not be read as natural lengths. **The
  threshold is `>= 767`, not `== 768`** — offline re-tokenization is approximate at the
  boundary, and on Llama α=0 an exact test reads 82.0% against 94.0% at `>=767` and 94.3%
  at `>=760`; the plateau above 760 is the real cap population. This does **not** affect
  the P3 accuracy comparison (all ten cells paired per question under one convention).
  **Cap-hit does not invalidate the paired P3 comparison as a FIXED-BUDGET estimand** —
  every dose was evaluated under the same 768-token budget. **But it can affect absolute
  accuracy and dose differences**, so the result may NOT be read as unconstrained
  reasoning capability. Llama's cap rate spans 4.7 pp across doses (91.3–96.0%); the one
  pair the fixed-workpoint test uses, `−6 vs 0`, is not significantly different (18/23,
  exact McNemar p=.53), and that covers that pair only. **A corollary for AdaDopamine_gsm8k.md §5.5's Llama
  −8 reading: "generation length did not shorten" is a CEILING EFFECT, not an
  observation** — every dose is flattened against the same cap. The conclusion (the
  collapse is not from generating less) still holds, but must rest on `posN med=0` and
  `post-commit=100%`, which are length-independent. For the CoT supplement, generations
  are longer at the same cap, so a degraded CoT result must report truncation and the
  commitment reading side by side rather than resolving it in favour of either. **Do NOT
  raise the cap now** — that would change the frozen primary question. Finish the
  768-token condition; if Llama comes out null or negative, build a larger-budget
  sensitivity SEPARATELY, never substituting it for the main result.

- **POST-UNSEAL RULE: the main analysis 口径 is CLOSED.** The predictor, features, marker
  adapter, workpoints and success criteria may not be changed in light of this result.
  Any further analysis is **exploratory** and must be labelled so.

## P3 supplement: CoT condition transfer (COMPLETE + FROZEN 2026-08-30)

> **`docs/PREREG_P3_SUPPLEMENT.md` is the protocol** (`p3-supp-v1`, tag `p3-supp-frozen`),
> amended additively by `docs/p3_supp_amendment_0{1,2,3}.json` — none overwrites its
> predecessor. This section is provenance and the run order.

<!-- Why: the supplement's whole claim to being "locked" rests on the CoT cells not
existing when the direction was frozen; a reader who treats it as another blind
validation will over-claim, and one who treats it as post-hoc will under-claim.
Evidence: p3_supp_predictions.json cot_cells_exist_at_freeze_time=false.
Scope: every supplement citation. -->
- **TWO PARTS, DIFFERENT EVIDENTIAL STATUS, and the split is the point.** Part 1 (the four
  CoT cells) is a **locked prospective condition test** — they did not exist when
  `p3_supp_predictions.json` was frozen. Part 2's No-CoT half (`AdaDopamine_gsm8k.md` §5.5's commitment
  panel) is **EXPLORATORY and permanently so** — those ten cells were unsealed first.
  **This is NOT a new blind dataset validation**: same 300 questions, same gold. What is
  prospective is the CONDITION.

- **FOUR CELLS ONLY: llama `0`/`−6`, qwen `0`/`+8`.** The α is the workpoint already
  established on GSM8K and already used in P3 — **not re-searched, and no CoT dose curve
  is run**. Everything else is held identical to the P3 No-CoT cells (same questions and
  order, `768`/`0.0`/bs `24`, prefill-only `tail=1` at the `Answer: ` anchor). **Both cells
  of one model share a card** — the primary metric is a per-question paired contrast
  between them.

- **`--cot` inserts EXACTLY one line, `Let's think step by step.`** Verified by diffing the
  two frozen templates: the `####` directive and the `Answer: ` anchor are untouched, so
  the injection site (token 220) and the commitment features keep the meaning they had
  when the predictor was fitted. The flag defaults to False, so every P3 caller is
  byte-identical, and it is recorded in meta — `prompt_template` alone would leave a
  consumer that reads only `alpha` unable to tell the two batches apart.

- **`meta.protocol` is `p3-supp-v1` when `--cot`, else `p3-v1`** (amend-01). A CoT cell
  carrying `p3-v1` would claim provenance from the CLOSED blind validation, and `cot=true`
  does not repair that — a consumer filtering on `protocol` would still pick it up as P3.

- **The freeze is TWO-STAGE, because the one-stage version was impossible** (amend-01).
  §1.5 originally listed "the frozen predictor's mean predicted score per cell" among the
  items frozen BEFORE generation — not computable, since the cells have no generations and
  therefore no features. The artifact was already correct (it contains no such field); only
  the protocol text overstated it. Stage 1 = freeze the **direction** of `ΔAcc` before
  generation (done, `fcf8b9c9b8fa8b70`); stage 2 = freeze the per-cell commitment score
  after generation but **before** accuracy.

- **Statistics.** Primary `ΔAcc = Acc(CoT+α) − Acc(CoT)` on `first_acc`, paired per
  question; exact two-sided McNemar with discordant counts; **the two models are ONE Holm
  family (m=2)**. The interaction `[CoT+α − CoT] − [NoCoT+α − NoCoT]` is **descriptive and
  EXCLUDED from Holm** — its No-CoT half is already unsealed, so it is not a locked
  prediction. Its four readings are pre-registered so the result cannot pick its own frame.

- **BUDGET POLICY IS STAGED AND CONDITIONAL (amend-03).** `768` stays primary. **A single
  extra cell at 1024 is explicitly rejected** — adding only No-CoT α=0 cannot separate a
  budget increase from a steering effect, because there is no α contrast at the new budget;
  a complete 1024 comparison needs a same-budget **2×2 per model**, `{No-CoT, CoT} ×
  {α=0, α=workpoint}`. Trigger order: run the 768 CoT cells → add llama CoT `0`/`−6` at
  1024 **only if** llama's CoT cap-hit makes truncation a live alternative explanation →
  add llama No-CoT `0`/`−6` at 1024 **only if** the interaction then needs a common budget.
  **Qwen is NOT auto-included** (its No-CoT cap-hit is 13–23% against llama's 91–96%); it
  gets a 1024 arm on its own evidence, never for symmetry. **Predictor scores on 1024
  output are EXPLORATORY** — the coefficients were fitted on 768-token generations, and
  `posN` is a normalized position while `early_candidate` is a first-line rule.

- **PREFLIGHT PASSED 2026-08-30** (5 samples, format only): `protocol=p3-supp-v1`,
  `cot=true`, the template carries the CoT line, `####` parses, and
  `steering_fires` read **0 / 45** (llama, `9×5×1`) and **0 / 30** (qwen, `6×5×1`).
  The four formal cells then read exactly **2700 / 0** (llama) and **1800 / 0** (qwen),
  n=300 each, one shared digest `48cc7635…`, no label fields.

### RESULT (2026-08-30) — `AdaDopamine_gsm8k.md` §5.6 is the results document

- **BOTH MODELS MATCHED THE LOCKED DIRECTION AND BOTH SURVIVE HOLM (m=2).** Llama
  α=−6: `.2000 → .2600`, **+6.00 pp**, discordant 27/9, McNemar p=.00393, Holm
  p=.00393, bootstrap [+2.33, +10.00]. Qwen α=+8: `.3800 → .5133`, **+13.33 pp**,
  58/18, p=4.71e−06, Holm p=9.42e−06, [+8.00, +19.00].
- **The interaction was NOT DETECTED and is EXCLUDED from Holm** — its No-CoT half
  was already unsealed, so it is not a locked prediction. Llama −0.33 pp
  [−6.00, +5.33], Qwen −3.00 pp [−9.67, +4.00]; both CIs contain 0, landing on the
  pre-registered `≈ 0` row ("steering's effect does not depend on CoT"). **Write
  "not detected", never an equivalence claim** — the CIs are wide.
<!-- Why: the trigger is about whether truncation EXPLAINS the result, not about the
cap rate; reading the 95-96% figure alone would fire stage 2 unnecessarily.
Evidence: docs/p3_supp_result_20260830.json budget_and_truncation.
Scope: amendment 03 only. -->
- **Amendment 03's stage-2 trigger DOES NOT FIRE, and the reason is a measurement,
  not a judgement call.** Llama's two cells do **not** differ in cap-hit (.953 vs
  .963, paired discordant 13/10, **p=.678**), and the effect is unchanged on the
  **276 questions where BOTH cells truncated** (+6.16 pp, p=.00455, against +6.00 pp
  overall) — truncation is a ceiling both cells sit against equally. Qwen's cap-hit
  *falls* under steering (.233 → .103) and its effect is **larger** on the 207
  untruncated questions (+14.98 pp, p=5.5e−06). **No 1024 cells are run**; the claim
  stays a fixed-768-budget result.
- **Stage 2 commitment (frozen BEFORE accuracy):** llama score .6454 → .6797,
  early-cand .437 → .303; qwen .7860 → .8830, early-cand .960 → .083. Both moved
  positive, the direction stage 1 predicted for accuracy. **The score is a
  descriptive commitment readout, NOT a calibrated accuracy estimate** — the
  predictor was fitted on 768-token No-CoT GSM8K output.
<!-- Why: posN=0 reads as the premature-lock-in failure signature, but here it sits at
the WORKING point with accuracy improving, so the obvious reading is backwards.
Evidence: docs/p3_supp_result_20260830.json llama_answer_first_pattern.
Scope: llama GSM-Hard CoT; check before reusing posN as a degradation marker. -->
- **EXPLORATORY: llama's answer-first pattern.** Under CoT α=−6, **91 of 156
  committed samples (58.3%) begin with a parseable `#### N`** and only then write
  the Step-by-step reasoning — `posN` median exactly **0.0000**. The pattern is present
  under both manipulations: No-CoT α=0/−6 reads 3.1%/24.4% of committed, while CoT
  α=0/−6 reads 11.3%/58.3%. This is AdaDopamine_gsm8k.md §5.5's premature-lock-in signature
  appearing at the **working point** rather than the overshoot point, **with accuracy
  still improving +6.00 pp** — so a low `posN` is not by itself a degradation marker,
  and the commitment score's rise must NOT be read as "reasons before answering".
  `answer-first` is frozen as the first non-whitespace content being a parseable
  `#### <number>`; it is deliberately NOT called the "first token" because the marker
  spans multiple tokenizer tokens. Its denominator is the committed subset in every
  cell. `p3/answer_first_panel.py` recomputes all four accuracies through the frozen
  extractor and asserts exact agreement with `p3_evaluation.json` /
  `p3_supp_evaluation.json`.
- **Artifacts:** stage 1 `fcf8b9c9b8fa8b70`, stage 2 `6a16d4d862edbdaa`, evaluation
  `7e39f9e36edfd0c4`; full record `docs/p3_supp_result_20260830.json`.
- **`run_p3_supp_eval.py` REFUSES to start without `p3_supp_commit.json`** — the same
  ordering guard `run_p3_eval.py` has. Both new scripts were mutation-tested before
  use: ten guards on the stage-2 freeze and six on the evaluator each verified to
  fire with the right message, against a passing unmutated control.

**Both runbooks (server generation, offline freeze/eval chain) moved to
`docs/P_PHASE_ARCHIVE.md` (2026-09-02).** The offline scripts refuse to overwrite a
frozen file, so re-running the chain requires deliberately deleting it.


## P4 fixed-workpoint transfer to generative logical reasoning (LogiQA 2.0)

> **`docs/PREREG_P4_LOGIQA2.md` is the protocol** (`logiqa2-p4-v0`, stage 0,
> commit `cb796ea`), amended additively by `docs/p4_amendment_0{1,2,3,5,6}.json`
> — none overwrites its predecessor. **`04` is deliberately absent from that list:
> it is referenced by `p4-amend-02`/`05` but is NOT on disk — see the bullet at the
> end of this section before citing it.** `docs/P4_LOGIQA2_PREFLIGHT_OUTCOME.md` is
> the preflight record. **Status: COMPLETE + FROZEN 2026-09-01 — DOUBLE NULL.**

### RESULT (scored 2026-09-01, `docs/p4_logiqa2_evaluation.json`)

**Neither model survives Holm (m=2), and llama's raw p is not significant
either** — llama `−6` `.5633 → .5200` = **−4.33 pp**, discordant 13/26, exact
McNemar **p=.0533**, `p_adj=.107`, CI `[−8.33, −0.33]`; qwen `+8`
`.6400 → .6500` = **+1.00 pp**, 33/30, p=.8013, `p_adj=.801`, CI
`[−4.00, +6.33]`. **The GSM8K workpoint did not transfer to neutral
single-stage generative multiple-choice reasoning.**

**FIRST sensitivity is same-signed with MAIN in both models** (llama −3.33 pp
vs LAST-MAIN −4.33; qwen +0.67 vs +1.00), which is what excludes "the LAST
parser or a tail revision manufactured this". Given llama's degeneration rate
that exclusion is not a formality.

**`answer_first` .757 → .330 is a MARKER artifact and must NOT be cited as
answer-formation timing.** The field is frozen as "first non-whitespace content
is a parseable `Final answer: X`", so `B\nFinal answer: B` and
`The correct answer is A … Final answer: A` are FALSE by definition while the
model has plainly already committed. An exploratory text scan reads
**86.7% → 54.3%** instead. Correct wording: "α=−6 reduced canonical marker-first
and increased pre-marker text; explicit option candidates also moved later, but
a large share of outputs still name an option before or at the opening of the
argument."

**Marker delay did NOT buy better judgement**, which is the load-bearing
observation. On the 146 items that moved marker-first → non-marker-first,
9 improved and 17 worsened (52.1% → 46.6%); globally α changed the TEXT of
246/300 items but the final option of only 61/300, and within those 13 improved
and 26 worsened. Post-treatment stratification — consistent-with evidence,
never mediation.

**Qwen's null is the one that closes the "MCQ leaves no room to reason" escape
hatch.** It compared ≥2 options before the marker in 270/300 (α=0) and 258/300
(+8), had **zero** strictly answer-first outputs in both cells, and median
pre-marker chars barely moved (1401 → 1370). Reasoning WAS expanded; the
workpoint still did nothing. Its generation behaviour is not inert either
(degeneration .133 → .217, multi-marker .483 → .617, budget exhaustion
.157 → .260, 74/300 changed answers with a net ≈ 0) — so write "+8 broadly
perturbed the trajectory with a near-zero net effect on accuracy", never "+8
had no effect on Qwen".

**Llama's outputs mix two things and the record must say so**: task-specific
logical judgement, and severe continuation/termination degeneration under the
bare generative protocol (`Final answer: D` repeated, `Skill 1a / Skill 1b …`
training-corpus continuation, drifting into unrelated problems, marker loops to
the budget cap).

**What this result does NOT establish.** LogiQA changed the reasoning content
AND the answer space (constructed number → four-way choice among given
candidates) at once, so the null cannot separate a choice-interface effect from
a reasoning-type effect. **P4b (BBH numeric) is the control built for exactly
this**, and even it cannot fully separate them — that needs a within-item
with/without-options contrast, which has not been run.

**The question is FIXED-WORKPOINT TRANSFER, not selection.** Does a workpoint
established on GSM8K (llama `−6`, qwen `+8`) still help on generative logical
reasoning, without re-searching α? **Never conflate this with P2B**, which asks
whether commitment features can PICK the best dose — they already diverge on
Qwen, whose GSM8K workpoint is `+8` while its MATH optimum is `+6`.

**This is NOT a blind validation.** LogiQA 2.0 gold is public and reachable; P3
(GSM-Hard) was the blind test and is CLOSED. What is prospective here is the
task, not sealed data. A null on both models is a **task-boundary result** and
is reported with equal prominence.

- **Two-stage freeze, and the split is the point.** Stage 0 fixed the question,
  doses, sampling, parsing, statistics and the token-budget upgrade rule BEFORE
  any loader existed. Stage 1 (`p4-amend-04`) added ONLY the budget the blind
  preflight selected. Anything else changing between them would be an
  adaptive edit.
- **`data_logiqa.py` (LogiQA 1.0-era) downloads the same file and is left
  byte-unchanged.** It is unusable here for four reasons: it writes to the
  stale `/data2/.../RolePlaying` tree, drops `id` / raw fields / `type`
  provenance, freezes no manifest, and has no generative runner. `data_logiqa2.py`
  is new.
- **The item key is a COMPOSITE, and this is load-bearing.** Measured on the
  1572-row test split: `id` is unique only 1568/1572 and `(passage, question)`
  only 1557/1572 — **neither is a key**. The frozen key is
  `sha256(salt:id US passage US question US options.join(RS))`, unique 1572/1572.
  Never `hash()` (process-salted). Both the official `id` and a salt-free
  `content_sha256` are stored, so an upstream edit is detectable.
- **Sample: 300 items, exactly 75 per gold label**, taken as ranks 1–75 by key
  within each label; the 20 preflight items are ranks 76–80, so **disjointness is
  structural, not checked after the fact**. Frozen digest `4d4b25e071a2a6dd`.
  Label pools are 347/384/417/424, so no stratum is starved.
- **`type` is a MULTI-LABEL dict (3–4 reasoning types per row) and is provenance
  only, never a stratification axis.** Splitting 300 items across an unbalanced
  multi-label space is the sampling problem that shelved MMLU-Pro; it is avoided
  here rather than re-imported.
- **The prompt is NEUTRAL-GENERATION and frozen verbatim** (`p4-amend-02`,
  sha256 `c42dc9c81f117a6c`, anchor `Response: `). `p4-amend-01`'s original
  prompt said "Think step by step." with a `Reasoning: ` anchor — an explicit
  CoT elicitation, so a reasoning trajectory could not be attributed to the model
  rather than the instruction. Deleting the sentence alone was insufficient:
  `Reasoning: ` itself names the expected output as reasoning. **Measured, not
  assumed:** the tail is token **220 `' '`** on BOTH tokenizers (llama
  `382/2647/25/220`, qwen `382/2582/25/220`), and `rstrip()` moves it to `':'`
  (25) on both — the trailing space is an assertion. **Do not write that the two
  models share "the same decision-bottleneck token"**: an identical integer in
  two tokenizers is a numbering coincidence.
- **Parsing: MAIN = LAST match, sensitivity = FIRST, and the reason differs from
  MATH's on purpose.** MATH uses FIRST because tail loops pollute the last
  `\boxed{}`; LogiQA uses LAST because the generative protocol invites revision
  and FIRST would lock a pre-reasoning guess. Zero markers scores incorrect,
  denominator stays 300, **no rescue generation**. The instruction still names
  `Final answer: X` verbatim, which is also why **no stop string may be used** —
  HF `stop_strings` matches anywhere in the output and the model restates the
  instruction (the CGT failure, invalid_rate 0.02→0.11).
- **The label firewall is STRUCTURAL on both blind paths, and the distinction is
  one I got wrong once.** `data_logiqa2.py` emits the formal sample twice from
  one selection: `logiqa2_p4_formal_blind.json` (gold absent, whitelist-built,
  payload-scanned) for generation and `logiqa2_p4_formal.json` (gold present)
  for `eval_logiqa2.py` only. An earlier runner read the gold-bearing file and
  merely declined to touch `answer_letter` — **"the code does not access gold"
  is much weaker than "gold is not reachable"**, which is the whole point of the
  P2 firewall. `get_answer_logiqa2.py` refuses any input whose meta does not say
  `contains_labels: false`.
- **Generation and scoring are separate scripts**, mirroring P3: a generation run
  cannot quietly become an accuracy run.
- **PREFLIGHT RESULT (20 held-out items × 4 cells, 512 tokens, gold-blind):
  Llama is SCORABLE BUT ITS TERMINATION IS DEGRADED; Qwen is normal.** llama
  α=0 20/20 cap-hit (min length 512, **no output stopped naturally**),
  `answer_first` 13/20, degenerate 17/20, first marker at position **0.000**;
  α=−6 `answer_first` 4/20, degenerate 14/20. qwen 0/20 `answer_first`,
  degenerate 0–1/20, first marker at 0.94–0.98, cap-hit 2–4/20. FIRST and LAST
  agree in **39/40** scorable outputs.
- **Degeneration does NOT make it unreadable — this repo's GSM8K precedent is
  strictly more extreme** (97% cap-hit, raw loop rate 74–88% with no clean α
  trend) and remains a main result source. Accuracy is readable because the
  answer is complete BEFORE the degeneration. Adopted with GSM8K's convention:
  **the degeneration rate is descriptive and must NOT be read as perseveration
  absent a clean α trend**, and it uses the same strict `is_loop` detector
  (final 40-char block recurring ≥4×) so the two tasks stay comparable — a
  permissive n-gram proxy read 80–86% on GSM8K and was all false positives.
  One real difference is recorded rather than smoothed over: GSM8K's `####` is
  terminal so its loop is always post-submission, while llama α=0 here is 13/20
  `answer_first`, a different shape.
- **The format gate was RELAXED and the budget rule was EXECUTED AS WRITTEN —
  opposite directions, deliberately** (`p4-amend-05`). The gate was written
  against "cannot produce the format", which is not what happened, so one valid
  marker now makes an output scorable and only a cell with NO marker anywhere is
  a hard stop. The budget rule fires because of repetition rather than long
  reasoning, but it states a MECHANICAL condition; keeping 512 would have been a
  SECOND adaptive change made in response to preflight output. **Budget = 1024**,
  with longer llama tails as the accepted, recorded cost.
- **`answer_first` 13/20 → 4/20 at α=−6 is a PILOT-GENERATED TIMING HYPOTHESIS,
  not effect evidence**: n=20, held-out items, a sample authorised only for
  fixing the budget, and degeneration stayed high (17/20 → 14/20) rather than
  falling with it. Direction is consistent with GSM8K's `early_candidate` minimum
  at the same workpoint.
- **Do NOT write "externalized reasoning emergence increased."** Text before the
  marker is not necessarily reasoning, and no frozen content-judgement rule
  exists. Report morphological fields only: `answer_first`, `pre_marker_chars`,
  `first_marker_pos`, `multi_marker` / `degenerate` rate. **`first_marker_pos` is
  AUXILIARY** — a longer repetition tail lowers it independently of anything
  preceding the marker, so it cannot carry a "more reasoning" reading alone.
  Qwen's 0.94–0.98 means the marker is LATE, **not** that the preceding text is
  complete or correct reasoning.
- **Descriptive fields are PREFLIGHT-INFORMED SECONDARY**: outside the Holm
  family, and themselves outcomes of α, so stratifying accuracy on them is
  post-treatment stratification — consistent-with evidence, never mediation.
- **Statistics**: primary `ΔAcc` on MAIN parsing paired per item; exact two-sided
  McNemar with discordant counts; question-level paired bootstrap 95% CI
  (B=10000, seed 0); **Holm over the TWO MODELS (m=2), judged only when both are
  complete** — a partial family reports an `m=1` adjustment under an `m=2` label.
- **`p4_amendment_04.json` (the stage-1 budget freeze) is REFERENCED BY `p4-amend-02` and `p4-amend-05` BUT IS NOT ON DISK (verified 2026-09-03).** It was produced by the earlier warning-then-write decider build, which warned on llama α=−6's 19/20 AND wrote the artifact anyway; re-running the current gate-corrected code reproduced it with NO differing keys, which is why it was kept rather than renumbered. That reproduction is the only surviving evidence — **the file itself is absent, so the numbering is preserved by the two amendments that name it, not by an artifact.** Do not renumber `04` and do not silently regenerate it; if it is ever needed, regenerate it deliberately and record that it is a reconstruction. The decider defect is fixed at `407e6ab`. Provenance in `p4-amend-06`.
- **Two frozen-artifact rules this line established the hard way.** A generator of
  frozen artifacts must ship **no `--allow_overwrite` escape hatch** — "refuses to
  overwrite" is not true while a flag bypasses it. And a **format violation is a
  hard stop BEFORE any write**: warning-then-write both downgrades the rule and
  leaves a stage-1 freeze derived from non-conforming output on disk.
- **A launcher bug worth not repeating:** `${1:?usage: ... {llama3|qwen2.5}}`
  ends the parameter expansion at the FIRST `}`, so `MODEL` became the literal
  `llama3}`. `bash -n` is clean on this — the syntax is valid, the semantics are
  not. Always invoke a launcher with real arguments, not just a syntax check.
- **`MODEL_DIR` is an HF REPO ID and the mask dir is `${MODEL}_non_logits`** (not
  `_nmd_logits`), matching every other launcher; there is no filesystem model
  tree. A nonexistent path is handed to transformers, which parses it as a repo
  id and raises an `HFValidationError` naming neither the launcher nor the
  missing directory. All cheap checks now run BEFORE the model loads.

```bash
# Server, from /data1/paveen/Dopamine. One model per card; both cells of a
# model stay together (bf16 greedy is not byte-reproducible across GPUs and the
# two cells are a paired contrast). The two MODELS may run on two cards.
python data_logiqa2.py --out_dir components/benchmark   # digest must read 4d4b25e071a2a6dd

CUDA_VISIBLE_DEVICES=0 nohup bash run_logiqa2_formal.sh llama3  > p4_formal_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_logiqa2_formal.sh qwen2.5 > p4_formal_qwen.log  2>&1 &
cat p4_formal_llama.log   # immediately -- a wrong PY exits 127 and the log looks empty

# Scoring is a SEPARATE script and the only one that reads gold. Run it once,
# after BOTH models finish (a single model is scored with Holm WITHHELD).
python eval_logiqa2.py \
  --generations components/logiqa2/formal/formal_llama3.json \
                components/logiqa2/formal/formal_qwen2.5.json \
  --formal_file components/benchmark/logiqa2_p4_formal.json \
  --out docs/p4_logiqa2_evaluation.json
```

## P4b fixed-workpoint transfer to BBH numeric reasoning (COMPLETE — DOUBLE NULL)

> **`docs/PREREG_P4B_BBH.md` is the protocol** (`bbh-p4b-v0`, frozen 2026-09-02
> before any α=0 cell existed), amended additively by
> `docs/p4b_amendment_01.json`. **`AdaDopamine_gsm8k.md` §5.8 is the results
> document.** **Status: `object_counting` is COMPLETE — all six cells run, a
> DOUBLE NULL on the primary test.** The stage-0 and audit blocks below are
> RETAINED AS HISTORY and were not rewritten in light of the result; the RESULT
> block is appended after them.

### STAGE-0 RESULT (`object_counting`, 2026-09-02, `docs/bbh_p4b_object_counting_stage0.json`)

**Both models PASS the frozen `[0.30, 0.85]` headroom gate on α=0 `first_acc`:
llama3 `.4160` (last `.4120`), qwen2.5 `.5520` (last `.5600`).** Both sit ~4–5×
the `.104` majority-class rate and 30–43 pp below the ceiling, so a later null on
either model **cannot** be a baseline-ceiling artifact — the only thing this gate
exists to establish. The majority-class rate stayed descriptive and did not move
the interval.

**The public ≈`.90` figure is NOT comparable to `.416`** — it uses BBH's 3-shot
CoT format, which this protocol deliberately does not inherit. It explains why a
headroom gate was set at all, nothing more.

**Diagnostics — recorded, NOT a gate, and not grounds to change prompt, parser or
budget.** Llama shows a looping/truncation **PHENOTYPE SIMILAR TO GSM-Hard**;
similar phenotype, **not** an established shared mechanism. cap-hit `.956` (decode
median exactly 768), no-marker `.092`, multi-marker `.840`, chars med 1988. Qwen
does not loop (cap-hit `.036`, chars med 423). **Consequence:** Llama's accuracy
stays valid as a **fixed-768-budget estimand**, but its length-derived readouts
press against that cap and are **not natural lengths**; if α moves truncation or
marker-absence, both travel with the main result. `last_acc` sensitivity is
retained even though first/last currently agree to ≤0.8 pp.

**Gold provenance:** the server gold is byte-identical to a locally regenerated
copy (`questions_sha256 4cfbf739e1fe7870`, matching both generation cells), so
identity is proved by digest rather than assumed.

### EARLYCAND AUDIT: PASS, with a ceiling limitation that constrains its use

Order preserved: 30 items frozen from the blind file → blind CSV with **no**
detector flag / gold / accuracy → manual labels written → **only then** was
`earlycand-v1` run. **Precision 1.000, recall 1.000, agreement 30/30, detector
positives 29** — clearing the frozen minimum of 10, so the `INCONCLUSIVE` branch
does not fire. The single FALSE (`id 39`) is the one generation opening with
working rather than a bare number; the other 29 open with a standalone digit and
justify afterwards.

<!-- Why: the audit validates the detector but its baseline is saturated, so the
readout can only fall; reading a flat rate as "α does not move timing" is the
baseline-ceiling error the accuracy gate exists to exclude, relocated to timing.
Evidence: docs/p4b_earlycand_audit_result_object_counting.json.
Scope: every P4b early_candidate number on object_counting. -->
**CRITICAL: the α=0 base rate is at CEILING (llama `.952`, qwen `1.000`).** The
audit shows the detector reads this task correctly, but a saturated baseline
leaves headroom only for a **decrease**. **A flat rate under α must NOT be read as
"α does not move answer-formation timing."** Scope: **EXPLORATORY, outside Holm**;
`early_candidate` is an OUTCOME of α, so stratifying accuracy on it is
post-treatment stratification — consistent-with evidence, never mediation. The
detector was not re-tuned. Labels come from **one annotator**; the 29/30 split is
morphologically stark but no inter-rater agreement was measured.

### `p4b-amend-01`: one opposite-signed DIAGNOSTIC cell per model

Additive. The primary test is unchanged — fixed-workpoint transfer, α read from
the frozen GSM8K record (llama `−6`, qwen `+8`), **Holm m=2 over the two workpoint
contrasts only**. Adds ONE reverse dose per model (**llama `+4`, qwen `−6`**) to
read whether the direction ordering continues (`−6 > 0 > +4`; `+8 > 0 > −6`).

**Binding:** the reverse cell is **outside the Holm family**, its p is
**unadjusted**, and it **MUST NOT redefine the workpoint** — a reverse cell that
scored higher would still not become the workpoint, because this protocol never
searches doses on BBH. `run_bbh_numeric.sh REVERSE` refuses until **both** the α=0
and the workpoint cell exist, so it cannot be mistaken for the search that
produced the workpoint. **One point is not a curve:** it can show an ordering
continues or breaks, never a peak, an inverted-U, or an "overshoot point".

<!-- Why: with two steered cells `[x for x in byalpha if x != 0][0]` returns the
SMALLEST alpha, which for qwen is the DIAGNOSTIC -6 -- it would have been reported
as the primary transfer result while the real workpoint went unreported.
Evidence: eval_bbh_numeric.py workpoint selection; verified on a fixture where the
reverse dose beats the workpoint and +8 still occupies the Holm table.
Scope: any scorer holding more than one steered cell per model. -->
**A latent hazard the reverse cell exposed:** the scorer selected the steered α as
`[x for x in byalpha if x != 0][0]`, fine with one steered cell but wrong with
two. It now selects `WORKPOINT[model]` **explicitly**. Verified on synthetic
fixtures including the case that matters — a qwen fixture where the reverse dose
(`.696`) BEATS the workpoint (`.532`) still puts `+8` in the Holm table, reports
`−6` separately as `BREAKS` with an unadjusted p, and does not promote it. An
unfrozen α (`−4`) is refused; Holm stays m=2 with reverse cells present.

**All THREE cells of one model stay on that model's original physical GPU**
(llama GPU 0, qwen GPU 1) — paired per-item contrasts, and bf16 greedy is not
byte-reproducible across GPUs. `WORKPOINT` and `REVERSE` of one model run
**sequentially** (same card, and `REVERSE` checks the workpoint cell exists).

### RESULT (`object_counting`, 2026-09-02, `docs/bbh_p4b_object_counting_result.json`)

> **`AdaDopamine_gsm8k.md` §5.8 is the results document.** This block is
> provenance and the 口径 traps. The stage-0 and audit blocks ABOVE are history
> and were not rewritten in light of this result.

**DOUBLE NULL. Neither model survives Holm (m=2), and neither raw p is
significant either** — llama `−6` `.4160 → .4080` = **−0.80 pp**, discordant
27/29, exact McNemar **p=.8939**, `p_adj=1.0000`, CI `[−6.80, +5.20]`; qwen `+8`
`.5520 → .5760` = **+2.40 pp**, 34/28, p=.5258, `p_adj=1.0000`, CI
`[−4.00, +8.40]`. **The GSM8K workpoint did not transfer to BBH numeric
reasoning.**

**LAST sensitivity is same-signed with MAIN in both models** (llama −0.40 pp vs
MAIN −0.80; qwen +0.40 vs +2.40), which is what excludes "the LAST parser or a
tail revision manufactured this". Given llama's 88–95% degenerate-tail rate that
exclusion is not a formality. **First/last marker revision is small on this
task**: llama `first .4160 / last .4120`, qwen `.5520 / .5600` — later markers
rarely flip correctness, which is NOT the same as "reasons before answering".

**Six cells, one shared digest `4cfbf739e1fe7870`, n=250 each,
`steering_fires` exactly 2250 (llama, L=9) / 1500 (qwen, L=6) at α≠0 and 0 at
α=0.** All carry `protocol=bbh-p4b-v0`, `cot=False`, `max_new_tokens=768`.

| model | α | role | first_acc | Δ pp | disc. | raw p | Holm p_adj | CI95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| llama3 | 0 | baseline | .4160 | — | — | — | — | — |
| llama3 | **−6** | **workpoint** | .4080 | **−0.80** | 27/29 | .8939 | **1.0000** | [−6.80, +5.20] |
| llama3 | +4 | reverse diag. | .3280 | −8.80 | 10/32 | .0009 | *unadjusted* | [−13.60, −4.00] |
| qwen2.5 | 0 | baseline | .5520 | — | — | — | — | — |
| qwen2.5 | **+8** | **workpoint** | .5760 | **+2.40** | 34/28 | .5258 | **1.0000** | [−4.00, +8.40] |
| qwen2.5 | −6 | reverse diag. | .5680 | +1.60 | 27/23 | .6718 | *unadjusted* | [−4.00, +7.20] |

**THE REVERSE CELL DOES NOT REDEFINE THE WORKPOINT, and on qwen it would be
tempting to let it.** Ordering **BREAKS on both**: llama is `0 (.416) > −6 (.408)
> +4 (.328)` — α=0 already beats the workpoint, so `−6 > 0 > +4` fails; qwen is
`+8 (.576) > −6 (.568) > 0 (.552)` — the reverse dose sits ABOVE α=0, so
`+8 > 0 > −6` fails too. **Llama `+4` (p_raw=.0009) is the ONLY significant
contrast anywhere in this task and it is a DEGRADATION**, consistent with
over-steering damage rather than with a direction ordering; it is outside Holm
and its p is unadjusted. One point is not a curve: it can show an ordering
continues or breaks, never a peak or an inverted-U.

<!-- Why: qwen +8's early-candidate rate falls 100.0 -> 44.4 and bare-first-line
54.4 -> 0.0, which reads as "reasoning before marker" until you open the text --
a large share of the turned-off group merely restates the answer in prose or as an
English number word. Writing "reasons first" would overstate a mixed regime.
Evidence: docs/bbh_p4b_object_counting_behavior.json; qwen +8 sid=1 (genuine),
sid=2 'three', sid=17 prose-first.
Scope: every BBH timing claim. -->
**BEHAVIOUR DID MOVE, and this is the finding accuracy cannot show — but it is a
MIXED REGIME.** Qwen `+8` drives bare-first-line **54.4% → 0.0%**,
`early_candidate` **100.0% → 44.4%**, multi-marker **92.0% → 36.0%**, while the
reverse cell `−6` does not move at all (100.0 / 54.8 / 92.4) — so this is
`+α`-specific, not arbitrary perturbation. **But it must NOT be summarised as
"先算后答".** Of the 139 samples that turned off, only some genuinely expand the
count first (`sid=1`: itemised list, then the sum, marker last); others merely
change the SURFACE FORM of an answer still stated first — `sid=2` opens with the
English word `three`, `sid=17` opens with
`You have a total of 9 objects (3 fridges + 1 bed + 5 stoves = 9)`. Llama `−6`
moves the same direction far more weakly (95.2 → 84.4) while `+4` saturates it
(99.2); its three cells' morphology is near-identical (digit → `Explanation:` →
degenerate `####.####…` tail), representative `sid=7`.

**`pre_marker_chars` (313 → 44) and `posN` (.757 → .699) are FORMAT DIAGNOSTICS
ONLY and must never be cited as "the answer moved later" — they move the OTHER
way.** They mix the "early empty `####`, first parseable marker later" case, so
the fall is an artifact of marker bookkeeping. The timing reading rests on the
audited `early_candidate` flag plus raw-text morphology.

**`early_candidate` is frozen as `earlycand-v1` and was NOT re-tuned for this
task.** Definition: the FIRST NON-EMPTY LINE is (1) ≤60 chars stripped,
(2) contains a number token, (3) is NOT a numbered reasoning opening
(`1. To find …`), (4) is NOT a bare heading (`Step 1:` / `Solution:`) — an
answer-shaped bare number written before any derivation. Its blind audit on THIS
task passed 30/30 (precision 1.000, recall 1.000), **but its α=0 base rate is at
CEILING (llama .952, qwen 1.000)**, so only a DECREASE is measurable and a flat
rate must NOT be read as "α does not move answer-formation timing".

**Stratifying accuracy on `early_candidate` is POST-TREATMENT stratification —
consistent-with evidence, never mediation.** The flag is an OUTCOME of α. The
paired split (EXPLORATORY, outside Holm, unadjusted p) is nonetheless recorded
because it cuts BOTH ways and a one-sided reading would be wrong: qwen `+8`
turned_off n=139 `.554 → .691` (gained 30 / lost 11, p_raw .0043) while
stayed_early n=111 `.549 → .432` (gained 4 / lost 17, p_raw .0072); llama `−6`
turned_off n=39 `.231 → .590` (14/0, p_raw .0001) while stayed_early n=211
`.450 → .374` (13/29, p_raw .0195). **There is no matched α=0 control group** —
qwen's α=0 early-candidate rate is 1.000 — so the between-group difference
cannot be attributed to α.

**Qwen carries ~50% training-corpus continuation in ALL THREE cells** (`You are
an AI assistant …`): 50.8 / 42.8 / 50.4 at 0 / +8 / −6. **α neither created nor
removed it.** It sits AFTER the marker, so it does not affect first-caliber
extraction, but it means this batch's termination behaviour is not clean
independently of steering.

**LLAMA IS A FIXED-768-BUDGET ESTIMAND.** Degenerate-tail rate 92.8 / 88.4 / 94.8
and `chars_median` 1988 / 1928 / 2003 across 0 / −6 / +4. The paired comparison
stays valid — every cell was evaluated under the same 768-token budget — but the
accuracies may NOT be read as unconstrained reasoning capability, and llama's
length-derived readouts are NOT natural lengths. **Frozen wording: "Llama 出现与
GSM-Hard 相似的循环/截断表型" — a SIMILAR PHENOTYPE, never an established shared
mechanism.** Qwen does not loop (degenerate 3.6–4.8%, `chars_median` 322–422).

**FROZEN OUTCOME WORDING.** Restoring the answer space and the submission
interface was **not sufficient** to restore transfer; this does **NOT** establish
that reasoning content is the binding constraint — separating a choice-interface
effect from a reasoning-type effect needs a within-item with/without-options
contrast, which is a different experiment and is not authorised here. Write
**"GSM8K 上的行为 signature 部分迁移"**, never "mechanism transferred". And:
**改善答案提交时序不是跨任务准确率提升的充分条件.**

**Behaviour numbers are RECOMPUTED, never hand-copied into Markdown.**
`analyze_bbh_behavior.py` reads the six generation files and writes
`docs/bbh_p4b_object_counting_behavior.json`; it IMPORTS the frozen
`early_candidate_detector` (`--detector_dir`) rather than reimplementing it, and
`--expect_acc` makes it recompute `first_acc` per cell and **die** unless it
reproduces the frozen eval exactly — that reproduction is the acceptance check
for any future edit. It refuses to overwrite an existing output. Its
`degenerate_tail` uses the strict GSM8K detector (final 40-char block recurring
≥4×), so the rate stays comparable with GSM8K; a permissive n-gram proxy read
80–86% there and was all false positives.

**POST-RESULT RULE: the P4b main analysis 口径 is CLOSED.** The predictor,
features, marker adapter, workpoints and success criteria may not be changed in
light of this result. Do not add doses, do not re-search α on BBH, do not promote
the reverse cell. Any further analysis is **exploratory** and must be labelled so.

**P4b IS NOT A NEW PHASE — it is the SECOND TASK of P4's question.** It was
briefly numbered `P5`; that was renamed before any artifact existed, because
`P5` implied a stage that does not exist. Do not renumber it back.

**Why it exists.** P4's LogiQA null cannot say *why*, because LogiQA changed two
things at once relative to GSM8K:

| | answer space | reasoning content | submission |
|---|---|---|---|
| GSM8K | model constructs an integer | arithmetic word problems | `####` |
| LogiQA 2.0 | choose among 4 **given** candidates | textual logic | `Final answer: X` |
| **BBH numeric** | model constructs an integer | counting / nested arithmetic | `####` |

The two BBH tasks restore the answer space and the submission interface to
GSM8K's. **That table compares three CHOSEN dimensions and is not exhaustive** —
BBH also differs in item format, domain, text distribution and generation shape,
so this is a between-task comparison, not a controlled single-factor
manipulation.

**FROZEN OUTCOME WORDING, both weaker than a causal claim.** Transfer → "the
workpoint transferred to an option-free numeric reasoning task, **consistent
with** the choice interface being part of the LogiQA boundary" — **never**
"LogiQA failed because of the options". Null → "removing the option interface
was **not sufficient** to restore transfer", which does **not** establish that
reasoning content is the binding constraint. **Separating a choice-interface
effect from a reasoning-type effect needs a within-item contrast (the same items
with and without options); that is a different experiment and is not authorised
here.**

- **Task tier is fixed in advance so task choice is not a post-hoc narrative**:
  `object_counting` → `multistep_arithmetic_two` (only if the first fails the
  gate for BOTH models) → `dyck_languages`, the last **unimplemented and
  unauthorised** (it changes the answer space to a symbol string and needs its
  own parser, prompt and gate). `--task` is an allowlist of the two numeric
  configs; a third cannot be smuggled through it.
- **`lukaemon/bbh` pinned to `982bb89fd79532a8ac676a61fc42eb1aeec63f99`**, `test`
  split, **all 250 items, no sampling** (so no salt and no selection step).
  `maveriq/bigbenchhard` is script-based and would hit current `datasets`'
  rejection of loading scripts — the same trap SIQA hit.
- **Measured at freeze:** `object_counting` 17 gold values 2…18, majority-class
  **.104**, digest `4cfbf739e1fe7870`; `multistep_arithmetic_two` 185 values
  −39960…250992, majority-class **.016**, digest `e69d300b94274ce3`. Both
  all-integer gold, so `normalize_gsm8k` applies unchanged.
- **Stage-0 gate is `α=0 first_acc ∈ [0.30, 0.85]`, judged PER MODEL, and the
  interval is FROZEN.** Its purpose is not to test α but to confirm a later null
  is interpretable rather than a baseline-ceiling artifact (the Qwen-MMLU-betting
  and pv6-Easy-bare failure). **The majority-class rate is recorded but does NOT
  move the interval** — it is the trivial constant-guess baseline, not a
  random-guess rate, and letting it adjust the gate after inspecting the data
  would make the gate itself adjustable. At .104/.016 it is moot, which is a fact
  about the data, not a licence to revisit the rule.
- **Format/truncation diagnostics are NOT a gate.** They exist only to confirm
  the 768 budget and the shared extractor did not fail *technically*. Seeing the
  output shape must not lead to a changed prompt, a redefined parser, or a
  re-tuned budget.
- **Per-model eligibility, but the two-model panel is the design.** Both eligible
  → Holm m=2. Exactly one → its workpoint cell still runs, pre-specified as
  **single-model exploratory transfer** with Holm **WITHHELD** and the raw p
  labelled unadjusted; running Holm at m=1 under an m=2 label is the error this
  prevents. Neither → no workpoint cell, move to the next task in the tier.
- **α is expressible ONLY as 0 or the frozen GSM8K workpoint** (llama `−6`, qwen
  `+8`). The launcher cannot name another dose and the scorer rejects one.
  `WORKPOINT` also refuses to run before the α=0 cell exists — the gate is judged
  on α=0, so running the workpoint first would make the gate unfalsifiable.
- **`cot=False` and `wording="plain"` are HARDCODED in the runner, not flags.**
  BBH ships 3-shot CoT prompts; inheriting them would add exemplars and explicit
  reasoning, re-mixing the variables this design exists to separate. The "pushy"
  wording is a known early-`####` inducer. Neither has a cell here.
- **Not a blind validation, and the files say so.** BBH gold is public; P3 was
  the blind test and is closed. What carries over is the OTHER half of the
  discipline: α read from the frozen record and never re-searched, predictor /
  features / marker adapter untouched, generation and scoring as separate
  scripts, sample frozen before any steered cell. The sample is still emitted as
  a blind copy (field whitelist, then asserted label-free) plus a gold copy, so
  generation cannot reach a label even by mistake.
- **`earlycand-v1` must be re-validated on this task BEFORE it may be called a
  commitment metric.** It was frozen on GSM8K, where a short first line with a
  number is answer-shaped; on `object_counting` the question IS a list of objects
  and the reasoning is a running count, so a leading number may merely restate
  the question. Its GSM8K blind audit (precision 1.000, recall .976) was on
  arithmetic text and does not transfer by assumption.
  `freeze_p4b_earlycand_audit.py` fixes **30 items from the frozen question
  digests before stage-0 generation** (ranked by salted `sha256` of the question
  text — never `hash()`, which is process-salted; `object_counting` selection
  digest `04508a19ce30b0b6`), because choosing them after seeing detector output
  would let the sample flatter the detector. The rubric is applied **without
  seeing the detector flag**; comparison happens only afterwards.
  **Three outcomes, all frozen**: pass → exploratory timing readout, outside
  Holm; fail → withdrawn as a timing metric, only marker/format description
  survives; **fewer than 10 detector positives among the 30 → INCONCLUSIVE**,
  report the raw counts (detector positives / manual positives / agreements) and
  claim no precision in either direction. In every case the accuracy main test is
  unaffected (it never uses the detector), the sample is not re-drawn or enlarged
  to chase positives, and **the detector is NOT re-tuned** — that would fork the
  definition against every stored GSM8K and MATH number.
- **`early_candidate` is an OUTCOME of α**, so stratifying accuracy on it is
  post-treatment stratification: consistent-with evidence, never mediation.

```bash
# Server, from /data1/paveen/Dopamine. Steps 1-2 precede ANY generation.
python data_bbh_numeric.py --task object_counting --out_dir components/benchmark
python data_bbh_numeric.py --task multistep_arithmetic_two --out_dir components/benchmark
#   digests must read 4cfbf739e1fe7870 / e69d300b94274ce3
python freeze_p4b_earlycand_audit.py --task object_counting --bench components/benchmark
#   selection_digest must read 04508a19ce30b0b6

# Stage-0: the two MODELS may share nothing but the task -- they are never
# compared per item, so they parallelise. A model's own two cells must stay on
# ONE card (paired per-item contrast; bf16 greedy is not byte-reproducible).
CUDA_VISIBLE_DEVICES=0 nohup bash run_bbh_numeric.sh llama3  object_counting STAGE0 > p4b_oc_l_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_bbh_numeric.sh qwen2.5 object_counting STAGE0 > p4b_oc_q_s0.log 2>&1 &
cat p4b_oc_l_s0.log      # immediately -- a wrong PY exits 127 and the log looks empty

# The gate, and the ONLY script that reads gold:
python eval_bbh_numeric.py \
  --generations components/llama3/bbh/object_counting/mdf_0/bbh_object_counting_8B_11_20.json \
                components/qwen2.5/bbh/object_counting/mdf_0/bbh_object_counting_7B_16_22.json \
  --gold_file components/benchmark/bbh_p4b_object_counting.json \
  --out docs/bbh_p4b_object_counting_stage0.json

# WORKPOINT only for models that printed PASS, and only after the manual audit.
# Then REVERSE (p4b-amend-01), which REFUSES until alpha=0 AND the workpoint
# cell exist. Both cells of one model are SEQUENTIAL on that model's own card.
CUDA_VISIBLE_DEVICES=0 nohup bash run_bbh_numeric.sh llama3  object_counting WORKPOINT > p4b_oc_l_wp.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_bbh_numeric.sh qwen2.5 object_counting WORKPOINT > p4b_oc_q_wp.log 2>&1 &
# after each finishes, on the SAME card:
CUDA_VISIBLE_DEVICES=0 nohup bash run_bbh_numeric.sh llama3  object_counting REVERSE > p4b_oc_l_rv.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_bbh_numeric.sh qwen2.5 object_counting REVERSE > p4b_oc_q_rv.log 2>&1 &

# steering_fires must read L*250 at alpha!=0: llama 2250 (L=9), qwen 1500 (L=6).
# The runner asserts this itself; 0 means the hook never fired.

# Score all SIX cells at once. The scorer accepts only three alphas per model
# (0, the frozen workpoint, the frozen reverse dose) and refuses anything else.
python eval_bbh_numeric.py \
  --generations components/llama3/bbh/object_counting/mdf_{0,neg6,4}/bbh_object_counting_8B_11_20.json \
                components/qwen2.5/bbh/object_counting/mdf_{0,8,neg6}/bbh_object_counting_7B_16_22.json \
  --gold_file components/benchmark/bbh_p4b_object_counting.json \
  --out docs/bbh_p4b_object_counting_result.json
```

**Runtime, measured off the stage-0 length distributions:** Llama ≈25–40 min per
cell (250 items, 95.6% running to the 768 cap), Qwen ≈5–10 min (median 131
tokens). Sequential per model: Llama ≈1–1.5 h for its two cells, Qwen ≈15–20 min;
with the two models on two cards the wall clock is Llama's. α can move length in
either direction, so treat these as estimates.

**Prior worth knowing before reading the gate**: public reports put
Llama-3.1-8B near **.90** on `object_counting` under *their* prompt, budget and
extractor — not comparable to ours, but enough that a `> .85` ceiling failure is
a live outcome, not a formality. If it fails, `multistep_arithmetic_two`'s sample
is already frozen; switch `--task`, no code change.

## P4c fixed-workpoint transfer to program-execution reasoning (CRUXEval-O)

> **`docs/PREREG_P4C_CRUXEVAL.md` is the protocol** (`cruxeval-p4c-v0`, frozen
> 2026-09-03 before any cell existed), amended additively by
> `docs/p4c_amendment_0{1,2}.json`. **Status: PREPARED, format-only preflight
> PASSED, formal cells NOT RUN.** This section is provenance and the 口径 traps.

**P4c IS NOT A NEW PHASE — it is the THIRD TASK of P4's question**, after LogiQA 2.0
and BBH numeric. Do not renumber it P5.

**It cannot isolate what LogiQA left ambiguous, and that limit is frozen in advance.**
CRUXEval-O removes the option interface but changes reasoning content AND answer space
at once (constructed integer → arbitrary Python literal). It is a **third, harder point
on the transfer boundary**, not a controlled single-factor manipulation. Frozen wording:
transfer → "the workpoint transferred to open-ended program-execution reasoning", never
"to code reasoning in general"; null → "the transfer range does not extend to open-ended
program-execution reasoning", which does **not** establish which of the two changes binds.

**Data.** `cruxeval-org/cruxeval`, revision pinned `b96af0450242eb4da433032b90998f25588a5d0f`,
800 rows, columns `code/input/output/id`. **300 selected** by `sha256(salt:id:code:input)`
— never `hash()`, which is process-salted. Digests `4580b7a9a9ef6054` (questions) /
`a214d1fc7d84a2d9` (gold). Measured and asserted at freeze: **800/800 gold `literal_eval`,
0 gold contain a newline, `####` appears nowhere in the source** — those three facts are
what make a single-line `#### <literal>` marker safe, and the loader checks them rather
than assuming them.

**Matrix: 8 cells.** llama `0/−6/−4/+4` (band 11–20), qwen `0/+8/+6/−6` (band 16–22).
**Holm m=2 covers ONLY the two workpoint contrasts** (llama −6, qwen +8). The neighbour
(`−4`/`+6`) and reverse (`+4`/`−6`) cells are reported always, sit **outside Holm with
unadjusted p**, and **MUST NOT redefine the workpoint** — α is read from the frozen GSM8K
record and never re-searched. Four sampled doses are not a dose-response curve.

<!-- Why: the official CRUXEval scorer execs model output; a later reader may "restore
fidelity" by porting it. The gap it would close is already reported as nonliteral_rate.
Evidence: docs/PREREG_P4C_CRUXEVAL.md §5; test_cruxeval_p4c.py exec-payload guards.
Scope: P4c scoring only. -->
- **SCORING IS `ast.literal_eval` + PYTHON OBJECT EQUALITY, NOT the official
  execution-based `pass@1`.** Official CRUXEval runs
  `exec(f"{code}\nassert {gold} == {candidate}")`, interpolating **model-generated text**
  into an executed expression. This protocol never does that, and **`reliability_guard`
  is not a security sandbox** — do not describe it as one. The recorded gap: a candidate
  that is a non-literal *expression* evaluating correctly (`[1]*3`) scores **incorrect**
  here and would score correct officially; its size is reported per cell as
  `nonliteral_rate` rather than assumed. The test suite pins the security property by
  feeding `__import__('os').system(...)` and asserting it neither parses nor executes.

<!-- Why: the parser was right BY COINCIDENCE before amend-01 -- '#' opens a Python
comment, so literal_eval silently truncated a trailing '####'. 5 of the 300 gold values
contain '#', so the coincidence was unsafe.
Evidence: docs/p4c_amendment_01.json; the REGRESSION checks in test_cruxeval_p4c.py.
Scope: P4c parsing. -->
- **`p4c-amend-01` strips TWO decoding artifacts, and the second one is subtle.** (a) a
  trailing EOS token text — Qwen returned `[1, 1, 1, 1]<|endoftext|>`, a correct answer
  failing to parse purely because the decoder surfaced EOS (5 of 8 preflight payloads per
  Qwen cell). (b) a trailing `####` occurring **outside** a string literal — Llama writes
  `#### <literal> ####` then loops. **Without (b) the payload parsed anyway**, because `#`
  opens a Python comment and `literal_eval` truncates there — correct by coincidence, and
  unsafe because **5 of the 300 gold values contain `#`** (`sample_755` is
  `'ph>t#A#BiEcDefW#ON#iiNCU'`). The cut is quote-parity-guarded, so a `####` **inside** a
  string literal is preserved; all five `#`-bearing gold values were verified to round-trip.

<!-- Why: the tempting move is to widen the parser or switch FIRST->LAST, and both
would hide a steering effect: qwen format obedience is 8/8 at a=0 but 6/8 at +8.
Evidence: docs/p4c_amendment_02.json (no code change); REGRESSION amend-02 checks.
Scope: P4c parsing; the same logic applies to any task where format obedience is dosed. -->
- **`p4c-amend-02` is a decision NOT to change code, and it is the load-bearing one.**
  Two Qwen `+8` preflight payloads still fail: prose after the payload
  (`#### 'ohesteo' The function f removes…`), and a **bare `####`** followed by a second
  marker carrying the answer. Both are the **MODEL failing the frozen "exactly one line"
  format**, not decoding artifacts. **Widening the parser would systematically hide a
  steering effect** — Qwen reads 8/8 at α=0 and 6/8 at α=+8, so the failures are
  dose-specific. **Both parse under LAST and fail under FIRST**, which is exactly the shape
  that tempts a caliber switch: **FIRST stays MAIN** (frozen, matching GSM8K/GSM-Hard/BBH
  production), and the difference is already visible as `nonliteral_rate` + `last_acc`.
  The rule generalises: *a decoding artifact is a parser bug; a model failing the format
  is a result.*

- **PREFLIGHT is FORMAT-ONLY and structurally cannot become an accuracy read.**
  `--preflight` forces n=8, writes to a separate `_preflight/` tree with a `preflight`
  filename prefix and meta flag, and **the scorer refuses any preflight cell**. It checks
  the frozen digest and the 300-item count on the FULL file *before* truncating — the
  reverse order would let a preflight run on anything.

- **PREFLIGHT MEASURED (2026-09-03, after amend-01):** marker 8/8 in all four cells;
  literal 8/8 / 8/8 / 8/8 / **6/8** (llama 0, llama −6, qwen 0, qwen +8);
  `steering_fires` 0 / 72 / 0 / 48, matching `L×n`. **Llama is truncation-bound**:
  capped 8/8 and 7/8, `chars` median ≈2450, tails degenerating to `[/PYTHON] [/PYTHON]…`.
  Qwen is not (1/8, median 50/241).

<!-- Why: GSM-Hard froze this wording after a ceiling effect was nearly read as an
observation; llama P4c sits against the same 768 cap.
Evidence: the P3 GSM-Hard truncation bullet; preflight capped 8/8 and 7/8.
Scope: every llama P4c length or accuracy claim. -->
- **Llama accuracy is a FIXED-768-BUDGET ESTIMAND.** The paired comparison stays valid
  (every cell has the same budget), but it may **not** be read as unconstrained reasoning
  capability, and llama's length-derived readouts are **not natural lengths**. The answer
  completes *before* the degenerate tail, so accuracy is readable — this is **not** a
  Mistral-style extraction floor. **Do not raise the budget now**: that changes the frozen
  primary question. Per the P3-supp precedent, finish 768 first and build a larger-budget
  sensitivity separately if needed, never substituting it for the main result.

- **NO ACCURACY GATE, deliberately** — unlike P4b's `[0.30, 0.85]`. A low baseline is a
  limitation on the reading, not a cancelled test; this removes the failure mode where a
  gate interval becomes adjustable after seeing data. **Hard stops are technical only**
  (six, listed in §7), and a hard stop is a stop — not a licence to redesign the prompt,
  parser or budget.

- **Cells are NOT required to share a GPU** (per the repo-wide rule); `host` and
  `CUDA_VISIBLE_DEVICES` are recorded as provenance, and a cross-device contrast is
  reported as a **cross-run pairing**. Pairing means alignment by the frozen item order.

```bash
# Server, from /data1/paveen/Dopamine. Loader first -- digests must match.
python data_cruxeval.py --out_dir components/benchmark

# Format-only preflight (no accuracy exists to read), then inspect:
CUDA_VISIBLE_DEVICES=2 bash run_cruxeval.sh llama3  PREFLIGHT
CUDA_VISIBLE_DEVICES=3 bash run_cruxeval.sh qwen2.5 PREFLIGHT
python inspect_p4c_preflight.py --root components
python dump_p4c_nonliteral.py --root components --preflight   # classify failures

# Formal cells. The two MODELS may run on two cards.
CUDA_VISIBLE_DEVICES=2 nohup bash run_cruxeval.sh llama3  ALL > p4c_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash run_cruxeval.sh qwen2.5 ALL > p4c_qwen.log  2>&1 &
cat p4c_llama.log      # immediately -- a wrong PY exits 127 and the log looks empty

# Score ONCE, after BOTH models finish (a single model withholds Holm).
python eval_cruxeval.py \
  --generations components/llama3/cruxeval/mdf_{0,neg6,neg4,4}/cruxeval_o_8B_11_20.json \
                components/qwen2.5/cruxeval/mdf_{0,8,6,neg6}/cruxeval_o_7B_16_22.json \
  --gold_file components/benchmark/cruxeval_p4c_formal.json \
  --out docs/p4c_cruxeval_evaluation.json
```

## Local checks (no GPU, no server)

There is no pytest suite and no linter config. The tests that exist are standalone scripts that exit non-zero on failure, and they are the fast way to verify a change before touching the server. **Use `python3.10`** — plain `python3` on the analysis box has no numpy.

Two things about running them that have each cost a wrong verdict: **(a) these scripts print their own `ok` lines and signal failure through the EXIT CODE**, so redirecting output to `/dev/null` and reading only the status is fine, but reading only the last printed line is not — check `$?`. **(b) `timeout` does NOT exist on this macOS box**; wrapping a check in it yields exit 127 for every script, which looks exactly like a mass test failure. Verified passing on 2026-08-21: `test_cgt_seq_v5.py`, `test_bandit_pv10.py`, `test_bandit_pv10c.py`, `test_bandit_pv11.py`, `test_pv10_stop_parity.py`, plus `bash -n` on the Qwen launchers.

```bash
# wps-v0 workpoint-stability supplement. STDLIB ONLY (runs under bare python3).
# It builds each of the SEVEN cells' REAL argv by running the launcher with a
# stub interpreter, then parses that argv with the GENERATOR'S OWN parser.
#
# This is the check bash -n cannot do. --n_samples exists only on
# get_answer_regenerate_math.py; carrying it over to the GSM8K cell was valid
# SYNTAX and invalid SEMANTICS, so the cell died on the server after loading
# nothing. Mutation-tested: restoring that flag makes this exit 1 with the same
# message the server produced.
python3 test_wps_launcher_argv.py      # 7 cells: configs / cot / budget / ans_file
bash -n run_wps_llama3.sh && bash -n run_wps_gsm_hard.sh
# Also drive them with REAL arguments -- the cell-name allowlist and the
# overwrite guard are semantic, and bash -n passes both regardless:
CUDA_VISIBLE_DEVICES=0 bash run_wps_gsm_hard.sh gsm8k_cot_neg2   # must be REFUSED
                                       # (that cell belongs to run_wps_llama3.sh)
# Near-optimal regions across all six curves. Run from RoleAnswer/; it asserts
# that 10 published values reproduce exactly and dies otherwise.
python3.10 wps_region.py               # -> wps_near_optimal_region.txt

# P4 LogiQA 2.0. The loader's --check re-runs every stage-0 assertion and
# reprints the frozen digests; it writes nothing. It is the fastest way to
# confirm the upstream split has not moved under you.
python3.10 data_logiqa2.py --check     # must print formal digest 4d4b25e071a2a6dd
                                       # and "preflight and blind formal carry
                                       # NO gold (whitelist + payload scan)"
bash -n run_logiqa2_preflight.sh && bash -n run_logiqa2_formal.sh
# bash -n is NOT sufficient for a launcher -- it passed the ${1:?...} brace bug
# that made every invocation die. Also invoke with real arguments.

# P4c CRUXEval-O. --check re-runs every loader assertion (schema, 800 rows,
# gold all literal_eval-able, no newline in gold, no '####' in the source,
# frozen selection digests) and reprints the digests; it writes nothing. It
# DOES hit the network to fetch the pinned revision.
python3.10 data_cruxeval.py --check    # must print questions 4580b7a9a9ef6054
                                       # and gold a214d1fc7d84a2d9
# The guard suite is STDLIB ONLY and runs under bare python3 too -- deliberately,
# per the test_p3_label_firewall lesson (an import of numpy-via-utils once made
# a crash exit 0). Guards are AST-extracted, not imported; every one is
# mutation-tested, including the security property that a payload like
# __import__('os').system(...) does NOT parse and is NOT executed.
python3 test_cruxeval_p4c.py           # 100 checks, ~2s, no GPU/server/network
bash -n run_cruxeval.sh
# bash -n is NOT sufficient for a launcher -- it once passed the ${1:?...} brace
# bug that made MODEL literally 'llama3}'. Drive it with real arguments; the
# model allowlist, the step allowlist and the need_baseline ordering guard are
# all semantic:
CUDA_VISIBLE_DEVICES=0 bash run_cruxeval.sh llama3 SEARCH_DOSES   # must REFUSE
CUDA_VISIBLE_DEVICES=0 bash run_cruxeval.sh llama3 WORKPOINT      # must REFUSE
                                       # until the alpha=0 cell exists

# P4b BBH numeric. --check re-runs every stage-0 assertion (schema, 250 rows,
# unique questions, integer gold, gold-distribution drift, content digest) and
# reprints the digests; it writes nothing and reads no gold. It DOES hit the
# network to fetch the pinned revision.
python3.10 data_bbh_numeric.py --task object_counting --check
python3.10 data_bbh_numeric.py --task multistep_arithmetic_two --check
# The audit freeze is deterministic and must reprint 04508a19ce30b0b6; run it
# twice if you touched the selection, since a process-salted hash() would give
# a different 30 items each run and the digest is what catches that. It reads
# the BLIND file, so it needs --bench pointing at a tree where that file
# exists; components/benchmark/ is SERVER-side, so locally write one to a temp
# dir first (the loader refuses to overwrite, hence the temp dir).
TMP=$(mktemp -d) && python3.10 data_bbh_numeric.py --task object_counting --out_dir "$TMP" >/dev/null \
  && python3.10 freeze_p4b_earlycand_audit.py --task object_counting --bench "$TMP" --check \
  && rm -rf "$TMP"
bash -n run_bbh_numeric.sh
# The step guard (STAGE0/WORKPOINT/REVERSE) and the REVERSE ordering guard sit
# AFTER the mask/blind-file checks, whose paths are server-side -- locally they
# refuse first, for an unrelated reason. To exercise them, stub the paths:
#   T=$(mktemp -d); mkdir -p "$T/mask/llama3_non_logits" "$T/benchmark"
#   touch "$T/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy" \
#         "$T/benchmark/bbh_p4b_object_counting_blind.json"
#   CUDA_VISIBLE_DEVICES=0 BASE_DIR=$T BENCH=$T/benchmark \
#     MODEL_DIR=stub/model WORK_DIR=$T PY=python3.10 \
#     bash run_bbh_numeric.sh llama3 object_counting REVERSE   # refuses: no a0
# Driving the launcher with REAL arguments is the only test that catches a
# semantic break -- bash -n once passed a ${1:?} brace bug that made MODEL
# literally 'llama3}'.
CUDA_VISIBLE_DEVICES=0 bash run_bbh_numeric.sh llama3 dyck_languages STAGE0
                                           # must be REFUSED: the task
                                           # allowlist is the guard, and
                                           # bash -n cannot test it

# P3 blind validation (GSM-Hard). STDLIB ONLY -- runs under bare python3 too,
# deliberately: it once imported the generation module (numpy via utils) and
# under a numpy-less interpreter CRASHED WITH EXIT CODE 0, i.e. CI would have
# read a crash as a pass. Guards are AST-extracted and an excepthook forces
# exit 2 on any crash. Every guard is mutation-tested.
python3 test_p3_label_firewall.py          # 82 checks, ~1s: label firewall,
                                           # 2^53 exactness, float64 hard stop,
                                           # 40-hex revision pin, schema hard
                                           # stop, sample determinism, and the
                                           # CLI contract (parses the launchers'
                                           # real argv -- a source grep missed a
                                           # live --configs nargs bug)
python3.10 p3/p3_bigint_audit.py --gsm8k_root llama3/gsm8k   # needs ROLEANSWER
                                           # or --roleanswer; verdict-equivalence
                                           # of norm_exact on stored GSM8K

# P3 evaluator. Synthetic fixtures only -- reads no real gold. Run from
# RoleAnswer/ (it imports the frozen extractor from analyze_first_last_acc).
# The load-bearing check is the WRONG-prediction fixture: it proves the
# evaluator reports failure as failure instead of passing it silently.
python3.10 p3/test_p3_eval.py              # 35 checks, ~5s

# Manifold pilot (sections 1-3). All CPU-only, no GPU, no server.
# check_hs_llama.py itself runs on the SERVER (conda `python`); its test does not.
python3.10 test_check_hs_llama.py          # 27 guards: metadata fail-closed,
                                           # layer indices [10..18,31], the
                                           # projection MEAN-not-sum identity,
                                           # agreement-as-rate
python3.10 manifold/split_manifest.py --check   # split reproduces from its
                                           # generator; add --roleanswer <dir>
                                           # to also verify the text digest
                                           # (absent tree = skipped, not passed)
python3.10 manifold/test_split_manifest.py # 21 guards incl. the process-salted
                                           # hash() control and n-extension
                                           # stability
python3.10 test_manifold_fit.py            # 74 guards: phase windows, frozen
                                           # commit-locator EQUIVALENCE,
                                           # per-question weighting (with a
                                           # row-weighted control), coord_t /
                                           # re_by_k export, manifest + H5
                                           # digest, overwrite, base-cell alpha
bash -n run_manifold_pilot.sh

# Section 3 exact-geometry scripts (server, read-only on the H5, CPU-only).
# Both pin BLAS threads HARD (not setdefault -- an inherited wrong value is
# exactly the failure mode) and fail closed on: a non-orthonormal basis
# (||W_k d||^2 would not be an energy), incomplete question coverage vs the
# manifest (checking only that CELLS AGREE would pass a run where every cell
# is missing the SAME question), a duplicate question_idx, and an existing
# output. --split_manifest is REQUIRED on the exact script: a 300-question
# total pools the 185 TRAIN questions the basis was fit on, so it is circular;
# roles are train = QC, val = k sensitivity, test = PRIMARY, all = descriptive.
python manifold_prefill_exact.py --h5_dir <hs> --basis <dir>/basis.npz \
    --split_manifest manifold/split_manifest.json --out prefill_exact.json
python manifold_prefill_direction.py --h5_dir <hs> \
    --split_manifest manifold/split_manifest.json --out prefill_direction.json

# Transfer a frozen basis to a NEW condition (this is what makes the CoT run a
# confirmation rather than a new model). Writes coordinates only; basis.npz and
# basis_meta.json are NOT rewritten.
python manifold_fit.py --h5_dir <hs> --out_dir <new_dir> \
    --split_manifest manifold/split_manifest.json \
    --base_cell nocot --cells cot,cot_aneg4 \
    --reuse_basis <frozen_dir> --model_dir meta-llama/Llama-3.1-8B-Instruct

# Qwen (pre-registered, NOT yet run): its own band, size and commit locator.
# --model_dir is an HF repo id, not a filesystem path (tokenizer only, no
# weights are downloaded for the fit).
python manifold_fit.py --h5_dir components/hidden_states/gsm8k/qwen25_signal_v1 \
    --out_dir components/qwen2.5/manifold/qwen25_signal_v1 \
    --split_manifest manifold/split_manifest.json --base_cell nocot \
    --size 7B --layer_start 16 --layer_end 22 --commit_locator qwen \
    --model_dir Qwen/Qwen2.5-7B-Instruct

# Offline analysis (RoleAnswer/manifold/, python3.10, no server):
#   incremental.py / incremental2.py  section 3.5
#   confirm_cot.py                    section 3.6
#   decode_minimal.py                 section 3.7

# Qwen2.5 signal replication (the ACTIVE line). The tracker's hook path is NOT
# covered by check_gsm8k_qwen.py -- see the Qwen-signal bullet above.
# Qwen HS backfill (COLLECTED + ACCEPTED). CHECK is read-only: interpreter+deps,
# mask presence, single-card guard, disk estimate. G1/G2/G3 are a
# token-balanced 3-way split; each covers a disjoint set of the 7 cells.
bash -n run_track_hidden_states_qwen25.sh
python3.10 -c "import ast;ast.parse(open('track_hidden_states.py').read())"

# HS acceptance -- READ-ONLY (mode="r" everywhere), SERVER-side, conda `python`.
# Must pass before any geometry analysis: check [2a] is the one that catches a
# wrong mask or wrong band, which would otherwise yield hours of
# uninterpretable geometry. --n_probe 0 reads every sample (minutes per cell)
# and is the full check to run once before freezing.
python check_hs_qwen25.py --n_probe 8      # sampled probe
python check_hs_qwen25.py --n_probe 0      # full probe, pre-freeze
python check_hs_qwen25.py --no_agreement   # skip [2b] if the light cells
                                           # are not uploaded yet
# --expect_n is argparse.SUPPRESS'd and exists for synthetic fixtures only;
# never pass it on real data.

# Qwen output decisiveness (§5.5 Result 2-3). Offline, read-only, no GPU;
# run from RoleAnswer/qwen_signal/. Reuses commit_aligned.commit_step by
# IMPORT -- a local re-implementation reads a different commit on Qwen.
# Fails closed on cell count AND on the alpha set (metrics_hs and a
# lightweight metrics tree share a schema and a naming convention).
python3.10 logit_family.py                 # reproduces logit_family_RESULT.txt

python3.10 check_signal_qwen.py --mask <nmd_0.5_16_22_7B.npy> \
    --run_tag qwen25_signal_v1 --skip_tokenizer
                                           # real hooks on real nn.Modules:
                                           # alpha=0 bit-identical, alpha!=0
                                           # last prefill token only, decode
                                           # never injected, exactly L layers,
                                           # projection == alpha*mean||mask||^2.
                                           # --skip_tokenizer downgrades the
                                           # verdict to PARTIAL; the server
                                           # pre-flight MUST pass --model_dir.
bash -n run_track_dopamine_signal_qwen25.sh

# CGT-Sequential (the ACTIVE line). v5 touches a driver whose v1-v4 results are
# frozen, so the load-bearing check is not "does v5 work" but "did adding v5
# change anything else". Byte-identity is checked against git HEAD, not a
# hand-copied baseline that can drift.
python3.10 test_cgt_seq_v5.py              # v1-v4 builders + make_box_sequence
                                           # unchanged; STRICT per-phase order
                                           # balance (random assignment drifts
                                           # -- a mutation test produced 36/28
                                           # in one run); all order-bearing
                                           # surfaces agree per round; neutral
                                           # example; ORDER_VERSION in iface for
                                           # v5 only; full FakeVC episode
bash -n run_cgt_seq.sh && bash -n run_cgt_seq_qwen25.sh
```

**Bandit pv6–pv11 checks moved to `docs/LOCAL_CHECKS_ARCHIVE.md` (2026-09-02).**
That line is CLOSED, so its ~160 lines of launcher/gate/analyzer invocations were
costing context on every session without being run. Nothing was deleted; the
command block there is byte-identical to what stood here. Read it only if you are
reusing a pv-era seed bank, gate rule or analyzer.

Individual checks are plain `if` statements inside those scripts, so to run one in
isolation, import the module and call the function directly rather than looking for
a test-selection flag.

**The FakeVC in `test_bandit_pv6_episode.py` mirrors the real `VicundaModel` contract on purpose** — `generate()` takes no `diff_matrices`, `regenerate()` raises on `diff_matrices=None` like llms.py:821, and the chat template serializes a BOS. A fake that is more permissive than the real API hides exactly the bugs these tests exist to catch; if you extend it, keep it at least as strict.

**A FakeVC models INTENT, not arithmetic.** Its `CountingVC` subclass reimplements the site formula rather than executing `llms.py`, so it will happily agree with a wrong counter. The site-counter bug (counting hook calls, so 32 layers and K candidates both collapsed to one number) passed a green FakeVC suite. To check anything whose value depends on real hook internals, drive the actual closures with a synthetic full-length mask (32 rows, non-zero only on the steered band) instead of trusting the fake.

## Editing guidance

- **Layer indexing convention** (踩過大坑,務必先讀): `LAYER_START` / `LAYER_END` follow **HF hidden_states semantics** (index 0 = embedding, 1..N = decoder layer outputs). Saved masks drop the embedding row (`detection/nmd.py: return mask[1:]`), so saved-index `i` ↔ `decoder_layers[i]` ↔ `hidden_states[i+1]`. **Always use `utils.mask_slice_for(mask, ls, le)` and `utils.decoder_layer_range(ls, le)`** instead of raw `mask[ls:le]` / `range(ls, le)` — they encode the `layer_start-1` offset so hook registration and mask slicing stay in sync. The `regenerate` family (`get_answer_regenerate_*.py`) sidesteps the offset by `zip(decoder_layers, mask)` full-length and is the canonical alignment reference. Verify on server with `sanity_mask_indexing.py` before changing any layer-indexing code.
- Don't refactor `llms.VicundaModel` loading branches casually — Mistral3, dream-diffusion and CausalLM each rely on slightly different hook surfaces.
- `template.py` 大部分變體不要原地改(歷史 baseline 綁定);但 GSM8K/MATH 的 `build_gsm8k_default_suite` 和 `build_math_suite` 在 2026-05-30 已修正為**對稱**(No-CoT 與 CoT 唯一差別是 `Let's think step by step.` 一行),vanilla / action / confidence 三個 suite 維持不動。舊不對稱模板跑的數字(含早期 61.7% / 76.0% baseline)與當前 pipeline 不可比。
- New plans go in `closed_loop_gsm8k.py` behind a new `--plan` value (added to the `choices=[...]` list and as a branch in `_compute_alpha()`); keep prior plans callable for ablation reproducibility.
- The injection–observation loop has a hard 1-step lag (pre-hook injects `α_t` before forward; post-hook reads `x_t`/`ema_t` after, so `α_{t+1}` is decided from `x_t`). Any new controller must assume `α` based on step-`t` observation only takes effect at step `t+1` — controllers that try to react to fast (per-token) signal changes diverge (see Plan D failure in `AdaptiveThinking.md` §4.3).
- Tracking projection and steering injection share the same `nmd_mask` (sparse, ~0.5% of neurons per layer). This co-design means injecting `+α` directly raises next-step `x_{t+1}` by `α × ‖mask‖²` — do not change one without the other.
- Result file naming includes plan + k1 + k2 + ema_alpha + layer range, so renaming any of these breaks the analysis scripts (`analyze_plan{D,EF,G,H1,H2,H3}.py` filter by exact filename).
- HS recorder writes `(P, n_stored, H)` and `(T, n_stored, H)` fp16 blocks for **middle layers `[LAYER_START, LAYER_END)` + final layer only** (10 layers for Llama3-8B). The on-the-fly NMD projection in HDF5 is a sanity scalar — analysis recomputes projections offline against any mask. `P` is variable-length per sample, so do not stack across samples without per-sample loops. If you need cross-layer ablation outside the stored set, re-run the tracker with a wider `[LAYER_START, LAYER_END)`.
- `track_hidden_states.py` is **bs=1** by design: forward hooks read last-token HS and append per step, and decode length varies per sample (different EOS). Batching would require per-sample masks + per-sample EOS bookkeeping in the hook — not worth it for 300 samples × 5 runs.
