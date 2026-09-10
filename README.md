# Dopamine: Is RSN Steering a Dopamine-like Gain Mechanism?

This repository extends the published **Role-Sensitive Neurons (RSN)** work into an
empirical test of a single hypothesis:

> Does the sparse mid-layer subspace that RSN identifies behave like a **dopaminergic
> gain axis** — a motivational "wanting" knob that changes *engagement, commitment
> timing and decisiveness* — rather than like a knowledge or capability knob?

The parent paper (`ACLARR/main.tex`, *Role-Sensitive Neurons: A Neuron-Level Gain
Control Mechanism for Confidence Steering*) established the mechanism: a ~0.5% subset
of mid-layer neurons acts as a bidirectional gain switch over **willingness to answer**,
with conditional accuracy largely preserved ("Confidence–Performance Decoupling"). It
raised the dopamine analogy only as a metaphor in its discussion. **This repo is the
attempt to turn that metaphor into evidence** — and to mark, honestly, where it does
not yet hold.

Intervention is a single scalar α applied to the RSN mask at the last prompt token
(prefill-only). Positive α ≈ *over-wanting*, negative α ≈ *under-wanting*.

---

## What is and is not established

The evidence is deliberately reported in three tiers. Overclaiming any tier as the one
below it is the most common failure mode in this project's own history.

### Supported

- **α is a controllable task-entry gain axis.** Entry gain `G_prefill` is near-linear in
  α on both Llama3.1-8B and Qwen2.5-7B (`R² ≈ 0.999`, holding out to Qwen `+12`), so a
  behavioral plateau is never an artifact of the injection failing.
- **A linear input produces a non-linear behavioral response.** Llama shows an
  *asymmetric peaked working point* on GSM8K — 60.0% at α=0, **78.0% at α=−6**, then
  collapse to 40.3% at α=−8. Qwen shows a *high-dose plateau* instead (86.0/88.3/87.7%
  at +8/+10/+12). Same lever, different dose–response shape.
- **The lever moves commitment timing, not knowledge.** Qwen's gain is mechanically an
  answer-first → reason-first switch (first `####` moves from decode step ≈3 to ≈187;
  early-candidate rate 96% → 5%). Llama's optimum is where premature commitment is
  lowest.
- **Wanting–knowing dissociation in behavioral economics.** Confidence Betting is the
  cleanest cell: α=+4 raises mean bet substantially while **accuracy is unchanged**
  (McNemar Holm `p_adj = 1.00` in every cell, GPQA n=646 and MMLU n=14,042). Replicated
  on Qwen, where the model's usable dose band breaks at the *opposite* end.
- **Role and Confidence directions share a sparse, high-density core.** Concatenated
  cosine 0.6063 over layers 11–19; top-neuron overlap 46/180 (Jaccard 0.147), far above
  chance, with shared neurons contributing ~8× per-neuron versus either exclusive set.

### Partial / model-dependent

- **Cross-task transfer of a frozen workpoint is real but bounded.** Reading α from the
  GSM8K record and never re-searching it: GSM-Hard transfers on both models
  (+6.33 / +16.33 pp), MATH on Llama only, CRUXEval-O on Qwen only, and **LogiQA 2.0 and
  BBH object-counting are double nulls** under No-CoT. Adding explicit CoT rescues
  several of these (BBH: +16.0 / +14.0 pp). There is no universal α and no single
  boundary condition.
- **Confidence neurons vs role neurons is a structural result, not yet a functional
  one.** Cross-steering on MMLU-E moves abstention hard in the predicted direction
  (E-rate 99.4% → 9.0% → 0.02%) but **also degrades competence on the confident prompt**
  (65.2% → 57.0% → 32.4%). Both halves must be reported together.

### Not established (recorded as such)

- **The thinking curve did not yield a hormone-like waveform.** The original goal — find
  an internal trajectory resembling tonic/phasic dopamine dynamics and control it — did
  not produce a significant effect. Closed-loop control (Plans A–H3) showed that
  *shaping* the waveform does not control accuracy, and is shelved. What survived is
  narrower and useful: commitment is a genuine, event-locked generation-state transition
  in both models.
- **Geometry does not explain peak-vs-plateau.** Last-prefill displacement is smooth,
  linear and single-axis in *both* models, so the difference lives downstream in
  commitment/decode dynamics, not at the injection point.
- **Bandit / directed exploration is closed as boundary evidence.** Across five
  intervention classes α changed policy stance and commitment without ever moving
  information acquisition. Kept because a negative result that sharp is informative.
- **No biological claim.** Everything here is *behavioral isomorphism*. α steering is not
  dopamine, no brain region is localized, and no subjective or physiological state is
  implied.

---

## Documentation map

Read the document for the question you have; none of them is a summary of the others.

| Document | Question it answers | Status |
|---|---|---|
| [`Note/AdaDopamine.md`](Note/AdaDopamine.md) | Do classic behavioral-economics paradigms (betting, CGT, IGT, bandit) show a wanting–knowing dissociation? | Behavioral core |
| [`Note/AdaDopamine_gsm8k.md`](Note/AdaDopamine_gsm8k.md) | What does α do to reasoning accuracy and commitment on GSM8K / MATH, and how far does a frozen workpoint transfer? | **Authoritative** for accuracy & transfer |
| [`Note/AdaptiveThinking.md`](Note/AdaptiveThinking.md) | What happens *inside* the trajectory — entry gain, slow state, commitment, release? | Thinking curve; largely a null on the waveform goal |
| [`Note/AdaManifold.md`](Note/AdaManifold.md) | Is steering a scalar gain or a directional reorganization of hidden state? | Extension of the above; closed |
| [`Note/AdaNeurons.md`](Note/AdaNeurons.md) | Are confidence neurons the same substrate as role neurons? | Newest line |

Supporting: [`Note/AdaBandit.md`](Note/AdaBandit.md) (bandit design and literature — no
usable positive result), [`Note/AdaLogitsLens.md`](Note/AdaLogitsLens.md) (analysis from
the RSN paper era), [`Note/AdaLiterature.md`](Note/AdaLiterature.md) (reading list),
[`Note/Todo.md`](Note/Todo.md) (execution plan), `ACLARR/` (parent paper).

**`CLAUDE.md` is the operational ledger** — per-experiment frozen results, retracted
readings, and the measurement conventions (口径) that cannot be re-derived from the code.
Read the section for the experiment you are touching; it is not meant to be read
end-to-end.

---

## Repository layout

```
data_<benchmark>.py            dataset loader -> JSON of question dicts
template.py                    prompt families (default / vanilla / action suites)
llms.py                        VicundaModel: loading + the steering hook surface
get_answer_*.py                answer extraction (MCQ / free-form / action scalars)
get_answer_regenerate_*.py     the canonical prefill-only steering path
mean/, detection/              diff vectors and mask selectors (NMD, KL, LR, PCA, ...)
track_dopamine_signal.py       lightweight signal collection (projected scalars)
track_hidden_states.py         full hidden-state dump (HDF5)
manifold/, p3/, zebralogic/,   self-contained per-experiment subtrees
  proofwriter_owa/, finqa/,      (own loader, runner, scorer, tests)
  gsm_symbolic/, cross_steering_confidence/
docs/                          pre-registrations, amendments, frozen evaluations
run_*.sh                       launchers — the source of truth for hyperparameters
```

A run is wired as: **loader → template → `VicundaModel` → forward hooks on the decoder
layers carrying `α × mask` → JSON output**. Accuracy is then recomputed *offline* by the
frozen extractors, never read from the inline fields the generation scripts store.

Analysis lives outside this repo in `~/Documents/RSNResult/RoleAnswer/` (not in git).

---

## Setup and running

```bash
bash setup_env.sh          # conda env "roleplaying" (py3.10) + bf16/CUDA stack
```

Models are referenced by Hugging Face repo ID (`meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen2.5-7B-Instruct`); masks are local files under `${BASE_DIR}/mask/`.

```bash
# A dose sweep. The launcher, not the .py, is the reproducible unit.
bash run_gsm8k.sh
bash run_gsm8k_qwen25.sh --baseline     # then --nocot / --cot

# Local verification without a GPU (python3.10 -- plain python3 has no numpy):
python3.10 test_cruxeval_p4c.py
python3.10 zebralogic/test_zebralogic.py
```

Layer bands are per-model mask facts, not tunable knobs: Llama `[11,20)` (L=9),
Qwen `[16,22)` (L=6). A raw α is **not** a common dose across models — different masks,
layer counts and activation scales.

---

## Methodological conventions

These exist because each was violated once and cost a wrong conclusion.

- **Pre-registration before generation.** Transfer tests (`docs/PREREG_*.md`) freeze the
  question, doses, parsing and statistics before any cell exists. Workpoints are read
  from the frozen record and never re-searched on the target task.
- **Structural label firewalls.** Blind protocols emit a gold-free question file and a
  sealed gold file; the generation script cannot reach a label, rather than merely
  declining to use one.
- **Frozen extractors, imported not reimplemented.** Accuracy is `first_acc` via the
  shared offline extractor; `last_acc` is sensitivity only. A second copy of a detector
  is how definitions silently fork.
- **Holm families are declared in advance and never pooled.** Diagnostic and exploratory
  cells sit outside the family with unadjusted p, and cannot be promoted.
- **A post-treatment variable is not a mediator.** Commitment metrics are outcomes of α;
  stratifying accuracy on them is consistent-with evidence, never causal mediation.
- **Negative and retracted results stay in the record**, with the reason they were
  retracted, so they are not re-derived.

---

## Citing the parent work

```bibtex
@inproceedings{huang2026rsn,
  title     = {Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism
               for Confidence Steering},
  author    = {Huang, Peiwen and Hsu, Chih-Hao and Huang, Tzu-Hung and Lin, Shou-De},
  note      = {Code and data: \url{https://github.com/paveenH/RSN}},
  year      = {2026}
}
```
