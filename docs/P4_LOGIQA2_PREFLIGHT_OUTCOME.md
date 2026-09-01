# P4 LogiQA 2.0 — Preflight Outcome Record

**Protocol `logiqa2-p4-v0` + `p4-amend-02` (prompt) + `p4-amend-03` (wording).
Run 2026-09-01. Format-only. NO ACCURACY WAS COMPUTED.**

This file records what the blind preflight observed. It makes no claim about
accuracy, about any α effect, or about the formal 300-item sample. Protocol
decisions taken in response live in `docs/p4_amendment_05.json`, not here.

---

## 1. What was run

20 items × 4 cells, `max_new_tokens=512`, greedy, bare-string, prefill-only
`tail_len=1` at the frozen `Response: ` anchor.

**The 20 items are HELD-OUT and disjoint from the formal 300 by construction**
— they are ranks 76–80 by composite key within each gold label, where the
formal sample takes ranks 1–75. Overlap is 0 by the sampling algorithm, not by
a check after the fact. **Nothing observed here transfers to the formal set.**

The preflight input file carries no gold field (built from a whitelist in
`data_logiqa2.py`, and re-scanned by the runner), so accuracy was not merely
unused — it was unreachable.

| artifact | sha256 |
|---|---|
| `logiqa2_p4_preflight.json` | digest `f5225130d765c992…` (20 items) |
| `logiqa2_p4_formal.json` | digest `4d4b25e071a2a6dd…` (300 items, untouched) |
| prompt template | `c42dc9c81f117a6c…` (`p4-amend-02`) |

Steering was verified per cell before any output was read: `steering_fires`
0 / 180 / 0 / 120, matching `L·B·1` exactly (llama L=9, qwen L=6).

---

## 2. Observations (descriptive, gold-blind)

| cell | format | budget exhausted | len min/med/max | markers med/max | answer_first | degenerate | first≠last | first_marker_pos med |
|---|---|---|---|---|---|---|---|---|
| llama α=0 | 20/20 | **20/20** | 512/512/512 | 4 / 127 | **13/20** | 17/20 | 0/20 | **0.000** |
| llama α=−6 | **19/20** | 19/20 | 60/512/512 | 2 / 102 | 4/20 | 14/20 | 1/20 | 0.064 |
| qwen α=0 | 20/20 | 2/20 | 82/303/512 | 1 / 23 | 0/20 | 1/20 | 0/20 | 0.975 |
| qwen α=+8 | 20/20 | 4/20 | 82/279/512 | 2 / 9 | 0/20 | 1/20 | 1/20 | 0.941 |

Field definitions: `answer_first` = the first non-whitespace content is a
`Final answer:` marker. `degenerate` = the final 40-character block recurs ≥4×
in the text (the repo's `is_loop` convention). `first_marker_pos` = character
offset of the first marker / total characters.

### 2.1 Llama: scorable, with degraded termination

Not "cannot answer" and not "unreadable". 19 of 20 outputs in each cell carry
at least one marker, and FIRST/LAST agree in 39 of 40. What fails is stopping:
**no Llama output terminated naturally at α=0** (min length = 512), and the
tails degrade into repetition — `Final answer: D` repeated 85×, an enumerated
list counting to 64, or one sentence restated indefinitely.

One α=−6 sample (id 11) produced no marker at all: it argues continuously and
is cut at the budget. That is the single 0-marker case in 80 generations.

### 2.2 Qwen: normal termination, marker at the end

Most outputs stop naturally; degeneration is 1/20 and 0/20. The first marker
sits at 0.94–0.98 of the text. **This says the marker is LATE, not that the
preceding text is complete or correct reasoning** — no content judgement was
made on it.

### 2.3 A pilot-generated timing hypothesis (NOT evidence)

> In this 20-item held-out preflight, Llama's `answer_first` rate descriptively
> fell from 13/20 at α=0 to 4/20 at α=−6, with more text preceding the marker.
> This raises a hypothesis to be tested — that −6 may delay explicit answer
> submission — whose direction is consistent with GSM8K's `early_candidate`
> minimum at the same workpoint. **It is not evidence of an α effect.**

Four limits, all binding:

1. n=20, and a 13→4 shift at that size is well within sampling noise.
2. These are held-out items; the formal 300 are different questions.
3. The preflight's only authorised purpose is fixing the token budget.
   Reading an effect from it would use a sample designed for another question.
4. Degeneration did NOT fall correspondingly (17/20 → 14/20): **the degeneration
   rate remains high with no clear synchronous decrease.** Whatever moved,
   submission timing and termination behaviour did not move together.

**Do not write "externalized reasoning emergence increased."** Text appearing
before the marker is not necessarily reasoning; establishing that would require
a frozen content-judgement rule, which does not exist. Report the morphological
fields only: `answer_first`, `pre_marker_chars`, `first_marker_pos`,
`multi_marker` / `degenerate` rate.

`first_marker_pos` is additionally confounded by tail length — a longer
repetition tail pushes the normalised position down independently of anything
before the marker — so it may not carry a "more reasoning" reading on its own,
and is reported as auxiliary.

### 2.4 The Llama/Qwen contrast is gold-blind and provisional

It is used here to adjust the readability protocol and to freeze descriptive
fields for the formal run. **The cross-model result waits for the 300 items.**

---

## 3. Two protocol problems this surfaced

Recorded here; resolved in `docs/p4_amendment_05.json`.

1. **The format gate treats repetition as a violation.** PREREG v0 §4 makes a
   missing marker a hard stop, and llama α=−6 reads 19/20. But 19 of those 20
   outputs are scorable; the gate was written against "the model cannot produce
   the format", which is not what happened.

2. **The budget rule fires for a reason its premise did not anticipate.** The
   rule reads "any output ≥511 → 1024", intended as "reasoning needs more
   room". Llama reaches 512 by repeating, not by reasoning, and Qwen's medians
   (279–303) do not need 1024. The rule's mechanical condition is met; its
   rationale is not. **It is executed as written anyway** — see the amendment
   for why changing it would be the worse error.
