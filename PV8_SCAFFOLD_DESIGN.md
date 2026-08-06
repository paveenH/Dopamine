# pv8 scaffold design — Choice History × Beta Calculator (2×2, α=0)

Status: **design only, nothing run.** Written before any prompt was frozen, so
the predictions below are pre-registered rather than fitted.

## Why a new protocol, not a pv7 variant

Both factors change the Stage-1 prompt, and the History factor makes prompt
length grow with the round. That is a different protocol: pv8 gets its own
prompt version, resume key, and output tree. pv7 files stay byte-unchanged and
pv8 numbers are never pooled with pv7 numbers.

The environment, seed banks and frozen state bank are **unchanged**, which is
what keeps the algorithmic baselines and the α=0 competence-gate rules valid.

## The question this answers

pv7 α=0 localised the failure precisely:

- `uncertainty_recognition` = .996 — the model says arms are uncertain
- `policy_targets_low_sample` = .017 — it almost never acts on that
- 5/5 one-shot-zero arms abandoned, 0/485 later Policies even mention them

and the pv7 α sweep narrowed it further:

- α moves the policy toward **UNTRIED** arms (.041 → .073 at −4)
- α does **not** move it toward **ONE-SHOT-ZERO** arms (.017 at every α)

So the deficit is specific: **an arm with one observed 0 is treated as settled
bad news, while an arm with no observation is treated as uncertain.** The
model is not missing the concept of uncertainty; it is missing it for exactly
one evidence pattern.

Two candidate causes, which the 2×2 separates:

1. **Representation** — `0/1 = 0.00` reads as a point estimate of zero.
   Beta posterior rewrites it as `0.33, very high uncertainty`, putting it on
   the same scale as an untried arm's `0.50, maximal`.
2. **Self-monitoring** — the model cannot see that it has pulled the same arm
   40 times in a row, because trial counts are order-blind. Choice history
   makes the lock-in itself visible.

## Design

|  | choice history | Beta calculator |
|---|---|---|
| `base` | – | – |
| `history` | ✓ | – |
| `calculator` | – | ✓ |
| `both` | ✓ | ✓ |

Same 123 frozen states, α=0 only, temperature 0. 4 arms × 123 = 492 Stage-1
generations + 492 Stage-2 scorings; roughly 2× the pv7 α run, so ~20 min per
GPU, or one GPU per two conditions.

`base` must reproduce pv7's α=0 cell **byte for byte** — it is the same prompt.
That is the protocol check, and it is free.

### Factor A — choice history

```
Round 51 of 100. Future choices after this one: 49.

CHOICE HISTORY (oldest → newest):
[A A B C C C C A ...]

OPTIONS
- Button A: 1 reward / 3 trials, empirical rate 0.33
...
```

Round 1: `CHOICE HISTORY: none`.

Compact letters, no `Button` prefix, no per-round reward. Rewards are excluded
deliberately: the OPTIONS table already carries the reward totals, and adding
per-round outcomes would turn this into a full-transcript condition, changing
several things at once.

**This factor is NOT neutral, and that is the point.** It is a live
experimental variable with two opposite plausible effects:

- *self-monitoring*: "I have chosen A 40 times running; I should re-check"
- *inertia / imitation*: a token sequence `[... A A A A]` is a next-token
  continuation cue, and continuing it is the locally likely completion

### Factor B — Beta calculator

Adds two columns, computed by the program under a Beta(1,1) prior:

```
OPTIONS
- Button A: 1/3 observed, posterior mean 0.40, uncertainty high
- Button B: 0/1 observed, posterior mean 0.33, uncertainty very high
- Button D: UNTRIED,      posterior mean 0.50, uncertainty maximal
- Button C: 33/46 observed, posterior mean 0.71, uncertainty low
```

posterior mean = (s+1)/(n+2); uncertainty binned from the posterior sd.

It does **not** provide UCB/Thompson scores or a recommendation — that would
be layer 3 (`algorithm-guided`), which executes the bandit algorithm for the
model and cannot answer whether the model can do it.

## Known confound: the calculator also changes the greedy ordering

Beta smoothing does not only express uncertainty; it **moves the point
estimates**, and it moves them most for small n. On seed 31's critical state,
`A: 1/2 = 0.50` vs `C: 0/1 = 0.00` becomes `A: 0.50` vs `C: 0.33` — the gap
shrinks from 0.50 to 0.17.

So a model that switches to C under `calculator` may simply be following a
smaller numeric gap, not reading the uncertainty column.

This is why the earlier single-factor "posterior mean only vs mean+uncertainty"
split was proposed, and it is **still needed**. It is folded in as an ablation
run only if `calculator` moves the revisit rate:

- `calc_mean_only` — posterior mean column, no uncertainty column

If `calc_mean_only` already lifts revisiting, the effect is re-ranking. Only
`calculator > calc_mean_only` licenses "the model used the uncertainty
information". Running it unconditionally costs another 123 generations; running
it only on a positive result keeps the default cheap.

## Known limit: inertia and exploit-the-best mostly coincide

In this bank the last-chosen arm IS the empirical-best in **103/123** states.
Only **20/123** dissociate them. So for most states, "continue the sequence"
and "exploit the best arm" predict the same button, and a History effect
cannot be attributed between them.

The tail structure is strong and grows with the round:

| state type | mean tail run | tail == dominant | distinct arms in last 10 |
|---|---:|---:|---:|
| round_5 | 1.8 | 11/20 | 2.85 |
| round_11 | 4.7 | 16/20 | 3.35 |
| round_31 | 21.4 | 20/20 | 1.15 |
| round_51 | 38.9 | 20/20 | 1.05 |
| round_76 | 59.5 | 20/20 | 1.00 |
| round_96 | 64.6 | 18/20 | 1.20 |

By r76 the history line is ~60 identical letters. That is close to a worst case
for an inertia cue, arriving exactly where lock-in is measured.

Consequences, decided in advance:

- The **20 dissociating states** are the pre-registered subset for reading
  inertia vs self-monitoring. n=20, so: proportions and per-state listing only,
  no significance testing.
- Report History effects **split by state type**. r5/r11 (short runs) versus
  r76/r96 (long runs) is the dose axis for inertia: if History raises
  persistence monotonically with run length, that is the imitation reading.
- A History effect on the pooled 123 is **not** attributable and must be
  reported as such.

## Primary readouts, in order

Same discipline as pv7: mechanism first, outcome last.

1. `policy_targets_low_sample_n1` — does it leave the .017 floor
2. `policy_targets_untried` — the channel that already responds, as a control
3. `low_sample_revisit_choice_n1` — executed
4. `uncertainty_recognition` — expected to stay at ceiling; a DROP would mean
   the calculator replaced the model's own uncertainty talk
5. late-state adherence to the empirical best — does the scaffold break
   working exploitation into uniform flailing
6. grounding error / format / hashtag — the calculator adds numbers to
   misquote, so grounding is a live risk, not a formality
7. margin, entropy — last

**Explicitly not a readout: reward or true-best-arm outcome.** The frozen
states were sampled from α=0 pv7 trajectories, so selecting a prompt by
outcome fits it to its own history. Same rule that governed the pv7 selection.

## Gate for continuing

α is tested **only if** a condition lifts `policy_targets_low_sample` off the
floor without collapsing late-state adherence. If the scaffold is required for
any α effect to appear, the finding is **scaffold-dependent modulation**, not
an improvement in native capability, and must be worded that way.

If no condition moves the floor, the conclusion is that the deficit is not a
calculation deficit and not a visibility deficit — it is in the
uncertainty-to-action policy itself, and a different intervention class is
needed.

## Order

1. `base` on the 123 states → assert byte-identical to pv7 α=0
2. `history`, `calculator`, `both` → the 2×2
3. `calc_mean_only` ablation, only if `calculator` moved the floor
4. α=0 competence gate on full episodes, only for a winning condition
5. Stage-1 α −4/0/+4, only after that gate passes
