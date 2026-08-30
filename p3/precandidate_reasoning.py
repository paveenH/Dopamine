#!/usr/bin/env python3.10
"""Does the optimal alpha make the model reason BEFORE forming an answer?

EXPLORATORY. This tests a mechanism question that `posN` and `answer-first`
cannot answer, because both are anchored on the `####` MARKER -- a formatting
event -- rather than on when an answer VALUE first appears.

Llama GSM-Hard CoT alpha=-6 is the counterexample that forces this: 58.3% of
its committed samples open with `#### N` (posN median .0000) and accuracy still
RISES. A marker-anchored readout calls that premature commitment; it may not be.

THREE METRICS, all located on the FIRST ANSWER CANDIDATE, not on `####`:

  1. cand_pos    normalized char position of the first answer-shaped number
                 that is not part of an ongoing computation (see below).
  2. pre_chars / pre_eqs / pre_steps
                 how much reasoning exists BEFORE that candidate.
  3. reason_before_answer
                 whether at least one equation or reasoning step precedes it.

WHAT COUNTS AS A CANDIDATE, AND WHY IT IS NOT "any number".
A grade-school solution is full of numbers that are operands, not answers
("15 + 25 = 40"). Counting those would make cand_pos a measure of where
arithmetic starts. So a candidate is a number that is EITHER
  (a) the first non-whitespace content of the generation (a bare opening
      answer -- the `3\nExplanation:` shape), OR
  (b) the right-hand side of an `=`, OR
  (c) preceded by an answer-declaring phrase ("the answer is", "####").
(b) is deliberately included: in this corpus the model commits by finishing a
computation, and the LAST such value is its answer -- but the FIRST one is the
earliest point a concrete answer-shaped value exists. This is a proxy with a
known bias, stated rather than hidden: it can fire on an intermediate result.
That bias is CONSTANT across alpha, so a paired within-question contrast
remains interpretable even though the absolute level is an underestimate.

PAIRING. Same 300 questions in the same order in every cell (asserted). All
contrasts are paired per question; the unit of inference is the question.

NOT MEDIATION. cand_pos and pre_* are outcomes of alpha, so stratifying
accuracy on them is post-treatment stratification. The strongest admissible
wording is "accuracy gain travels with a change in answer-formation timing".
"""
import json, glob, re, sys, statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k  # noqa: E402

NUM = r"[+-]?\d[\d,]*\.?\d*"
_RHS   = re.compile(r"=\s*(" + NUM + r")")
_DECL  = re.compile(r"(?:the answer is|answer:|####)\s*(" + NUM + r")", re.I)
_OPEN  = re.compile(r"^\s*(" + NUM + r")\s*(?:$|\n|[^\d.,])")
_EQ    = re.compile(r"=")
_STEP  = re.compile(r"(?:^|\n)\s*(?:Step\s*\d|\d+\s*[.)]\s+[A-Z])")


def first_candidate(text):
    """Char offset of the earliest answer-shaped value, or None."""
    cands = []
    m = _OPEN.match(text)
    if m:
        cands.append(m.start(1))
    for rx in (_RHS, _DECL):
        m = rx.search(text)
        if m:
            cands.append(m.start(1))
    return min(cands) if cands else None


def per_sample(text, gold):
    h = all_hash(text)
    pred = h[0] if h else fallback_gsm8k(text)
    ok = (gold is not None and pred is not None
          and norm_gsm8k(pred) == norm_gsm8k(str(gold)))
    c = first_candidate(text)
    if c is None or not text:
        return dict(correct=ok, cand_pos=None, pre_chars=None,
                    pre_eqs=None, pre_steps=None, rba=None)
    pre = text[:c]
    eqs, steps = len(_EQ.findall(pre)), len(_STEP.findall(pre))
    return dict(correct=ok, cand_pos=c / len(text), pre_chars=len(pre),
                pre_eqs=eqs, pre_steps=steps, rba=int(eqs > 0 or steps > 0))


def load(tree, tag):
    f = glob.glob(str(BASE / tree / tag / "*.json"))
    f = [x for x in f if "summary" not in x]
    if not f:
        return None
    d = json.load(open(f[0], encoding="utf-8"))
    return d["data"] if isinstance(d, dict) else d


def cell(tree, tag):
    data = load(tree, tag)
    if data is None:
        return None
    rows = []
    for s in data:
        t = s.get("generated") or s.get("generated_neutral") or ""
        g = s.get("gold_answer") or s.get("answer")
        rows.append(per_sample(t, g))
    return rows


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else float("nan")


def summarise(rows):
    ok = [r for r in rows if r["cand_pos"] is not None]
    return dict(n=len(rows), acc=sum(r["correct"] for r in rows) / len(rows),
                n_cand=len(ok),
                cand_pos=med([r["cand_pos"] for r in ok]),
                pre_chars=med([r["pre_chars"] for r in ok]),
                pre_eqs=med([r["pre_eqs"] for r in ok]),
                rba=sum(r["rba"] for r in ok) / len(ok) if ok else float("nan"))


def paired(a, b, key):
    """Per-question deltas b-a on questions where both are defined."""
    d = [(y[key] - x[key]) for x, y in zip(a, b)
         if x[key] is not None and y[key] is not None]
    return d


# ---------------------------------------------------------------- denominators
# THREE DIFFERENT DENOMINATORS COEXIST HERE AND MIXING THEM PRODUCES A WRONG
# READING. This is not hypothetical -- it produced one during development:
#
#   all samples            300      accuracy, coverage rates
#   committed subset       varies   posN, answer-first  (Table 5.6c)
#   candidate-covered      varies   cand_pos, pre_*, reason-first
#
# GSM-Hard CoT a=-6 reads posN median .0000 on 156 committed samples (91 of
# them open with `#### N`) but cand_pos median .1032 on 262 candidate-covered
# samples. Both are correct; they are not the same population. A paired
# contrast therefore runs on the COMMON candidate-covered subset, reported with
# its own n, and coverage is printed for every cell.
#
# The locator does NOT skip a leading `#### N`: on those 91 samples it returns
# offset 4-6, i.e. it treats the marker's number AS the candidate. An earlier
# note calling them "placeholders" was wrong and is retracted -- the shift in
# the median is a denominator effect, not the locator seeing through the marker.

def _wilcoxon(d):
    from scipy.stats import wilcoxon
    return wilcoxon(d).pvalue if any(x != 0 for x in d) else float("nan")


def _mcnemar(a, b, key="rba"):
    from scipy.stats import binomtest
    pairs = [(x[key], y[key]) for x, y in zip(a, b)
             if x[key] is not None and y[key] is not None]
    b01 = sum(1 for x, y in pairs if x == 0 and y == 1)
    b10 = sum(1 for x, y in pairs if x == 1 and y == 0)
    if b01 + b10 == 0:
        return b01, b10, float("nan")
    p = min(1.0, binomtest(min(b01, b10), b01 + b10, 0.5).pvalue * 2)
    return b01, b10, p


def common(a, b, key="cand_pos"):
    """Rows where BOTH cells have a locatable candidate -- the paired subset."""
    return [(x, y) for x, y in zip(a, b)
            if x[key] is not None and y[key] is not None]


def report(a, b, label, acc=None):
    both = common(a, b)
    n = len(both)
    print(f"\n{label}   common candidate-covered n={n}"
          f"   (coverage {sum(1 for r in a if r['cand_pos'] is not None)}/{len(a)}"
          f" -> {sum(1 for r in b if r['cand_pos'] is not None)}/{len(b)})")
    if acc:
        print(f"   accuracy {acc[0]:.4f} -> {acc[1]:.4f}   [frozen artifacts]")
    for k, name in (("cand_pos", "cand_pos "), ("pre_chars", "pre_chars"),
                    ("pre_eqs", "pre_eqs  ")):
        d = [y[k] - x[k] for x, y in both]
        print(f"   {name}  {med([x[k] for x, _ in both]):9.4f} ->"
              f" {med([y[k] for _, y in both]):9.4f}   median Δ={st.median(d):+9.4f}"
              f"   p={_wilcoxon(d):.2e}")
    b01, b10, p = _mcnemar([x for x, _ in both], [y for _, y in both])
    r0 = sum(x["rba"] for x, _ in both) / n
    r1 = sum(y["rba"] for _, y in both) / n
    print(f"   reason-first  {r0:8.1%} -> {r1:8.1%}"
          f"   0→1={b01} 1→0={b10}   McNemar p={p:.2e}")


# Accuracy is NEVER recomputed for GSM-Hard: those generations carry no gold
# (label firewall -- {sample_id, question, generated} only), so it is read from
# the frozen evaluation artifacts instead.
HARD_ACC = {("gsm_hard", "mdf_0"): .1800, ("gsm_hard", "mdf_neg6"): .2433,
            ("gsm_hard", "mdf_0_cot"): .2000, ("gsm_hard", "mdf_neg6_cot"): .2600}

PAIRS = [
    ("llama3/gsm8k", "mdf_0", "mdf_-6", "GSM8K    No-CoT  a=0 -> -6"),
    ("llama3/gsm8k", "mdf_0_cot", "mdf_-4_cot", "GSM8K    CoT     a=0 -> -4"),
    ("llama3/gsm_hard", "mdf_0", "mdf_neg6", "GSM-Hard No-CoT  a=0 -> -6"),
    ("llama3/gsm_hard", "mdf_0_cot", "mdf_neg6_cot", "GSM-Hard CoT     a=0 -> -6"),
]

if __name__ == "__main__":
    print("=" * 76)
    print("Pre-candidate reasoning -- EXPLORATORY, read-only, no frozen artifact")
    print("touched. cand_pos/pre_*/reason-first are NEW readouts; they do NOT")
    print("modify P2's frozen `early_candidate` feature or any predictor.")
    print("=" * 76)
    for tree, ta, tb, label in PAIRS:
        A, B = cell(tree, ta), cell(tree, tb)
        if A is None or B is None:
            print(f"\n{label}: missing cell -- skipped")
            continue
        key = tree.split("/")[-1]
        acc = None
        if (key, ta) in HARD_ACC:
            acc = (HARD_ACC[(key, ta)], HARD_ACC[(key, tb)])
        else:
            acc = (summarise(A)["acc"], summarise(B)["acc"])
        report(A, B, label, acc)
    print("\nNOT MEDIATION: cand_pos and pre_* are outcomes of alpha.")
