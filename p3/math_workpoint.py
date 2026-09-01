#!/usr/bin/env python3.10
"""MATH fixed-workpoint transfer: does the GSM8K workpoint beat alpha=0 on MATH?

This is the MATH analogue of P3's fixed-workpoint test
(docs/p3_amendment_05.json). The alpha is READ FROM THE FROZEN GSM8K RECORD --
Llama -6, Qwen +8 -- and is NOT re-searched, and NOT taken from any predicted
score. One pre-declared contrast per model, paired per question.

WHAT THIS IS NOT
----------------
It is NOT commitment-based workpoint SELECTION. That is P2B, already frozen
(p2/p2b_predictions.json -> p2b_evaluation.json), where the predictor ranked
the doses and its argmax was scored against the observed curve. The two tests
answer different questions and must never be pooled:

  selection        "can commitment features PICK the best dose on a new task?"
                   -> P2B. Qwen picked +6 (observed best, regret 0).
  fixed transfer   "does an ALREADY-ESTABLISHED workpoint still help?"
                   -> this script. Qwen tests +8, NOT +6.

Note the consequence for Qwen: its GSM8K workpoint (+8) is NOT its MATH optimum
(+6). So the fixed-workpoint test is a genuinely different -- and harder --
question than selection, and can fail while selection succeeded.

ACCURACY 口径 (frozen, per CLAUDE.md): first_acc is MAIN via the shared
offline extractor, imported never reimplemented; last_acc is tail-pollution
sensitivity only. utils.extract_math_answer takes the FIRST \\boxed{}.

STATISTICS. Exact two-sided McNemar on discordant pairs; paired bootstrap 95%
CI with the QUESTION as the unit (B=10000, seed 0). The two models form ONE
Holm family (m=2), matching the P3 supplement's convention.

PAIRING is verified, not assumed: same n, same order, identical gold vector.
Fails closed on any mismatch.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze_first_last_acc import all_boxed, norm_math, fallback_math

import numpy as np
from scipy.stats import binomtest

# alpha read from the frozen GSM8K record -- NOT re-searched, NOT predicted.
CELLS = {
    "llama": {
        "dir": ROOT / "llama3/math/math_eot",
        "file": "math_8B_11_20.json",
        "base": "mdf_0", "wp": "mdf_neg6", "wp_label": "-6",
        "band": "11-20 (L=9)",
        "workpoint_source": "GSM8K No-CoT peak alpha=-6 (182 production curve)",
    },
    "qwen": {
        "dir": ROOT / "qwen2.5/math",
        "file": "math_7B_16_22.json",
        "base": "mdf_0", "wp": "mdf_8", "wp_label": "+8",
        "band": "16-22 (L=6)",
        "workpoint_source": "GSM8K No-CoT saturation workpoint alpha=+8",
    },
}


def load(cfg, cell):
    p = cfg["dir"] / cell / cfg["file"]
    if not p.exists():
        return None, p
    return json.load(open(p))["data"], p


def vec_first(data):
    """Per-item first_acc correctness -- the MAIN readout."""
    out = []
    for s in data:
        m = all_boxed(s["generated"])
        pred = m[0] if m else fallback_math(s["generated"])
        out.append(norm_math(pred) == norm_math(s["gold_answer"]))
    return out


def vec_last(data):
    """Sensitivity only: LAST boxed. Never the headline."""
    out = []
    for s in data:
        m = all_boxed(s["generated"])
        pred = m[-1] if m else fallback_math(s["generated"])
        out.append(norm_math(pred) == norm_math(s["gold_answer"]))
    return out


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    p = 1.0 if n == 0 else binomtest(b01, n, 0.5).pvalue
    return b01, b10, p


def boot(a, b, B=10000, seed=0):
    d = np.array(b, float) - np.array(a, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return np.percentile(bs, [2.5, 97.5])


def holm(pairs):
    s = sorted(pairs, key=lambda t: t[1]); m = len(s); out = {}; run = 0.0
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(run, (m - i) * p)); run = adj; out[k] = adj
    return out


def main():
    print("MATH FIXED-WORKPOINT TRANSFER")
    print("alpha read from the frozen GSM8K record; NOT re-searched.")
    print("first_acc is MAIN (frozen offline extractor); last_acc is sensitivity.")
    print("=" * 74)

    res, missing = {}, []
    for model, cfg in CELLS.items():
        d0, p0 = load(cfg, cfg["base"])
        dw, pw = load(cfg, cfg["wp"])
        if d0 is None:
            missing.append(f"{model}: {p0}"); continue
        if dw is None:
            missing.append(f"{model}: {pw}"); continue

        # pairing: fail closed rather than silently comparing different problems
        assert len(d0) == len(dw), f"{model}: n differs {len(d0)} vs {len(dw)}"
        g0 = [s["gold_answer"] for s in d0]
        gw = [s["gold_answer"] for s in dw]
        assert g0 == gw, f"{model}: gold vectors differ -- not the same problems in the same order"
        q0 = [s.get("question", "") for s in d0]
        qw = [s.get("question", "") for s in dw]
        assert q0 == qw, f"{model}: question vectors differ -- pairing is invalid"

        a, b = vec_first(d0), vec_first(dw)
        al, bl = vec_last(d0), vec_last(dw)
        n = len(a)
        b01, b10, p = mcnemar(a, b)
        lo, hi = boot(a, b)
        res[model] = dict(
            n=n, acc0=sum(a) / n, accw=sum(b) / n,
            acc0_last=sum(al) / n, accw_last=sum(bl) / n,
            b01=b01, b10=b10, p=p, lo=lo, hi=hi, cfg=cfg,
        )

    if missing:
        print("\nMISSING CELLS (run them first):")
        for m in missing:
            print("  ", m)
        if not res:
            print("\nnothing to report."); return

    adj = holm([(m, r["p"]) for m, r in res.items()])

    print(f"\n{'model':>6} {'alpha':>6} {'acc0':>7} {'acc_wp':>7} {'Delta':>8} "
          f"{'0->1':>5} {'1->0':>5} {'p':>10} {'p_adj':>9}  95% CI (pp)")
    print("-" * 90)
    for m, r in res.items():
        d = (r["accw"] - r["acc0"]) * 100
        star = "*" if adj[m] < .05 else " "
        print(f"{m:>6} {r['cfg']['wp_label']:>6} {r['acc0']*100:7.2f} {r['accw']*100:7.2f} "
              f"{d:+8.2f} {r['b01']:5d} {r['b10']:5d} {r['p']:10.4g} {adj[m]:9.4g}{star} "
              f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}]")

    print(f"\n{'model':>6}  last_acc sensitivity (never the headline)")
    for m, r in res.items():
        print(f"{m:>6}  alpha=0 {r['acc0_last']*100:6.2f}   "
              f"alpha={r['cfg']['wp_label']} {r['accw_last']*100:6.2f}   "
              f"Delta {(r['accw_last']-r['acc0_last'])*100:+.2f} pp")

    print("\nProvenance")
    for m, r in res.items():
        print(f"  {m:>6}  band {r['cfg']['band']:<14} n={r['n']}  "
              f"workpoint from: {r['cfg']['workpoint_source']}")
    print("\nHolm family = the two models, m=2 (matches the P3 supplement).")
    print("Cross-run pairing: the stored cells' physical GPU is unrecoverable")
    print("(summary CSV carries no device field). At temperature=0 the drift is")
    print("small, but this is a limitation and must travel with the result.")


if __name__ == "__main__":
    main()
