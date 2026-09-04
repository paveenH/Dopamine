#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for proofwriter_owa/scoring.py. No GPU, no network, stdlib only.

    python3 test_scoring.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import (mcnemar_exact, holm, question_paired_bootstrap_ci,
                     discrete_argmax, near_optimal_region)

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def test_mcnemar_no_discordant_pairs():
    a = [1, 0, 1, 0]
    b = [1, 0, 1, 0]
    b01, b10, p = mcnemar_exact(a, b)
    check(b01 == 0 and b10 == 0, f"expected no discordant pairs, got {b01},{b10}")
    check(p == 1.0, f"expected p=1.0, got {p}")


def test_mcnemar_all_improve():
    a = [0, 0, 0, 0]
    b = [1, 1, 1, 1]
    b01, b10, p = mcnemar_exact(a, b)
    check(b01 == 4 and b10 == 0, f"got {b01},{b10}")
    # exact binomial two-sided: n=4, k=0 -> p = 2 * (1/16) = 0.125
    check(abs(p - 0.125) < 1e-9, f"expected p=0.125, got {p}")


def test_mcnemar_symmetric():
    a = [1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 1, 0]
    b01_ab, b10_ab, p_ab = mcnemar_exact(a, b)
    b01_ba, b10_ba, p_ba = mcnemar_exact(b, a)
    check(b01_ab == b10_ba and b10_ab == b01_ba,
          "swapping a/b should swap discordant directions")
    check(abs(p_ab - p_ba) < 1e-12, "p-value should be symmetric under a<->b swap")


def test_mcnemar_length_mismatch_raises():
    try:
        mcnemar_exact([1, 0], [1, 0, 1])
        check(False, "mismatched lengths should raise ValueError")
    except ValueError:
        check(True, "correctly raised on length mismatch")


def test_holm_monotone_and_bounded():
    pairs = [("a", 0.001), ("b", 0.04), ("c", 0.5)]
    adj = holm(pairs)
    check(set(adj) == {"a", "b", "c"}, f"got keys {set(adj)}")
    for k, v in adj.items():
        check(0.0 <= v <= 1.0, f"adjusted p for {k} out of [0,1]: {v}")
    # Holm-adjusted p-values must be non-decreasing in raw-p rank
    ordered = sorted(pairs, key=lambda t: t[1])
    adjs_in_order = [adj[k] for k, _ in ordered]
    check(all(adjs_in_order[i] <= adjs_in_order[i + 1]
              for i in range(len(adjs_in_order) - 1)),
          f"Holm-adjusted p-values must be monotone non-decreasing: {adjs_in_order}")
    # hand-computed: m=3. sorted raw = [0.001,0.04,0.5]
    # step1: max(0, 3*0.001)=0.003
    # step2: max(0.003, 2*0.04)=0.08
    # step3: max(0.08, 1*0.5)=0.5
    check(abs(adj["a"] - 0.003) < 1e-9, f"expected 0.003, got {adj['a']}")
    check(abs(adj["b"] - 0.08) < 1e-9, f"expected 0.08, got {adj['b']}")
    check(abs(adj["c"] - 0.5) < 1e-9, f"expected 0.5, got {adj['c']}")


def test_holm_m3_all_significant_survives():
    # a case matching the ProofWriter m=3 per-model family shape, all clearly
    # significant raw p -- every one should stay significant after Holm.
    pairs = [(-6, 1e-6), (-4, 1e-5), (4, 1e-4)]
    adj = holm(pairs)
    check(all(v < 0.05 for v in adj.values()),
          f"all three should survive Holm(m=3) when raw p are this small: {adj}")


def test_bootstrap_ci_reproducible_with_seed():
    a = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
    lo1, hi1 = question_paired_bootstrap_ci(a, b, B=500, seed=42)
    lo2, hi2 = question_paired_bootstrap_ci(a, b, B=500, seed=42)
    check(lo1 == lo2 and hi1 == hi2, "same seed must give identical CI")
    check(lo1 <= hi1, f"lo should be <= hi, got {lo1},{hi1}")


def test_bootstrap_ci_identical_vectors_is_degenerate():
    a = [1, 0, 1, 1, 0]
    lo, hi = question_paired_bootstrap_ci(a, a, B=200, seed=0)
    check(lo == 0.0 and hi == 0.0,
          f"identical paired vectors -> zero difference in every resample, "
          f"got [{lo},{hi}]")


def test_discrete_argmax_simple():
    d = {-6: 0.3, -4: 0.4, 0: 0.35, 4: 0.2}
    check(discrete_argmax(d) == -4, f"got {discrete_argmax(d)}")


def test_discrete_argmax_tie_prefers_smaller_abs_alpha():
    d = {-6: 0.5, 0: 0.5, 4: 0.3}
    check(discrete_argmax(d) == 0,
          f"tie between -6 and 0 should prefer |alpha| smaller (0), got {discrete_argmax(d)}")

    d2 = {-6: 0.5, -4: 0.5, 4: 0.5}
    check(discrete_argmax(d2) == -4,
          f"tie between -6,-4,4 should prefer smallest |alpha| (-4), got {discrete_argmax(d2)}")


def test_near_optimal_region_exact_ties_only():
    d = {-6: 0.5, -4: 0.5, 0: 0.4, 4: 0.3}
    region = near_optimal_region(d, tolerance_pp=0.0)
    check(sorted(region) == [-6, -4], f"expected [-6,-4], got {sorted(region)}")


def test_near_optimal_region_with_tolerance():
    d = {-6: 0.50, -4: 0.49, 0: 0.30, 4: 0.20}
    region = near_optimal_region(d, tolerance_pp=1.5)
    check(sorted(region) == [-6, -4],
          f"within 1.5pp of best (0.50) should include -4 (0.49, 1pp gap), got {sorted(region)}")
    region_strict = near_optimal_region(d, tolerance_pp=0.0)
    check(sorted(region_strict) == [-6],
          f"zero tolerance should give only the exact best, got {sorted(region_strict)}")


def test_near_optimal_region_empty_input():
    check(near_optimal_region({}) == [], "empty input -> empty region, not a crash")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
