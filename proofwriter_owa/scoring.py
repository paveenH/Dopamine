#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring primitives for ProofWriter OWA: exact McNemar, Holm correction,
question-paired bootstrap CI. No GPU, no network -- pure stdlib.

Matches the conventions already used across this repo's fixed-workpoint
transfer scripts (eval_bbh_numeric.py / eval_cruxeval.py / eval_logiqa2.py):
paired exact two-sided McNemar on 0/1 accuracy vectors, Holm within a
declared family size, bootstrap over per-item differences in percentage
points.
"""

from __future__ import annotations

import random
from math import comb


def mcnemar_exact(a: list[int], b: list[int]):
    """a, b: paired 0/1 accuracy vectors (same length, same item order).
    Returns (discordant_0to1, discordant_1to0, two_sided_p)."""
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired)")
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def holm(pairs: list[tuple]) -> dict:
    """pairs: list of (key, raw_p). Returns {key: holm_adjusted_p}."""
    s = sorted(pairs, key=lambda t: t[1])
    m = len(s)
    out = {}
    running = 0.0
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = adj
    return out


def question_paired_bootstrap_ci(a: list[int], b: list[int], B: int = 10000,
                                  seed: int = 0, alpha_level: float = 0.05):
    """Item-level (question-paired) bootstrap 95% CI on (mean(b) - mean(a))
    in percentage points. Resamples ITEM INDICES with replacement, so the
    unit of resampling is the question (matches the P3/P4/P4b/P4c convention)."""
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired)")
    n = len(a)
    rng = random.Random(seed)
    d = [(y - x) * 100.0 for x, y in zip(a, b)]
    draws = []
    for _ in range(B):
        draws.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo_idx = int((alpha_level / 2) * B)
    hi_idx = int((1 - alpha_level / 2) * B)
    return draws[lo_idx], draws[min(hi_idx, B - 1)]


def discrete_argmax(alpha_to_acc: dict) -> int | float:
    """Argmax over sampled alpha; ties broken toward the smaller |alpha|."""
    best = None
    for al, acc in alpha_to_acc.items():
        key = (-acc, abs(al))
        if best is None or key < best[0]:
            best = (key, al)
    return best[1]


def near_optimal_region(alpha_to_acc: dict, tolerance_pp: float = 0.0) -> list:
    """Alpha values whose accuracy is within `tolerance_pp` percentage points
    of the sampled-set best. tolerance_pp=0.0 -> exact-tie set only. Defined
    ONLY over the actually sampled points, never interpolated, and kept
    separate from Holm significance (a caller must combine the two)."""
    if not alpha_to_acc:
        return []
    best_acc = max(alpha_to_acc.values())
    return sorted(al for al, acc in alpha_to_acc.items()
                  if (best_acc - acc) * 100.0 <= tolerance_pp)
