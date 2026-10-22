#!/usr/bin/env python3
"""Exact prefill displacement decomposition in AMBIENT space (Manifold section 4).

Runs on the SERVER with the conda env's `python`. READ-ONLY on the H5.

WHY THIS EXISTS. The exported per-phase scalars (`coord`, `energy`, `re`,
`re_by_k`) cannot recover the displacement's normal component. `energy -
||coord||^2` is a residual ENERGY, so differencing two cells leaves out the
cross term -2<n_alpha, n_0>: it is not an upper bound, not a lower bound, it is
not a bound at all. A residual NORM difference would at least be a lower bound
on the normal displacement, but it is still not the quantity we want. The only
correct route is the ambient displacement itself.

FROZEN DEFINITION
    d   = h(alpha) - h(0)                     ambient, same question, same token
    f_k = ||W_k d||^2 / ||d||^2               k = 5, 10, 20

  W_k are the first k components of the SAME alpha=0 TRAIN basis every cell was
  projected onto. mu cancels in d and is therefore irrelevant here -- which is
  exactly why prefill is the phase that licenses a displacement claim.

  PRIMARY is the ENERGY-POOLED ratio sum||W_k d||^2 / sum||d||^2, not the mean
  of per-question f_k. A per-question ratio weights a question whose
  displacement is near zero exactly as heavily as one that moved a lot, so a
  handful of tiny-displacement questions can dominate the mean.

WORDING. Report as energy INSIDE / OUTSIDE the alpha=0 top-k PCA subspace.
Never "off-manifold": k=20 spans only ~50% of the alpha=0 variance, so the
complement contains directions that are ordinary natural variation.
"""
import argparse
import json
import os
import sys

# Hard-set, NOT setdefault: an inherited wrong value is exactly the failure
# mode here (the server's OpenBLAS inverts on high thread counts, 117x), and
# setdefault would silently keep it.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifold_fit import iter_samples  # noqa: E402

KS = (5, 10, 20)
SLOT = 8            # storage slot 8 == decoder layer 18 (primary)


def cell_stem(task, size, cell, ls, le):
    return f"hs_{task}_{size}_{cell}_L{ls}-{le}.h5"


def read_prefill(path, slot, expect):
    """{question_idx: last-prefill state at `slot`} -- read-only, fp32.

    Reuses section 3's iterator so the `samples/` group layout is handled
    identically. Fails closed on a duplicate question_idx and on any deviation
    from `expect`: checking only that the cells AGREE would pass a run where
    every cell is missing the SAME question, which silently shrinks the
    cohort while looking perfectly paired.
    """
    out = {}
    with h5py.File(path, "r") as f:
        for qi, g in iter_samples(f):
            if qi in out:
                raise SystemExit(f"[FAIL] {os.path.basename(path)}: duplicate "
                                 f"question_idx {qi}")
            out[qi] = g["prefill_hs"][-1][slot].astype(np.float32)
    got, want = set(out), set(expect)
    if got != want:
        raise SystemExit(
            f"[FAIL] {os.path.basename(path)}: question set mismatch "
            f"(missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--split_manifest", required=True,
                   help="REQUIRED. No default: 300-question totals pool the "
                        "185 TRAIN questions the basis was fit on.")
    p.add_argument("--basis", required=True, help="basis.npz from section 3")
    p.add_argument("--base_cell", default="nocot")
    p.add_argument("--cells", default="nocot_aneg8,nocot_aneg6,nocot_a6")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--size", default="8B")
    p.add_argument("--layer_start", type=int, default=11)
    p.add_argument("--layer_end", type=int, default=20)
    p.add_argument("--slot", type=int, default=SLOT)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default="prefill_exact.json")
    a = p.parse_args()

    if os.path.exists(a.out):
        print(f"[FAIL] {a.out} exists; move it aside deliberately")
        return 1

    # The band has (layer_end - layer_start) middle slots; the H5 additionally
    # stores the model's FINAL layer after them. A slot at or past n_middle is
    # the final layer, which reads legally and silently yields band-looking
    # geometry from the wrong layer (Qwen: n_middle=6, so slot 6 is decoder 27).
    n_middle = a.layer_end - a.layer_start
    if not 0 <= a.slot < n_middle:
        print(f"[FAIL] --slot {a.slot} is outside the middle band "
              f"[0,{n_middle}) for L{a.layer_start}-{a.layer_end}; slot "
              f"{n_middle} would be the stored FINAL layer, not a band layer")
        return 1

    W = np.load(a.basis)[f"{a.slot}|prefill|components"].astype(np.float64)
    print(f"[basis] slot {a.slot} prefill components {W.shape}")
    orth = np.abs(W @ W.T - np.eye(W.shape[0])).max()
    print(f"[basis] orthonormality max|WW^T - I| = {orth:.2e}")
    if orth > 1e-4:
        print("[FAIL] basis is not orthonormal; a non-orthonormal W makes "
              "||W_k d||^2 not an energy")
        return 1

    man = json.load(open(a.split_manifest))
    split = {k: sorted(man["split"][k]) for k in ("train", "val", "test")}
    allq = sorted(set().union(*split.values()))
    n_exp = len(allq)
    if any(set(split[x]) & set(split[y])
           for x, y in (("train", "val"), ("train", "test"), ("val", "test"))):
        print("[FAIL] split buckets overlap")
        return 1
    print(f"[split] {man.get('version')} "
          f"train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} total={n_exp}")

    stem = lambda c: os.path.join(
        a.h5_dir, cell_stem(a.task, a.size, c, a.layer_start, a.layer_end))
    base = read_prefill(stem(a.base_cell), a.slot, allq)
    print(f"[read] {a.base_cell}: {len(base)} questions")

    # ROLE OF EACH SPLIT (frozen). The basis was fit on TRAIN, so a total over
    # all 300 is contaminated by the questions that defined the subspace and is
    # DESCRIPTIVE ONLY. TEST at k=20 is the single primary number.
    ROLE = {"train": "QC only (basis was fit here)",
            "val":   "k sensitivity",
            "test":  "PRIMARY (k=20)",
            "all":   "descriptive only (pools TRAIN)"}

    rng = np.random.default_rng(0)
    results = {}
    for c in a.cells.split(","):
        cur = read_prefill(stem(c), a.slot, allq)
        D_all = {q: cur[q].astype(np.float64) - base[q].astype(np.float64)
                 for q in allq}
        row = {}
        print(f"\n=== {c} ===")
        for name in ("train", "val", "test", "all"):
            qs = allq if name == "all" else split[name]
            D = np.stack([D_all[q] for q in qs])
            tot = (D ** 2).sum(axis=1)
            Z = D @ W.T
            sub = {"n": len(qs), "role": ROLE[name],
                   "mean_disp_norm": float(np.sqrt(tot).mean())}
            print(f"  [{name}] n={len(qs)}  mean||d||="
                  f"{sub['mean_disp_norm']:.3f}   {ROLE[name]}")
            for k in KS:
                tan = (Z[:, :k] ** 2).sum(axis=1)
                pooled = float(tan.sum() / tot.sum())
                bs = np.array([
                    (lambda i: tan[i].sum() / tot[i].sum())(
                        rng.integers(0, len(qs), len(qs)))
                    for _ in range(a.n_boot)])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                perq = float((tan / tot).mean())
                sub[f"k{k}"] = {"pooled": pooled, "ci": [float(lo), float(hi)],
                                "per_question_mean": perq}
                star = " *" if (name == "test" and k == 20) else ""
                print(f"    k={k:>2}: inside {pooled*100:>5.1f}% "
                      f"[{lo*100:.1f},{hi*100:.1f}]   "
                      f"outside {(1-pooled)*100:>5.1f}%"
                      f"   (per-q {perq*100:.1f}%){star}")
            row[name] = sub
        results[c] = row

    with open(a.out, "w") as f:
        json.dump({"slot": a.slot, "decoder_layer": a.layer_start - 1 + a.slot,
                   "base_cell": a.base_cell, "ks": list(KS),
                   "primary": "TEST split, k=20, pooled energy ratio",
                   "split_manifest_version": man.get("version"),
                   "split_counts": {k: len(v) for k, v in split.items()},
                   "cells": results}, f, indent=2)
    print(f"\n[ok] wrote {a.out}")
    print("     Wording: energy INSIDE / OUTSIDE the alpha=0 top-k PCA "
          "subspace. Not 'off-manifold'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
