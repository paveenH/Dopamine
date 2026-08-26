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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import h5py

KS = (5, 10, 20)
SLOT = 8            # storage slot 8 == decoder layer 18 (primary)


def cell_stem(task, size, cell, ls, le):
    return f"hs_{task}_{size}_{cell}_L{ls}-{le}.h5"


def read_prefill(path, slot):
    """{question_idx: last-prefill state at `slot`} -- read-only, fp32."""
    out = {}
    with h5py.File(path, "r") as f:
        keys = [k for k in f.keys() if k != "meta"]
        for k in keys:
            g = f[k]
            qi = g.attrs.get("question_idx")
            if qi is None:
                continue
            out[int(qi)] = g["prefill_hs"][-1][slot].astype(np.float32)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
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

    W = np.load(a.basis)[f"{a.slot}|prefill|components"].astype(np.float64)
    print(f"[basis] slot {a.slot} prefill components {W.shape}")
    orth = np.abs(W @ W.T - np.eye(W.shape[0])).max()
    print(f"[basis] orthonormality max|WW^T - I| = {orth:.2e}")
    if orth > 1e-4:
        print("[FAIL] basis is not orthonormal; a non-orthonormal W makes "
              "||W_k d||^2 not an energy")
        return 1

    stem = lambda c: os.path.join(
        a.h5_dir, cell_stem(a.task, a.size, c, a.layer_start, a.layer_end))
    base = read_prefill(stem(a.base_cell), a.slot)
    print(f"[read] {a.base_cell}: {len(base)} questions")

    rng = np.random.default_rng(0)
    results = {}
    for c in a.cells.split(","):
        cur = read_prefill(stem(c), a.slot)
        qs = sorted(set(base) & set(cur))
        if len(qs) != len(base) or len(qs) != len(cur):
            print(f"[FAIL] {c}: pairing is not complete "
                  f"({len(qs)} vs {len(base)}/{len(cur)})")
            return 1

        D = np.stack([cur[q].astype(np.float64) - base[q].astype(np.float64)
                      for q in qs])                       # (n, dim)
        tot = (D ** 2).sum(axis=1)                        # ||d||^2 per question
        Z = D @ W.T                                       # (n, k_max)

        row = {"n": len(qs), "mean_disp_norm": float(np.sqrt(tot).mean())}
        print(f"\n{c}: n={len(qs)}  mean||d||={row['mean_disp_norm']:.3f}")
        for k in KS:
            tan = (Z[:, :k] ** 2).sum(axis=1)
            pooled = float(tan.sum() / tot.sum())
            bs = np.array([
                (lambda i: tan[i].sum() / tot[i].sum())(
                    rng.integers(0, len(qs), len(qs)))
                for _ in range(a.n_boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            perq = float((tan / tot).mean())
            row[f"k{k}"] = {"pooled": pooled, "ci": [float(lo), float(hi)],
                            "per_question_mean": perq}
            print(f"  k={k:>2}: inside {pooled*100:>5.1f}% "
                  f"[{lo*100:.1f},{hi*100:.1f}]   outside {(1-pooled)*100:>5.1f}%"
                  f"   (per-question mean {perq*100:.1f}%, secondary)")
        results[c] = row

    with open(a.out, "w") as f:
        json.dump({"slot": a.slot, "decoder_layer": a.layer_start - 1 + a.slot,
                   "base_cell": a.base_cell, "ks": list(KS),
                   "primary": "pooled energy ratio", "cells": results}, f,
                  indent=2)
    print(f"\n[ok] wrote {a.out}")
    print("     Wording: energy INSIDE / OUTSIDE the alpha=0 top-k PCA "
          "subspace. Not 'off-manifold'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
