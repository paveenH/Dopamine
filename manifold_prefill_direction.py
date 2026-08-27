#!/usr/bin/env python3
"""Cross-dose direction test for the prefill displacement (Manifold section 4).

Runs on the SERVER with the conda env's `python`. READ-ONLY on the H5.

WHY. An equal INSIDE ratio does not imply an equal direction: two
displacements can occupy the same top-k subspace to the same degree and still
point different ways. And a differing inside ratio does not by itself refute
scalar gain -- that refutation needs the scalar model fitted and its residual
measured, not one of its corollaries.

WHAT IT TESTS
  1. cos(d_a, d_b) per question, then pooled -- do two doses point the same way?
  2. The FIXED-DIRECTION SCALAR model. If steering is scalar gain along one
     direction u, then d_a = c_a * u for every dose, so:
       - per question, cos(d_a, d_b) ~ 1 (or -1 for opposite signs);
       - the best rank-1 fit of the dose matrix leaves little residual.
     Reported as: residual = ||d_a - k * d_b||^2 / ||d_a||^2 at the
     least-squares k, which is exactly 1 - cos^2. So cos and residual are the
     same fact; k is the independent number (the amplitude ratio).

  This mirrors the repo's existing scalar-compression residual convention
  (CLAUDE.md, Qwen per-layer response): report k ALONGSIDE the residual, and
  never the residual alone, because 1 - cos^2 and cos carry no extra
  information beyond each other.

  3. A per-question SIGN check on the dose pair, since a pooled cosine can hide
     a mixture of aligned and anti-aligned questions.

NOT TESTED HERE. Whether any of this adds information beyond Z_prefill. That
is the incremental-prediction step and it is deliberately separate.
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifold_fit import iter_samples  # noqa: E402

SLOT = 8


def read_prefill(path, slot, expect):
    out = {}
    with h5py.File(path, "r") as f:
        for qi, g in iter_samples(f):
            if qi in out:
                raise SystemExit(f"[FAIL] duplicate question_idx {qi}")
            out[qi] = g["prefill_hs"][-1][slot].astype(np.float64)
    if set(out) != set(expect):
        raise SystemExit(f"[FAIL] {os.path.basename(path)}: question set "
                         f"mismatch vs the manifest")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--base_cell", default="nocot")
    p.add_argument("--cells", default="nocot_aneg8,nocot_aneg6,nocot_a6")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--size", default="8B")
    p.add_argument("--layer_start", type=int, default=11)
    p.add_argument("--layer_end", type=int, default=20)
    p.add_argument("--slot", type=int, default=SLOT)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default="prefill_direction.json")
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

    man = json.load(open(a.split_manifest))
    split = {k: sorted(man["split"][k]) for k in ("train", "val", "test")}
    allq = sorted(set().union(*split.values()))
    print(f"[split] {man.get('version')} "
          f"{ {k: len(v) for k, v in split.items()} }")

    stem = lambda c: os.path.join(
        a.h5_dir, f"hs_{a.task}_{a.size}_{c}_L{a.layer_start}-{a.layer_end}.h5")
    base = read_prefill(stem(a.base_cell), a.slot, allq)
    cells = a.cells.split(",")
    D = {}
    for c in cells:
        cur = read_prefill(stem(c), a.slot, allq)
        D[c] = np.stack([cur[q] - base[q] for q in allq])
    qidx = {q: i for i, q in enumerate(allq)}
    rng = np.random.default_rng(0)

    def block(name, qs):
        rows = [qidx[q] for q in qs]
        print(f"\n--- {name} (n={len(qs)}) ---")
        out = {"n": len(qs)}
        print(f"{'pair':>16} {'cos':>7} {'95% CI':>14} {'k':>7} "
              f"{'resid':>7}  {'sign':>10}")
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                A, B = D[cells[i]][rows], D[cells[j]][rows]
                num = (A * B).sum(1)
                na, nb = (A ** 2).sum(1), (B ** 2).sum(1)
                # Pooled cosine: energy-weighted, matching the pooled ratio
                # convention used for the inside fraction.
                cos_p = float(num.sum() / np.sqrt(na.sum() * nb.sum()))
                # least-squares k for A ~ k*B, and its residual (= 1 - cos^2)
                k_ls = float(num.sum() / nb.sum())
                resid = float(((A - k_ls * B) ** 2).sum() / na.sum())
                perq = num / np.sqrt(na * nb)
                frac_pos = float((perq > 0).mean())
                bs = np.array([
                    (lambda r: num[r].sum() / np.sqrt(na[r].sum() * nb[r].sum()))(
                        rng.integers(0, len(rows), len(rows)))
                    for _ in range(a.n_boot)])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                tag = f"{cells[i].split('_')[-1]}|{cells[j].split('_')[-1]}"
                print(f"{tag:>16} {cos_p:>7.3f} [{lo:>6.3f},{hi:>6.3f}] "
                      f"{k_ls:>7.3f} {resid:>7.3f}  {frac_pos*100:>8.1f}%+")
                out[tag] = {"cos_pooled": cos_p, "ci": [float(lo), float(hi)],
                            "k_ls": k_ls, "residual": resid,
                            "frac_positive_cos": frac_pos,
                            "per_question_cos_mean": float(perq.mean())}
        return out

    res = {"test": block("TEST (primary)", split["test"]),
           "val": block("VAL", split["val"]),
           "train": block("TRAIN (QC only)", split["train"]),
           "all": block("ALL 300 (descriptive, pools TRAIN)", allq)}

    with open(a.out, "w") as f:
        json.dump({"slot": a.slot, "decoder_layer": a.layer_start - 1 + a.slot,
                   "base_cell": a.base_cell,
                   "note": "residual == 1 - cos^2 exactly at the LS k; k is "
                           "the only independent number of the three",
                   "splits": res}, f, indent=2)
    print(f"\n[ok] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def orthogonal_component_report(D, W, cells, rows, base_cell="nocot_aneg6",
                                target="nocot_a6"):
    """Where does the target's EXTRA direction live, inside or outside top-k?

    The inside ratio and the cross-dose cosine are two SEPARATE observations
    that happen to agree. They are the same fact only if the target's
    orthogonal component (what is left after removing the shared axis) is
    mostly OUTSIDE the alpha=0 top-k subspace. That has to be measured, not
    assumed.
    """
    A, B = D[target][rows], D[base_cell][rows]
    k_ls = (A * B).sum() / (B ** 2).sum()
    R = A - k_ls * B                       # the target's extra direction
    out = {"k_ls": float(k_ls),
           "resid_energy_frac": float((R ** 2).sum() / (A ** 2).sum())}
    for k in (5, 10, 20):
        Wk = W[:k]
        out[f"resid_inside_k{k}"] = float(((R @ Wk.T) ** 2).sum() / (R ** 2).sum())
        out[f"shared_inside_k{k}"] = float(((B @ Wk.T) ** 2).sum() / (B ** 2).sum())
    return out
