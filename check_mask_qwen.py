#!/usr/bin/env python
"""check_mask_qwen.py — verify a freshly generated mask BEFORE any sweep uses it.

Layer misalignment is the classic bug in this repo (the 2026-05-30 offset bug),
and it is silent: a misaligned mask still loads, still steers, and still produces
plausible-looking numbers -- they are just measuring the wrong layers.

Checks, none of which need a GPU or the model weights:
  1. FILE       — exists, loads, dtype/shape sane, finite.
  2. ROW COUNT  — rows == the model's decoder-layer count (embedding row dropped
                  by detection/nmd.py `return mask[1:]`, so a mask with L+1 rows
                  means the drop did not happen).
  3. ALIGNMENT  — non-zero rows are EXACTLY `decoder_layer_range(start, end)`,
                  the half-open range the hooks actually register on. This is the
                  check that catches an off-by-one band.
  4. SPARSITY   — every steered row has the same non-zero count, and it matches
                  top_k = floor(H * percentage / 100).
  5. COMPARISON — if a second mask is given (--compare), report band overlap and
                  per-layer support overlap, so "is this a different band or the
                  same neurons relabelled" is answered explicitly.

Usage:
  python check_mask_qwen.py --layer_start 11 --layer_end 18
  python check_mask_qwen.py --layer_start 11 --layer_end 18 --compare 16 22
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import decoder_layer_range  # noqa: E402

N_DECODER_LAYERS = {"7B": 28, "8B": 32}


def load(base_dir, hs, typ, mask_type, pct, st, en, size):
    d = os.path.join(base_dir, "mask", f"{hs}_{typ}_logits")
    p = os.path.join(d, f"{mask_type}_{pct}_{st}_{en}_{size}.npy")
    return p, (np.load(p) if os.path.exists(p) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--hs", default="qwen2.5")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--size", default="7B")
    ap.add_argument("--layer_start", type=int, required=True)
    ap.add_argument("--layer_end", type=int, required=True)
    ap.add_argument("--compare", nargs=2, type=int, metavar=("START", "END"),
                    help="second mask's band, to contrast supports")
    a = ap.parse_args()

    fails = []
    path, m = load(a.base_dir, a.hs, a.type, a.mask_type, a.percentage,
                   a.layer_start, a.layer_end, a.size)
    print(f"mask : {path}")
    if m is None:
        print("!! NOT FOUND — generate it first (bash run_nmd_qwen25.sh)")
        return 1

    # 1. file
    print(f"shape: {m.shape}  dtype={m.dtype}")
    if not np.all(np.isfinite(m)):
        fails.append("mask contains non-finite values")

    # 2. row count
    exp_rows = N_DECODER_LAYERS.get(a.size)
    print(f"\n--- rows vs decoder layers ---")
    print(f"  rows={m.shape[0]}  expected={exp_rows} (embedding row must be dropped)")
    if exp_rows and m.shape[0] != exp_rows:
        fails.append(f"rows {m.shape[0]} != decoder layers {exp_rows}"
                     + (" (looks like the embedding row was NOT dropped)"
                        if exp_rows and m.shape[0] == exp_rows + 1 else ""))

    # 3. alignment
    nz = [i for i, r in enumerate(m) if np.count_nonzero(r)]
    band = list(decoder_layer_range(a.layer_start, a.layer_end))
    print(f"\n--- layer alignment ---")
    print(f"  non-zero rows : {nz}")
    print(f"  hooked band   : {band}   (decoder_layer_range, half-open)")
    print(f"  L (steered)   : {len(band)}")
    if sorted(nz) != sorted(band):
        fails.append(f"non-zero rows {nz} != hooked band {band}")
        print("  !! MISALIGNED — steering would hit different layers than intended")
    else:
        print("  alignment: ok")

    # 4. sparsity
    H = m.shape[1]
    top_k = max(1, int(H * a.percentage / 100.0))
    counts = sorted({int(np.count_nonzero(m[i])) for i in nz})
    print(f"\n--- sparsity ---")
    print(f"  H={H}  percentage={a.percentage}%  -> expected top_k={top_k}")
    print(f"  distinct non-zero counts across steered rows: {counts}")
    if counts != [top_k]:
        fails.append(f"non-zero counts {counts} != expected top_k {top_k}")

    # 5. comparison
    if a.compare:
        cs, ce = a.compare
        p2, m2 = load(a.base_dir, a.hs, a.type, a.mask_type, a.percentage, cs, ce, a.size)
        print(f"\n--- comparison vs [{cs},{ce}) ---")
        print(f"  {p2}")
        if m2 is None:
            print("  (not found — skipped)")
        else:
            b2 = list(decoder_layer_range(cs, ce))
            shared = sorted(set(band) & set(b2))
            print(f"  band overlap  : {shared or 'NONE (disjoint layers)'}")
            for l in shared:
                s1 = set(np.nonzero(m[l])[0].tolist())
                s2 = set(np.nonzero(m2[l])[0].tolist())
                j = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else float("nan")
                print(f"    layer {l:2d}: |A|={len(s1)} |B|={len(s2)} "
                      f"shared={len(s1 & s2)} Jaccard={j:.3f}")
            if not shared:
                print("  -> fully disjoint bands: this is a genuine layer-position test,")
                print("     not the same neurons relabelled.")

    print(f"\n{'=' * 60}")
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASSED — mask is well-formed and correctly aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
