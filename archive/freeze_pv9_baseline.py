#!/usr/bin/env python3
"""Write the FROZEN PV9 algorithmic baseline manifest.

Run ONCE, before any PV9 model run. Same role as `freeze_bandit_baseline.py`
has for pv6: it fixes the competence gate's comparison basis (seed bank +
Random/Greedy/Oracle point estimates and bootstrap CIs) so the basis cannot
drift after model behaviour is observed.

WHY A SEPARATE FILE AND A SEPARATE MANIFEST
-------------------------------------------
`bandit_pv6_baseline_manifest.json` is the frozen basis for the pv6/pv7/pv8
Easy and Hard gate verdicts that are already cited. Adding `neartie` to it
would change that file, and `freeze_bandit_baseline.py --check` would then
fail against every stored citation. PV9 therefore gets its own manifest and
leaves the pv6 one byte-unchanged.

The two manifests AGREE on `easy`: both call the same deterministic
`build_baseline_manifest`, so `easy` is duplicated, not forked. `--check`
asserts that agreement explicitly, because a silent divergence there would
mean PV9-Easy is being judged against a different basis than pv8-Easy.

    python3.10 freeze_pv9_baseline.py            # write if absent
    python3.10 freeze_pv9_baseline.py --check    # verify, never write
    python3.10 freeze_pv9_baseline.py --force    # overwrite
"""
import argparse
import json
import math
import os

import bandit_reference as br
from freeze_bandit_baseline import summarise

DEFAULT_PATH = "bandit_pv9_baseline_manifest.json"
PV6_PATH = "bandit_pv6_baseline_manifest.json"
ENV_KEYS = ("easy", "neartie")


def _roundtrip(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


# Floats are compared with a relative tolerance; everything else exactly.
#
# WHY THIS IS NOT A RELAXATION OF THE FREEZE
# ------------------------------------------
# The Easy cells ran on one host (python 3.10, arm64) and later cells on
# another (python 3.12, x86_64). Identical inputs through identical code give
# summation results that differ in the last ULP -- 0.04 vs 0.04000000000000001,
# 0.9696969696969692 vs ...697. An exact `!=` reads that as "the frozen basis
# does not reproduce" and refuses to run, on two machines that in fact agree.
#
# The tolerance is 1e-9 RELATIVE. The gate compares point estimates at the 0.05
# scale (SuffFail 0.200 vs Greedy 0.250), so a 1e-9 relative difference cannot
# move any verdict; a real basis change -- a different seed bank, environment
# or policy -- is orders of magnitude larger and still fails. Structure is
# unaffected: a missing/extra key, a changed seed list, a type change or a
# length change is still an exact mismatch.
_REL_TOL = 1e-9


def _diff(a, b, path=""):
    """Paths where `a` and `b` disagree. Floats within _REL_TOL agree.

    Returns a list of human-readable differences so a genuine mismatch says
    WHERE, rather than only that the manifest failed.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a is b else [f"{path}: {a!r} vs {b!r}"]
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_REL_TOL):
                return []
            return [f"{path}: {a!r} vs {b!r}  (delta {b - a:+.3e})"]
        return [f"{path}: {a!r} vs {b!r}"]
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: missing from stored")
            elif k not in b:
                out.append(f"{path}/{k}: missing from recompute")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def check_easy_agrees_with_pv6(man: dict) -> bool:
    """PV9-Easy and pv8-Easy must share one basis, or they are not comparable."""
    if not os.path.exists(PV6_PATH):
        print(f"  (skip) {PV6_PATH} absent, cannot cross-check easy")
        return True
    with open(PV6_PATH) as f:
        pv6 = json.load(f)
    a = _roundtrip(man["environments"]["easy"])
    b = _roundtrip(pv6["environments"]["easy"])
    d = _diff(a, b, "easy")
    if not d:
        print("  OK: easy basis is identical to the pv6 manifest")
        return True
    print("  MISMATCH: easy basis differs from the pv6 manifest")
    for line in d[:20]:
        print(f"    {line}")
    if len(d) > 20:
        print(f"    ... and {len(d) - 20} more")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    man = br.build_baseline_manifest(env_keys=ENV_KEYS)

    if args.check:
        if not os.path.exists(args.path):
            print(f"MISSING: {args.path}")
            return 1
        with open(args.path) as f:
            stored = json.load(f)
        d = _diff(stored, _roundtrip(man))
        if d:
            print(f"MISMATCH: {args.path} differs from a fresh recompute")
            for line in d[:20]:
                print(f"  {line}")
            if len(d) > 20:
                print(f"  ... and {len(d) - 20} more")
            return 1
        print(f"OK: {args.path} matches a fresh recompute")
        ok = check_easy_agrees_with_pv6(man)
        summarise(man)
        return 0 if ok else 1

    if os.path.exists(args.path) and not args.force:
        print(f"EXISTS: {args.path} (use --check to verify, --force to rewrite)")
        return 1

    with open(args.path, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"WROTE: {args.path}")
    check_easy_agrees_with_pv6(man)
    summarise(man)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
