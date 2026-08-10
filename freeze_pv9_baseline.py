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
import os

import bandit_reference as br
from freeze_bandit_baseline import summarise

DEFAULT_PATH = "bandit_pv9_baseline_manifest.json"
PV6_PATH = "bandit_pv6_baseline_manifest.json"
ENV_KEYS = ("easy", "neartie")


def _roundtrip(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


def check_easy_agrees_with_pv6(man: dict) -> bool:
    """PV9-Easy and pv8-Easy must share one basis, or they are not comparable."""
    if not os.path.exists(PV6_PATH):
        print(f"  (skip) {PV6_PATH} absent, cannot cross-check easy")
        return True
    with open(PV6_PATH) as f:
        pv6 = json.load(f)
    a = _roundtrip(man["environments"]["easy"])
    b = _roundtrip(pv6["environments"]["easy"])
    if a == b:
        print("  OK: easy basis is identical to the pv6 manifest")
        return True
    print("  MISMATCH: easy basis differs from the pv6 manifest")
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
        if stored != _roundtrip(man):
            print(f"MISMATCH: {args.path} differs from a fresh recompute")
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
