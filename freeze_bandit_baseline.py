#!/usr/bin/env python3
"""Write the FROZEN pv6 algorithmic baseline manifest (§3.5 / §3.7).

Run ONCE, before any model run. The manifest fixes the competence gate's
comparison basis (seed bank + Random/Greedy/Oracle point estimates and
bootstrap CIs) so it cannot drift with later model behaviour.

    python3.10 freeze_bandit_baseline.py                 # write if absent
    python3.10 freeze_bandit_baseline.py --check         # verify, never write
    python3.10 freeze_bandit_baseline.py --force         # overwrite

It is fully deterministic (no model, no GPU), so --check recomputes and
diffs rather than trusting the stored file.
"""
import argparse
import json
import os
import sys

import bandit_reference as br

DEFAULT_PATH = "bandit_pv6_baseline_manifest.json"


def _fmt(ci: dict) -> str:
    return f"{ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def summarise(man: dict) -> None:
    for key, e in man["environments"].items():
        print(f"\n== {e['name']}  K={e['k']}  T={e['horizon']}  "
              f"probs={e['probs']}")
        rep = e["bank_report"]
        print(f"   bank n={rep['n']}  cells {rep['n_cross_cells_used']}"
              f"/{rep['n_cross_cells_total']}  max_repeat={rep['max_cell_repeat']}"
              f"  pos_bal={rep['position_balanced']}"
              f"  id_bal={rep['identity_balanced']}")
        print(f"   seeds {e['seed_bank']}")
        print(f"   smoke {e['smoke_bank']}")
        hdr = f"   {'policy':7s} {'SuffFail(T/2)':>22s} {'KxMinFrac':>22s} {'late_opt':>22s}"
        print(hdr)
        for pol, s in e["policies"].items():
            print(f"   {pol:7s} {_fmt(s['suff_fail_freq_half']):>22s} "
                  f"{_fmt(s['k_min_frac_full']):>22s} "
                  f"{_fmt(s['late_opt_frac']):>22s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the stored file")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest")
    args = ap.parse_args()

    man = br.build_baseline_manifest()

    if args.check:
        if not os.path.exists(args.path):
            print(f"MISSING: {args.path}")
            return 1
        with open(args.path) as f:
            stored = json.load(f)
        # Compare in ROUND-TRIPPED form: bank_report keys position_counts by
        # int, and JSON has no int keys, so a fresh in-memory manifest can
        # never compare equal to a reloaded one. Serialising both puts them in
        # the same representation, which is also the representation that
        # actually gets stored and cited.
        fresh = json.loads(json.dumps(man, sort_keys=True))
        if stored == fresh:
            print(f"OK: {args.path} matches a fresh recompute")
            summarise(man)
            return 0
        print(f"MISMATCH: {args.path} differs from a fresh recompute")
        return 1

    if os.path.exists(args.path) and not args.force:
        print(f"EXISTS: {args.path} (use --check to verify, --force to rewrite)")
        return 1

    with open(args.path, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"wrote {args.path}")
    summarise(man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
