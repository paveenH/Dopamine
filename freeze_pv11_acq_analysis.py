#!/usr/bin/env python3.10
"""Verify the PV11-Acq analysis freeze.

    python3.10 freeze_pv11_acq_analysis.py --check

Recomputes the hashes recorded in pv11_acq_analysis_manifest.json. A MISMATCH
means the analysis code changed after it was frozen, so no PV11-Acq number is
citable against that manifest until the change is either reverted or recorded
in a dated amendment.

Freezing prevents tampering; it does NOT establish that the analyzer is
correct or that it was ever wired to real output. That is what
test_pv11_acq_analyzer.py is for -- the PV10 gate loader shipped frozen and
still could not read what the driver wrote.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "pv11_acq_analysis_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()

    if not MANIFEST.exists():
        print(f"MISSING: {MANIFEST}")
        return 1
    m = json.loads(MANIFEST.read_text())

    bad = []
    print("frozen artifacts")
    for key, entry in m["frozen_artifacts"].items():
        p = HERE / entry["file"]
        if not p.exists():
            print(f"  MISSING  {entry['file']}")
            bad.append(entry["file"])
            continue
        got = sha256(p)
        ok = got == entry["sha256"]
        print(f"  {'ok  ' if ok else 'DIFF'}     {entry['file']}")
        if not ok:
            print(f"           recorded {entry['sha256']}")
            print(f"           on disk  {got}")
            bad.append(entry["file"])

    print("\nreferenced amendment")
    a = m["references_amendment"]
    p = HERE / a["file"]
    if not p.exists():
        print(f"  MISSING  {a['file']}")
        bad.append(a["file"])
    else:
        got = sha256(p)
        ok = got == a["sha256"]
        print(f"  {'ok  ' if ok else 'DIFF'}     {a['file']}")
        if not ok:
            print(f"           recorded {a['sha256']}")
            print(f"           on disk  {got}")
            print("           The amendment is supposed to be immutable "
                  "once committed.")
            bad.append(a["file"])

    print()
    if bad:
        print(f"MISMATCH ({len(bad)}): {sorted(set(bad))}")
        print("No PV11-Acq number is citable against this manifest until "
              "this is resolved.")
        return 1
    print("freeze verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
