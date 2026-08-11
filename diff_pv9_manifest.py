#!/usr/bin/env python
"""Show WHERE the stored PV9 manifest differs from a fresh recompute.

Diagnostic only: reads and compares, never writes. `--check` reports a
boolean; this reports the paths and both values so the cause (env drift vs
numpy version vs stale file) is readable.
"""
import json
import bandit_reference as br
import freeze_pv9_baseline as fz

stored = json.load(open(fz.DEFAULT_PATH))
fresh = json.loads(json.dumps(br.build_baseline_manifest(env_keys=fz.ENV_KEYS),
                              sort_keys=True))


def walk(a, b, path=""):
    if type(a) is not type(b):
        print(f"{path}: TYPE {type(a).__name__} vs {type(b).__name__}")
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                print(f"{path}/{k}: only in FRESH = {b[k]!r}")
            elif k not in b:
                print(f"{path}/{k}: only in STORED = {a[k]!r}")
            else:
                walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            print(f"{path}: LEN {len(a)} vs {len(b)}")
            print(f"    stored={a!r}")
            print(f"    fresh ={b!r}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif a != b:
        extra = ""
        if isinstance(a, float) and isinstance(b, float):
            extra = f"   (delta {b - a:+.3e})"
        print(f"{path}: stored={a!r}  fresh={b!r}{extra}")


if stored == fresh:
    print("IDENTICAL")
else:
    walk(stored, fresh)

import numpy, platform, sys
print(f"\npython {sys.version.split()[0]}  numpy {numpy.__version__}  "
      f"{platform.platform()}")
