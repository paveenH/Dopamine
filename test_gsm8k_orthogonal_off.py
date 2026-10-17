#!/usr/bin/env python3
"""Local synthetic checks for validate_masks(support=...). Needs numpy+torch; stubs llms/template/utils.
This is a numerical check of the mask validator, NOT a real-injection check."""
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

for name in ("llms", "template", "utils"):
    m = types.ModuleType(name)
    m.VicundaModel = object
    m.select_templates_gsm8k = None
    sys.modules[name] = m
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import get_answer_gsm8k_orthogonal as g  # noqa: E402
spec = importlib.util.spec_from_file_location("nmd", ROOT / "detection" / "nmd.py")
nmd = importlib.util.module_from_spec(spec); spec.loader.exec_module(nmd)

L, H, K, S, E = 12, 256, 8, 4, 8   # saved rows L; dense rows L+1; band [4,8)
rng = np.random.default_rng(0)
exp = rng.standard_normal((1, 1, L + 1, H)).astype(np.float32)
non = rng.standard_normal((1, 1, L + 1, H)).astype(np.float32)
tmp = Path(tempfile.mkdtemp())
np.save(tmp / "e.npy", exp); np.save(tmp / "n.npy", non)
rsn = nmd.get_nmd_mask(exp, non, K, S, E)
same = nmd.get_ortho_gauss_mask(exp, non, K, S, E, 1, True)
off = nmd.get_ortho_gauss_mask(exp, non, K, S, E, 1, False)
for n, a in (("rsn", rsn), ("same", same), ("off", off)):
    np.save(tmp / f"{n}.npy", a.astype(np.float32))
fails = 0


def check(name, fn, expect_ok, needle=""):
    global fails
    try:
        fn(); ok, msg = True, ""
    except Exception as e:
        ok, msg = False, str(e)
    good = ok == expect_ok and (expect_ok or needle in msg)
    fails += not good
    print(("ok  " if good else "FAIL"), name, "" if good else f"(ok={ok} msg={msg})")


def val(mask, support="same", rsn_arr=None, alpha=-6.0):
    np.save(tmp / "m.npy", mask.astype(np.float32))
    p = tmp / "r.npy"; np.save(p, (rsn if rsn_arr is None else rsn_arr).astype(np.float32))
    return g.validate_masks(str(tmp / "m.npy"), str(p), str(tmp / "e.npy"), str(tmp / "n.npy"), S, E, True, alpha, support)


check("off passes support=off", lambda: val(off, "off"), True)
rep = val(off, "off"); assert rep["support_kind"] == "off" and all(r["support_overlap"] == 0 for r in rep["layers"])
check("off rejected by default (same) check", lambda: val(off), False, "support")
check("same passes default (v1 behaviour)", lambda: val(same), True)
check("same masquerading as off rejected", lambda: val(same, "off"), False, "overlaps")
bad = off.copy(); r = S - 1 + 1; j = np.flatnonzero(rsn[r])[0]; k = np.flatnonzero(off[r])[0]
bad[r, j], bad[r, k] = off[r, k], 0.0
check("one coordinate moved onto RSN support rejected", lambda: val(bad, "off"), False, "overlaps")
cnt = off.copy(); cnt[r, np.flatnonzero(cnt[r])[0]] = 0.0
check("wrong nonzero count rejected", lambda: val(cnt, "off"), False, "count")
check("wrong norm rejected", lambda: val(off * 1.01, "off"), False, "norm")
dense = (exp - non).squeeze()[r + 1]
nonorth = off.copy(); nonorth[r] += 0.05 * dense * (nonorth[r] != 0)
check("non-orthogonal rejected", lambda: val(nonorth, "off"), False, "orthogonality")
nf = off.copy(); nf[r, np.flatnonzero(nf[r])[0]] = np.nan
check("non-finite rejected", lambda: val(nf, "off"), False, "non-finite")
oob = off.copy(); oob[0, 3] = 0.1
check("out-of-band nonzero rejected", lambda: val(oob, "off"), False, "outside requested band")
check("bad support kind rejected", lambda: val(off, "weird"), False, "unknown support")
check("seeds differ -> distinct masks", lambda: (_ for _ in ()).throw(AssertionError()) if np.array_equal(
    off, nmd.get_ortho_gauss_mask(exp, non, K, S, E, 2, False)) else None, True)
print("FAILED" if fails else "all passed")
sys.exit(1 if fails else 0)
