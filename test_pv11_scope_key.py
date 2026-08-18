#!/usr/bin/env python3.10
"""PV11 scope-identity regression tests.

WHY THIS FILE EXISTS
--------------------
`--block` shipped as a smoke/debug flag. The driver filtered the state list
but handed the UNFILTERED bank to `resume_key`, so `--block acquisition` and
`--block all` produced a byte-identical key and would silently resume into
each other. Harmless while `--block` was debug-only; a blocker the moment a
single block became a formal cell (PV11-Acq, pv11_amendment_01.json).

The fix makes scope enter the identity, which means it can also BREAK the
stored alpha=0 cell if it changes the `block=all` strings. The golden values
below are the strings the stored A0 was written with, transcribed from
`bandit_pv11_alpha0.json` BEFORE the change. They are the thing that must
not move.

Exit non-zero on failure, like every other test in this repo.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bandit_pv11 as p11                      # noqa: E402
import run_bandit_pv11_episodes as drv         # noqa: E402

# ── golden: read off the STORED alpha=0 result, not recomputed ──────────────
GOLD_TAG = "pv11_p11_pv11-states-v1_pv10-strict-v2_k4_H5-20_s64e7fb3e4947"
GOLD_KEY_A0 = (
    "pv11_p11_pv11-states-v1_pv10-strict-v2_k4_H5-20_s64e7fb3e4947"
    "_bankc324c058fc4ac9bc_a0.0_L11-20_m51cdd819d200b365")
MODEL_CFG = "51cdd819d200b365"

# Acquisition is H=20 only, so its horizon segment is `H20`, not `H5-20`.
GOLD_TAG_ACQ = ("pv11_p11_pv11-states-v1_pv10-strict-v2_k4_H20"
                "_s8699154dee9b_blkacquisition")

failures = []


def check(name, got, want):
    if got == want:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          got  {got}\n          want {want}")
        failures.append(name)


def check_true(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


bank = json.loads((HERE / "pv11_state_bank.json").read_text())
states_all = bank["states"]
states_acq = [s for s in states_all if s["block"] == "acquisition"]
states_com = [s for s in states_all if s["block"] == "commitment"]

print("\n[1] full-bank key/tag are byte-identical to the stored A0")
check("tag(block=all)", drv.scope_tag(states_all, "all"), GOLD_TAG)
check("key(block=all, alpha=0)",
      drv.resume_key(0.0, 11, 20, bank, MODEL_CFG,
                     states=states_all, block="all"),
      GOLD_KEY_A0)
check("resume_key default args still full-bank",
      drv.resume_key(0.0, 11, 20, bank, MODEL_CFG),
      GOLD_KEY_A0)

# The stored file is the real authority; recomputation must match it.
a0_path = Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer/llama3"
               "/bandit/pv11/pv11_a0/bandit_pv11_alpha0.json")
if a0_path.exists():
    stored = json.loads(a0_path.read_text())
    check("stored A0 resume_key reproduces", stored["resume_key"], GOLD_KEY_A0)
    check("stored A0 interface_tag reproduces",
          stored["config"]["interface_tag"], GOLD_TAG)
else:
    print(f"  SKIP  stored-A0 cross-check ({a0_path} not present)")

print("\n[2] acquisition scope enters the tag AND the key")
check("tag(acquisition)", drv.scope_tag(states_acq, "acquisition"),
      GOLD_TAG_ACQ)
key_acq = drv.resume_key(-4.0, 11, 20, bank, MODEL_CFG,
                         states=states_acq, block="acquisition")
check_true("key contains _blkacquisition", "_blkacquisition" in key_acq,
           key_acq)
check_true("acq tag carries H20 not H5-20",
           "_H20_" in GOLD_TAG_ACQ and "H5-20" not in GOLD_TAG_ACQ)
check_true("acq uid digest differs from full",
           "s8699154dee9b" in GOLD_TAG_ACQ
           and "s64e7fb3e4947" not in GOLD_TAG_ACQ)

print("\n[3] full data cannot be resumed by an acquisition run, and back")
for alpha in (0.0, -4.0, 4.0):
    k_all = drv.resume_key(alpha, 11, 20, bank, MODEL_CFG,
                           states=states_all, block="all")
    k_acq = drv.resume_key(alpha, 11, 20, bank, MODEL_CFG,
                           states=states_acq, block="acquisition")
    k_com = drv.resume_key(alpha, 11, 20, bank, MODEL_CFG,
                           states=states_com, block="commitment")
    check_true(f"all != acquisition (alpha={alpha})", k_all != k_acq)
    check_true(f"all != commitment (alpha={alpha})", k_all != k_com)
    check_true(f"acquisition != commitment (alpha={alpha})", k_acq != k_com)

print("\n[4] resume validates the UID SET, not the count")
exp_acq = [s["state_uid"] for s in states_acq]
# a full-bank file offered to an acquisition cell: same count is impossible,
# but the check must reject on IDENTITY even when counts would agree.
done_wrong_block = {s["state_uid"]: {} for s in states_com}
try:
    drv.check_resumed_uids(done_wrong_block, exp_acq, "acquisition")
    check_true("commitment runs rejected by acquisition cell", False,
               "no SystemExit raised")
except SystemExit:
    check_true("commitment runs rejected by acquisition cell", True)

# 80 stored, 80 expected, but one uid swapped: a count check would pass.
done_swapped = {s["state_uid"]: {} for s in states_acq[:-1]}
done_swapped[states_com[0]["state_uid"]] = {}
check_true("count-equal but identity-wrong set is rejected",
           len(done_swapped) == len(exp_acq))
try:
    drv.check_resumed_uids(done_swapped, exp_acq, "acquisition")
    check_true("swapped-uid set rejected", False, "no SystemExit raised")
except SystemExit:
    check_true("swapped-uid set rejected", True)

# a genuine partial resume must be ACCEPTED
try:
    drv.check_resumed_uids({u: {} for u in exp_acq[:30]}, exp_acq,
                           "acquisition")
    check_true("partial in-scope resume accepted", True)
except SystemExit as e:
    check_true("partial in-scope resume accepted", False, str(e))

print("\n[5] the pre-fix defect is actually gone")
# Pre-fix, resume_key ignored its scope entirely. Reproduce that call shape
# and assert it no longer collapses the two scopes.
check_true("scope is not ignored",
           drv.resume_key(-4.0, 11, 20, bank, MODEL_CFG,
                          states=states_acq, block="acquisition")
           != drv.resume_key(-4.0, 11, 20, bank, MODEL_CFG,
                             states=states_all, block="all"))

print()
if failures:
    print(f"FAILED ({len(failures)}): {failures}")
    sys.exit(1)
print("all scope-identity checks passed")
