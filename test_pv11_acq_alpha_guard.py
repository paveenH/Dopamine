#!/usr/bin/env python3.10
"""PV11 alpha x block guard: only three cells are runnable.

    block=all          alpha=0             the original 160-state baseline
    block=acquisition  alpha in {-4, +4}   PV11-Acq (pv11_amendment_01.json)

The guard started as alpha=0-only and was WIDENED, not removed. What must not
happen is that widening it for the authorized follow-up quietly makes any
alpha reachable, so the three refusals that look harmless are tested
explicitly:

  * full-bank +-4       -- the gate FAILED; only the Acquisition follow-up
                           was authorized, not the whole protocol
  * acquisition alpha=0 -- already exists inside the full A0 file; a second
                           baseline invites quoting whichever one suits
  * commitment anything -- withdrawn on construct grounds

Two independent layers are checked: the DRIVER (which validates the pair) and
the LAUNCHER (which exposes no --alpha at all and hardcodes each step). The
point of two layers is that editing one must not unlock the other, so the
launcher is checked by source inspection rather than by trusting the driver.

Runs the driver as a subprocess with a nonexistent model dir: the guard must
fire during argument validation, long before anything tries to load a model.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "run_bandit_pv11_episodes.py"
LAUNCHER = HERE / "run_bandit_pv11.sh"

failures = []


def ck(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


def invoke(block, alpha):
    """Run the driver far enough to hit the guard. Returns (rc, output)."""
    r = subprocess.run(
        [sys.executable, str(DRIVER),
         "--model_dir", "/nonexistent/model",
         "--block", block, "--alpha", str(alpha),
         "--base_dir", "/nonexistent/base",
         "--ans_file", "/nonexistent/out.json"],
        capture_output=True, text=True, cwd=str(HERE))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def refused(block, alpha):
    rc, out = invoke(block, alpha)
    return rc != 0 and "REFUSED" in out, out


def passed_guard(block, alpha):
    """The pair is authorized: the run must fail LATER, not at the guard."""
    rc, out = invoke(block, alpha)
    return "REFUSED" not in out, out


print("\n[1] the three AUTHORIZED cells clear the guard")
for block, alpha in (("all", 0), ("acquisition", -4), ("acquisition", 4)):
    ok, out = passed_guard(block, alpha)
    ck(f"block={block} alpha={alpha} accepted", ok,
       out.strip().split("\n")[-1][:90] if not ok else "")

print("\n[2] full-bank steering is refused (the gate FAILED)")
for alpha in (-4, 4, -8, 8, 2):
    ok, out = refused("all", alpha)
    ck(f"block=all alpha={alpha} refused", ok)
ok, out = refused("all", -4)
ck("full-bank message names the acquisition alternative",
   "--block acquisition" in out, out[:120])

print("\n[3] acquisition alpha=0 is refused (baseline already exists)")
ok, out = refused("acquisition", 0)
ck("block=acquisition alpha=0 refused", ok)
ck("message explains the baseline already exists",
   "ALREADY EXISTS" in out, out[:120])

print("\n[4] commitment is refused at every alpha")
for alpha in (0, -4, 4):
    ok, _ = refused("commitment", alpha)
    ck(f"block=commitment alpha={alpha} refused", ok)
ok, out = refused("commitment", -4)
ck("message says WITHDRAWN", "WITHDRAWN" in out, out[:120])

print("\n[5] unauthorized magnitudes are refused for acquisition too")
for alpha in (-8, -6, -2, 2, 6, 8, 1):
    ok, _ = refused("acquisition", alpha)
    ck(f"block=acquisition alpha={alpha} refused", ok)

print("\n[6] the launcher is an INDEPENDENT layer")
src = LAUNCHER.read_text()
ck("launcher exposes no --alpha passthrough",
   not re.search(r'--alpha\s+"?\$\{?(?:ALPHA|3|4)\b', src))
ck("AM4 step exists", re.search(r'\bAM4\b', src) is not None)
ck("AP4 step exists", re.search(r'\bAP4\b', src) is not None)
ck("acq steps hardcode --block acquisition",
   "--block acquisition" in src)
ck("usage line lists the new steps",
   "CHECK SMOKE A0 GATE AM4 AP4" in src)
# The acq alpha must come from a hardcoded branch, never from user input.
ck("acq alpha is set by the step, not by an argument",
   'ACQ_ALPHA="-4"' in src and 'ACQ_ALPHA="4"' in src)
ck("separate output dir per alpha",
   "pv11_acq_am4" in src and "pv11_acq_ap4" in src)
ck("launcher requires the A0 baseline before an acq cell",
   'no alpha=0 baseline at' in src)
ck("launcher verifies the analyzer freeze before spending GPU time",
   "freeze_pv11_acq_analysis.py --check" in src)

print("\n[7] the launcher restates the amendment's limits at run time")
for phrase, label in (
        ("NOT a gate", "not-a-gate-PASS"),
        ("pv11_amendment_01.json", "names the amendment"),
        ("low-power baseline", "null wording"),
        ("closes regardless of outcome", "termination rule")):
    ck(f"launcher states: {label}", phrase in src)

print()
if failures:
    print(f"FAILED ({len(failures)}): {failures}")
    sys.exit(1)
print("all alpha x block guard checks passed")
