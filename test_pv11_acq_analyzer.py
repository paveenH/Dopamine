"""Verify analyze_bandit_pv11_acq.py fails closed. Mutates a COPY of A0.

The analyzer is frozen (pv11_acq_analysis_manifest.json), so its guards can
never be exercised by a real bad cell before the data that would need them
exists. Every guard is therefore driven here by a synthetic mutation of the
stored alpha=0 file, which is only ever read.

Two controls come first and must SUCCEED: an unmutated baseline, and a
well-formed alpha=+4 pair. Without them a file that rejected everything
would look perfect.

One guard is worth reading: a cell declaring block=acquisition that also
holds a commitment run is REJECTED, not silently filtered. Counting after
filtering would report 80/80 and drop the stray run unseen.
"""
import json, sys, copy, subprocess, tempfile
from pathlib import Path
REPO = Path("/Users/paveenhuang/Downloads/Dopamine")
sys.path.insert(0, str(REPO))
A0 = Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer/llama3/bandit"
          "/pv11/pv11_a0/bandit_pv11_alpha0.json")
base = json.loads(A0.read_text())

def run(payload_a0, payload_steer=None, expect_fail=True, name=""):
    with tempfile.TemporaryDirectory() as td:
        p0 = Path(td)/"a0.json"; p0.write_text(json.dumps(payload_a0))
        cmd = [sys.executable, str(REPO/"analyze_bandit_pv11_acq.py"), "--a0", str(p0)]
        if payload_steer is not None:
            p4 = Path(td)/"ap4.json"; p4.write_text(json.dumps(payload_steer))
            cmd += ["--ap4", str(p4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        failed = r.returncode != 0
        ok = failed == expect_fail
        print(("  ok    " if ok else "  FAIL  ") + name +
              ("" if ok else f"  (rc={r.returncode})"))
        if not ok and r.stdout: print("      ", r.stdout.strip()[-200:])
        if ok and failed:
            msg = (r.stderr or r.stdout).strip().split("\n")[-1]
            print(f"          -> {msg[:100]}")
        return ok

fails=[]
def t(name, ok):
    if not ok: fails.append(name)

# make a fake steered cell from A0's acquisition half
def make_steered(alpha=4.0, fires=873):
    p = copy.deepcopy(base)
    acq = [r for r in p["runs"] if r["block"]=="acquisition"]
    for r in acq:
        r["alpha"]=alpha; r["steered"]=True
        r["attestation"]["steering_fires"]=fires
        r["attestation"]["expected_fires"]=fires
    p["runs"]=acq
    p["config"]["block"]="acquisition"
    p["config"]["interface_tag"]=("pv11_p11_pv11-states-v1_pv10-strict-v2_k4"
                                  "_H20_s8699154dee9b_blkacquisition")
    return p

print("\n[control] unmutated A0 alone must SUCCEED")
t("control a0", run(base, expect_fail=False, name="a0 baseline runs"))
print("\n[control] a0 + well-formed steered cell must SUCCEED")
t("control pair", run(base, make_steered(), expect_fail=False, name="paired run"))

print("\n[guards]")
# 1. commitment contamination
p=copy.deepcopy(base); s=make_steered()
s["runs"]=s["runs"][:79]+[r for r in base["runs"] if r["block"]=="commitment"][:1]
for r in s["runs"]:
    if r["block"]=="commitment": r["alpha"]=4.0
t("commitment contamination", run(p, s, name="commitment state rejected"))

# 2. short cell
s=make_steered(); s["runs"]=s["runs"][:79]
t("short cell", run(copy.deepcopy(base), s, name="79 states rejected"))

# 3. duplicate uid
s=make_steered(); s["runs"][1]=copy.deepcopy(s["runs"][0])
t("duplicate uid", run(copy.deepcopy(base), s, name="duplicate uid rejected"))

# 4. wrong alpha
s=make_steered(alpha=6.0)
t("wrong alpha", run(copy.deepcopy(base), s, name="alpha=6 in --ap4 rejected"))

# 5. steered but zero fires
s=make_steered(fires=0)
t("zero fires", run(copy.deepcopy(base), s, name="steered w/ 0 fires rejected"))

# 6. None fire count
s=make_steered(); s["runs"][0]["attestation"]["steering_fires"]=None
t("none fires", run(copy.deepcopy(base), s, name="None fire count rejected"))

# 7. fires != expected
s=make_steered(); s["runs"][0]["attestation"]["steering_fires"]=800
t("fires mismatch", run(copy.deepcopy(base), s, name="fires!=expected rejected"))

# 8. state field differs (opening_counts)
s=make_steered(); s["runs"][0]["opening_counts"]["A"]=[99,99]
t("state drift", run(copy.deepcopy(base), s, name="opening_counts drift rejected"))

# 9. tape_key differs
s=make_steered(); s["runs"][0]["tape_key"]="TAMPERED"
t("tape drift", run(copy.deepcopy(base), s, name="tape_key drift rejected"))

# 10. display_order differs
s=make_steered(); s["runs"][0]["display_order"]=["D","C","B","A"]
t("display drift", run(copy.deepcopy(base), s, name="display_order drift rejected"))

# 11. mask fingerprint differs
s=make_steered(); s["config"]["model_config"]="DIFFERENT"
t("model cfg", run(copy.deepcopy(base), s, name="model_config drift rejected"))

# 12. bank hash differs
s=make_steered(); s["state_bank_canonical_sha256"]="deadbeef"
t("bank hash", run(copy.deepcopy(base), s, name="bank hash drift rejected"))

# 13. tag/scope mismatch: acquisition block wearing a full-bank tag
s=make_steered(); s["config"]["interface_tag"]=base["config"]["interface_tag"]
t("tag scope", run(copy.deepcopy(base), s, name="acq block w/ full tag rejected"))

# 14. a0 alpha!=0 must fire the alpha=0-registers-no-hook rule
p=copy.deepcopy(base)
for r in p["runs"]:
    if r["block"]=="acquisition": r["attestation"]["steering_fires"]=5; r["attestation"]["expected_fires"]=5
t("a0 fired", run(p, name="alpha=0 with nonzero fires rejected"))

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
