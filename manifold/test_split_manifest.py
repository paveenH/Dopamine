#!/usr/bin/env python3
"""
Mutation suite for split_manifest.py. python3.10, no GPU, no server, ~1s.

Every guard is mutation-tested: the test BUILDS the specific defect and asserts
rejection, because a guard never shown to fire on a real defect is not a guard.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SM = os.path.join(HERE, "split_manifest.py")

spec = importlib.util.spec_from_file_location("sm", SM)
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def run(args, cwd=None):
    return subprocess.run([sys.executable, SM] + args, cwd=cwd,
                          capture_output=True, text=True)


print("[1] split structure")
split = sm.build(sm.N_QUESTIONS)
allq = sorted(split["train"] + split["val"] + split["test"])
check(allq == list(range(sm.N_QUESTIONS)),
      "every question appears exactly once across the three buckets")
check(len(set(split["train"]) & set(split["val"])) == 0
      and len(set(split["train"]) & set(split["test"])) == 0
      and len(set(split["val"]) & set(split["test"])) == 0,
      "buckets are disjoint")
c = {k: len(v) for k, v in split.items()}
check(all(v > 0 for v in c.values()), f"no empty bucket ({c})")
# Loose band: thresholds are on the hash value, so counts vary around the
# nominal 180/60/60. A tight equality here would be wrong (see the module
# docstring on why re-balancing is forbidden), but a wild deviation would mean
# the thresholding is broken.
check(140 <= c["train"] <= 220 and 30 <= c["val"] <= 90 and 30 <= c["test"] <= 90,
      f"counts near nominal 180/60/60 without being forced to it ({c})")

print("\n[2] determinism -- the property hash() would silently break")
check(sm.build(sm.N_QUESTIONS) == split, "same-process rebuild is identical")
r1 = run(["--check"])
r2 = run(["--check"])
check(r1.returncode == 0 and r2.returncode == 0 and r1.stdout == r2.stdout,
      "separate PROCESSES agree (a salted hash() would differ here)")
# Directly demonstrate the failure mode being avoided.
p = subprocess.run([sys.executable, "-c",
                    "print(hash('rsn-manifold-pilot-v1:7'))"],
                   capture_output=True, text=True).stdout
q = subprocess.run([sys.executable, "-c",
                    "print(hash('rsn-manifold-pilot-v1:7'))"],
                   capture_output=True, text=True).stdout
check(p != q, "control: builtin hash() IS process-salted, so it was right to "
              "use sha256 (if this ever fails, PYTHONHASHSEED is pinned and "
              "the control is void, not the design)")

print("\n[3] per-index stability -- extension must not re-roll existing questions")
small = sm.build(100)
for q in range(100):
    b_small = [k for k, v in small.items() if q in v][0]
    b_full = [k for k, v in split.items() if q in v][0]
    if b_small != b_full:
        check(False, f"question {q} moved bucket when n grew 100->300")
        break
else:
    check(True, "all 100 questions keep their bucket when n grows to 300")

print("\n[4] salt is load-bearing")
orig = sm.SALT
sm.SALT = "different-salt"
other = sm.build(sm.N_QUESTIONS)
sm.SALT = orig
check(other != split, "changing the salt changes the split (so it is frozen "
                      "as a constant, not exposed as a flag)")

print("\n[5] --write refuses to clobber a frozen manifest")
check(os.path.exists(sm.MANIFEST), "manifest exists (written by --write)")
r = run(["--write"])
check(r.returncode != 0 and "frozen" in r.stdout,
      "--write on an existing manifest FAILS with the freeze rationale")
r = run(["--write", "--force"])
check(r.returncode == 0, "--force overrides deliberately")

print("\n[6] --check catches a tampered manifest")
with open(sm.MANIFEST) as f:
    good = json.load(f)

with tempfile.TemporaryDirectory() as td:
    backup = os.path.join(td, "backup.json")
    with open(backup, "w") as f:
        json.dump(good, f)

    def restore():
        with open(sm.MANIFEST, "w") as f:
            json.dump(good, f, indent=2)

    mutations = [
        ("moved one question between buckets",
         lambda m: (m["split"]["test"].append(m["split"]["train"].pop()), m)[1]),
        ("changed the recorded salt",
         lambda m: {**m, "salt": "tampered"}),
        ("changed n_questions",
         lambda m: {**m, "n_questions": 250}),
        ("changed the fractions",
         lambda m: {**m, "fractions": {"train": 0.5, "val": 0.25, "test": 0.25}}),
        ("changed a recorded count without changing the split",
         lambda m: {**m, "counts": {**m["counts"], "train": 999}}),
        ("changed the question-text digest (benchmark reordered)",
         lambda m: {**m, "question_text_sha256": "0" * 64}),
    ]
    for name, mut in mutations:
        m = json.loads(json.dumps(good))
        with open(sm.MANIFEST, "w") as f:
            json.dump(mut(m), f, indent=2)
        r = run(["--check"])
        check(r.returncode != 0, f"--check rejects: {name}")
        restore()

    r = run(["--check"])
    check(r.returncode == 0, "restored manifest passes again")

print("\n[7] question-text digest is a real join guard")
ref = os.path.join(sm.DEFAULT_ROLEANSWER, sm.REF_SIGNAL_REL)
d, n = sm.question_digest(ref)
if d is None:
    print("  skip (reference signal JSON not present locally)")
else:
    check(n == sm.N_QUESTIONS,
          f"reference cell has {sm.N_QUESTIONS} questions (got {n})")
    with open(ref) as f:
        qs = [r["question"] for r in json.load(f)["data"]]
    swapped = qs[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    h2 = hashlib.sha256("\n".join(swapped).encode()).hexdigest()
    check(h2 != d, "digest changes when two questions are REORDERED (order "
                   "matters -- a set-hash would miss this, and the join is "
                   "positional)")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
