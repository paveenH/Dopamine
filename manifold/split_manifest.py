#!/usr/bin/env python3
"""
Manifold Plan section 2 -- the FROZEN question-level split.

Runs LOCALLY with python3.10 (python3 on this box has no numpy). Writes one
JSON that every later stage consumes; it reads no hidden states and needs no
server access.

WHAT THIS FIXES, AND WHY IT IS ITS OWN ARTIFACT
-----------------------------------------------
The manifold is FIT on alpha=0 and then EVALUATED on steered cells. If the fit
saw every question, the reconstruction error of a held-out trajectory is not
held out at all, and NRE stops being evidence about anything. So the split has
to be decided once, before any fitting, and stored where both the server-side
fit and the local analysis read the same copy.

  train 60%  -- fit the alpha=0 PCA basis (per layer, per phase)
  val   20%  -- choose k (the PCA dimension), and only from alpha=0
  test  20%  -- the NRE denominator, and every reported number

THE SPLIT IS BY QUESTION, AND THAT IS THE WHOLE POINT
------------------------------------------------------
Tokens within one question are strongly correlated, so a token-level split
leaks: the "held-out" tokens sit in the same trajectory the basis was fit on.
Effective sample size is counted in QUESTIONS (n=300), never tokens. This is
the same rule the plan sets for choosing k.

ONE SPLIT, SHARED BY EVERY CELL
--------------------------------
All alpha cells, CoT cells and role cells use THIS split -- question 7 is in
`test` for alpha=0 and for alpha=-6 and for CoT alike. If each cell drew its
own split, a cross-alpha comparison would compare different question subsets
and the dose effect would be confounded with question difficulty. That is the
same failure that made PV10-A's cross-alpha accuracy uninterpretable (each
cell dropped a different seed set), so it is enforced by construction here:
the manifest stores question indices, not per-cell anything.

DETERMINISM
-----------
Assignment is `sha256(f"{salt}:{question_idx}")` -> uniform in [0,1) ->
bucket. Not `random.shuffle`, not `hash()`:

  * `hash()` on str is salted per process in Python 3, so it gives a DIFFERENT
    split on every run. Silent, and catastrophic here.
  * an RNG-based shuffle reproduces only if the RNG implementation never
    changes, and it makes each index's assignment depend on every other index,
    so extending to n=500 later would re-shuffle the existing 300.

With a per-index hash, question 7's bucket depends on question 7 alone. Adding
questions later leaves every existing assignment untouched.

The 60/20/20 boundaries are on the hash value, so the realised counts are
close to but not exactly 180/60/60. That is correct and must not be "fixed" by
sorting-and-slicing to exact counts: exact counts would make each assignment
depend on the whole set again, reintroducing the extension problem above. The
realised counts are recorded in the manifest.

QUESTION IDENTITY
-----------------
The H5 carry a per-sample `question_idx`; the lightweight signal JSON carries
NONE and is purely positional. check_hs_llama.py already refuses to compare
the two unless the H5 `question_idx` is the identity permutation AND the
question TEXT matches, so for the four accepted primary cells
`question_idx == row position` is VERIFIED, not assumed. This manifest keys on
that index, and stores a sha256 of the question text list so a consumer can
detect a different benchmark ordering rather than silently mis-joining.

LOCATION
  This lives in the main repo, not in RoleAnswer/, because manifold_fit.py
  REQUIRES the manifest and that script is versioned server-side code. A
  required input of versioned code must not sit in an unversioned tree.

USAGE
  python3.10 manifold/split_manifest.py --write   # create (refuses to overwrite)
  python3.10 manifold/split_manifest.py --check   # re-derive and diff
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The manifest and its generator live IN THIS REPO: manifold_fit.py refuses to
# run without the manifest, so a required input of versioned server code cannot
# itself sit in an unversioned tree. It is 4 KB of frozen definition, not a data
# product -- the plan's "not in git" list covers H5, PCA caches, bootstrap
# intermediates and figures, none of which this is.
#
# The reference signal JSON is a different matter: it IS a data product and
# correctly stays in the (unversioned) RoleAnswer tree. It is used only for an
# identity digest, so its absence downgrades one optional check and never
# changes the split. Override with --roleanswer when the tree is elsewhere.
DEFAULT_ROLEANSWER = os.path.expanduser("~/Documents/RSNResult/RoleAnswer")

# The salt is part of the frozen definition. Changing it re-rolls every
# assignment, which invalidates every stored manifold result -- so it is a
# constant here rather than a CLI flag.
SALT = "rsn-manifold-pilot-v1"

N_QUESTIONS = 300
FRAC_TRAIN = 0.60
FRAC_VAL = 0.20  # test is the remainder

# Reference cell for the question-text digest. alpha=0 No-CoT is the cell the
# manifold basis is fit on, so it is the natural identity anchor.
REF_SIGNAL_REL = os.path.join(
    "llama3", "dopamine", "signal",
    "dopamine_signal_gsm8k_8B_nocot_ema0.95_L11-20.json")

MANIFEST = os.path.join(HERE, "split_manifest.json")


def bucket(qidx: int) -> str:
    """Deterministic per-question bucket. Depends on qidx alone."""
    h = hashlib.sha256(f"{SALT}:{qidx}".encode()).digest()
    # First 8 bytes as a big-endian unsigned int, scaled to [0,1).
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    if u < FRAC_TRAIN:
        return "train"
    if u < FRAC_TRAIN + FRAC_VAL:
        return "val"
    return "test"


def build(n: int):
    split = {"train": [], "val": [], "test": []}
    for q in range(n):
        split[bucket(q)].append(q)
    return split


def question_digest(path: str):
    """sha256 over the ordered question texts, so a consumer can tell it is
    joining against the same benchmark ordering. Returns None if the reference
    cell is not present locally -- the split itself does not depend on it."""
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)["data"]
    qs = [r["question"] for r in data]
    h = hashlib.sha256("\n".join(qs).encode()).hexdigest()
    return h, len(qs)


def make_manifest(roleanswer=DEFAULT_ROLEANSWER):
    split = build(N_QUESTIONS)
    digest, n_seen = question_digest(os.path.join(roleanswer, REF_SIGNAL_REL))
    return {
        "version": "manifold-split-v1",
        "salt": SALT,
        "n_questions": N_QUESTIONS,
        "fractions": {"train": FRAC_TRAIN, "val": FRAC_VAL,
                      "test": round(1.0 - FRAC_TRAIN - FRAC_VAL, 10)},
        "method": ("bucket(q) = sha256(f'{salt}:{q}')[:8] as big-endian uint "
                   "/ 2**64, thresholded at the cumulative fractions. Per-index, "
                   "so assignments are stable under extending n."),
        "counts": {k: len(v) for k, v in split.items()},
        "split": split,
        "question_text_sha256": digest,
        "question_text_n": n_seen,
        "question_text_source": REF_SIGNAL_REL if digest else None,
        "notes": [
            "Split is by QUESTION. Token-level splitting leaks: tokens within "
            "one question are strongly correlated, so held-out tokens would sit "
            "in a trajectory the basis was fit on. Effective n is 300 questions.",
            "ALL cells (every alpha, CoT, role) share this split. Per-cell "
            "splits would confound the dose effect with question difficulty.",
            "Realised counts are not exactly 180/60/60 because thresholds are "
            "on the hash value. Do NOT re-balance by sorting and slicing -- that "
            "would make each assignment depend on the whole set again.",
            "train fits the alpha=0 PCA basis; val chooses k (alpha=0 only); "
            "test carries the NRE denominator and every reported number.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true",
                   help="create the manifest (refuses to overwrite)")
    g.add_argument("--check", action="store_true",
                   help="re-derive and diff against the stored manifest")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing manifest")
    ap.add_argument("--roleanswer", default=DEFAULT_ROLEANSWER,
                    help="offline data tree holding the reference signal JSON "
                         "(identity digest only; absence never changes the split)")
    args = ap.parse_args()

    fresh = make_manifest(args.roleanswer)

    if args.write:
        if os.path.exists(MANIFEST) and not args.force:
            print(f"[FAIL] {MANIFEST} exists. The split is frozen once written "
                  f"-- overwriting it invalidates every stored manifold result. "
                  f"Pass --force only if you intend exactly that.")
            return 1
        with open(MANIFEST, "w") as f:
            json.dump(fresh, f, indent=2)
            f.write("\n")
        print(f"[ok] wrote {MANIFEST}")
        print(f"     counts: {fresh['counts']}")
        print(f"     question_text_sha256: {fresh['question_text_sha256']}")
        return 0

    # --check
    if not os.path.exists(MANIFEST):
        print(f"[FAIL] {MANIFEST} not found; run --write first")
        return 1
    with open(MANIFEST) as f:
        stored = json.load(f)

    bad = []
    for key in ("version", "salt", "n_questions", "fractions", "counts", "split"):
        if stored.get(key) != fresh.get(key):
            bad.append(key)

    # The text digest is checked separately: a mismatch means the benchmark
    # ordering changed, which is a DIFFERENT and more serious problem than a
    # split-derivation mismatch, and it can legitimately be absent locally.
    if fresh["question_text_sha256"] is None:
        print(f"[!] reference signal JSON not found under {args.roleanswer}; "
              f"question-text digest NOT verified this run (the split itself "
              f"does not depend on it)")
    elif stored.get("question_text_sha256") != fresh["question_text_sha256"]:
        bad.append("question_text_sha256 (BENCHMARK ORDERING CHANGED -- any "
                   "join keyed on question_idx is now suspect)")

    if bad:
        print(f"[FAIL] stored manifest does not reproduce: {bad}")
        return 1
    print(f"[ok] manifest reproduces: {stored['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
