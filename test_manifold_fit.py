#!/usr/bin/env python3
"""
Mutation suite for manifold_fit.py. Synthetic H5, no GPU, no server, no real
model. python3.10 locally / conda `python` on the server. ~3s.

Every guard is mutation-tested: the test BUILDS the defect and asserts
rejection. A guard never shown to fire on a real defect is not a guard.

The tokenizer-dependent paths (commit location) are exercised with a FAKE
tokenizer whose contract matches the real one closely enough for the slicing
logic: ids are character indices, so decode(ids[:i+1]) has length i+1 and
char_to_step becomes the identity. That makes the expected window arithmetic
checkable by hand, which is the point -- a real tokenizer would make the
assertions depend on Llama's BPE rather than on the code under test.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mf",
                                              os.path.join(HERE, "manifold_fit.py"))
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


class FakeTok:
    """ids == char indices, so char_to_step is the identity."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}

    def decode(self, ids):
        return "x" * len(ids)


TOK = FakeTok()
DIM, N_MID = 8, 9        # band width 11..20 -> 9 middle layers


def make_h5(path, n=12, T=60, commit_at=30, dim=DIM, n_mid=N_MID,
            no_commit_idx=(), seed=0, steer_alpha=0.0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["stored_layer_indices"] = list(range(10, 10 + n_mid)) + [31]
        meta.attrs["steer_alpha"] = float(steer_alpha)
        grp = f.create_group("samples")
        for i in range(n):
            g = grp.create_group(f"{i:04d}")
            g.create_dataset("prefill_hs",
                             data=rng.normal(size=(5, n_mid + 1, dim)).astype(np.float16))
            g.create_dataset("decode_hs",
                             data=rng.normal(size=(T, n_mid + 1, dim)).astype(np.float16))
            if i in no_commit_idx:
                text = "a" * T            # no #### and no digit -> no commit
            else:
                text = "a" * commit_at + "####" + "b" * (T - commit_at - 4)
            g.attrs["generated"] = text
            g.attrs["question"] = f"question number {i}"
            g.attrs["question_idx"] = i
            g.attrs["correct"] = bool(i % 2)
    return path


print("[1] phase windows -- the frozen option-B definition")
check(mf.WINDOW == 20, "window is 20 tokens per side")
check(mf.phase_rows(60, 30, "pre_commit") == (10, 30),
      "pre_commit is [c-20, c) for a late commit")
check(mf.phase_rows(60, 30, "post_commit") == (30, 50),
      "post_commit is [c, c+20)")
# The load-bearing one: a fast commit is TRUNCATED, never dropped.
check(mf.phase_rows(60, 5, "pre_commit") == (0, 5),
      "commit at token 5 KEEPS the sample on its 5 actual tokens "
      "(dropping c<20 would systematically delete fast commitment)")
check(mf.phase_rows(60, 0, "pre_commit") is None,
      "commit at token 0 has no pre-commit rows -> absent, not a crash")
check(mf.phase_rows(60, 55, "post_commit") == (55, 60),
      "post_commit truncates at the end of decode")
check(mf.phase_rows(60, None, "pre_commit") is None
      and mf.phase_rows(60, None, "post_commit") is None,
      "a no-commit sample is ABSENT from both aligned phases")
check(mf.phase_rows(60, None, "decode_all") == (0, 60),
      "a no-commit sample still enters decode_all (option-A sensitivity)")

print("\n[2] commit locator matches the Llama convention")
check(mf.commit_char("aaa####bbb") == 3, "finds ####")
check(mf.commit_char("the answer is 42") > 0,
      "falls back to an answer-candidate when #### is absent")
check(mf.commit_char("no marker here") == -1, "returns -1 when neither exists")
check(mf.char_to_step(TOK, "abcdef", -1) is None,
      "char_to_step(-1) is None, so 'no commit' propagates as None")

print("\n[2b] commit locator is BYTE-EQUIVALENT to the frozen Llama one")
# The locator is deliberately COPIED from analyze_wrong_right_commit.py, not
# improved. Its `>=` boundary can land one token early in principle, but every
# published Llama commit-aligned number uses that definition, so changing it
# here would silently redefine the event the manifold phases are built on. The
# test therefore pins EQUIVALENCE, not correctness.
_FROZEN = os.path.expanduser(
    "~/Documents/RSNResult/RoleAnswer/analyze_wrong_right_commit.py")
if not os.path.exists(_FROZEN):
    print("  skip (frozen reference not present locally)")
else:
    import re as _re
    src = open(_FROZEN).read()
    def _grab(name):
        m = _re.search(rf"^{name} = (re\.compile\(.*?\))$", src, _re.M | _re.S)
        return m.group(1) if m else None
    ref_hash, ref_cand = _grab("HASH"), _grab("CAND")
    check(ref_hash is not None and eval(ref_hash, {'re': _re}).pattern == mf.HASH.pattern,
          "HASH pattern is identical to the frozen implementation")
    check(ref_cand is not None and eval(ref_cand, {'re': _re}).pattern == mf.CAND.pattern
          and eval(ref_cand, {'re': _re}).flags == mf.CAND.flags,
          "CAND pattern AND flags are identical to the frozen implementation")
    # Behavioural equivalence on cases that separate the two branches.
    ns = {}
    exec(compile(_re.search(r"def commit_char.*?return m\.start\(1\) if m else -1",
                            src, _re.S).group(0), "<frozen>", "exec"),
         {"HASH": mf.HASH, "CAND": mf.CAND}, ns)
    cases = ["aaa####bbb", "the answer is 42", "no marker here",
             "12 apples\n#### 7", "  5\nthe answer is: 9", ""]
    check(all(ns["commit_char"](c) == mf.commit_char(c) for c in cases),
          f"commit_char agrees with the frozen version on {len(cases)} cases "
          f"spanning both branches and the no-match path")

print("\n[3] per-question weighting -- long trajectories must not dominate")
a = mf.Accum(DIM)
rng = np.random.default_rng(1)
long_q = rng.normal(size=(20, DIM)) + 10.0     # 20 rows, far from origin
short_q = rng.normal(size=(2, DIM)) - 10.0     # 2 rows, opposite side
a.add(long_q)
a.add(short_q)
# Enough extra questions that k is not rank-capped below the requested 4.
for j in range(6):
    a.add(rng.normal(size=(3 + j, DIM)))
b = a.finish(4)
mu = b["mu"]
# Equal weight => mu sits near the midpoint (~0), not dragged to the long side.
check(abs(float(mu.mean())) < 3.0,
      f"mean is near the questions' midpoint (|mu.mean()|={abs(float(mu.mean())):.2f}), "
      f"not dragged toward the 20-row question")
# Mutation: row-weighted accumulation instead of question-weighted.
a2 = mf.Accum(DIM)
a2.w = 2.0
a2.s = (long_q.sum(axis=0) + short_q.sum(axis=0)).astype(np.float64)
row_mu = a2.s / 22.0
check(abs(float(row_mu.mean())) > 5.0,
      f"control: ROW-weighted mean IS dragged to the long question "
      f"(|{float(row_mu.mean()):.2f}|), which is the bias the design avoids")
check(b["n_questions"] == 8, "n_questions counts QUESTIONS, not rows")

print("\n[4] basis shape and centering")
check(b["components"].shape == (4, DIM), "components is (k, dim)")
check(b["explained"].shape == (4,), "explained has k entries")
check(np.all(b["explained"] >= 0), "eigenvalues are non-negative")
ortho = b["components"] @ b["components"].T
check(np.allclose(ortho, np.eye(4), atol=1e-4), "components are orthonormal")
check(mf.Accum(DIM).finish(4) is None,
      "a basis with <2 questions is None rather than a rank-0 fit")
# Rank is set by ROWS and the numerical spectrum, NOT by question count: two
# questions of 5 tokens each span up to 9 dimensions, so an "n_questions - 1"
# cap would be wrong (and was -- it silently discarded real directions). What
# must hold is that no basis exceeds the rank actually present in the data.
rank = mf.Accum(DIM)
rank.add(rng.normal(size=(5, DIM)))
rank.add(rng.normal(size=(5, DIM)))
rb = rank.finish(4)
check(rb["components"].shape[0] == 4,
      "2 questions x 5 tokens DO support 4 directions (rank comes from rows, "
      "not from question count)")
thin = mf.Accum(DIM)
thin.add(rng.normal(size=(1, DIM)))
thin.add(rng.normal(size=(1, DIM)))
tb = thin.finish(4)
check(tb["components"].shape[0] == 1,
      "2 questions x 1 token support only 1 direction, so a thin phase cannot "
      "fabricate 4 directions from 2 rows")
check(np.allclose(tb["components"] @ tb["components"].T, np.eye(1), atol=1e-6),
      "the rank-limited basis is still orthonormal")

print("\n[5] end to end on a synthetic tree")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    h5d, outd = td / "h5", td / "out"
    h5d.mkdir()
    base = h5d / mf.cell_stem("gsm8k", "8B", "nocot", 11, 20)
    make_h5(base, n=12, no_commit_idx=(11,), seed=0)
    make_h5(h5d / mf.cell_stem("gsm8k", "8B", "nocot_a6", 11, 20),
            n=12, commit_at=10, seed=1, steer_alpha=6.0)

    split = {"train": list(range(8)), "val": [8, 9], "test": [10, 11]}
    man = td / "split.json"
    man.write_text(json.dumps({
        "version": "test-v1", "salt": "t", "counts":
        {k: len(v) for k, v in split.items()}, "split": split,
        "question_text_sha256": None}))

    basis, stored, n_mid = mf.fit_basis(base, split, 11, 20, 4, TOK, verbose=False)
    check(n_mid == N_MID, f"n_middle is the half-open band width ({N_MID})")
    check(all(ph in {ph2 for (_, ph2) in basis} for ph in mf.PHASES),
          "a basis exists for every phase")
    check(len({li for (li, _) in basis}) == N_MID,
          "a basis exists for every middle layer")
    b0 = basis[(0, "prefill")]
    check(b0["n_questions"] == 8,
          f"prefill basis saw exactly the 8 TRAIN questions (got {b0['n_questions']})")
    check(all(v["n_questions"] <= 8 for v in basis.values()),
          "NO basis saw more than the 8 train questions (val/test never leak "
          "into the fit)")

    recs, n_commit, n_total = mf.project_cell(base, basis, split, n_mid,
                                              TOK, verbose=False)
    check(n_total == 12, "projection covers every question, not just train")
    check(n_commit == 11 and abs(n_commit / n_total - 11 / 12) < 1e-9,
          f"commit coverage is REPORTED as {n_commit}/{n_total}, not gated")
    by_q = {r["question_idx"]: r for r in recs}
    check(by_q[11]["commit_step"] is None,
          "the no-commit question has commit_step None")
    check("pre_commit" not in by_q[11]["phases"]
          and "post_commit" not in by_q[11]["phases"],
          "the no-commit question is ABSENT from the aligned phases")
    check("decode_all" in by_q[11]["phases"],
          "the no-commit question IS present in decode_all")
    check({r["split"] for r in recs} == {"train", "val", "test"},
          "every record is tagged with its split")
    ph = by_q[0]["phases"]["pre_commit"]["0"]
    check(len(ph["coord"]) == 4 and ph["re"] >= 0 and ph["energy"] >= 0,
          "each phase-layer cell carries coord (k), re and energy")
    check(ph["re"] <= ph["energy"] + 1e-3,
          "reconstruction error never exceeds total centered energy "
          "(the NRE numerator is a residual, so this must hold)")


    print("\n[7] BLOCKER 1 -- per-token coordinates survive the export")
    ph_pre = by_q[0]["phases"]["pre_commit"]["0"]
    check("coord_t" in ph_pre,
          "pre_commit exports coord_t (a phase MEAN destroys token order, and "
          "speed/curvature/turning are defined on the ordered sequence)")
    check(len(ph_pre["coord_t"]) == ph_pre["n_rows"],
          f"coord_t has one row per token ({ph_pre['n_rows']})")
    check(all(len(r) == len(ph_pre["coord"]) for r in ph_pre["coord_t"]),
          "each coord_t row has k entries")
    recon_mean = np.mean(np.array(ph_pre["coord_t"]), axis=0)
    check(np.allclose(recon_mean, ph_pre["coord"], atol=1e-3),
          "coord is exactly the mean of coord_t (they are consistent, so the "
          "summary never disagrees with the trajectory)")
    check("coord_t" in by_q[0]["phases"]["post_commit"]["0"],
          "post_commit also exports coord_t")
    check("coord_t" not in by_q[0]["phases"]["decode_all"]["0"],
          "decode_all does NOT (unbounded rows), so option-A is a LEVEL "
          "sensitivity, not a trajectory-shape one")

    print("\n[8] BLOCKER 2 -- k can be chosen on validation")
    rk = ph_pre["re_by_k"]
    check(len(rk) == len(ph_pre["coord"]),
          f"re_by_k has one entry per k (1..{len(rk)})")
    check(all(rk[i] >= rk[i + 1] - 1e-6 for i in range(len(rk) - 1)),
          "re_by_k is non-increasing in k (more dimensions never reconstruct "
          "worse)")
    check(abs(rk[-1] - ph_pre["re"]) < 1e-6,
          "re_by_k[-1] equals the reported re at k_max (same quantity)")
    check(rk[0] <= ph_pre["energy"] + 1e-6,
          "even k=1 residual is bounded by the total centered energy")

    print("\n[9] BLOCKER 3 -- the manifest is VALIDATED, not merely read")
    good_split = {"train": list(range(8)), "val": [8, 9], "test": [10, 11]}
    check(mf.validate_manifest({"split": good_split, "counts":
                                {"train": 8, "val": 2, "test": 2}}, 12) == [],
          "a well-formed manifest passes")
    muts = [
        ("overlapping buckets",
         {"train": list(range(9)), "val": [8, 9], "test": [10, 11]}),
        ("a question in no bucket",
         {"train": list(range(7)), "val": [8, 9], "test": [10, 11]}),
        ("an index outside 0..n-1",
         {"train": list(range(8)), "val": [8, 9], "test": [10, 99]}),
        ("an empty train bucket",
         {"train": [], "val": list(range(8)), "test": [8, 9, 10, 11]}),
    ]
    for name, sp in muts:
        check(mf.validate_manifest({"split": sp}, 12) != [],
              f"validate_manifest rejects: {name}")
    check(mf.validate_manifest({"split": good_split, "counts":
                                {"train": 99, "val": 2, "test": 2}}, 12) != [],
          "validate_manifest rejects: counts disagreeing with the split")
    check(mf.validate_manifest({"split": {"train": [1], "val": [2]}}, 3) != [],
          "validate_manifest rejects: a missing bucket")

    print("\n[10] BLOCKER 3 -- the H5 digest guard actually fires")
    d1, n1 = mf.h5_question_digest(base)
    check(d1 is not None and n1 == 12, "digest computed over the H5 questions")
    other = h5d / "hs_gsm8k_8B_other_L11-20.h5"
    make_h5(other, n=12, seed=5)
    d2, _ = mf.h5_question_digest(other)
    check(d1 == d2, "identical question text gives an identical digest "
                    "(the digest tracks QUESTIONS, not hidden states)")
    reordered = h5d / "hs_gsm8k_8B_reorder_L11-20.h5"
    make_h5(reordered, n=12, seed=6)
    with h5py.File(reordered, "a") as f:
        g = f["samples"]
        g["0000"].attrs["question"], g["0001"].attrs["question"] = (
            g["0001"].attrs["question"], g["0000"].attrs["question"])
    d3, _ = mf.h5_question_digest(reordered)
    check(d3 != d1, "swapping two questions CHANGES the digest, so a manifest "
                    "built on another ordering cannot be joined silently")
    clipped = h5d / "hs_gsm8k_8B_clip_L11-20.h5"
    make_h5(clipped, n=4, seed=7)
    with h5py.File(clipped, "a") as f:
        f["samples"]["0000"].attrs["question"] = "x" * 2500
    dc, _ = mf.h5_question_digest(clipped)
    check(dc == "TRUNCATED",
          "a question hitting the tracker's 2000-char clip reports TRUNCATED "
          "rather than a mismatch that would be an artifact")

    print("\n[11] basis_meta records what the comments promise")
    # The reduction and the trajectory-phase set were described in comments but
    # not emitted, so a consumer had to infer them from whether coord_t
    # happened to be present. Fields, not folklore.
    import json as _json
    meta_rows = {f"{li}|{ph}": {
        "n_questions": b["n_questions"], "n_rows": b["n_rows"],
        "reduced_to_mean": b["reduced_to_mean"],
        "trajectory_exported": ph in mf.TRAJECTORY_PHASES,
        "k": int(b["components"].shape[0]),
    } for (li, ph), b in basis.items()}
    check(all("n_rows" in v and "reduced_to_mean" in v for v in meta_rows.values()),
          "every basis carries n_rows and reduced_to_mean")
    check(meta_rows["0|decode_all"]["reduced_to_mean"] is True,
          "decode_all is marked reduced_to_mean")
    check(meta_rows["0|pre_commit"]["reduced_to_mean"] is False,
          "pre_commit is NOT reduced (it keeps per-token rows)")
    check(meta_rows["0|decode_all"]["n_rows"] == meta_rows["0|decode_all"]["n_questions"],
          "reduced decode_all has exactly one row per question, which is what "
          "makes its spectrum a per-question point cloud")
    check(meta_rows["0|pre_commit"]["n_rows"] > meta_rows["0|pre_commit"]["n_questions"],
          "pre_commit has more rows than questions (its Gram is ~3700x3700 at "
          "full scale, so it is NOT the thin-phase speedup case)")
    check(meta_rows["0|prefill"]["n_rows"] == meta_rows["0|prefill"]["n_questions"],
          "prefill is one row per question by construction (last-prefill only)")
    check(meta_rows["0|decode_all"]["trajectory_exported"] is False
          and meta_rows["0|pre_commit"]["trajectory_exported"] is True,
          "trajectory_exported agrees with which phases actually carry coord_t")

    print("\n[6] CLI fails closed")
    def run(extra):
        import subprocess
        return subprocess.run(
            [sys.executable, os.path.join(HERE, "manifold_fit.py"),
             "--h5_dir", str(h5d), "--out_dir", str(outd),
             "--model_dir", "/nonexistent", "--n_questions", "12"] + extra,
            capture_output=True, text=True)

    r = run(["--cells", "nocot"])
    check(r.returncode != 0 and "split_manifest" in (r.stderr + r.stdout),
          "--split_manifest is REQUIRED (no silent fit on all questions)")

    r = run(["--split_manifest", str(man), "--cells", "nocot,does_not_exist"])
    check(r.returncode != 0 and "missing H5" in r.stdout,
          "a missing cell fails BEFORE any fitting work is done")

    outd.mkdir(exist_ok=True)
    (outd / "manifold_nocot.json").write_text("{}")
    r = run(["--split_manifest", str(man), "--cells", "nocot"])
    check(r.returncode != 0 and "allow_overwrite" in r.stdout,
          "an existing output is not clobbered without --allow_overwrite")
    (outd / "manifold_nocot.json").unlink()

    # BLOCKER 4: the basis is an output too.
    for extra in ("basis.npz", "basis_meta.json"):
        (outd / extra).write_text("stale")
        r = run(["--split_manifest", str(man), "--cells", "nocot"])
        check(r.returncode != 0 and extra in r.stdout,
              f"{extra} is guarded too (a stale basis beside fresh coordinates "
              f"still loads and still looks reasonable)")
        (outd / extra).unlink()

    # BLOCKER 3 at the CLI: a structurally broken manifest must not run.
    badman = td / "bad_split.json"
    badman.write_text(json.dumps({"version": "x", "salt": "t",
                                  "counts": {"train": 9, "val": 2, "test": 2},
                                  "split": {"train": list(range(9)),
                                            "val": [8, 9], "test": [10, 11]}}))
    r = run(["--split_manifest", str(badman), "--cells", "nocot"])
    check(r.returncode != 0 and "not usable" in r.stdout,
          "a manifest with overlapping buckets is rejected at the CLI")

    # --base_cell must really be alpha=0.
    r = run(["--split_manifest", str(man), "--base_cell", "nocot_a6",
             "--cells", "nocot_a6"])
    check(r.returncode != 0 and ("steer_alpha" in r.stdout or "not 0" in r.stdout),
          "a steered cell is refused as the basis (fitting the natural "
          "manifold on a steered cell would hide the effect under test)")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
