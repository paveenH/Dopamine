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
            no_commit_idx=(), seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["stored_layer_indices"] = list(range(10, 10 + n_mid)) + [31]
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
rank = mf.Accum(DIM)
rank.add(rng.normal(size=(5, DIM)))
rank.add(rng.normal(size=(5, DIM)))
rb = rank.finish(4)
check(rb["components"].shape[0] == 1,
      "k is rank-capped to n_questions-1 (2 questions support 1 component), "
      "so a small phase cannot fabricate 4 directions from 2 samples")

print("\n[5] end to end on a synthetic tree")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    h5d, outd = td / "h5", td / "out"
    h5d.mkdir()
    base = h5d / mf.cell_stem("gsm8k", "8B", "nocot", 11, 20)
    make_h5(base, n=12, no_commit_idx=(11,), seed=0)
    make_h5(h5d / mf.cell_stem("gsm8k", "8B", "nocot_a6", 11, 20),
            n=12, commit_at=10, seed=1)

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

    print("\n[6] CLI fails closed")
    def run(extra):
        import subprocess
        return subprocess.run(
            [sys.executable, os.path.join(HERE, "manifold_fit.py"),
             "--h5_dir", str(h5d), "--out_dir", str(outd),
             "--model_dir", "/nonexistent"] + extra,
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

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
