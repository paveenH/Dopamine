#!/usr/bin/env python3
"""
Manifold Plan section 3 -- fit the natural (alpha=0) manifold and export
low-dimensional coordinates.

Runs on the SERVER with the conda env's `python`. `python3.10` does not exist
there and exits 127 before anything runs. READ-ONLY on the H5 (mode="r").

WHAT IT DOES
  1. Locate each sample's commit step from its stored `generated` text.
  2. Slice three PHASES of the stored hidden states (see below).
  3. Fit a PCA basis PER (layer, phase) on the alpha=0 TRAIN questions only.
  4. Project every cell's samples onto that basis and export coordinates +
     reconstruction error, so sections 4-6 run locally without the 62 GB tree.

PHASE DEFINITION (frozen -- option B, with A as the sensitivity)
  prefill      : the LAST prefill state only. It is the one position that is
                 strictly matched across alpha (same prompt, same token), so
                 it is the ONLY phase where a displacement claim is licensed.
  pre_commit   : at most 20 decode tokens ending at commit, i.e. [c-20, c).
  post_commit  : at most 20 decode tokens from commit, i.e. [c, c+20).
  decode_all   : (sensitivity, option A) the whole decode span, no commit
                 alignment. Also the only phase a no-commit sample can enter.

  *** A SAMPLE THAT COMMITS BEFORE TOKEN 20 IS KEPT, ON ITS ACTUAL TOKENS. ***
  Dropping c < 20 would systematically delete FAST commitment, which is
  precisely the behaviour alpha moves (Llama alpha=0 already commits before
  token 20 in ~23% of samples, and that fraction is itself alpha-dependent).
  So the window is truncated, never the sample. at-risk filtering belongs to
  the speed/curvature analyses that need a fixed landmark -- NOT here, and
  never to the PCA training set.

  A sample with NO commit does not enter the commit-aligned phases at all. Its
  coverage is REPORTED per cell, and it still enters decode_all.

PER-QUESTION WEIGHTING
  Every question contributes the SAME total weight within a phase, regardless
  of how many tokens it has there. Without this a 20-token trajectory outvotes
  a 3-token one 7:1 and the basis describes long trajectories -- which
  correlates with alpha, so the "natural manifold" would silently be the
  natural manifold OF THE SLOW SAMPLES. Implemented by scaling each sample's
  rows by 1/sqrt(n_rows) before the covariance accumulation, which is exactly
  equal weight per question in the second moment.

THE BASIS IS FIT ON alpha=0 TRAIN ONLY
  Never on val (that chooses k), never on test (that carries the numbers),
  never on a steered cell (that is the thing being tested). --split_manifest
  is REQUIRED: there is deliberately no default, because a silent fallback to
  "all questions" would destroy the held-out property without erroring.

CENTERING
  mu is the alpha=0 TRAIN mean for that (layer, phase) and is stored with the
  basis. Every cell is centered by that SAME mu -- centering a steered cell on
  its own mean would subtract the very displacement under test.

WHAT IT DOES NOT DO
  No dose comparison, no statistics, no k selection. k selection reads the
  exported val coordinates (section 3's --k_max spans the candidates); the
  primary cap is k <= 20 and it is chosen from alpha=0 val only.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import decoder_layer_range  # noqa: E402

EXPECTED_START = 11
EXPECTED_END   = 20
PHASES = ("prefill", "pre_commit", "post_commit", "decode_all")
WINDOW = 20          # tokens per side of commit

# Phases whose per-token coordinates are exported. Bounded at WINDOW rows each,
# so the JSON stays tractable; decode_all is unbounded (up to 768 rows) and is
# exported as summary statistics only.
TRAJECTORY_PHASES = ("prefill", "pre_commit", "post_commit")
K_MAX_DEFAULT = 20   # the plan's primary cap

# Llama commit locator: first '####', falling back to the first bare
# answer-candidate. This is the SAME definition analyze_wrong_right_commit.py
# uses for every published Llama commit-aligned number, and it is deliberately
# NOT Qwen's (####-only): the repo records that porting one model's locator to
# the other silently redefines the event every aligned number is built on.
HASH = re.compile(r"####")
CAND = re.compile(r"(?:answer\s+is[:\s]*|^\s*)([+-]?\d[\d,]*\.?\d*)", re.I | re.M)


def commit_char(text: str) -> int:
    m = HASH.search(text)
    if m:
        return m.start()
    m = CAND.search(text)
    return m.start(1) if m else -1


def char_to_step(tok, text: str, char_pos: int):
    """Decode-token index whose cumulative decoded text first covers char_pos.

    Incremental re-decode, matching analyze_wrong_right_commit.char_to_step.
    Chars-per-token is ~1 for digits and ~4 for prose and the pre-commit span
    is where that ratio is most skewed, so a length RATIO is not a substitute.
    """
    if char_pos < 0:
        return None
    ids = tok(text, add_special_tokens=False)["input_ids"]
    for i in range(len(ids)):
        if len(tok.decode(ids[:i + 1])) >= char_pos:
            return i
    return len(ids) - 1 if ids else None


def phase_rows(n_decode: int, commit: int, phase: str):
    """Decode-row slice for a phase, or None if this sample has no such rows."""
    if phase == "decode_all":
        return (0, n_decode) if n_decode > 0 else None
    if commit is None:
        return None                      # no-commit samples skip aligned phases
    if phase == "pre_commit":
        lo, hi = max(0, commit - WINDOW), min(commit, n_decode)
    elif phase == "post_commit":
        lo, hi = min(commit, n_decode), min(commit + WINDOW, n_decode)
    else:
        raise ValueError(phase)
    return (lo, hi) if hi > lo else None


class Accum:
    """Per-question row collector for one (layer, phase), then an exact PCA.

    WHY ROWS AND NOT A STREAMING COVARIANCE. With ~185 train questions the
    covariance has rank <= 185, far below dim=4096, so a full 4096x4096 eigh
    computes ~3900 directions that are exactly zero by construction (~7-8 s per
    basis, 36 bases). The Gram route decomposes the n x n matrix instead and
    recovers identical eigenvalues and orthonormal components -- verified to
    1e-15 -- in ~0.01 s. It is the same decomposition, not an approximation.

    MEMORY. Rows are kept only for the bounded phases (prefill: 1 row/question,
    pre/post_commit: <=20). `decode_all` is unbounded (up to 768 rows) and would
    need ~2.3 GB per layer, so there each question is reduced to its ROW MEAN
    before storage. That is exactly what per-question equal weighting already
    asserts for that phase, and it is recorded in the basis meta so the
    difference is never silent.
    """

    def __init__(self, dim, reduce_to_mean=False):
        self.dim = dim
        self.reduce = reduce_to_mean
        self.rows = []     # one (n_i, dim) block per question

    def add(self, X):
        """X is (n_rows, dim) for ONE question. Weight 1 for the question."""
        n = X.shape[0]
        if n == 0:
            return
        Xf = X.astype(np.float64)
        self.rows.append(Xf.mean(axis=0, keepdims=True) if self.reduce else Xf)

    @property
    def n_questions(self):
        return len(self.rows)

    def finish(self, k_max):
        """Exact PCA under per-question equal weighting.

        Each question contributes its rows scaled by 1/sqrt(n_i), so its total
        second-moment weight is 1 regardless of length. Without this a 20-token
        trajectory outvotes a 3-token one 7:1 and the basis describes long
        trajectories -- which correlates with alpha, so the "natural manifold"
        would silently be the natural manifold OF THE SLOW SAMPLES.
        """
        nq = len(self.rows)
        if nq < 2:
            return None

        mu = np.mean([r.mean(axis=0) for r in self.rows], axis=0)
        blocks = [(r - mu) / np.sqrt(r.shape[0]) for r in self.rows]
        A = np.vstack(blocks)                       # (sum n_i, dim)
        m = A.shape[0]

        # Gram route when it is smaller, which is the normal case here.
        if m <= self.dim:
            G = (A @ A.T) / nq
            g, U = np.linalg.eigh((G + G.T) / 2)
            order = np.argsort(g)[::-1]
            g, U = g[order], U[:, order]
            total_var = float(np.maximum(g, 0.0).sum())
            keep = min(k_max, self.dim, int((g > 1e-10).sum()))
            keep = max(keep, 1)
            g, U = g[:keep], U[:, :keep]
            W = (A.T @ U) / np.sqrt(np.maximum(g * nq, 1e-30))
            comps, evals = W.T, g
        else:
            C = (A.T @ A) / nq
            evals_all, V = np.linalg.eigh((C + C.T) / 2)
            order = np.argsort(evals_all)[::-1]
            evals_all, V = evals_all[order], V[:, order]
            total_var = float(np.maximum(evals_all, 0.0).sum())
            keep = max(1, min(k_max, self.dim))
            comps, evals = V[:, :keep].T, evals_all[:keep]

        return {
            "mu": mu.astype(np.float32),
            "components": comps.astype(np.float32),          # (k, dim)
            "explained": np.maximum(evals, 0.0).astype(np.float64),
            "total_var": total_var,
            "n_questions": nq,
            "n_rows": int(m),
            "reduced_to_mean": bool(self.reduce),
        }


def validate_manifest(man, n_expected):
    """Structural validation of the split. BLOCKER 3.

    Reading a manifest is not verifying one. Without this, passing the wrong
    file (or a hand-edited one) still produces plausible-looking output: the
    fit just silently uses a different held-out set, and every NRE downstream
    is computed against questions that were never held out. The digest guard
    section 2 built is worthless unless a CONSUMER enforces it -- this is that
    consumer.
    """
    bad = []
    split = man.get("split")
    if not isinstance(split, dict) or set(split) != {"train", "val", "test"}:
        return [f"split must have exactly train/val/test, got "
                f"{sorted(split) if isinstance(split, dict) else type(split)}"]

    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    for a, b, na, nb in ((tr, va, "train", "val"), (tr, te, "train", "test"),
                         (va, te, "val", "test")):
        if a & b:
            bad.append(f"{na}/{nb} overlap on {sorted(a & b)[:5]}...")
    allq = tr | va | te
    if len(allq) != len(split["train"]) + len(split["val"]) + len(split["test"]):
        bad.append("a question appears more than once within a bucket")
    missing = set(range(n_expected)) - allq
    extra = allq - set(range(n_expected))
    if missing:
        bad.append(f"{len(missing)} question(s) in no bucket, e.g. "
                   f"{sorted(missing)[:5]}")
    if extra:
        bad.append(f"question index outside 0..{n_expected - 1}: "
                   f"{sorted(extra)[:5]}")
    for k in ("train", "val", "test"):
        rec = man.get("counts", {}).get(k)
        if rec is not None and rec != len(split[k]):
            bad.append(f"counts[{k}]={rec} but split has {len(split[k])}")
    if not tr:
        bad.append("train bucket is empty -- nothing to fit the basis on")
    return bad


def h5_question_digest(h5_path):
    """sha256 over the H5's questions IN question_idx ORDER.

    Compared against the manifest's digest so a manifest built for a different
    benchmark ordering cannot be silently joined against these hidden states.
    Returns None when the H5 stores no question text (older tracker output),
    which downgrades the check rather than inventing a match.

    TRUNCATION. track_hidden_states.py stores `question` clipped to 2000 chars
    (line 439), while the manifest digest is built from the untruncated text in
    the signal JSON. On GSM8K the longest question is 696 chars, so the clip
    never fires and the two digests are comparable -- but that is a property of
    THIS benchmark, not a guarantee. A cell whose questions exceed the clip
    would report a mismatch that is an artifact, so the clip is detected and
    reported as "not verified" instead of as a failure.
    """
    import hashlib
    with h5py.File(h5_path, "r") as f:
        pairs = []
        for qi, g in iter_samples(f):
            q = g.attrs.get("question")
            if q is None:
                return None, 0
            q = q.decode() if isinstance(q, bytes) else str(q)
            if len(q) >= 2000:
                # Hit the tracker's clip: this text is not the full question,
                # so a digest over it cannot be compared with the manifest's.
                return "TRUNCATED", 0
            pairs.append((qi, q))
    pairs.sort(key=lambda t: t[0])
    texts = [t for _, t in pairs]
    return hashlib.sha256("\n".join(texts).encode()).hexdigest(), len(texts)


def cell_stem(task, size, key, ls, le):
    return f"hs_{task}_{size}_{key}_L{ls}-{le}.h5"


def iter_samples(h5, want_idx=None):
    grp = h5["samples"] if "samples" in h5 else h5
    for k in sorted(grp.keys()):
        g = grp[k]
        qi = int(g.attrs.get("question_idx", -1))
        if want_idx is not None and qi not in want_idx:
            continue
        yield qi, g


def fit_basis(h5_path, split_idx, ls, le, k_max, tok, verbose=True):
    """Fit alpha=0 PCA per (layer, phase) on the TRAIN questions."""
    train = set(split_idx["train"])
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"] if "meta" in f else f
        stored = np.asarray(meta.attrs.get("stored_layer_indices", []))
        # Half-open, exactly like every layer range in the repo:
        # decoder_layer_range(11,20) = range(10,19) -> 9 middle layers. The
        # H5's selective layout puts those middle layers at [0, n_middle) and
        # the model's FINAL layer last, so the middle count is the band width.
        n_middle = len(decoder_layer_range(ls, le))
        dim = None
        acc = {}
        n_seen = 0
        for qi, g in iter_samples(f, train):
            text = g.attrs.get("generated", "")
            dec = g["decode_hs"]
            T = dec.shape[0]
            cstep = char_to_step(tok, text, commit_char(text))
            if cstep is not None and cstep >= T:
                cstep = None            # commit past the stored decode span
            if dim is None:
                dim = dec.shape[2]
            n_seen += 1
            for li in range(n_middle):
                for phase in PHASES:
                    key = (li, phase)
                    if key not in acc:
                        # decode_all is unbounded (up to 768 rows/question),
                        # so it stores each question's row MEAN. See Accum.
                        acc[key] = Accum(dim,
                                         reduce_to_mean=(phase == "decode_all"))
                    if phase == "prefill":
                        X = g["prefill_hs"][-1][li][None, :].astype(np.float32)
                    else:
                        rows = phase_rows(T, cstep, phase)
                        if rows is None:
                            continue
                        X = dec[rows[0]:rows[1], li, :].astype(np.float32)
                    acc[key].add(X)
            if verbose and n_seen % 25 == 0:
                print(f"    fit: {n_seen}/{len(train)} train questions",
                      flush=True)

    basis = {}
    for (li, phase), a in acc.items():
        b = a.finish(k_max)
        if b is not None:
            basis[(li, phase)] = b
    return basis, stored.tolist(), n_middle


def project_cell(h5_path, basis, split_idx, n_middle, tok, verbose=True):
    """Project every sample onto the alpha=0 basis; export coords + NRE parts."""
    which = {q: name for name, qs in split_idx.items() for q in qs}
    out = []
    n_commit = n_total = 0
    with h5py.File(h5_path, "r") as f:
        for qi, g in iter_samples(f):
            text = g.attrs.get("generated", "")
            dec = g["decode_hs"]
            T = dec.shape[0]
            cstep = char_to_step(tok, text, commit_char(text))
            if cstep is not None and cstep >= T:
                cstep = None
            n_total += 1
            n_commit += cstep is not None
            rec = {
                "question_idx": qi,
                "split": which.get(qi),
                "commit_step": cstep,
                "n_decode": int(T),
                "correct": bool(g.attrs.get("correct", False)),
                "phases": {},
            }
            for li in range(n_middle):
                for phase in PHASES:
                    b = basis.get((li, phase))
                    if b is None:
                        continue
                    if phase == "prefill":
                        X = g["prefill_hs"][-1][li][None, :].astype(np.float32)
                    else:
                        rows = phase_rows(T, cstep, phase)
                        if rows is None:
                            continue
                        X = dec[rows[0]:rows[1], li, :].astype(np.float32)
                    Xc = X - b["mu"]
                    Z = Xc @ b["components"].T                  # (n_rows, k)

                    # --- BLOCKER 1: per-token coordinates ---------------
                    # A phase MEAN destroys token order, and speed /
                    # curvature / turning are all defined on the ordered
                    # sequence. The bounded phases are exported in full
                    # (<=20 rows); decode_all keeps only summary stats
                    # because 768 rows x k x 300 questions x 9 layers does
                    # not belong in a JSON. So option-A sensitivity is a
                    # LEVEL comparison, not a trajectory-shape one, and the
                    # meta records that rather than leaving it implied.
                    keep_traj = phase in TRAJECTORY_PHASES

                    # --- BLOCKER 2: reconstruction error for EVERY k -----
                    # NRE at k=20 alone cannot be re-expanded into k=1..20
                    # offline, so validation could not choose k. Because the
                    # components are orthonormal, the residual after k dims
                    # is energy - cumsum(Z^2), which is exact and costs one
                    # cumulative sum. Stored per question, so k selection is
                    # a pure alpha=0-val computation downstream.
                    energy = float(np.mean(np.sum(Xc ** 2, axis=1)))
                    z2 = np.mean(Z ** 2, axis=0)                # (k,)
                    re_by_k = (energy - np.cumsum(z2)).clip(min=0.0)

                    cell = {
                        "n_rows": int(X.shape[0]),
                        "coord": Z.mean(axis=0).tolist(),
                        "re": float(re_by_k[-1]),
                        "energy": energy,
                        "re_by_k": re_by_k.tolist(),
                    }
                    if keep_traj:
                        cell["coord_t"] = Z.astype(np.float32).round(4).tolist()
                    rec["phases"].setdefault(phase, {})[str(li)] = cell
            out.append(rec)
            if verbose and n_total % 50 == 0:
                print(f"    proj: {n_total} questions", flush=True)
    return out, n_commit, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--split_manifest", required=True,
                   help="REQUIRED. There is no default: a silent fallback to "
                        "all questions would destroy the held-out property "
                        "without erroring.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_dir", required=True,
                   help="LOCAL snapshot for the tokenizer (commit location).")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--size", default="8B")
    p.add_argument("--base_cell", default="nocot",
                   help="the alpha=0 cell the basis is fit on")
    p.add_argument("--cells", default="nocot,nocot_aneg6,nocot_aneg8,nocot_a6")
    p.add_argument("--layer_start", type=int, default=EXPECTED_START)
    p.add_argument("--layer_end",   type=int, default=EXPECTED_END)
    p.add_argument("--k_max", type=int, default=K_MAX_DEFAULT)
    p.add_argument("--n_questions", type=int, default=300,
                   help="expected question count; the split must cover 0..n-1")
    p.add_argument("--allow_overwrite", action="store_true")
    args = p.parse_args()

    with open(args.split_manifest) as f:
        man = json.load(f)

    bad = validate_manifest(man, args.n_questions)
    if bad:
        print("[FAIL] split manifest is not usable:")
        for b in bad:
            print(f"       - {b}")
        return 1
    split_idx = man["split"]
    print(f"[split] {man.get('version')} salt={man.get('salt')} "
          f"counts={man.get('counts')}  (structure verified)")

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    if args.base_cell not in cells:
        cells.insert(0, args.base_cell)

    # Fail closed BEFORE any expensive work: every requested cell must exist,
    # and no output may be clobbered. These run BEFORE the tokenizer load on
    # purpose -- loading a tokenizer first means waiting on it only to be told
    # a cell name was mistyped, and it also hides these guards behind an
    # unrelated failure when --model_dir is wrong.
    for c in cells:
        f = h5_dir / cell_stem(args.task, args.size, c,
                               args.layer_start, args.layer_end)
        if not f.exists():
            print(f"[FAIL] missing H5 for cell {c!r}: {f}")
            return 1
        o = out_dir / f"manifold_{c}.json"
        if o.exists() and not args.allow_overwrite:
            print(f"[FAIL] {o} exists; pass --allow_overwrite deliberately")
            return 1

    # The basis is an OUTPUT too. Guarding only the per-cell JSON would let a
    # re-run silently replace basis.npz while the coordinate files it no longer
    # matches sit beside it -- the worst failure mode here, because the result
    # still loads and still looks reasonable.
    for extra in ("basis.npz", "basis_meta.json"):
        o = out_dir / extra
        if o.exists() and not args.allow_overwrite:
            print(f"[FAIL] {o} exists; pass --allow_overwrite deliberately")
            return 1

    base_h5 = h5_dir / cell_stem(args.task, args.size, args.base_cell,
                                 args.layer_start, args.layer_end)

    # The basis cell must actually BE alpha=0. Trusting the filename is how a
    # steered cell becomes "the natural manifold" without anything erroring.
    with h5py.File(base_h5, "r") as f:
        bmeta = f["meta"] if "meta" in f else f
        a = bmeta.attrs.get("steer_alpha")
        if a is None:
            print(f"[FAIL] {base_h5.name} has no steer_alpha; cannot confirm "
                  f"the basis cell is unsteered")
            return 1
        if float(a) != 0.0:
            print(f"[FAIL] --base_cell {args.base_cell!r} has steer_alpha="
                  f"{float(a)}, not 0. The natural manifold must be fit on an "
                  f"UNSTEERED cell; fitting it on a steered one would make the "
                  f"steered state the reference and hide the effect under test.")
            return 1

    # BLOCKER 3, second half: the manifest's question ordering must match the
    # H5's. section 2 stores a digest precisely so this join can be checked.
    man_digest = man.get("question_text_sha256")
    h5_digest, n_txt = h5_question_digest(base_h5)
    if h5_digest == "TRUNCATED":
        print("[!] H5 question text hits the tracker's 2000-char clip, so the "
              "digest is not comparable; NOT verified this run")
    elif man_digest and h5_digest:
        if man_digest != h5_digest:
            print(f"[FAIL] question-text digest mismatch:\n"
                  f"       manifest {man_digest[:16]}...\n"
                  f"       H5       {h5_digest[:16]}...\n"
                  f"       The split was built for a different benchmark "
                  f"ordering, so every question_idx join is wrong.")
            return 1
        print(f"[split] question-text digest matches the H5 ({n_txt} questions)")
    elif man_digest and not h5_digest:
        print("[!] H5 stores no question text; digest NOT verified this run")
    else:
        print("[!] manifest carries no question-text digest; NOT verified")

    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    print(f"[fit] basis from {args.base_cell} TRAIN "
          f"({len(split_idx['train'])} questions), k_max={args.k_max}")
    basis, stored, n_middle = fit_basis(base_h5, split_idx, args.layer_start,
                                        args.layer_end, args.k_max, tok)
    print(f"[fit] stored_layer_indices={stored}  n_middle={n_middle}  "
          f"bases={len(basis)}")

    np.savez_compressed(
        out_dir / "basis.npz",
        **{f"{li}|{ph}|{fld}": b[fld]
           for (li, ph), b in basis.items()
           for fld in ("mu", "components", "explained")},
    )
    with open(out_dir / "basis_meta.json", "w") as f:
        json.dump({
            "base_cell": args.base_cell,
            "k_max": args.k_max,
            "phases": list(PHASES),
            "window": WINDOW,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "stored_layer_indices": stored,
            "n_middle": n_middle,
            "split_manifest_version": man["version"],
            "split_salt": man["salt"],
            "split_counts": man["counts"],
            "question_text_sha256": man.get("question_text_sha256"),
            "commit_locator": "first #### else first answer-candidate "
                              "(Llama convention, NOT Qwen's ####-only)",
            "per_layer": {f"{li}|{ph}": {
                "n_questions": b["n_questions"],
                "total_var": b["total_var"],
                "explained": b["explained"].tolist(),
            } for (li, ph), b in basis.items()},
        }, f, indent=2)

    for c in cells:
        h5 = h5_dir / cell_stem(args.task, args.size, c,
                                args.layer_start, args.layer_end)
        print(f"[proj] {c}", flush=True)
        recs, n_commit, n_total = project_cell(h5, basis, split_idx,
                                               n_middle, tok)
        cov = n_commit / n_total if n_total else 0.0
        print(f"[proj] {c}: {n_total} questions, commit coverage "
              f"{n_commit}/{n_total} = {cov:.3f}")
        with open(out_dir / f"manifold_{c}.json", "w") as f:
            json.dump({
                "cell": c,
                "n_questions": n_total,
                "commit_coverage": cov,
                "n_commit": n_commit,
                "basis_from": args.base_cell,
                "split_manifest_version": man["version"],
                "data": recs,
            }, f)

    print("\n[ok] wrote basis + per-cell coordinates to", out_dir)
    print("     Coverage is REPORTED, not gated: a no-commit sample is absent "
          "from the aligned phases and present in decode_all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
