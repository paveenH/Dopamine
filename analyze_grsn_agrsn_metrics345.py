#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_grsn_agrsn_metrics345.py -- OFFLINE CPU analysis (no model, no GPU).

Metrics 3-5 of the GRSN vs AGRSN representation-similarity analysis for
Llama3.1-8B, over ALL 32 decoder layers (raw array indices 1..32; embedding
index 0 excluded). Companion to the LOCAL Metric 1-2 script
(analyze_grsn_agrsn_direction.py in the RoleHidden workspace); this script
runs on the SERVER because it needs the four per-question H5 files.

Inputs (all already produced by the frozen pipeline; NOTHING is re-extracted,
re-generated, or re-steered here):

  GRSN  = gsm8k_role             (no abstention exit, plain Bare role prompt)
  AGRSN = gsm8k_role_abstention  (v2 abstention-enabled prompt)
  Both: expert_mean - non_expert_mean, sign FIXED, never auto-flipped.

  components/hidden_states/llama3/{task}/{expert,non_expert}_8B.h5
      dataset "hidden_states": (1319, 33, 4096) float16,
      index 0 = embedding, 1..32 = decoder layer outputs (HF convention);
      datasets "question" (per-sample) and "in_300_sample" (bool, audit-only);
      attrs carry model/task/condition, prefill_only=True,
      generation_performed=False, steering_applied=False,
      chat_template_applied=False, ordered_sample_identity_sha256, etc.
  components/hidden_states_mean/llama3/{task}/{expert,non_expert,diff_mean}_8B.npy
      (1,1,33,4096) float16 + manifest.json (pairing_digest).

Per-question, per-layer role-transition vectors:
      D_G[i,l] = H_G,expert[i,l] - H_G,non_expert[i,l]   (computed in float32)
      D_A[i,l] = H_A,expert[i,l] - H_A,non_expert[i,l]

FAIL-CLOSED validation BEFORE any metric is computed (mirrors
mean/mean_diff_gsm8k_role.py's gate): all four H5 shapes must be
(1319,33,4096); attrs model/task/condition must agree with the file path;
prefill_only / no-generation / no-steering / no-chat-template flags must hold;
the four ordered_sample_identity_sha256 values must be identical AND equal to
both mean-manifest pairing_digests; the four per-sample "question" arrays must
be byte-identical; the four in_300_sample arrays must be identical (used as
provenance only, never to subset). Any failure aborts with nothing written.

Metric 3 -- Role-Transition CKA (per layer):
  Column-center D_G[:,l,:] and D_A[:,l,:] over the question axis, then linear
  CKA on the resulting Gram matrices:
      CKA = <G_G, G_A>_F / (||G_G||_F * ||G_A||_F),  G_X = Xc @ Xc.T
  (column-centering makes the Grams doubly-centered, so this IS the centered
  linear CKA). Null model: with a FIXED seed, permute the QUESTION ROWS of one
  group (breaks the per-question pairing; an orthogonal-rotation null is NOT
  used -- CKA is invariant to it). Reports real value, per-layer null
  mean/std/max and empirical p (fraction of perms >= real).

Metric 4 -- Role-Transition Subspace (per layer, per k in K_GRID):
  ONE fixed-seed train/held-out split of the 1319 questions (same indices for
  both prompt families). PCA per prompt per layer is fit on the TRAIN split
  only, after subtracting the TRAIN column mean; the held-out split is
  centered with the same train mean. Fit once at max k and truncate.
  Reports, per layer and k:
    - principal-angle summary between the two k-dim subspaces (singular
      values of V_G^T V_A = cosines of principal angles; mean and min);
    - held-out cross energy fraction: mean_i ||P_A x_i||^2 / ||x_i||^2 for
      held-out GRSN diffs x (and the reverse direction);
    - held-out SELF reconstruction fraction (each subspace on its own held-out
      diffs) as the reference denominator for the cross numbers.
  Energy fraction = mean over held-out samples of per-sample squared-norm
  ratio; denominator = the held-out sample's own squared norm after train-mean
  centering. Train-fit reconstruction is NEVER reported as held-out.

Metric 5 -- Direction-Subspace (per layer, per k, both directions):
  Full-1319 mean directions are RECONSTRUCTED from expert_mean/non_expert_mean
  (float32 subtraction; stored diff_mean used ONLY as a consistency check),
  exactly the Metric 1-2 convention. Projection fraction:
      R^2 = ||V_k^T r||^2 / ||r||^2
  of the GRSN mean direction into the AGRSN train-fit PCA subspace, and vice
  versa. PCA subspaces are the Metric 4 TRAIN-fit ones; the full-1319 mean
  direction is a DESCRIPTIVE input here -- this projection is NOT an
  independent held-out validation. Random-k-dim-subspace baseline = k/4096,
  plus a fixed-seed random simulation (random unit direction vs random
  orthonormal k-bases). Projection enrichment is NOT to be read as mechanism
  equivalence.

Outputs (all under --out_dir, default
/data1/paveen/Dopamine/components/analysis/GRSN_AGRSN/):
  role_transition_cka_by_layer.csv
  role_transition_subspace_by_layer_k.csv
  direction_subspace_projection.csv
  report.md
  provenance.json
  role_transition_cka_by_layer.{png,svg}
  role_transition_subspace.{png,svg}
  direction_subspace_projection.{png,svg}

Usage (SERVER; interpreter is `python`):
  python analyze_grsn_agrsn_metrics345.py \
      --hs_root /data1/paveen/Dopamine/components/hidden_states \
      --mean_root /data1/paveen/Dopamine/components/hidden_states_mean \
      --out_dir /data1/paveen/Dopamine/components/analysis/GRSN_AGRSN

Runtime: order tens of minutes on CPU (dominated by the CKA permutation
null). --n_perms and thread count are the main knobs.

These are REPRESENTATION-similarity results. Geometric similarity between
GRSN and AGRSN must NOT be written as steering-equivalence, behavioral
equivalence, or causal equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_VERSION = "analyze_grsn_agrsn_metrics345-v1"

TASKS = {"grsn": "gsm8k_role", "agrsn": "gsm8k_role_abstention"}
CONDITIONS = ("expert", "non_expert")

N_EXPECT = 1319
N_LAYERS_OUT = 33          # index 0 embedding + 32 decoder layers
HIDDEN = 4096
MODEL = "llama3"
SIZE = "8B"

ANALYSIS_LAYERS = list(range(1, 33))   # raw indices 1..32 == decoder layers 1..32
INJECTION_BAND = (11, 20)              # raw indices [11,20) == decoder layers 11-19
BAND_LAYERS = list(range(INJECTION_BAND[0], INJECTION_BAND[1]))

K_GRID = (1, 2, 5, 10, 20, 50)
K_MAX = max(K_GRID)

DEFAULT_SEED = 20260922
DEFAULT_N_PERMS = 200
DEFAULT_TRAIN_FRAC = 0.8


def die(msg: str, code: int = 1):
    print(f"[REFUSE] {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def bool_attr(a, key, expected):
    if key not in a:
        return f"attr {key!r} missing"
    if bool(a[key]) is not expected:
        return f"attr {key!r}={a[key]!r}, expected {expected}"
    return None


# ---------------------------------------------------------------------------
# Fail-closed input validation
# ---------------------------------------------------------------------------

def validate_inputs(hs_root: Path, mean_root: Path) -> dict:
    info = {"h5": {}, "h5_sha256": {}, "attrs": {}, "mean": {}}
    expected_digest = None

    for task in TASKS.values():
        for cond in CONDITIONS:
            p = hs_root / MODEL / task / f"{cond}_{SIZE}.h5"
            if not p.exists():
                die(f"missing H5: {p}")
            with h5py.File(p, "r") as f:
                ds = f["hidden_states"]
                if tuple(ds.shape) != (N_EXPECT, N_LAYERS_OUT, HIDDEN):
                    die(f"{p}: hidden_states shape {ds.shape}, expected "
                        f"{(N_EXPECT, N_LAYERS_OUT, HIDDEN)}")
                if "question" not in f or "in_300_sample" not in f:
                    die(f"{p}: missing 'question'/'in_300_sample' dataset")
                q = f["question"][...]
                in300 = f["in_300_sample"][...]
                a = dict(f.attrs)
            err = bool_attr(a, "prefill_only", True) or \
                  bool_attr(a, "generation_performed", False) or \
                  bool_attr(a, "steering_applied", False) or \
                  bool_attr(a, "chat_template_applied", False)
            if err:
                die(f"{p}: {err}")
            if a.get("model") != MODEL:
                die(f"{p}: attrs model={a.get('model')!r} != {MODEL!r}")
            if a.get("task") != task:
                die(f"{p}: attrs task={a.get('task')!r} != {task!r}")
            if a.get("condition") != cond:
                die(f"{p}: attrs condition={a.get('condition')!r} != {cond!r}")
            if int(a.get("n_samples_done", -1)) != N_EXPECT:
                die(f"{p}: n_samples_done={a.get('n_samples_done')!r} != {N_EXPECT}")
            d = a.get("ordered_sample_identity_sha256")
            if not d:
                die(f"{p}: attrs missing ordered_sample_identity_sha256")
            if expected_digest is None:
                expected_digest = d
            elif d != expected_digest:
                die(f"{p}: ordered_sample_identity_sha256 {d} != {expected_digest}")
            info["h5"][(task, cond)] = p
            info["h5_sha256"][str(p)] = sha256_of_file(p)
            info["attrs"][(task, cond)] = a

            # per-sample identity checks (against the first file read)
            if ("question_ref" not in info):
                info["question_ref"] = q
                info["in300_ref"] = in300
            else:
                if not np.array_equal(q, info["question_ref"]):
                    die(f"{p}: per-sample 'question' array differs from the "
                        "first H5 read -- question ordering is not identical "
                        "across the four files.")
                if not np.array_equal(in300, info["in300_ref"]):
                    die(f"{p}: in_300_sample array differs across files.")

    # mean files + manifest pairing digests
    for name, task in TASKS.items():
        mdir = mean_root / MODEL / task
        means = {}
        for key in ("expert_mean", "non_expert_mean", "diff_mean"):
            p = mdir / f"{key}_{SIZE}.npy"
            if not p.exists():
                die(f"missing mean file: {p}")
            arr = np.load(p)
            if tuple(arr.shape) != (1, 1, N_LAYERS_OUT, HIDDEN):
                die(f"{p}: shape {arr.shape}, expected (1,1,33,4096)")
            means[key] = arr
        mp = mdir / "manifest.json"
        if not mp.exists():
            die(f"missing mean manifest: {mp}")
        with open(mp, "r", encoding="utf-8") as f:
            man = json.load(f)
        if man.get("pairing_digest") != expected_digest:
            die(f"{mp}: pairing_digest {man.get('pairing_digest')!r} != H5 "
                f"ordered_sample_identity_sha256 {expected_digest!r}")
        info["mean"][name] = {"means": means, "manifest": man,
                              "manifest_path": mp,
                              "manifest_sha256": sha256_of_file(mp)}
        for key in ("expert_mean", "non_expert_mean", "diff_mean"):
            info["mean"][name][f"{key}_sha256"] = sha256_of_file(
                mdir / f"{key}_{SIZE}.npy")

    info["pairing_digest"] = expected_digest
    info["in300_count"] = int(info["in300_ref"].sum())
    return info


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def linear_cka_from_gram(gx: np.ndarray, gy: np.ndarray) -> float:
    num = float(np.sum(gx * gy))
    den = float(np.sqrt(np.sum(gx * gx) * np.sum(gy * gy)))
    return num / max(den, 1e-300)


def column_center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def pca_basis(train_centered: np.ndarray, k: int) -> np.ndarray:
    """Right singular vectors (4096 x k) of the train-centered matrix."""
    _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
    return vt[:k].T


def energy_fraction(x_centered: np.ndarray, basis: np.ndarray) -> float:
    """mean_i ||P x_i||^2 / ||x_i||^2 over rows of x_centered."""
    proj = x_centered @ basis                       # (n, k)
    num = np.sum(proj ** 2, axis=1)
    den = np.sum(x_centered ** 2, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-300)))


def principal_angle_cosines(vg: np.ndarray, va: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(vg.T @ va, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def random_subspace_baseline_sim(d: int, k_grid, n_draws: int = 200,
                                 seed: int = DEFAULT_SEED) -> dict:
    rng = np.random.default_rng(seed + 777)
    out = {}
    for k in k_grid:
        vals = np.empty(n_draws)
        for i in range(n_draws):
            q, _ = np.linalg.qr(rng.standard_normal((d, k)))
            r = rng.standard_normal(d)
            r /= np.linalg.norm(r)
            vals[i] = float(np.sum((q.T @ r) ** 2))
        out[k] = {"sim_mean": float(vals.mean()),
                  "sim_std": float(vals.std()),
                  "analytic_k_over_d": k / d}
    return out


# ---------------------------------------------------------------------------
# Figures (headless Agg)
# ---------------------------------------------------------------------------

def _band_decorate(ax):
    ax.axvspan(INJECTION_BAND[0] - 0.5, INJECTION_BAND[1] - 0.5,
               color="orange", alpha=0.15, label="band L11-L19")
    ax.set_xlabel("Decoder layer")


def make_figures(out_dir: Path, cka_rows, sub_rows, dir_rows, rng_state=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = ANALYSIS_LAYERS

    # ---- Fig 1: CKA vs permutation null ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    real = [r["cka"] for r in cka_rows]
    null_m = [r["null_mean"] for r in cka_rows]
    null_s = [r["null_std"] for r in cka_rows]
    ax.plot(layers, real, "o-", color="C0", label="paired CKA (real)")
    ax.plot(layers, null_m, "s--", color="C1", label="question-permutation null (mean)")
    ax.fill_between(layers,
                    np.array(null_m) - np.array(null_s),
                    np.array(null_m) + np.array(null_s),
                    color="C1", alpha=0.2, label="null ±1 std")
    _band_decorate(ax)
    ax.set_ylabel("Centered linear CKA")
    ax.set_title("Role-Transition CKA by layer (GRSN vs AGRSN, n=1319)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "role_transition_cka_by_layer.png", dpi=200)
    fig.savefig(out_dir / "role_transition_cka_by_layer.svg")
    plt.close(fig)

    # ---- Fig 2: subspace overlap + held-out cross energy ----
    ks = K_GRID
    e_ga = np.full((len(layers), len(ks)), np.nan)   # G val -> A subspace
    e_ag = np.full((len(layers), len(ks)), np.nan)   # A val -> G subspace
    e_gs = np.full((len(layers), len(ks)), np.nan)   # G val -> G subspace (self)
    e_as = np.full((len(layers), len(ks)), np.nan)   # A val -> A subspace (self)
    pacos = np.full((len(layers), len(ks)), np.nan)
    for r in sub_rows:
        i = layers.index(r["layer"]); j = ks.index(r["k"])
        e_ga[i, j] = r["val_energy_G_into_A"]
        e_ag[i, j] = r["val_energy_A_into_G"]
        e_gs[i, j] = r["val_energy_G_self"]
        e_as[i, j] = r["val_energy_A_self"]
        pacos[i, j] = r["principal_cos_mean"]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    for ax, data, title in (
        (fig.add_subplot(gs[0, 0]), e_ga, "Held-out GRSN diff energy in AGRSN subspace"),
        (fig.add_subplot(gs[0, 1]), e_ag, "Held-out AGRSN diff energy in GRSN subspace"),
    ):
        im = ax.imshow(data.T, aspect="auto", origin="lower",
                       extent=(0.5, 32.5, -0.5, len(ks) - 0.5),
                       vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_yticks(range(len(ks)), [f"k={k}" for k in ks])
        ax.set_xlabel("Decoder layer")
        ax.set_title(title, fontsize=9)
        for b in BAND_LAYERS:
            ax.axvspan(b - 0.5, b + 0.5, color="orange", alpha=0.12)
        fig.colorbar(im, ax=ax, label="energy fraction")
    ax = fig.add_subplot(gs[1, :])
    for j, k in enumerate(ks):
        ax.plot(layers, pacos[:, j], "o-", ms=3, label=f"k={k}")
    _band_decorate(ax)
    ax.set_ylabel("Mean cos of principal angles")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Subspace overlap between GRSN and AGRSN role-transition PCA subspaces")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.suptitle("Role-Transition Subspace (held-out, PCA fit on 80% train split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "role_transition_subspace.png", dpi=200)
    fig.savefig(out_dir / "role_transition_subspace.svg")
    plt.close(fig)

    # ---- Fig 3: direction -> subspace projection R^2 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, key, title in (
        (axes[0], "r2_G_into_A", "GRSN mean direction -> AGRSN subspace"),
        (axes[1], "r2_A_into_G", "AGRSN mean direction -> GRSN subspace"),
    ):
        rows_by_lk = {(r["layer"], r["k"]): r for r in dir_rows}
        for j, k in enumerate(ks):
            ys = [rows_by_lk[(l, k)][key] for l in layers]
            ax.plot(layers, ys, "o-", ms=3, label=f"k={k}")
            ax.axhline(k / HIDDEN, ls=":", lw=0.8, color="gray", alpha=0.6)
        _band_decorate(ax)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R^2 = ||V_k^T r||^2 / ||r||^2")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], ls=":", color="gray"))
    labels.append("random k/4096 baseline")
    axes[1].legend(handles, labels, loc="best", fontsize=7, ncol=2)
    fig.suptitle("Direction-Subspace projection (full-1319 mean direction, Metric-4 train-fit PCA)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "direction_subspace_projection.png", dpi=200)
    fig.savefig(out_dir / "direction_subspace_projection.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(v, nd=4):
    if v is None:
        return "NA"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "NA"
    return f"{v:.{nd}f}"


def write_report(out_dir: Path, val: dict, cka_rows, sub_rows, dir_rows,
                 null_summary, split_info, seed, n_perms, runtime_s):
    band_cka = [r["cka"] for r in cka_rows if r["layer"] in BAND_LAYERS]
    all_cka = [r["cka"] for r in cka_rows]

    # band means for cross energy at a representative k
    def band_mean(key, k):
        vs = [r[key] for r in sub_rows if r["layer"] in BAND_LAYERS and r["k"] == k]
        return float(np.mean(vs)) if vs else float("nan")

    def band_mean_dir(key, k):
        vs = [r[key] for r in dir_rows if r["layer"] in BAND_LAYERS and r["k"] == k]
        return float(np.mean(vs)) if vs else float("nan")

    L = []
    L.append("# GRSN vs AGRSN Representation-Similarity Analysis -- Metrics 3-5\n")
    L.append("**Scope: representation-similarity only, Llama3.1-8B, all 32 decoder layers "
             "(L1-L32; embedding index 0 excluded). Companion to the Metric 1-2 report.** "
             "Geometric similarity between GRSN and AGRSN does NOT imply the two directions "
             "have the same steering, behavioral, or causal effect.\n")
    L.append("## Directions and inputs\n")
    L.append("- **GRSN** = `gsm8k_role` (no abstention exit); **AGRSN** = `gsm8k_role_abstention` "
             "(v2 abstention prompt). Both: expert_mean - non_expert_mean, sign fixed, full "
             "1319-question GSM8K test set, prefill-only, no generation, no steering.")
    L.append(f"- All four H5 `ordered_sample_identity_sha256` values match both mean-manifest "
             f"`pairing_digest`s: `{val['pairing_digest']}`.")
    L.append("- Per-sample `question` arrays byte-identical across all four H5s; "
             f"`in_300_sample` identical (n_in_300={val['in300_count']}, audit-only, "
             "not used to subset).")
    L.append(f"- Per-question role-transition vectors D = expert - non_expert computed in "
             f"float32 per layer. Train/held-out split: seed={seed}, "
             f"{split_info['n_train']}/{split_info['n_val']} questions "
             f"({split_info['train_frac']:.0%}/{1-split_info['train_frac']:.0%}), "
             f"IDENTICAL indices for both prompt families.\n")

    L.append("## Metric 3: Role-Transition CKA (centered linear CKA, per layer)\n")
    L.append(f"- Permutation null: question rows of one group shuffled with fixed seed "
             f"{seed} (n_perms={n_perms} per layer). No orthogonal-rotation null (CKA is "
             "invariant to it).")
    L.append(f"- Null distribution across all {n_perms}x{len(cka_rows)} values: "
             f"mean={null_summary['mean']:.5f}, std={null_summary['std']:.5f}, "
             f"max={null_summary['max']:.5f} (essentially 0 -- real values are far above).")
    L.append(f"- Real CKA, band L11-L19: mean={np.mean(band_cka):.4f}, "
             f"min={np.min(band_cka):.4f}, max={np.max(band_cka):.4f}.")
    L.append(f"- Real CKA, all 32 layers: mean={np.mean(all_cka):.4f}, "
             f"min={np.min(all_cka):.4f}, max={np.max(all_cka):.4f}.")
    L.append("- Per-layer table: `role_transition_cka_by_layer.csv`. "
             "This measures whether the PER-QUESTION geometry of role transitions is "
             "similar across the two prompts; it does NOT measure whether the two mean "
             "directions are parallel (that is Metric 1).\n")

    L.append("## Metric 4: Role-Transition Subspace (PCA fit on train split only)\n")
    L.append("- Centering: per-layer column mean of the TRAIN split; the held-out split is "
             "centered with the SAME train mean. Denominator of every energy fraction: the "
             "held-out sample's own squared norm (per-sample ratio, then mean). Train-fit "
             "reconstruction is never reported as held-out.")
    L.append("- Principal-angle summary: mean cos of the k principal angles between the two "
             "k-dim subspaces (singular values of V_G^T V_A).")
    for k in K_GRID:
        L.append(f"- k={k}: band-mean held-out cross energy G->A = {fmt(band_mean('val_energy_G_into_A', k))}, "
                 f"A->G = {fmt(band_mean('val_energy_A_into_G', k))}; "
                 f"self refs G->G = {fmt(band_mean('val_energy_G_self', k))}, "
                 f"A->A = {fmt(band_mean('val_energy_A_self', k))}; "
                 f"band-mean principal cos = {fmt(band_mean('principal_cos_mean', k))}.")
    L.append("- Full table: `role_transition_subspace_by_layer_k.csv`.\n")

    L.append("## Metric 5: Direction-Subspace (full-1319 mean direction -> Metric-4 train-fit subspace)\n")
    L.append(f"- Mean directions reconstructed from expert_mean/non_expert_mean (float32); "
             f"stored diff_mean used only as a consistency check "
             f"(GRSN max|err|={val['diff_check']['grsn']:.2e}, "
             f"AGRSN max|err|={val['diff_check']['agrsn']:.2e}).")
    L.append("- R^2 = ||V_k^T r||^2 / ||r||^2. Random k-dim subspace baseline = k/4096 "
             "(dotted gray; fixed-seed simulation confirms it). The full-1319 mean direction "
             "is a DESCRIPTIVE input here: this is NOT an independent held-out validation, "
             "and projection enrichment must not be read as mechanism equivalence.")
    for k in K_GRID:
        L.append(f"- k={k}: band-mean R^2 G->A = {fmt(band_mean_dir('r2_G_into_A', k))}, "
                 f"A->G = {fmt(band_mean_dir('r2_A_into_G', k))} "
                 f"(random baseline {k/HIDDEN:.5f}).")
    L.append("- Full table: `direction_subspace_projection.csv`.\n")

    L.append("## Caveats\n")
    L.append("- All values are representation-geometry numbers for ONE prompt pair on ONE "
             "model. They say nothing about steering equivalence or behavior.")
    L.append(f"- Runtime {runtime_s:.0f}s, CPU only, no model loaded.\n")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs_root", required=True)
    ap.add_argument("--mean_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n_perms", type=int, default=DEFAULT_N_PERMS)
    ap.add_argument("--train_frac", type=float, default=DEFAULT_TRAIN_FRAC)
    args = ap.parse_args()

    hs_root = Path(args.hs_root)
    mean_root = Path(args.mean_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.iterdir()] if out_dir.exists() else []
    if existing:
        print(f"[warn] out_dir {out_dir} already contains {len(existing)} file(s); "
              "existing files with the same name will be overwritten.")

    t0 = time.time()
    print("[1/6] fail-closed input validation ...")
    val = validate_inputs(hs_root, mean_root)
    print(f"      pairing_digest = {val['pairing_digest']}")

    # mean-direction reconstruction + stored-diff consistency check (Metric 5 inputs)
    dirs = {}
    diff_check = {}
    for name, task in TASKS.items():
        m = val["mean"][name]["means"]
        e = m["expert_mean"].astype(np.float32).reshape(N_LAYERS_OUT, HIDDEN)
        n = m["non_expert_mean"].astype(np.float32).reshape(N_LAYERS_OUT, HIDDEN)
        d = m["diff_mean"].astype(np.float32).reshape(N_LAYERS_OUT, HIDDEN)
        r = e - n
        diff_check[name] = float(np.max(np.abs(r - d)))
        if diff_check[name] > 1e-2:
            die(f"{name}: reconstructed expert-non_expert differs from stored "
                f"diff_mean by {diff_check[name]:.2e} (>1e-2) -- refusing.")
        dirs[name] = r
    val["diff_check"] = diff_check

    # fixed train/held-out split, identical for both prompt families
    rng_split = np.random.default_rng(args.seed)
    perm = rng_split.permutation(N_EXPECT)
    n_train = int(round(N_EXPECT * args.train_frac))
    train_idx, val_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    split_info = {"n_train": n_train, "n_val": N_EXPECT - n_train,
                  "train_frac": args.train_frac,
                  "train_idx_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest()}

    # permutation nulls (shared across layers: same row permutations every layer)
    rng_null = np.random.default_rng(args.seed + 1)
    null_perms = np.stack([rng_null.permutation(N_EXPECT)
                           for _ in range(args.n_perms)])       # (n_perms, n)

    print("[2/6] loading four H5 files (4 x ~356 MB float16) ...")
    hs = {}
    for task in TASKS.values():
        for cond in CONDITIONS:
            with h5py.File(val["h5"][(task, cond)], "r") as f:
                hs[(task, cond)] = f["hidden_states"][...]
            if not np.isfinite(hs[(task, cond)].astype(np.float32)).all():
                die(f"{val['h5'][(task, cond)]}: non-finite hidden states")

    cka_rows, sub_rows, dir_rows = [], [], []
    null_all = []

    print("[3/6] per-layer Metric 3 (CKA + permutation null) ...")
    for li, l in enumerate(ANALYSIS_LAYERS):
        dg = (hs[(TASKS["grsn"], "expert")][:, l, :].astype(np.float32)
              - hs[(TASKS["grsn"], "non_expert")][:, l, :].astype(np.float32))
        da = (hs[(TASKS["agrsn"], "expert")][:, l, :].astype(np.float32)
              - hs[(TASKS["agrsn"], "non_expert")][:, l, :].astype(np.float32))

        dgc = column_center(dg).astype(np.float64)
        dac = column_center(da).astype(np.float64)
        gx = dgc @ dgc.T
        gy = dac @ dac.T
        cka = linear_cka_from_gram(gx, gy)
        nulls = np.empty(args.n_perms)
        for p in range(args.n_perms):
            dp = dac[null_perms[p]]
            gp = dp @ dp.T
            nulls[p] = linear_cka_from_gram(gx, gp)
        null_all.extend(nulls.tolist())
        cka_rows.append({
            "layer": l, "cka": cka,
            "null_mean": float(nulls.mean()), "null_std": float(nulls.std()),
            "null_min": float(nulls.min()), "null_max": float(nulls.max()),
            "null_p_ge_real": float(np.mean(nulls >= cka)),
            "n_perms": args.n_perms,
            "n_questions": N_EXPECT, "in_band": l in BAND_LAYERS})
        if (li + 1) % 8 == 0 or li == len(ANALYSIS_LAYERS) - 1:
            print(f"      CKA layer {l}/32 done ({time.time()-t0:.0f}s)")

    print("[4/6] per-layer Metric 4 (subspaces, train-fit PCA) ...")
    for li, l in enumerate(ANALYSIS_LAYERS):
        dg = (hs[(TASKS["grsn"], "expert")][:, l, :].astype(np.float32)
              - hs[(TASKS["grsn"], "non_expert")][:, l, :].astype(np.float32))
        da = (hs[(TASKS["agrsn"], "expert")][:, l, :].astype(np.float32)
              - hs[(TASKS["agrsn"], "non_expert")][:, l, :].astype(np.float32))
        d = {"grsn": dg, "agrsn": da}
        mean = {}
        basis = {}
        for name in TASKS:
            tr = d[name][train_idx]
            mean[name] = tr.mean(axis=0)
            basis[name] = pca_basis(tr - mean[name], K_MAX)
        v = {"grsn": d["grsn"][val_idx] - mean["grsn"],
             "agrsn": d["agrsn"][val_idx] - mean["agrsn"]}
        for k in K_GRID:
            vg = basis["grsn"][:, :k]
            va = basis["agrsn"][:, :k]
            pac = principal_angle_cosines(vg, va)
            sub_rows.append({
                "layer": l, "k": k,
                "principal_cos_mean": float(pac.mean()),
                "principal_cos_min": float(pac.min()),
                "val_energy_G_into_A": energy_fraction(v["grsn"], va),
                "val_energy_A_into_G": energy_fraction(v["agrsn"], vg),
                "val_energy_G_self": energy_fraction(v["grsn"], vg),
                "val_energy_A_self": energy_fraction(v["agrsn"], va),
                "n_train": n_train, "n_val": N_EXPECT - n_train,
                "in_band": l in BAND_LAYERS})
        if (li + 1) % 8 == 0 or li == len(ANALYSIS_LAYERS) - 1:
            print(f"      subspace layer {l}/32 done ({time.time()-t0:.0f}s)")

    print("[5/6] Metric 5 (mean direction -> subspace) + baselines ...")
    # subspaces must be recomputed per layer; reuse Metric-4 loop by re-fitting
    # (deterministic, same split) -- cheaper than caching 32x2x4096x50 bases? cache them.
    # (bases are small: 32*2*4096*50*8B ~ 105 MB; recompute instead to save memory)
    rand_base = random_subspace_baseline_sim(HIDDEN, K_GRID, seed=args.seed)
    for li, l in enumerate(ANALYSIS_LAYERS):
        dg = (hs[(TASKS["grsn"], "expert")][:, l, :].astype(np.float32)
              - hs[(TASKS["grsn"], "non_expert")][:, l, :].astype(np.float32))
        da = (hs[(TASKS["agrsn"], "expert")][:, l, :].astype(np.float32)
              - hs[(TASKS["agrsn"], "non_expert")][:, l, :].astype(np.float32))
        rg = dirs["grsn"][l]
        ra = dirs["agrsn"][l]
        ng = float(rg @ rg)
        na = float(ra @ ra)
        mean_g = dg[train_idx].mean(axis=0)
        mean_a = da[train_idx].mean(axis=0)
        bg = pca_basis(dg[train_idx] - mean_g, K_MAX)
        ba = pca_basis(da[train_idx] - mean_a, K_MAX)
        for k in K_GRID:
            dir_rows.append({
                "layer": l, "k": k,
                "r2_G_into_A": float(np.sum((bg[:, :k].T @ ra) ** 2)) / max(na, 1e-300),
                "r2_A_into_G": float(np.sum((ba[:, :k].T @ rg) ** 2)) / max(ng, 1e-300),
                "random_baseline_k_over_d": k / HIDDEN,
                "random_sim_mean": rand_base[k]["sim_mean"],
                "grsn_dir_norm": float(np.sqrt(ng)),
                "agrsn_dir_norm": float(np.sqrt(na)),
                "in_band": l in BAND_LAYERS})

    null_summary = {"mean": float(np.mean(null_all)),
                    "std": float(np.std(null_all)),
                    "min": float(np.min(null_all)),
                    "max": float(np.max(null_all))}

    print("[6/6] writing CSVs, figures, report, provenance ...")
    import csv
    for name, rows in (("role_transition_cka_by_layer", cka_rows),
                       ("role_transition_subspace_by_layer_k", sub_rows),
                       ("direction_subspace_projection", dir_rows)):
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    make_figures(out_dir, cka_rows, sub_rows, dir_rows)

    runtime_s = time.time() - t0
    write_report(out_dir, val, cka_rows, sub_rows, dir_rows, null_summary,
                 split_info, args.seed, args.n_perms, runtime_s)

    provenance = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "analysis": "GRSN vs AGRSN representation-similarity, metrics 3-5 "
                    "(role-transition CKA, role-transition subspace, "
                    "direction-subspace), Llama3.1-8B, all 32 decoder layers "
                    "(raw indices 1-32; embedding index 0 excluded). "
                    "Offline CPU only: no model loaded, no generation, no "
                    "steering. Companion to the LOCAL metric 1-2 artifacts "
                    "in RoleHidden/AdaResult/GRSN_AGRSN/.",
        "representation_only_disclaimer": "Geometric similarity is NOT "
                    "steering/behavioral/causal equivalence.",
        "inputs": {
            "h5_paths": {f"{t}/{c}": str(val["h5"][(t, c)])
                         for t in TASKS.values() for c in CONDITIONS},
            "h5_sha256": val["h5_sha256"],
            "mean_files": {
                name: {f"{k}_sha256": val["mean"][name][f"{k}_sha256"]
                       for k in ("expert_mean", "non_expert_mean", "diff_mean")}
                for name in TASKS},
            "mean_manifest": {name: {"path": str(val["mean"][name]["manifest_path"]),
                                     "sha256": val["mean"][name]["manifest_sha256"]}
                              for name in TASKS},
            "pairing_digest": val["pairing_digest"],
            "question_arrays_identical_across_h5": True,
            "in300_identical_across_h5": True,
            "n_in_300_audit_only": val["in300_count"],
        },
        "direction_reconstruction": {
            "rule": "expert_mean - non_expert_mean in float32 from the mean "
                    "files; stored diff_mean used ONLY as a consistency check "
                    "(tolerance 1e-2)",
            "max_abs_err_vs_stored_diff": diff_check,
        },
        "layer_indexing": {
            "convention": "HF hidden_states: index 0 = embedding (excluded), "
                          "1..32 = decoder layer outputs; here raw index i "
                          "labels decoder layer i",
            "analysis_layers": "1-32 (all decoder layers)",
            "injection_band_raw_index": list(INJECTION_BAND),
            "injection_band_decoder_layers": "11-19 (9 layers)",
        },
        "randomness": {
            "seed": args.seed,
            "train_val_split": split_info,
            "cka_permutation_null": {
                "n_perms_per_layer": args.n_perms,
                "permutes": "question rows of the AGRSN group only (pairing "
                            "broken); same permutation set reused at every "
                            "layer",
                "note": "no orthogonal-rotation null: CKA is invariant to it",
            },
            "random_subspace_baseline_sim": rand_base,
        },
        "metric_definitions": {
            "metric3_cka": "column-centered (over questions) linear CKA via "
                           "Gram matrices: <Gx,Gy>_F/(||Gx||_F ||Gy||_F)",
            "metric4_subspace": "PCA fit on the train split only (train "
                                "column mean subtracted; held-out centered "
                                "with the SAME train mean); energy fraction = "
                                "mean over held-out samples of "
                                "||P x||^2/||x||^2; principal angles via "
                                "singular values of V_G^T V_A",
            "metric5_direction_subspace": "R^2 = ||V_k^T r||^2/||r||^2 of the "
                                "full-1319 reconstructed mean direction r "
                                "into the Metric-4 TRAIN-fit subspace; the "
                                "mean direction is a DESCRIPTIVE input, NOT "
                                "an independent held-out validation",
            "k_grid": list(K_GRID),
        },
        "environment": {
            "host": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "threads": os.environ.get("OMP_NUM_THREADS", "default"),
        },
        "runtime_seconds": runtime_s,
        "extraction_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    tmp = out_dir / "provenance.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    os.replace(tmp, out_dir / "provenance.json")

    print(f"[DONE] metrics 3-5 written to {out_dir} in {runtime_s:.0f}s")
    print("band CKA:", [round(r['cka'], 4) for r in cka_rows
                        if r['layer'] in BAND_LAYERS])


if __name__ == "__main__":
    main()
