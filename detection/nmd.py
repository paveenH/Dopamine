#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate NMD-style sparse masks over a layer range [start, end),
using pre-computed mean hidden states (diff_char, diff_none).

- Only layers in [start, end) are edited.
- Embedding layer (layer 0) is removed in the saved mask => shape (L-1, H).
- Supports three mask types: nmd / random / diff_random.
"""
import os
import numpy as np
import argparse


def get_nmd_mask(diff_char: np.ndarray, diff_none: np.ndarray, top_k: int, start: int, end: int) -> np.ndarray:
    """
    NMD: per-layer pick top_k by |diff| in [start, end).

    diff_char, diff_none: np.ndarray with shape (1, 1, L, H)
    Returns: mask with shape (L-1, H), removing layer 0 (embedding).
    """
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    L, H = diff.shape
    mask = np.zeros_like(diff)

    for l in range(L):
        if start <= l < end:
            vec = diff[l]
            idxs = np.argsort(np.abs(vec))[-top_k:]
            mask[l, idxs] = vec[idxs]
            
    return mask[1:, :]


def get_random_mask(top_k: int, start: int, end: int, total_layers: int, hidden_dim: int, seed: int = 42) -> np.ndarray:
    """
    RANDOM: random positions in [start, end), values in {-1, +1}.
    Returns: (L-1, H), removing embedding layer 0.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((total_layers, hidden_dim))
    k = max(1, int(top_k))

    for l in range(total_layers):
        if start <= l < end:
            idxs = rng.choice(hidden_dim, k, replace=False)
            mask[l, idxs] = rng.choice([-1.0, 1.0], size=k)

    return mask[1:, :]


def get_diff_random_mask(
    diff_char: np.ndarray, diff_none: np.ndarray, top_k: int, start: int, end: int, seed: int = 42
) -> np.ndarray:
    """
    DIFF-RANDOM: random positions in [start, end), but values taken
    from the true diff (diff_char - diff_none) at those positions.

    Returns: (L-1, H), removing embedding layer 0.
    """
    rng = np.random.default_rng(seed)
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    L, H = diff.shape
    mask = np.zeros_like(diff)

    k = max(1, int(top_k))
    for l in range(L):
        if start <= l < end:
            idxs = rng.choice(H, k, replace=False)
            mask[l, idxs] = diff[l, idxs]

    return mask[1:, :]


def _ortho_gauss_layer(d_sub: np.ndarray, target_norm: float,
                       rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    """One layer's orthogonal-Gaussian sparse values on a k-position support.

    d_sub = dense role-diff Δ_l restricted to those k positions (float64).
    Returns m_sub (k,) s.t.  m_sub·d_sub = 0  and  ||m_sub|| = target_norm.
    Re-draws (not divides) if ||d_sub|| or the post-projection ||m_sub|| is tiny.
    """
    d_sub = np.asarray(d_sub, dtype=np.float64)
    dn2 = float(d_sub @ d_sub)
    if dn2 < 1e-12:
        raise ValueError("||d_sub||^2 too small — degenerate support, resample support")
    for _ in range(max_tries):
        g = rng.standard_normal(d_sub.shape[0]).astype(np.float64)
        m = g - (float(g @ d_sub) / dn2) * d_sub      # project out the Δ direction
        mn = float(np.sqrt(m @ m))
        if mn > 1e-8:
            return m * (target_norm / mn)             # norm-match (preserves ⊥)
    raise RuntimeError("orthogonal component vanished repeatedly — resample support")


def get_ortho_gauss_mask(diff_char: np.ndarray, diff_none: np.ndarray,
                         top_k: int, start: int, end: int, seed: int,
                         on_support: bool) -> np.ndarray:
    """ORTHO-GAUSS: per layer, Gaussian sparse values that are EXACTLY orthogonal
    to that layer's dense role-diff Δ_l and norm-matched to the NMD mask row.

    on_support=True  -> reuse NMD's top-|Δ| positions (same-support null)
    on_support=False -> random positions DISJOINT from NMD's (off-support null)

    Returns (L-1, H), embedding row removed (same convention as the others).
    """
    diff = (diff_char - diff_none).squeeze(0).squeeze(0).astype(np.float64)  # (L, H)
    L, H = diff.shape
    rng = np.random.default_rng(seed)
    mask = np.zeros_like(diff)
    k = max(1, int(top_k))

    for l in range(L):
        if not (start <= l < end):
            continue
        vec = diff[l]
        nmd_idx = np.argsort(np.abs(vec))[-k:]                 # NMD support this layer
        nmd_norm = float(np.sqrt(np.sum(vec[nmd_idx] ** 2)))   # NMD row norm (match target)
        if on_support:
            idx = nmd_idx
        else:
            pool = np.setdiff1d(np.arange(H), nmd_idx, assume_unique=False)
            idx = rng.choice(pool, k, replace=False)           # disjoint from NMD
        d_sub = vec[idx]                                        # dense Δ on these positions
        m_sub = _ortho_gauss_layer(d_sub, nmd_norm, rng)
        mask[l, idx] = m_sub

    return mask[1:, :]


def get_sparse_fv_mask(diff_char: np.ndarray, diff_none: np.ndarray,
                       top_k: int, start: int, end: int) -> np.ndarray:
    """
    Sparse FV: global top neurons across layers [start, end).

    top_k: number of neurons PER LAYER (already computed by caller)
    We convert this into a global-k across the whole block:
        global_k = top_k * num_layers
    """
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    L, H = diff.shape

    # Get block
    block = diff[start:end]                 # shape = (num_layers, H)
    num_layers = end - start

    # Global K = per-layer K × number of layers
    global_k = max(1, int(top_k * num_layers))

    print(f"[Sparse FV] Layer range = [{start}, {end}), layers = {num_layers}")
    print(f"[Sparse FV] Hidden dim = {H}")
    print(f"[Sparse FV] Per-layer top_k = {top_k}")
    print(f"[Sparse FV] Global top_k = {global_k}")

    # Flatten and select global top-K neurons
    flat = np.abs(block).reshape(-1)
    idxs = np.argpartition(-flat, global_k - 1)[:global_k]

    # Convert flat indices → (layer_offset, neuron_id)
    layer_offsets = idxs // H
    neuron_ids = idxs % H

    # Build mask
    mask = np.zeros_like(diff)
    for lo, nid in zip(layer_offsets, neuron_ids):
        real_layer = start + lo
        mask[real_layer, nid] = diff[real_layer, nid]

    final_mask = mask[1:, :]     # remove embedding layer

    print(f"[Sparse FV] Final non-zero neurons = {np.count_nonzero(final_mask)}")

    return final_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate NMD/diff_random/random masks from mean hidden states")
    parser.add_argument("--model", type=str, default="stablelm", help="Model name")
    parser.add_argument("--size", type=str, default="12B", help="Model size label")
    parser.add_argument("--type", type=str, default="non", help="Type: 'non' or 'exp'")
    parser.add_argument("--hs", type=str, default="stablelm", help="Hidden state folder prefix")
    parser.add_argument("--percentage", type=float, default=0.01, help="Percentage (in %) of neurons to keep per layer, e.g. 0.5 for 0.5%")
    parser.add_argument("--start_layer", type=int, default=16, help="Start layer index (inclusive)")
    parser.add_argument("--end_layer", type=int, default=22, help="End layer index (exclusive)")
    parser.add_argument("--logits", action="store_true", help="Use logits variant for HS_MEAN path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for random/diff_random)")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for components (overrides hardcoded path)")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="nmd",
        choices=["nmd", "random", "diff_random", "sparse_fv",
                 "ortho_gauss_same", "ortho_gauss_off"],
        help="Which mask to save: nmd / random / diff_random / sparse_fv / "
             "ortho_gauss_same (⊥ Gaussian on NMD support) / "
             "ortho_gauss_off (⊥ Gaussian off NMD support)",
    )

    args = parser.parse_args()

    # Determine HS mean path
    suffix = f"{args.hs}_{args.type}_logits" if args.logits else f"{args.hs}_{args.type}"
    # Base directory: use --base_dir if provided, otherwise derive from script location
    if args.base_dir:
        BASE_DIR = args.base_dir
    else:
        BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "components")
    HS_MEAN = os.path.join(BASE_DIR, "hidden_states_mean", suffix)
    diff_char = np.load(os.path.join(HS_MEAN, f"diff_mean_{args.size}.npy"))
    diff_none = np.load(os.path.join(HS_MEAN, f"none_diff_mean_{args.size}.npy"))

    # Shapes & top-k
    total_layers, hidden_dim = diff_char.squeeze(0).squeeze(0).shape
    top_k = max(1, int(hidden_dim * args.percentage / 100.0))
    print(f"Hidden size H={hidden_dim}, percentage={args.percentage}% -> top_k per layer = {top_k}")
    print(f"Total layers (INCLUDING embedding layer 0) L={total_layers}")
    print(f"Edit range: [{args.start_layer}, {args.end_layer})")

    # Build mask (returns shape (L-1, H), embedding removed)
    if args.mask_type == "nmd":
        mask = get_nmd_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer)
    elif args.mask_type == "random":
        mask = get_random_mask(top_k, args.start_layer, args.end_layer, total_layers, hidden_dim, seed=args.seed)
    elif args.mask_type == "diff_random":
        mask = get_diff_random_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer, seed=args.seed)
    elif args.mask_type == "sparse_fv":
        mask = get_sparse_fv_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer)
    elif args.mask_type in ("ortho_gauss_same", "ortho_gauss_off"):
        mask = get_ortho_gauss_mask(diff_char, diff_none, top_k,
                                    args.start_layer, args.end_layer, seed=args.seed,
                                    on_support=(args.mask_type == "ortho_gauss_same"))
    else:
        raise ValueError(f"Unknown mask_type: {args.mask_type}")

    print(f"{args.mask_type} Mask shape: {mask.shape}")  # (L-1, H)

    # ---- acceptance guards (ortho-gauss only): per-layer ⊥, norm-match, k nonzeros ----
    if args.mask_type in ("ortho_gauss_same", "ortho_gauss_off"):
        diff_full = (diff_char - diff_none).squeeze(0).squeeze(0).astype(np.float64)  # (L,H)
        nmd_full = get_nmd_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer)
        max_cos = 0.0
        # saved row = dense-diff layer - 1 (embedding row dropped); only the
        # in-range layers were populated, so iterate them directly.
        for l in range(args.start_layer, args.end_layer):
            row = l - 1
            m_l = mask[row].astype(np.float64)
            d_l = diff_full[l]
            nz = np.flatnonzero(m_l)
            assert nz.size == top_k, f"layer {l}: {nz.size} nonzeros, expected {top_k}"
            nmd_nz = np.flatnonzero(nmd_full[row])
            if args.mask_type == "ortho_gauss_same":
                assert np.array_equal(np.sort(nz), np.sort(nmd_nz)), \
                    f"layer {l}: same-support must match NMD support exactly"
            else:
                assert np.intersect1d(nz, nmd_nz).size == 0, \
                    f"layer {l}: off-support must be disjoint from NMD support"
            denom = (np.linalg.norm(m_l) * np.linalg.norm(d_l))
            cos = abs(float(m_l @ d_l) / denom) if denom > 0 else 0.0
            max_cos = max(max_cos, cos)
            nm, nn = np.linalg.norm(m_l), np.linalg.norm(nmd_full[row])
            # relative tolerance: nn ~ 0.2-0.7, and NMD row norm is recomputed from
            # float32 dense diff while target_norm scaled in float64 -> ~float32 eps
            # gap is expected. 1e-3 rel still catches a real mismatch (e.g. the 4x
            # gap a diff_random ¼-norm would show).
            assert abs(nm - nn) < 1e-3 * max(nn, 1e-8), \
                f"layer {l}: norm {nm:.6g} != NMD norm {nn:.6g} (rel {abs(nm-nn)/nn:.2e})"
        print(f"[ortho guard] max |cos(m_l, Δ_l)| over layers = {max_cos:.2e}")
        assert max_cos < 1e-5, f"orthogonality broke on reload dtype: max cos {max_cos:.2e}"

    # Save
    mask_suffix = f"{args.model}_{args.type}_logits" if args.logits else f"{args.model}_{args.type}"
    mask_dir = os.path.join(BASE_DIR, "mask", mask_suffix)
    os.makedirs(mask_dir, exist_ok=True)

    # For seed-driven null masks, append the seed so multiple draws don't overwrite.
    # ortho_gauss_* ALWAYS tag the seed (there is no canonical seed=42 baseline for them).
    seed_tag = ""
    if args.mask_type in ("ortho_gauss_same", "ortho_gauss_off"):
        seed_tag = f"_seed{args.seed}"
    elif args.mask_type in ("random", "diff_random") and args.seed != 42:
        seed_tag = f"_seed{args.seed}"
    mask_name = f"{args.mask_type}_{args.percentage}_{args.start_layer}_{args.end_layer}_{args.size}{seed_tag}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    np.save(mask_path, mask)
    print(f"Saved {args.mask_type} mask → {mask_path}")
