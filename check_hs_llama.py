#!/usr/bin/env python3
"""
Llama3-8B HS acceptance for the MANIFOLD PILOT (Manifold Plan section 1).
READ-ONLY -- opens every HDF5 with mode="r" and writes nothing.

Runs on the SERVER (the H5 live there) with the conda env's `python`.
`python3.10` does not exist there and exits 127 before anything runs.

SCOPE. The primary pilot is the FOUR No-CoT cells alpha = {0, -6, -8, +6}
under run tag phase1b_eot. The same tree also holds -4/-2/+2/+4/+8, CoT 0/-4
and three roles; those are SENSITIVITY cells and are checked only when named
explicitly via --cells. Restricting the default keeps the acceptance verdict
about the cells the pilot actually rests on, and stops a stray unrelated file
from failing a run.

WHY A SEPARATE SCRIPT FROM check_hs_qwen25.py rather than flags on it: that
one is the frozen Qwen acceptance whose verdict is recorded in
ACCEPTANCE_20260824.txt, and its constants (band [16,22), 7 stored layers,
Qwen's 28-layer final index 27, the seven-cell CELLS map) are all
model-specific. Parameterising it would fork the meaning of a frozen artifact.
The CHECKS are the same three, re-derived for Llama's numbers.

FOUR LLAMA-SPECIFIC FACTS, none inherited from the Qwen script:

  band [11,20)  -> utils.decoder_layer_range(11,20) = range(10,19), i.e.
                   decoder layers 10..18, L=9 steered layers. NOT 11..20, and
                   the last steered layer is decoder index 18 -- the manifold
                   pilot's PRIMARY layer. (decoder 10, the first steered layer,
                   is the sensitivity layer: it is the only one whose injection
                   carries no propagated residue from earlier steered layers.)
  10 stored     -> 9 middle + the model's FINAL layer.
  final index 31-> Llama-3.1-8B has 32 decoder layers, so the final layer's
                   model-space index is 31, NOT layer_end-1 (=19). A file
                   storing [10..19] would pass a layer-COUNT check while
                   carrying no final layer at all, so the indices are compared.
  mask          -> nmd_0.5_11_20_8B.npy under llama3_non_logits.

THREE CHECKS.

  1. INTEGRITY (per cell, no mask needed)
     n_samples_done == 300, n_stored_layers == 10, band == [11,20),
     max_new_tokens == 768, ema_alpha == 0.95, steer_alpha/cot/steer_mode
     agreeing with what the FILENAME claims, stored_layer_indices ==
     [10..18, 31], and PER SAMPLE: len(decode_hs) == len(x_decode_proj) ==
     len(ema_decode_proj), prefill_hs non-empty, decode_hs carrying 10 layers,
     question_idx a full 0..299 cover.

     TRUNCATION. Judge a flagged length by DECODE STEPS and the tail, never by
     the length alone: a real cap shows up as a CLUSTER at one value, while a
     lone sample whose tail carries an EOS and whose step count is far below
     max_new_tokens is a coincidence (the Qwen run had exactly one such false
     positive at 1000 chars, which is why 1000 is not probed). The old tracker
     capped `generated` at 4000 chars, so that cap is the one that can bite.

  2. PROJECTION REPRODUCTION (needs the mask)
     Re-project the stored raw HS against the NMD mask and compare against the
     scalars the tracker computed on the fly. This is what says the stored HS
     really are the states the published signal curves were read from.

     THE PROJECTION IS A MEAN OVER LAYERS, NOT A SUM. utils.project_rsn_numpy
     is `np.sum(hs * dirs, axis=-1).mean()`; summing instead is off by exactly
     n_middle (9x here) and fails every healthy file. This calls the shared
     helper rather than reimplementing it.

     Tolerance is RELATIVE and deliberately loose: it is a wrong-mask /
     wrong-band detector (on Qwen those read 3.5 and 38), not a precision
     test. The tracker casts HS to fp16 BEFORE projecting, so the stored
     states ARE the projected states and a healthy probe reads ~0.

  3. AGREEMENT vs the lightweight batch (needs the signal JSON)
     Keyed on question_idx, never on row order.

     *** A RATE, NOT A GATE. *** Qwen measured 1.000 on all seven cells, but
     that is an OBSERVED property of that batch -- this card, this code, that
     length distribution -- not a protocol guarantee. bf16 greedy can diverge
     at one critical token and that single divergence changes the whole chain.
     A cell at 100% and a cell at 85% are BOTH usable: each H5 carries its own
     readouts and the manifold is built on the H5's own trajectory. The rate
     exists so divergence is REPORTED rather than assumed away. Do not add a
     pass/fail threshold here.

WHAT THIS SCRIPT DOES NOT ESTABLISH. Passing means the four cells are
internally consistent and are the states the signal curves came from. It says
nothing about whether the alpha=0 manifold is stable enough to carry the dose
comparison -- that is Manifold Plan section 5, and its failure is a stop
condition regardless of this verdict.

The EMA stored in these files is PREFILL-SEEDED. It is checked for internal
length consistency only and must NOT be read as s_t downstream: recompute s_t
from x_decode with a decode-seeded EMA, per phase1_gain.decode_ema. The
contamination is alpha-DEPENDENT, so a cross-alpha decode comparison built on
the stored series mixes an entry-injection residual into the effect.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py

EXPECTED_N       = 300
EXPECTED_LAYERS  = 10       # 9 middle (decoder 10..18) + final (decoder 31)
EXPECTED_START   = 11
EXPECTED_END     = 20
EXPECTED_MNT     = 768
EXPECTED_EMA     = 0.95
EXPECTED_NUM_LAYERS = 32    # Llama-3.1-8B

# Historical `generated` attr caps worth probing for.
#
# 2000 was REMOVED after a verified false positive on this very batch: nocot
# alpha=0 sample 0148 and nocot_a6 sample 0106 are each exactly 2000 chars,
# but each is the ONLY such sample in its cell (a real cap makes a CLUSTER),
# each ran the full 767 of 768 decode steps, and each tail is a degenerate
# repetition loop ('.####.####.####...' / 'The final answer is $2.00. ####$2.00.'
# repeated). The loop is WHY they are short -- 767 steps of a short repeated
# fragment accumulates ~2000 chars where normal reasoning reaches 3600-3900 --
# so the length is a consequence of the loop plus the max_new_tokens ceiling,
# not of a character cut. Same reasoning that removed 1000 from the Qwen list.
#
# 1000 is likewise NOT probed (the Qwen false positive). 4000 is retained: the
# old tracker really did cap this attr at 4000 chars, and these cells reach
# 3895 -- 105 of headroom, i.e. the cap was about to bite.
ROUND_CAPS = (4000,)

# h5 stem suffix -> (steer_alpha, cot). The four PRIMARY pilot cells.
PRIMARY_CELLS = {
    "nocot_aneg8": (-8.0, False),
    "nocot_aneg6": (-6.0, False),
    "nocot":       (0.0,  False),
    "nocot_a6":    (6.0,  False),
}

# Everything else that legitimately lives in phase1b_eot. Checked only when
# named via --cells; listed so an unrecognised stem is still an error.
SENSITIVITY_CELLS = {
    "nocot_aneg4":     (-4.0, False),
    "nocot_aneg2":     (-2.0, False),
    "nocot_a2":        (2.0,  False),
    "nocot_a4":        (4.0,  False),
    "nocot_a8":        (8.0,  False),
    "cot":             (0.0,  True),
    "cot_aneg4":       (-4.0, True),
    # role cells: unsteered, so steer_alpha 0 / steer_mode "none"
    "nocot_expert":          (0.0, False),
    "nocot_non_expert":      (0.0, False),
    "nocot_primary_teacher": (0.0, False),
}

ALL_CELLS = {**PRIMARY_CELLS, **SENSITIVITY_CELLS}


def cell_key(h5_path: Path, ls: int, le: int) -> str:
    """hs_gsm8k_8B_nocot_aneg6_L11-20.h5 -> nocot_aneg6"""
    body = h5_path.stem[len("hs_"):]
    suffix = f"_L{ls}-{le}"
    if body.endswith(suffix):
        body = body[: -len(suffix)]
    parts = body.split("_", 2)          # strip the leading "<task>_<size>_"
    return parts[2] if len(parts) == 3 else body


def signal_name_for(h5_path: Path, ls: int, le: int, ema_alpha: float) -> str:
    """The lightweight JSON name extract_signal_json.py writes for this H5.

    hs_gsm8k_8B_nocot_aneg6_L11-20.h5
      -> dopamine_signal_gsm8k_8B_nocot_aneg6_ema0.95_L11-20.json

    Derived from the H5 stem exactly as that script does, so the two stay in
    step. An EXACT name, never a glob: '*_nocot_*' also matches nocot_a6 and
    '*_cot_*' also matches cot_aneg4, so a glob silently skips precisely the
    two alpha=0 cells while the footer still claims agreement ran.
    """
    body = h5_path.stem[len("hs_"):]
    suffix = f"_L{ls}-{le}"
    if body.endswith(suffix):
        body = body[: -len(suffix)]
    return f"dopamine_signal_{body}_ema{ema_alpha}_L{ls}-{le}.json"


def attr(meta, key, default=None):
    v = meta.attrs.get(key, default)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def check_integrity(h5_path: Path, ls: int, le: int, expect: dict):
    """Step 1. Returns (ok, list_of_problems, info_dict)."""
    bad = []
    info = {}
    key = cell_key(h5_path, ls, le)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        n_done   = int(attr(meta, "n_samples_done", -1))
        n_plan   = attr(meta, "n_samples_planned", None)
        n_layers = int(attr(meta, "n_stored_layers", -1))
        m_ls     = int(attr(meta, "layer_start", -1))
        m_le     = int(attr(meta, "layer_end", -1))
        mnt      = int(attr(meta, "max_new_tokens", -1))
        ema_a    = float(attr(meta, "ema_alpha", -1.0))
        s_alpha  = float(attr(meta, "steer_alpha", float("nan")))
        s_mode   = attr(meta, "steer_mode", None)
        cot      = bool(attr(meta, "cot", False))
        acc      = float(attr(meta, "accuracy", float("nan")))
        mask_nm  = attr(meta, "sanity_mask", "")
        n_model  = int(attr(meta, "num_layers", -1))
        stored   = np.asarray(meta.attrs.get("stored_layer_indices", []))

        info.update(cell=key, n=n_done, layers=n_layers, band=(m_ls, m_le),
                    mnt=mnt, alpha=s_alpha, cot=cot, acc=acc, mask=mask_nm,
                    stored_layers=stored.tolist(), steer_mode=s_mode,
                    num_layers=n_model)

        if n_done   != EXPECTED_N:      bad.append(f"n_samples_done={n_done} != {EXPECTED_N}")
        if n_layers != EXPECTED_LAYERS: bad.append(f"n_stored_layers={n_layers} != {EXPECTED_LAYERS}")
        if (m_ls, m_le) != (ls, le):
            bad.append(f"band=[{m_ls},{m_le}) != [{ls},{le})")
        if mnt != EXPECTED_MNT: bad.append(f"max_new_tokens={mnt} != {EXPECTED_MNT}")
        if abs(ema_a - EXPECTED_EMA) > 1e-9: bad.append(f"ema_alpha={ema_a} != {EXPECTED_EMA}")

        # A truncated cell is the failure mode that most resembles a healthy
        # one. n_samples_planned MISSING is itself fatal: checking it only when
        # present is fail-OPEN on exactly the condition the check exists for.
        if n_plan is None:
            bad.append("n_samples_planned missing -- cannot tell a complete cell "
                       "from an interrupted one")
        elif int(n_plan) != n_done:
            bad.append(f"n_samples_done={n_done} < n_samples_planned={int(n_plan)} "
                       f"-- interrupted collection")

        # The filename claims an alpha/cot; the metadata must agree, or a cell
        # was written to the wrong path and every downstream label is wrong.
        if key in expect:
            exp_a, exp_cot = expect[key]
            if abs(s_alpha - exp_a) > 1e-9:
                bad.append(f"steer_alpha={s_alpha} but filename says {exp_a}")
            if cot != exp_cot:
                bad.append(f"cot={cot} but filename says {exp_cot}")
            exp_mode = "prefill_only" if exp_a != 0.0 else "none"
            # .get(..., default) would be fail-OPEN here: a steered cell whose
            # steer_mode attr is absent would be recorded as unsteered.
            if s_mode is None:
                bad.append("steer_mode attr missing")
            elif s_mode != exp_mode:
                bad.append(f"steer_mode={s_mode!r} != {exp_mode!r}")
        else:
            bad.append(f"unrecognised cell key {key!r}")

        # The stored set is the middle band plus the model's FINAL layer, whose
        # model-space index is num_layers-1 (31 for Llama's 32), NOT layer_end-1
        # (19). Compare indices, not just the count.
        n_middle = m_le - m_ls
        if n_model != EXPECTED_NUM_LAYERS:
            bad.append(f"num_layers={n_model} != {EXPECTED_NUM_LAYERS}")
        exp_stored = list(range(m_ls - 1, m_ls - 1 + n_middle)) + [n_model - 1]
        if n_model > 0 and stored.tolist() != exp_stored:
            bad.append(f"stored_layer_indices={stored.tolist()} != {exp_stored}")
        fin_stored = int(attr(meta, "final_layer_idx_stored", -1))
        if fin_stored != n_middle:
            bad.append(f"final_layer_idx_stored={fin_stored} != {n_middle}")

        grp = f["samples"]
        keys = sorted(grp.keys())
        if len(keys) != n_done:
            bad.append(f"{len(keys)} sample groups but n_samples_done={n_done}")

        # question_idx must cover 0..n-1 exactly once. Checked here rather than
        # only in the agreement step, or a permuted / gapped cell looks clean
        # whenever --no_agreement is passed. A GAP means a question is missing
        # while n still looks plausible, silently misaligning every pairing.
        all_idx = sorted(int(grp[k].attrs.get("question_idx", -1)) for k in keys)
        if all_idx != list(range(len(keys))):
            missing = sorted(set(range(len(keys))) - set(all_idx))
            dupes = len(all_idx) - len(set(all_idx))
            bad.append(f"question_idx is not a full 0..{len(keys)-1} cover "
                       f"({dupes} duplicate(s), {len(missing)} missing"
                       + (f", first missing {missing[:3]}" if missing else "") + ")")

        max_gen = 0
        at_cap = []
        len_mismatch = 0
        empty_prefill = 0
        min_decode = None
        for k in keys:
            g = grp[k]
            gen = g.attrs.get("generated", "")
            if isinstance(gen, bytes):
                gen = gen.decode("utf-8", "replace")
            L = len(gen)
            max_gen = max(max_gen, L)
            if L in ROUND_CAPS:
                at_cap.append((k, L, int(g["decode_hs"].shape[0])))

            T_hs  = g["decode_hs"].shape[0]
            T_x   = g["x_decode_proj"].shape[0]
            T_ema = g["ema_decode_proj"].shape[0]
            if not (T_hs == T_x == T_ema):
                len_mismatch += 1
            min_decode = T_hs if min_decode is None else min(min_decode, T_hs)
            if g["prefill_hs"].shape[0] == 0:
                empty_prefill += 1
            if g["decode_hs"].shape[1] != n_middle + 1:
                bad.append(f"{k}: decode_hs has {g['decode_hs'].shape[1]} layers, "
                           f"expected {n_middle + 1}")
                break

        info["max_gen_chars"] = max_gen
        info["min_decode_steps"] = min_decode
        if at_cap:
            # (key, chars, decode_steps) so the reader can apply the cluster /
            # EOS-tail judgement the docstring describes.
            bad.append(f"{len(at_cap)} generation(s) sit EXACTLY at a round cap "
                       f"{ROUND_CAPS} -- inspect the tail and decode steps before "
                       f"calling it truncation: {at_cap[:3]}")
        if len_mismatch:
            bad.append(f"{len_mismatch} sample(s): decode_hs / x_decode_proj / "
                       f"ema_decode_proj lengths disagree")
        if empty_prefill:
            bad.append(f"{empty_prefill} sample(s) have empty prefill_hs")

    return (len(bad) == 0), bad, info


def check_projection(h5_path: Path, mask_path: str, ls: int, le: int,
                     n_probe: int, rtol: float):
    """Step 2a. Re-project raw HS and compare to the tracker's on-the-fly scalars."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import utils
    mask_mid = utils.mask_slice_for(np.load(mask_path), ls, le).astype(np.float32)
    n_middle = le - ls
    if mask_mid.shape[0] != n_middle:
        return False, [f"mask slice has {mask_mid.shape[0]} rows, expected {n_middle}"], {}

    bad = []
    worst_pre = worst_dec = 0.0
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        if "final_layer_idx_stored" not in meta.attrs:
            return False, ["no final_layer_idx_stored: not a selective HDF5"], {}
        msl = slice(0, n_middle)        # selective layout: middle at [0, n_middle)

        grp = f["samples"]
        keys = sorted(grp.keys())
        probe = keys if n_probe <= 0 else keys[:: max(1, len(keys) // n_probe)][:n_probe]

        for k in probe:
            g = grp[k]
            # prefill: LAST prompt token, middle layers only -- matching the
            # tracker, which projects layers_LH[:, -1, :] cast to fp32.
            pre_hs = g["prefill_hs"][-1][msl].astype(np.float32)     # (n_middle, H)
            got_pre = utils.project_rsn_numpy(pre_hs, mask_mid)
            exp_pre = float(g["x_prefill_proj"][()])
            scale = max(abs(exp_pre), 1.0)
            worst_pre = max(worst_pre, abs(got_pre - exp_pre) / scale)

            # decode: same formula PER TOKEN.
            dec_hs = g["decode_hs"][:, msl, :].astype(np.float32)    # (T, n_middle, H)
            exp_dec = g["x_decode_proj"][:].astype(np.float32)
            got_dec = np.array([utils.project_rsn_numpy(dec_hs[t], mask_mid)
                                for t in range(dec_hs.shape[0])], dtype=np.float32)
            sc = np.maximum(np.abs(exp_dec), 1.0)
            worst_dec = max(worst_dec, float(np.max(np.abs(got_dec - exp_dec) / sc)))

    if worst_pre > rtol:
        bad.append(f"prefill projection rel-err {worst_pre:.3e} > rtol {rtol:.0e}")
    if worst_dec > rtol:
        bad.append(f"decode projection rel-err {worst_dec:.3e} > rtol {rtol:.0e}")
    return (len(bad) == 0), bad, {"worst_prefill_relerr": worst_pre,
                                  "worst_decode_relerr": worst_dec,
                                  "n_probed": len(probe)}


_Q_CACHE = {}
def _h5_question(h5_path: Path, key: str) -> str:
    ck = (str(h5_path), key)
    if ck not in _Q_CACHE:
        with h5py.File(h5_path, "r") as f:
            q = f["samples"][key].attrs.get("question", "")
        _Q_CACHE[ck] = q.decode("utf-8", "replace") if isinstance(q, bytes) else q
    return _Q_CACHE[ck]


def check_agreement(h5_path: Path, signal_json: Path, ls: int, le: int):
    """Step 2b. Per-question agreement vs the lightweight cell. A RATE, not a gate."""
    with open(signal_json) as fh:
        light = json.load(fh)["data"]

    with h5py.File(h5_path, "r") as f:
        grp = f["samples"]
        keys = sorted(grp.keys())
        idxs, gens, cors = [], [], []
        for k in keys:
            g = grp[k]
            idxs.append(int(g.attrs.get("question_idx", -1)))
            gen = g.attrs.get("generated", "")
            gens.append(gen.decode("utf-8", "replace") if isinstance(gen, bytes) else gen)
            cors.append(bool(int(g.attrs.get("correct", 0))))

    out = {"n_h5": len(keys), "n_light": len(light)}
    if len(light) != len(keys):
        out["error"] = f"row counts differ: H5 {len(keys)} vs lightweight {len(light)}"
        return out

    # The lightweight JSON carries no question_idx, so its rows can only be
    # addressed positionally -- which is legitimate ONLY if the H5 side is the
    # identity permutation. Refuse rather than report a wrong rate.
    order_ok = idxs == list(range(len(idxs)))
    out["question_order_identity"] = order_ok
    if not order_ok:
        out["error"] = ("H5 question_idx is not the identity permutation, so the "
                        "lightweight rows (which carry no idx) cannot be paired "
                        "positionally. Refusing to report a rate.")
        return out

    # Question TEXT is the real order check: question_idx being the identity
    # only says the H5 rows are in order, not that the two batches drew the
    # same 300 questions.
    q_match = sum(1 for i, r in enumerate(light)
                  if r.get("question", "") == _h5_question(h5_path, keys[i]))
    gen_match = sum(1 for i, r in enumerate(light) if r.get("generated", "") == gens[i])
    cor_match = sum(1 for i, r in enumerate(light) if bool(r.get("correct", False)) == cors[i])

    n = len(keys)
    out.update(question_match=q_match / n,
               generated_match=gen_match / n,
               correct_match=cor_match / n,
               h5_accuracy=100.0 * sum(cors) / n,
               light_accuracy=100.0 * sum(1 for r in light if r.get("correct")) / n)
    return out


def main():
    global EXPECTED_N
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_dir", default="/data1/paveen/Dopamine/components/"
                                       "hidden_states/gsm8k/phase1b_eot")
    p.add_argument("--mask_path", default="/data1/paveen/Dopamine/components/mask/"
                                          "llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--layer_start", type=int, default=EXPECTED_START)
    p.add_argument("--layer_end",   type=int, default=EXPECTED_END)
    p.add_argument("--signal_dir",
                   default="/data1/paveen/Dopamine/components/llama3/signal/phase1b_eot",
                   help="Directory of the lightweight dopamine_signal_*.json cells.")
    p.add_argument("--no_agreement", action="store_true",
                   help="Skip step 2b. The rate is a recorded fact, not a "
                        "precondition for building the manifold.")
    p.add_argument("--cells", default="primary",
                   help="'primary' (the four pilot cells, default), 'all', or a "
                        "comma-separated list of cell keys such as "
                        "'nocot,nocot_aneg6'. Sensitivity cells are checked only "
                        "when named.")
    p.add_argument("--ema_alpha", type=float, default=EXPECTED_EMA,
                   help="Only used to build the expected lightweight JSON filename.")
    p.add_argument("--n_probe", type=int, default=8,
                   help="Samples per cell for the projection check (0 = all; all is "
                        "minutes per cell because it reads the full decode_hs). "
                        "Run 0 once before freezing.")
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="Relative tolerance for the projection check. A wrong-mask / "
                        "wrong-band DETECTOR, not a precision test -- a healthy probe "
                        "reads ~0 because the tracker stores the fp16 states it "
                        "projected.")
    p.add_argument("--skip_projection", action="store_true")
    p.add_argument("--expect_n", type=int, default=EXPECTED_N,
                   help=argparse.SUPPRESS)   # test fixtures only; never on real data
    args = p.parse_args()
    EXPECTED_N = args.expect_n

    if args.cells == "primary":
        want_cells = dict(PRIMARY_CELLS)
    elif args.cells == "all":
        want_cells = dict(ALL_CELLS)
    else:
        names = [c.strip() for c in args.cells.split(",") if c.strip()]
        unknown = [c for c in names if c not in ALL_CELLS]
        if unknown:
            raise SystemExit(f"[x] unknown cell key(s): {unknown}\n"
                             f"    known: {sorted(ALL_CELLS)}")
        want_cells = {c: ALL_CELLS[c] for c in names}

    h5_dir = Path(args.h5_dir)
    all_files = sorted(h5_dir.glob("hs_*.h5"))
    if not all_files:
        raise SystemExit(f"[x] no hs_*.h5 in {h5_dir}")
    files = [f for f in all_files
             if cell_key(f, args.layer_start, args.layer_end) in want_cells]

    # A stem that matches no known cell is FILTERED OUT by the line above, not
    # reported -- so a mis-renamed H5 would surface only as "requested cell
    # missing", pointing at the wrong thing. Name it here. Not an error: the
    # tree legitimately holds cells outside ALL_CELLS' remit only if someone
    # adds one, and refusing to run would make this checker a gate on an
    # unrelated file's existence.
    unregistered = sorted({cell_key(f, args.layer_start, args.layer_end)
                           for f in all_files} - set(ALL_CELLS))
    if unregistered:
        print(f"[!] {len(unregistered)} file(s) in this tree have an unregistered "
              f"cell key (mis-named, or a new cell needing a CELLS entry): "
              f"{unregistered}\n")

    print(f"H5 dir : {h5_dir}")
    print(f"mask   : {args.mask_path}")
    print(f"band   : [{args.layer_start},{args.layer_end}) -> decoder layers "
          f"{args.layer_start-1}..{args.layer_end-2} (L={args.layer_end-args.layer_start})"
          f"; PRIMARY layer = decoder {args.layer_end-2}, sensitivity = decoder "
          f"{args.layer_start-1}")
    print(f"cells  : {len(files)} matched of {len(all_files)} present, "
          f"{len(want_cells)} requested ({args.cells})\n")

    seen = set()
    failures = 0

    for h5 in files:
        key = cell_key(h5, args.layer_start, args.layer_end)
        seen.add(key)
        size_gb = h5.stat().st_size / 1e9
        print(f"=== {h5.name}  ({size_gb:.1f} GB)")

        ok, bad, info = check_integrity(h5, args.layer_start, args.layer_end, want_cells)
        print(f"  [1] integrity  : {'OK' if ok else 'FAIL'}   "
              f"n={info.get('n')} layers={info.get('layers')} "
              f"band={info.get('band')} alpha={info.get('alpha')} "
              f"cot={info.get('cot')} acc={info.get('acc')}%")
        print(f"      stored_layer_indices={info.get('stored_layers')}  "
              f"max_gen={info.get('max_gen_chars')} chars  "
              f"min_decode={info.get('min_decode_steps')} steps  "
              f"mask={info.get('mask')}")
        for b in bad:
            print(f"      [x] {b}")
        failures += 0 if ok else 1

        if not args.skip_projection:
            pok, pbad, pinfo = check_projection(h5, args.mask_path, args.layer_start,
                                                args.layer_end, args.n_probe, args.rtol)
            print(f"  [2a] projection: {'OK' if pok else 'FAIL'}   "
                  f"probed={pinfo.get('n_probed')} "
                  f"worst rel-err prefill={pinfo.get('worst_prefill_relerr', float('nan')):.2e} "
                  f"decode={pinfo.get('worst_decode_relerr', float('nan')):.2e}")
            for b in pbad:
                print(f"      [x] {b}")
            failures += 0 if pok else 1

        if args.signal_dir and not args.no_agreement:
            want = signal_name_for(h5, args.layer_start, args.layer_end, args.ema_alpha)
            cand = Path(args.signal_dir) / want
            if not cand.exists():
                print(f"  [2b] agreement : ERROR -- expected lightweight cell not found:")
                print(f"      {cand}")
                failures += 1
            else:
                a = check_agreement(h5, cand, args.layer_start, args.layer_end)
                if "error" in a:
                    print(f"  [2b] agreement : ERROR -- {a['error']}")
                    failures += 1
                else:
                    print(f"  [2b] agreement : question={a['question_match']:.3f}  "
                          f"generated={a['generated_match']:.3f}  "
                          f"correct={a['correct_match']:.3f}   "
                          f"(acc H5 {a['h5_accuracy']:.2f}% vs light "
                          f"{a['light_accuracy']:.2f}%)")
                    print(f"      RATE, NOT A GATE: a same-protocol re-run may diverge "
                          f"at one token; each H5 is analysed on its own trajectory.")
        print()

    missing = set(want_cells) - seen
    if missing:
        print(f"[!] {len(missing)} requested cell(s) absent: {sorted(missing)}")

    if failures:
        print(f"\n[x] {failures} check(s) FAILED -- do not start manifold analysis.")
        raise SystemExit(1)
    if missing:
        print(f"\n[!] all present cells pass, but the requested set is INCOMPLETE.")
        raise SystemExit(1)
    ran_agreement = bool(args.signal_dir) and not args.no_agreement
    print(f"\n[ok] all {len(seen)} requested cell(s) pass integrity + projection"
          + (" + agreement" if ran_agreement else " (agreement SKIPPED)"))
    print("     This says the cells are internally consistent and are the states")
    print("     the signal curves came from. It says NOTHING about whether the")
    print("     alpha=0 manifold is stable enough to carry a dose comparison --")
    print("     that is Manifold Plan section 5, whose failure is a stop condition.")


if __name__ == "__main__":
    main()
