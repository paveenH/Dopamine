#!/usr/bin/env python3
"""
Qwen2.5 HS backfill acceptance: integrity (step 1) + projection reproduction
and lightweight-batch agreement (step 2). READ-ONLY — opens every HDF5 with
mode="r" and writes nothing.

Runs on the SERVER (the H5 live there; ~24-30 GB for the seven cells) with the
conda env's `python`. `python3.10` does not exist there and exits 127.

WHY THIS IS A SEPARATE SCRIPT rather than flags on extract_signal_json.py:
that script is the frozen Llama extraction path and its out_meta deliberately
drops steer_alpha / n_samples_done / stored_layer_indices / question_idx. Those
are exactly the fields these checks read, and the seven Qwen cells differ ONLY
by alpha and CoT — a checker that cannot see steer_alpha cannot tell them apart.

THREE CHECKS, in the order the user pre-registered:

  1. INTEGRITY (per cell, no mask needed)
     n_samples_done == 300, n_stored_layers == 7 (6 middle + final),
     layer band == [16, 22), max_new_tokens == 768, ema_alpha == 0.95,
     steer_alpha/cot match what the filename claims, and PER SAMPLE:
     len(decode_hs) == len(x_decode_proj) == len(ema_decode_proj),
     prefill_hs non-empty, `generated` untruncated.

     "Untruncated" is checked as: no sample sits exactly at a round cap
     (4000/2000/1000) AND the longest generation is reported so a future
     reader can see the headroom. The old tracker capped attrs at 4000 chars
     and the longest Qwen signal generation is 3895 -- 105 of headroom, i.e.
     the cap was about to bite. A hard equality test is the only thing that
     distinguishes "long" from "cut".

  2. PROJECTION REPRODUCTION (needs the mask)
     Re-project the stored raw HS against the NMD mask and compare against the
     scalars the tracker computed on the fly. This is what says the stored HS
     really are the states the signal curves were read from.

     THE PROJECTION IS A MEAN OVER LAYERS, NOT A SUM. utils.project_rsn_numpy
     is `np.sum(hs * dirs, axis=-1).mean()`, which is what the tracker calls;
     summing instead is off by exactly n_middle (6x here) and fails every
     healthy file. This calls the shared helper rather than reimplementing it.

     The tracker casts HS to fp16 BEFORE projecting (track_hidden_states.py:206
     casts, 223/227 cast that fp16 back to fp32), so the stored states ARE the
     projected states and agreement should be near-exact -- the residual is
     fp32 summation order alone. Relative error is still the right judgement
     because the projection's magnitude spans two orders across alpha.

  3. AGREEMENT vs the lightweight batch (needs the local signal JSON)
     Keyed on question_idx, NEVER on row order. Reports the rate for
     generated-text / correct / question-order separately.

     *** A RATE, NOT A GATE. *** This is a same-protocol representative
     re-run, not a replay: bf16 greedy can diverge at one critical token and
     that single divergence changes the whole chain. A cell at 100% and a cell
     at 85% are BOTH usable -- each H5 carries its own readouts and is analysed
     on its own trajectory. The rate exists so the divergence is REPORTED
     rather than silently assumed away. Do not add a pass/fail threshold here.

The EMA stored in these files is PREFILL-SEEDED (track_hidden_states seeds its
running EMA with x_prefill). It is checked for internal consistency only and
must NOT be read as s_t downstream -- recompute s_t from x_decode with a
decode-seeded EMA. The contamination is alpha-DEPENDENT (|stored - decode-
seeded| at t=20 is 0.17 at alpha=0 but 97.18 at alpha=+12), so a cross-alpha
decode comparison built on the stored series mixes an entry-injection residual
into the effect.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py

EXPECTED_N          = 300
EXPECTED_LAYERS     = 7        # 6 middle (16..21) + final
EXPECTED_START      = 16
EXPECTED_END        = 22
EXPECTED_MNT        = 768
EXPECTED_EMA        = 0.95
ROUND_CAPS          = (1000, 2000, 4000)   # historical attr caps

# Seven cells: h5 stem suffix -> (steer_alpha, cot)
CELLS = {
    "nocot_aneg8": (-8.0, False),
    "nocot":       (0.0,  False),
    "nocot_a6":    (6.0,  False),
    "nocot_a8":    (8.0,  False),
    "nocot_a12":   (12.0, False),
    "cot":         (0.0,  True),
    "cot_a6":      (6.0,  True),
}


def cell_key(h5_path: Path, ls: int, le: int) -> str:
    """hs_gsm8k_7B_nocot_a12_L16-22.h5 -> nocot_a12"""
    body = h5_path.stem[len("hs_"):]
    suffix = f"_L{ls}-{le}"
    if body.endswith(suffix):
        body = body[: -len(suffix)]
    # strip the leading "<task>_<size>_"
    parts = body.split("_", 2)
    return parts[2] if len(parts) == 3 else body


def signal_name_for(h5_path: Path, ls: int, le: int, ema_alpha: float) -> str:
    """The lightweight JSON name extract_signal_json.py writes for this H5.

    hs_gsm8k_7B_nocot_a12_L16-22.h5
      -> dopamine_signal_gsm8k_7B_nocot_a12_ema0.95_L16-22.json

    Derived from the H5 stem exactly as that script does, so the two stay in
    step. Reconstructing it here rather than globbing is what keeps the two
    alpha=0 cells from being silently skipped.
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


def check_integrity(h5_path: Path, ls: int, le: int):
    """Step 1. Returns (ok, list_of_problems, info_dict)."""
    bad = []
    info = {}
    key = cell_key(h5_path, ls, le)

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        n_done   = int(attr(meta, "n_samples_done", -1))
        n_layers = int(attr(meta, "n_stored_layers", -1))
        m_ls     = int(attr(meta, "layer_start", -1))
        m_le     = int(attr(meta, "layer_end", -1))
        mnt      = int(attr(meta, "max_new_tokens", -1))
        ema_a    = float(attr(meta, "ema_alpha", -1.0))
        s_alpha  = float(attr(meta, "steer_alpha", float("nan")))
        s_mode   = attr(meta, "steer_mode", "")
        cot      = bool(attr(meta, "cot", False))
        acc      = float(attr(meta, "accuracy", float("nan")))
        mask_nm  = attr(meta, "sanity_mask", "")
        stored   = np.asarray(meta.attrs.get("stored_layer_indices", []))

        info.update(cell=key, n=n_done, layers=n_layers, band=(m_ls, m_le),
                    mnt=mnt, alpha=s_alpha, cot=cot, acc=acc, mask=mask_nm,
                    stored_layers=stored.tolist(), steer_mode=s_mode)

        if n_done   != EXPECTED_N:      bad.append(f"n_samples_done={n_done} != {EXPECTED_N}")
        if n_layers != EXPECTED_LAYERS: bad.append(f"n_stored_layers={n_layers} != {EXPECTED_LAYERS}")
        if (m_ls, m_le) != (EXPECTED_START, EXPECTED_END):
            bad.append(f"band=[{m_ls},{m_le}) != [{EXPECTED_START},{EXPECTED_END})")
        if mnt   != EXPECTED_MNT: bad.append(f"max_new_tokens={mnt} != {EXPECTED_MNT}")
        if abs(ema_a - EXPECTED_EMA) > 1e-9: bad.append(f"ema_alpha={ema_a} != {EXPECTED_EMA}")

        # The filename claims an alpha/cot; the metadata must agree, or a cell
        # was written to the wrong path and every downstream label is wrong.
        if key in CELLS:
            exp_a, exp_cot = CELLS[key]
            if not (np.isnan(s_alpha) and np.isnan(exp_a)) and abs(s_alpha - exp_a) > 1e-9:
                bad.append(f"steer_alpha={s_alpha} but filename says {exp_a}")
            if cot != exp_cot:
                bad.append(f"cot={cot} but filename says {exp_cot}")
            exp_mode = "prefill_only" if exp_a != 0.0 else "none"
            if s_mode != exp_mode:
                bad.append(f"steer_mode={s_mode!r} != {exp_mode!r}")
        else:
            bad.append(f"unrecognised cell key {key!r} (not one of the seven)")

        # The stored set is the middle band plus the model's FINAL layer, whose
        # model-space index is num_layers-1 (27 for Qwen's 28), NOT layer_end-1.
        # A file storing [15..21] would look plausible on a layer COUNT check
        # while carrying no final layer at all, so compare the indices.
        n_middle = m_le - m_ls
        n_model  = int(attr(meta, "num_layers", -1))
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

        # question_idx must cover 0..n-1 exactly once, checked here rather than
        # only in the agreement step -- otherwise a permuted or gapped cell looks
        # clean whenever --signal_dir is omitted.
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
        for k in keys:
            g = grp[k]
            gen = g.attrs.get("generated", "")
            if isinstance(gen, bytes):
                gen = gen.decode("utf-8", "replace")
            L = len(gen)
            max_gen = max(max_gen, L)
            if L in ROUND_CAPS:
                at_cap.append((k, L))

            T_hs  = g["decode_hs"].shape[0]
            T_x   = g["x_decode_proj"].shape[0]
            T_ema = g["ema_decode_proj"].shape[0]
            if not (T_hs == T_x == T_ema):
                len_mismatch += 1
            if g["prefill_hs"].shape[0] == 0:
                empty_prefill += 1
            # stored HS must carry middle + final
            if g["decode_hs"].shape[1] != n_middle + 1:
                bad.append(f"{k}: decode_hs has {g['decode_hs'].shape[1]} layers, "
                           f"expected {n_middle + 1}")
                break

        info["max_gen_chars"] = max_gen
        if at_cap:
            bad.append(f"{len(at_cap)} generation(s) sit EXACTLY at a round cap "
                       f"{ROUND_CAPS} -- likely truncated: {at_cap[:3]}")
        if len_mismatch:
            bad.append(f"{len_mismatch} sample(s): decode_hs / x_decode_proj / "
                       f"ema_decode_proj lengths disagree")
        if empty_prefill:
            bad.append(f"{empty_prefill} sample(s) have empty prefill_hs")

    return (len(bad) == 0), bad, info


def check_projection(h5_path: Path, mask_path: str, ls: int, le: int,
                     n_probe: int, rtol: float):
    """Step 2a. Re-project raw HS and compare to the tracker's on-the-fly scalars.

    fp16 storage vs fp32 projection means exact equality is NOT expected; we
    judge relative error against the projection's own magnitude.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import utils
    mask_mid = utils.mask_slice_for(np.load(mask_path), ls, le).astype(np.float32)
    # The projection is utils.project_rsn_numpy -- a per-layer dot product then
    # a MEAN over layers, matching track_hidden_states._project_middle_last.
    # Summing instead is off by exactly n_middle (6x here) and would fail every
    # healthy file. Call the shared helper rather than reimplementing it.
    n_middle = le - ls
    if mask_mid.shape[0] != n_middle:
        return False, [f"mask slice has {mask_mid.shape[0]} rows, expected {n_middle}"], {}

    bad = []
    worst_pre = worst_dec = 0.0
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        if "final_layer_idx_stored" not in meta.attrs:
            return False, ["no final_layer_idx_stored: not a selective HDF5"], {}
        # selective layout: middle layers live at storage [0, n_middle)
        msl = slice(0, n_middle)

        grp = f["samples"]
        keys = sorted(grp.keys())
        probe = keys if n_probe <= 0 else keys[:: max(1, len(keys) // n_probe)][:n_probe]

        for k in probe:
            g = grp[k]
            # prefill: LAST prompt token, middle layers only (matches the tracker,
            # which projects layers_LH[:, -1, :] cast to fp32).
            pre_hs = g["prefill_hs"][-1][msl].astype(np.float32)   # (n_middle, H)
            got_pre = utils.project_rsn_numpy(pre_hs, mask_mid)
            exp_pre = float(g["x_prefill_proj"][()])
            scale = max(abs(exp_pre), 1.0)
            worst_pre = max(worst_pre, abs(got_pre - exp_pre) / scale)

            # decode: same formula PER TOKEN. Vectorising the mean over layers is
            # identical to calling the helper in a loop, but the loop keeps the
            # single definition visible and T is small.
            dec_hs = g["decode_hs"][:, msl, :].astype(np.float32)  # (T, n_middle, H)
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


def check_agreement(h5_path: Path, signal_json: Path, ls: int, le: int):
    """Step 2b. Per-question agreement vs the lightweight cell. A RATE, not a gate.

    Keyed on question_idx. The lightweight JSON does not store question_idx, so
    its rows are addressed by position -- which is exactly why the H5 side's
    question_idx is checked to be the identity permutation first. If it is not,
    positional pairing is refused rather than silently producing a wrong rate.
    """
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

    order_ok = (idxs == sorted(idxs)) and idxs == list(range(len(idxs)))
    out["question_order_identity"] = order_ok
    if not order_ok:
        out["error"] = ("H5 question_idx is not the identity permutation, so the "
                        "lightweight rows (which carry no idx) cannot be paired "
                        "positionally. Refusing to report a rate.")
        return out

    # Question TEXT is the real order check -- question_idx being the identity
    # only says the H5 rows are in order, not that the two batches drew the
    # same 300 questions.
    q_match = sum(1 for i, r in enumerate(light) if r.get("question", "") ==
                  _h5_question(h5_path, keys[i]))
    gen_match = sum(1 for i, r in enumerate(light) if r.get("generated", "") == gens[i])
    cor_match = sum(1 for i, r in enumerate(light) if bool(r.get("correct", False)) == cors[i])

    n = len(keys)
    out.update(question_match=q_match / n,
               generated_match=gen_match / n,
               correct_match=cor_match / n,
               h5_accuracy=100.0 * sum(cors) / n,
               light_accuracy=100.0 * sum(1 for r in light if r.get("correct")) / n)
    return out


_Q_CACHE = {}
def _h5_question(h5_path: Path, key: str) -> str:
    ck = (str(h5_path), key)
    if ck not in _Q_CACHE:
        with h5py.File(h5_path, "r") as f:
            q = f["samples"][key].attrs.get("question", "")
        _Q_CACHE[ck] = q.decode("utf-8", "replace") if isinstance(q, bytes) else q
    return _Q_CACHE[ck]


def main():
    global EXPECTED_N
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_dir", default="/data1/paveen/Dopamine/components/"
                                       "hidden_states/gsm8k/qwen25_signal_v1")
    p.add_argument("--mask_path", default="/data1/paveen/Dopamine/components/mask/"
                                          "qwen2.5_non_logits/nmd_0.5_16_22_7B.npy")
    p.add_argument("--layer_start", type=int, default=EXPECTED_START)
    p.add_argument("--layer_end",   type=int, default=EXPECTED_END)
    p.add_argument("--signal_dir",
                   default="/data1/paveen/Dopamine/components/signal",
                   help="Directory of the lightweight dopamine_signal_*.json cells "
                        "(upload the 7 matching cells from the local analysis box). "
                        "Pass --no_agreement to skip step 2b instead.")
    p.add_argument("--no_agreement", action="store_true",
                   help="Skip the agreement check. Use while the lightweight cells "
                        "have not been uploaded yet -- the rate is a recorded fact, "
                        "not a precondition for the geometry analysis.")
    p.add_argument("--ema_alpha", type=float, default=EXPECTED_EMA,
                   help="Only used to build the expected lightweight JSON filename "
                        "(extract_signal_json.py puts it in the name).")
    p.add_argument("--n_probe", type=int, default=8,
                   help="Samples per cell for the projection check (0 = all; all is "
                        "minutes per cell because it reads the full decode_hs).")
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="Relative tolerance for the projection check. The tracker "
                        "casts HS to fp16 BEFORE projecting (it stores the same "
                        "values it projected), so re-reading the H5 should agree "
                        "to fp32 summation order -- expect « 1e-2. The tolerance "
                        "is loose on purpose: it is a wrong-mask/wrong-band "
                        "detector (those read 3.5 and 38), not a precision test.")
    p.add_argument("--skip_projection", action="store_true")
    p.add_argument("--expect_n", type=int, default=EXPECTED_N,
                   help=argparse.SUPPRESS)   # test fixtures only; never on real data
    args = p.parse_args()
    EXPECTED_N = args.expect_n

    h5_dir = Path(args.h5_dir)
    files = sorted(h5_dir.glob("hs_*.h5"))
    if not files:
        raise SystemExit(f"[x] no hs_*.h5 in {h5_dir}")

    print(f"H5 dir : {h5_dir}")
    print(f"mask   : {args.mask_path}")
    print(f"cells  : {len(files)} found, {len(CELLS)} expected\n")

    seen = set()
    failures = 0

    for h5 in files:
        key = cell_key(h5, args.layer_start, args.layer_end)
        seen.add(key)
        size_gb = h5.stat().st_size / 1e9
        print(f"=== {h5.name}  ({size_gb:.1f} GB)")

        ok, bad, info = check_integrity(h5, args.layer_start, args.layer_end)
        print(f"  [1] integrity  : {'OK' if ok else 'FAIL'}   "
              f"n={info.get('n')} layers={info.get('layers')} "
              f"band={info.get('band')} alpha={info.get('alpha')} "
              f"cot={info.get('cot')} acc={info.get('acc')}%")
        print(f"      stored_layer_indices={info.get('stored_layers')}  "
              f"max_gen={info.get('max_gen_chars')} chars  mask={info.get('mask')}")
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
            # EXACT name, not a glob. '*_nocot_*' also matches nocot_a6/a8/a12 and
            # '*_cot_*' also matches cot_a6, so a glob silently SKIPs exactly the
            # two alpha=0 cells while the footer still claims agreement ran.
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

    missing = set(CELLS) - seen
    if missing:
        print(f"[!] {len(missing)} expected cell(s) absent: {sorted(missing)}")
    extra = seen - set(CELLS)
    if extra:
        print(f"[!] unexpected cell(s) present: {sorted(extra)}")

    if failures:
        print(f"\n[x] {failures} check(s) FAILED — do not start geometry analysis.")
        raise SystemExit(1)
    if missing:
        print(f"\n[!] all present cells pass, but the set is INCOMPLETE.")
        raise SystemExit(1)
    ran_agreement = bool(args.signal_dir) and not args.no_agreement
    print("\n[ok] all seven cells pass integrity + projection"
          + (" + agreement" if ran_agreement else " (agreement SKIPPED)"))


if __name__ == "__main__":
    main()
