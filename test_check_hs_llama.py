#!/usr/bin/env python3
"""
Guard suite for check_hs_llama.py. Synthetic H5 + synthetic mask, no GPU, no
server, no real model -- runs in seconds on the analysis box or the server.

EVERY guard is MUTATION-TESTED: the test builds the specific defect and
asserts the checker rejects it. A guard never shown to fire on a real defect
is not a guard. The Qwen precedent is the reason -- its freeze verified the
evaluator's HASH but not that it could read what the driver actually writes,
and a "runs" vs "episodes" key mismatch produced a spurious FAIL that no hash
could catch.

The healthy fixture is built to be genuinely healthy: stored_layer_indices
[10..18, 31], 10 stored layers, band [11,20), and -- load-bearing -- the
x_*_proj scalars are computed with utils.project_rsn_numpy from the SAME fp16
states that are stored, exactly as track_hidden_states.py does. That is what
makes the ~0 projection residual the expected reading rather than an accident.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils  # noqa: E402
import check_hs_llama as C  # noqa: E402

H = 64            # small hidden size; the checks are shape/indices/relative-error
N = 6             # samples per fixture cell
LS, LE = 11, 20
N_MIDDLE = LE - LS          # 9
N_STORED = N_MIDDLE + 1     # 10
NUM_LAYERS = 32
FINAL_IDX = NUM_LAYERS - 1  # 31

_FAILS = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        _FAILS.append(msg)


def mut(td, tag):
    """A mutation fixture must carry a REALISTIC stem, or cell_key() splits on
    '_' and returns junk ('m_alpha.h5' -> 'lpha'), tripping the
    unrecognised-cell branch before the guard under test ever runs. Each
    mutation gets its own subdirectory so the stem can stay canonical."""
    d = td / f"mut_{tag}"
    d.mkdir(exist_ok=True)
    return d / "hs_gsm8k_8B_nocot_L11-20.h5"


def make_mask(path: Path, seed=0):
    """A saved mask has num_layers rows; rows outside the band are zero."""
    rng = np.random.default_rng(seed)
    m = np.zeros((NUM_LAYERS, H), dtype=np.float32)
    for r in utils.decoder_layer_range(LS, LE):
        idx = rng.choice(H, size=4, replace=False)
        m[r, idx] = rng.normal(size=4).astype(np.float32)
    np.save(path, m)
    return m


def make_h5(path: Path, mask, *, alpha=0.0, cot=False, n=N,
            n_planned=None, stored_idx=None, num_layers=NUM_LAYERS,
            n_stored=None, qidx=None, steer_mode="__auto__",
            gen_len=None, break_proj_len=False, decode_layers=None,
            drop_planned=False, drop_steer_mode=False, ema_alpha=0.95,
            mnt=768, band=(LS, LE)):
    """Write a fixture cell. Defaults are HEALTHY; kwargs inject one defect."""
    mask_mid = utils.mask_slice_for(mask, LS, LE).astype(np.float32)
    rng = np.random.default_rng(1234)

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["task"] = "gsm8k"
        meta.attrs["model"] = "llama3"
        meta.attrs["size"] = "8B"
        meta.attrs["role"] = "neutral"
        meta.attrs["cot"] = bool(cot)
        meta.attrs["layer_start"] = band[0]
        meta.attrs["layer_end"] = band[1]
        meta.attrs["ema_alpha"] = ema_alpha
        meta.attrs["steer_alpha"] = float(alpha)
        if not drop_steer_mode:
            mode = ("prefill_only" if alpha != 0.0 else "none") \
                if steer_mode == "__auto__" else steer_mode
            meta.attrs["steer_mode"] = mode
        meta.attrs["sanity_mask"] = "nmd_0.5_11_20_8B.npy"
        meta.attrs["max_new_tokens"] = mnt
        if not drop_planned:
            meta.attrs["n_samples_planned"] = n if n_planned is None else n_planned
        meta.attrs["n_samples_done"] = n
        meta.attrs["accuracy"] = 60.0
        meta.attrs["num_layers"] = num_layers
        meta.attrs["final_layer_idx_model"] = num_layers - 1
        meta.attrs["final_layer_idx_stored"] = N_MIDDLE
        si = stored_idx if stored_idx is not None else \
            list(range(LS - 1, LS - 1 + N_MIDDLE)) + [num_layers - 1]
        meta.attrs["stored_layer_indices"] = np.array(si, dtype=np.int32)
        meta.attrs["n_stored_layers"] = N_STORED if n_stored is None else n_stored

        grp = f.create_group("samples")
        for i in range(n):
            g = grp.create_group(f"{i:04d}")
            T = 5 + i
            n_lay = N_STORED if decode_layers is None else decode_layers
            # fp16 storage, exactly as the tracker does: it casts BEFORE
            # projecting, so the stored states ARE the projected states.
            pre = rng.normal(size=(3, n_lay, H)).astype(np.float16)
            dec = rng.normal(size=(T, n_lay, H)).astype(np.float16)
            g.create_dataset("prefill_hs", data=pre)
            g.create_dataset("decode_hs", data=dec)

            if n_lay == N_STORED:
                xp = utils.project_rsn_numpy(
                    pre[-1][:N_MIDDLE].astype(np.float32), mask_mid)
                xd = np.array([utils.project_rsn_numpy(
                    dec[t][:N_MIDDLE].astype(np.float32), mask_mid)
                    for t in range(T)], dtype=np.float32)
            else:
                xp, xd = 0.0, np.zeros(T, dtype=np.float32)

            g.create_dataset("x_prefill_proj", data=np.float32(xp))
            proj_T = T - 1 if break_proj_len else T
            g.create_dataset("x_decode_proj", data=xd[:proj_T])
            g.create_dataset("ema_decode_proj", data=xd[:proj_T])

            g.attrs["correct"] = int(i % 2)
            g.attrs["generated"] = "x" * (400 if gen_len is None else gen_len)
            g.attrs["pred_answer"] = "42"
            g.attrs["gold_answer"] = "42"
            g.attrs["difficulty"] = "easy"
            g.attrs["question"] = f"Q{i}"
            g.attrs["question_idx"] = i if qidx is None else qidx[i]
    return path


def light_json(path: Path, h5: Path, *, gen_mismatch=0, n=None):
    """The lightweight cell the agreement step pairs against."""
    with h5py.File(h5, "r") as f:
        grp = f["samples"]
        keys = sorted(grp.keys())
        rows = []
        for j, k in enumerate(keys):
            g = grp[k]
            gen = g.attrs["generated"]
            if j < gen_mismatch:
                gen = gen + "DIVERGED"
            rows.append({"question": g.attrs["question"],
                         "generated": gen,
                         "correct": bool(int(g.attrs["correct"]))})
    if n is not None:
        rows = rows[:n]
    path.write_text(json.dumps({"data": rows}))
    return path


def integrity(h5, expect=None):
    exp = expect or {"nocot": (0.0, False)}
    return C.check_integrity(h5, LS, LE, exp)


def main():
    C.EXPECTED_N = N        # fixtures are small; real runs use 300

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mask = make_mask(td / "nmd.npy")
        mask_path = str(td / "nmd.npy")

        print("[1] healthy fixture passes all three checks")
        h5 = make_h5(td / "hs_gsm8k_8B_nocot_L11-20.h5", mask)
        ok, bad, info = integrity(h5)
        check(ok, f"integrity OK on a healthy cell (problems: {bad})")
        check(info["stored_layers"] == list(range(10, 19)) + [31],
              f"stored_layer_indices reads [10..18, 31] -> {info['stored_layers']}")
        pok, pbad, pinfo = C.check_projection(h5, mask_path, LS, LE, 0, 1e-2)
        check(pok, f"projection OK (problems: {pbad})")
        check(pinfo["worst_decode_relerr"] < 1e-6,
              f"healthy projection residual is ~0, got "
              f"{pinfo['worst_decode_relerr']:.2e} -- the tracker stores the "
              f"fp16 states it projected")
        lj = light_json(td / "light.json", h5)
        a = C.check_agreement(h5, lj, LS, LE)
        check(a.get("generated_match") == 1.0 and a.get("question_match") == 1.0,
              f"agreement 1.000 when the two batches match: {a}")

        print("\n[2] cell_key / signal_name_for round-trip")
        for stem, want in [("hs_gsm8k_8B_nocot_aneg6_L11-20", "nocot_aneg6"),
                           ("hs_gsm8k_8B_nocot_L11-20", "nocot"),
                           ("hs_gsm8k_8B_cot_aneg4_L11-20", "cot_aneg4"),
                           ("hs_gsm8k_8B_nocot_primary_teacher_L11-20",
                            "nocot_primary_teacher")]:
            got = C.cell_key(Path(stem + ".h5"), LS, LE)
            check(got == want, f"cell_key({stem}) == {want!r} -> {got!r}")
        got = C.signal_name_for(Path("hs_gsm8k_8B_nocot_aneg6_L11-20.h5"), LS, LE, 0.95)
        check(got == "dopamine_signal_gsm8k_8B_nocot_aneg6_ema0.95_L11-20.json",
              f"signal_name_for -> {got}")
        # The exact-name rule: an alpha=0 cell's name must not be reachable by a
        # glob that also matches its steered siblings.
        check(C.signal_name_for(Path("hs_gsm8k_8B_nocot_L11-20.h5"), LS, LE, 0.95)
              != got, "alpha=0 and alpha=-6 map to DIFFERENT lightweight names")

        print("\n[3] MUTATION: each defect is rejected")

        # -- wrong final layer index: [10..19] has the right COUNT but no final layer
        h5m = make_h5(mut(td, "final"), mask, stored_idx=list(range(10, 20)))
        ok, bad, _ = integrity(h5m)
        check(not ok and any("stored_layer_indices" in b for b in bad),
              f"rejects stored_layer_indices=[10..19] (right count, NO final "
              f"layer): {bad}")

        # -- truncated cell
        h5m = make_h5(mut(td, "trunc"), mask, n=N - 1, n_planned=N)
        ok, bad, _ = integrity(h5m)
        check(not ok and any("interrupted" in b for b in bad),
              f"rejects n_samples_done < n_samples_planned: {bad}")

        # -- n_samples_planned MISSING is fatal, not skipped (fail-closed)
        h5m = make_h5(mut(td, "noplan"), mask, drop_planned=True)
        ok, bad, _ = integrity(h5m)
        check(not ok and any("n_samples_planned missing" in b for b in bad),
              f"rejects a MISSING n_samples_planned rather than skipping: {bad}")

        # -- steer_mode missing is fatal (a .get default would be fail-OPEN)
        h5m = make_h5(mut(td, "nomode"), mask, alpha=-6.0, drop_steer_mode=True)
        ok, bad, _ = integrity(h5m, {"nocot": (-6.0, False)})
        check(not ok and any("steer_mode attr missing" in b for b in bad),
              f"rejects a MISSING steer_mode on a steered cell: {bad}")

        # -- steer_mode contradicting steer_alpha
        h5m = make_h5(mut(td, "mode"), mask, alpha=-6.0, steer_mode="none")
        ok, bad, _ = integrity(h5m, {"nocot": (-6.0, False)})
        check(not ok and any("steer_mode" in b for b in bad),
              f"rejects steer_mode='none' on a steered cell: {bad}")

        # -- metadata alpha disagreeing with the filename's claim
        h5m = make_h5(mut(td, "alpha"), mask, alpha=-8.0)
        ok, bad, _ = integrity(h5m, {"nocot": (-6.0, False)})
        check(not ok and any("steer_alpha" in b for b in bad),
              f"rejects metadata alpha != the filename's alpha: {bad}")

        # -- question_idx GAP: n still looks plausible, pairing silently breaks
        h5m = make_h5(mut(td, "gap"), mask, qidx=[0, 1, 3, 4, 5, 6])
        ok, bad, _ = integrity(h5m)
        check(not ok and any("question_idx" in b for b in bad),
              f"rejects a question_idx GAP: {bad}")

        # -- duplicate question_idx
        h5m = make_h5(mut(td, "dup"), mask, qidx=[0, 1, 2, 2, 4, 5])
        ok, bad, _ = integrity(h5m)
        check(not ok and any("duplicate" in b for b in bad),
              f"rejects a duplicate question_idx: {bad}")

        # -- projection-length disagreement
        h5m = make_h5(mut(td, "len"), mask, break_proj_len=True)
        ok, bad, _ = integrity(h5m)
        check(not ok and any("lengths disagree" in b for b in bad),
              f"rejects decode_hs/x_decode_proj length mismatch: {bad}")

        # -- wrong stored layer COUNT in decode_hs
        h5m = make_h5(mut(td, "lay"), mask, decode_layers=7)
        ok, bad, _ = integrity(h5m)
        check(not ok and any("layers, expected" in b for b in bad),
              f"rejects decode_hs with the wrong layer count: {bad}")

        # -- generation sitting exactly at a historical cap
        h5m = make_h5(mut(td, "cap"), mask, gen_len=4000)
        ok, bad, _ = integrity(h5m)
        check(not ok and any("round cap" in b for b in bad),
              f"flags generations at exactly 4000 chars: {bad}")

        # -- wrong band / mnt / ema
        for kw, needle in [(dict(band=(16, 22)), "band"),
                           (dict(mnt=512), "max_new_tokens"),
                           (dict(ema_alpha=0.9), "ema_alpha"),
                           (dict(num_layers=28), "num_layers")]:
            h5m = make_h5(mut(td, needle), mask, **kw)
            ok, bad, _ = integrity(h5m)
            check(not ok and any(needle in b for b in bad),
                  f"rejects wrong {needle}: {bad}")

        print("\n[4] MUTATION: projection detects a wrong mask and a wrong band")
        other = make_mask(td / "other.npy", seed=99)
        h5 = make_h5(td / "hs_ok.h5", mask)
        pok, pbad, pinfo = C.check_projection(h5, str(td / "other.npy"), LS, LE, 0, 1e-2)
        check(not pok, f"rejects a WRONG mask (rel-err "
                       f"{pinfo.get('worst_decode_relerr', float('nan')):.2e}): {pbad}")

        # The 9x SUM-vs-MEAN trap: summing over layers instead of averaging is
        # off by exactly n_middle and would fail every healthy file. Verify the
        # helper really is the mean, so the checker cannot drift onto a sum.
        mask_mid = utils.mask_slice_for(mask, LS, LE).astype(np.float32)
        hs = np.random.default_rng(7).normal(size=(N_MIDDLE, H)).astype(np.float32)
        mean_v = utils.project_rsn_numpy(hs, mask_mid)
        sum_v = float(np.sum(hs * mask_mid))
        check(abs(sum_v - mean_v * N_MIDDLE) < 1e-3 and abs(sum_v - mean_v) > 1e-6,
              f"project_rsn_numpy is a MEAN over layers, not a sum "
              f"(sum/mean = {sum_v/mean_v:.1f}x, expected {N_MIDDLE}x)")

        print("\n[5] MUTATION: agreement refuses rather than reporting a wrong rate")
        # permuted question_idx -> positional pairing is illegitimate
        h5m = make_h5(mut(td, "perm"), mask, qidx=[5, 4, 3, 2, 1, 0])
        lj = light_json(td / "light_perm.json", h5m)
        a = C.check_agreement(h5m, lj, LS, LE)
        check("error" in a and "identity" in a["error"],
              f"refuses to pair when question_idx is permuted: {a.get('error')}")

        # row-count mismatch
        lj = light_json(td / "light_short.json", h5, n=N - 1)
        a = C.check_agreement(h5, lj, LS, LE)
        check("error" in a and "row counts" in a["error"],
              f"refuses on a row-count mismatch: {a.get('error')}")

        # a real divergence is REPORTED as a rate, not failed
        lj = light_json(td / "light_div.json", h5, gen_mismatch=2)
        a = C.check_agreement(h5, lj, LS, LE)
        check("error" not in a and abs(a["generated_match"] - (N - 2) / N) < 1e-9,
              f"reports partial generated_match as a RATE, not an error: {a}")

        print("\n[6] CLI: --cells rejects an unknown key and defaults to primary")
        r = subprocess.run([sys.executable, str(HERE / "check_hs_llama.py"),
                            "--h5_dir", str(td), "--cells", "nocot_bogus"],
                           capture_output=True, text=True)
        check(r.returncode != 0 and "unknown cell key" in r.stdout + r.stderr,
              "unknown --cells key exits non-zero with a clear message")
        check(set(C.PRIMARY_CELLS) == {"nocot", "nocot_aneg6", "nocot_aneg8",
                                       "nocot_a6"},
              f"primary set is the four pilot cells: {sorted(C.PRIMARY_CELLS)}")
        check(all(k in C.ALL_CELLS for k in
                  ["nocot_aneg4", "nocot_a4", "nocot_a8", "cot", "cot_aneg4",
                   "nocot_expert", "nocot_non_expert", "nocot_primary_teacher"]),
              "every file actually present in phase1b_eot has a known cell key")

    print()
    if _FAILS:
        print(f"[x] {len(_FAILS)} check(s) FAILED")
        for m in _FAILS:
            print(f"    - {m}")
        raise SystemExit(1)
    print("[ok] all guards fire on their defects")


if __name__ == "__main__":
    main()
