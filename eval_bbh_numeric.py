#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBH numeric-task scoring. The ONLY script that reads gold. Protocol bbh-p5-v0.

TWO MODES, and which one runs is decided by the cells supplied, not by a flag:

  STAGE 0 (alpha=0 only)  -- the headroom gate.
      Reports first_acc per model and applies the FROZEN interval
      [0.30, 0.85]. Above it: ceiling risk, do not run the workpoint.
      Below it: capability/format floor, do not run the workpoint.
      Records format and truncation rates for the sole purpose of confirming
      that the 768 budget and the shared extractor did not fail technically.
      Those rates are NOT a gate: seeing the model's output shape must not
      lead to a changed prompt, a redefined parser, or a re-tuned budget.

  TRANSFER (alpha=0 + one workpoint) -- the fixed-workpoint test.
      dAcc = Acc(alpha) - Acc(0), paired per item, exact two-sided McNemar
      with discordant counts, and an ITEM-LEVEL paired bootstrap 95% CI
      (B=10000, seed 0). Each item is an independent question here, so the
      bootstrap unit is the item; there is no clustering structure.

HOLM IS m=2 OVER THE TWO MODELS AND IS JUDGED ONLY WHEN BOTH ARE COMPLETE.
If only one model passed stage 0, its workpoint cell is still scored, but the
result is pre-specified as SINGLE-MODEL EXPLORATORY TRANSFER: raw McNemar and
the paired CI are reported, Holm is WITHHELD, and the raw p is labelled
unadjusted. Silently running Holm at m=1 would report an m=1 adjustment under
an m=2 label.

first_acc (the FIRST '####') is MAIN, matching GSM8K/GSM-Hard production.
last_acc is a tail-pollution SENSITIVITY readout and is never the headline.
The extractor, normalizer and fallback chain are IMPORTED from utils, never
reimplemented -- an independent copy is how the inline/offline caliber gap
opened on MATH.

alpha is read from the frozen GSM8K record (llama -6, qwen +8) and is never
re-searched here; this script cannot select a dose.

@author: paveenhuang
"""

import argparse
import json
import os
import random
import re
import sys
from math import comb

from utils import extract_gsm8k_answer, normalize_gsm8k

PROTOCOL = "bbh-p5-v0"
N = 250
GATE_LO, GATE_HI = 0.30, 0.85
B_BOOT, SEED = 10000, 0
# frozen GSM8K workpoints -- read, never searched
WORKPOINT = {"llama3": -6, "qwen2.5": 8}


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


def last_hash_answer(text: str) -> str:
    """Sensitivity readout: the LAST '#### <number>', else the shared chain.

    Kept deliberately parallel to extract_gsm8k_answer's FIRST-match branch so
    the only difference between MAIN and SENSITIVITY is which marker is taken.
    """
    ms = re.findall(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if ms:
        return ms[-1].replace(",", "")
    return extract_gsm8k_answer(text)


def mcnemar_exact(a, b):
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def boot_ci(a, b, B=B_BOOT, seed=SEED):
    """Item-level paired bootstrap on the per-item difference, in pp."""
    rng = random.Random(seed)
    n = len(a)
    d = [(y - x) * 100.0 for x, y in zip(a, b)]
    out = []
    for _ in range(B):
        out.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def holm(pairs):
    s = sorted(pairs, key=lambda t: t[1]); m = len(s); out = {}; run = 0.0
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(run, (m - i) * p)); run = adj; out[k] = adj
    return out


def med(xs):
    xs = sorted(xs)
    return None if not xs else xs[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True,
                    help="cell JSONs written by get_answer_bbh_numeric.py")
    ap.add_argument("--gold_file", required=True,
                    help="the gold-bearing bbh_p5_<task>.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    gblob = json.load(open(a.gold_file, encoding="utf-8"))
    gmeta = gblob["meta"]
    task = gmeta["task"]
    gold = {r["sample_id"]: r["gold"] for r in gblob["data"]}
    if sorted(gold) != list(range(N)):
        die(f"gold does not cover 0..{N-1}")

    # model -> alpha -> {sample_id: row}
    cells, cmeta = {}, {}
    for p in a.generations:
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        if m.get("protocol") != PROTOCOL:
            die(f"{p}: protocol {m.get('protocol')!r} != {PROTOCOL!r}")
        if m.get("task") != task:
            die(f"{p}: task {m.get('task')!r} != gold file's {task!r}")
        if m.get("accuracy_computed") is not False:
            die(f"{p}: generation file claims accuracy was already computed")
        if m.get("questions_sha256") != gmeta["questions_sha256"]:
            die(f"{p}: questions_sha256 differs from the gold file; the cell "
                "was generated from a different sample")
        rows = d["data"]
        if len(rows) != N:
            die(f"{p}: {len(rows)} rows, expected {N}")
        ids = [r["sample_id"] for r in rows]
        if sorted(ids) != list(range(N)):
            die(f"{p}: sample_ids do not cover 0..{N-1}")
        exp = 0 if m["alpha"] == 0 else m["L"] * N
        if m.get("steering_fires") != exp:
            die(f"{p}: steering_fires {m.get('steering_fires')} != {exp}; "
                "intervention unverified")
        mdl, al = m["model"], m["alpha"]
        if al != 0 and al != WORKPOINT.get(mdl):
            die(f"{p}: alpha {al} is not {mdl}'s frozen GSM8K workpoint "
                f"{WORKPOINT.get(mdl)}; this protocol does not search doses")
        if al in cells.get(mdl, {}):
            die(f"{mdl} alpha={al} supplied twice")
        cells.setdefault(mdl, {})[al] = {r["sample_id"]: r for r in rows}
        cmeta.setdefault(mdl, {})[al] = m

    res = {}
    for mdl, byalpha in sorted(cells.items()):
        if 0 not in byalpha:
            die(f"{mdl}: no alpha=0 cell; it is the baseline for both modes")
        r = {"alphas": sorted(byalpha)}

        def score(al, fn):
            return [1 if normalize_gsm8k(fn(byalpha[al][i]["generated"]))
                    == normalize_gsm8k(gold[i]) else 0 for i in range(N)]

        acc0 = score(0, extract_gsm8k_answer)
        r["stage0"] = {
            "first_acc": sum(acc0) / N,
            "last_acc": sum(score(0, last_hash_answer)) / N,
            "gate_interval": [GATE_LO, GATE_HI],
            "gate_pass": GATE_LO <= sum(acc0) / N <= GATE_HI,
        }
        # DIAGNOSTIC ONLY: confirms the budget and extractor did not fail
        # technically. Not a gate, and must not motivate a prompt/parser change.
        texts = [byalpha[0][i]["generated"] for i in range(N)]
        mnt = cmeta[mdl][0]["max_new_tokens"]
        r["stage0"]["diagnostics"] = {
            "no_marker_rate": sum(1 for t in texts if "####" not in t) / N,
            "parseable_marker_rate": sum(
                1 for t in texts
                if re.search(r"####\s*[+-]?[\d,]+", t)) / N,
            "multi_marker_rate": sum(
                1 for t in texts if len(re.findall(r"####", t)) > 1) / N,
            "gen_chars_med": med([len(t) for t in texts]),
            "max_new_tokens": mnt,
            "note": ("diagnostic only -- these rates are NOT a gate and must "
                     "not motivate a changed prompt, parser or budget"),
        }

        steer = [x for x in byalpha if x != 0]
        if steer:
            al = steer[0]
            accA = score(al, extract_gsm8k_answer)
            f0 = score(0, last_hash_answer)
            fA = score(al, last_hash_answer)
            b01, b10, p = mcnemar_exact(acc0, accA)
            lo, hi = boot_ci(acc0, accA)
            r["transfer"] = {
                "alpha": al, "alpha_source": "frozen GSM8K workpoint, not searched",
                "acc_base": sum(acc0) / N, "acc_steer": sum(accA) / N,
                "dAcc_pp": (sum(accA) - sum(acc0)) / N * 100,
                "discordant_0to1": b01, "discordant_1to0": b10,
                "p_raw": p, "ci95_pp": [lo, hi],
                "sensitivity_last": {
                    "acc_base": sum(f0) / N, "acc_steer": sum(fA) / N,
                    "dAcc_pp": (sum(fA) - sum(f0)) / N * 100},
            }
        res[mdl] = r

    # ---- stage-0 verdicts
    print(f"\n=== STAGE 0 GATE  task={task}  interval [{GATE_LO}, {GATE_HI}] "
          f"(FROZEN; majority-class rate {gmeta['majority_class_rate']:.3f} is "
          f"descriptive and does NOT move it)")
    print(f"{'model':9s} {'first':>7} {'last':>7} {'no_mk':>7} {'parse':>7} "
          f"{'multi':>7} {'chars':>7}  verdict")
    for mdl, r in sorted(res.items()):
        s = r["stage0"]; d = s["diagnostics"]
        v = ("PASS" if s["gate_pass"] else
             ("FAIL ceiling" if s["first_acc"] > GATE_HI else "FAIL floor"))
        print(f"{mdl:9s} {s['first_acc']:7.4f} {s['last_acc']:7.4f} "
              f"{d['no_marker_rate']:7.3f} {d['parseable_marker_rate']:7.3f} "
              f"{d['multi_marker_rate']:7.3f} {d['gen_chars_med']:7d}  {v}")
    failed = [m for m, r in res.items() if not r["stage0"]["gate_pass"]]
    if failed:
        print(f"\n[!] gate FAILED for {sorted(failed)}: do NOT run the workpoint "
              "cell for these. Record the risk and move to the next task in the "
              "tier (multistep_arithmetic_two, then dyck_languages).")

    # ---- transfer, only for models that actually have a steered cell
    have = {m: r for m, r in res.items() if "transfer" in r}
    holm_complete = len(have) == 2
    adj = holm([(m, r["transfer"]["p_raw"]) for m, r in have.items()]) \
        if holm_complete else None
    if have:
        print(f"\n=== FIXED-WORKPOINT TRANSFER  "
              f"({'Holm m=2' if holm_complete else 'SINGLE-MODEL EXPLORATORY'})")
        print(f"{'model':9s} {'a':>3} {'acc0':>7} {'acc_a':>7} {'dAcc':>8} "
              f"{'0>1':>4} {'1>0':>4} {'p':>9} {'p_adj':>9}  CI95")
        for m, r in sorted(have.items()):
            t = r["transfer"]
            pa = f"{adj[m]:9.4f}" if adj else "  WITHHELD"
            print(f"{m:9s} {t['alpha']:>3} {t['acc_base']:7.4f} "
                  f"{t['acc_steer']:7.4f} {t['dAcc_pp']:+8.2f} "
                  f"{t['discordant_0to1']:4d} {t['discordant_1to0']:4d} "
                  f"{t['p_raw']:9.4f} {pa}  "
                  f"[{t['ci95_pp'][0]:+.2f}, {t['ci95_pp'][1]:+.2f}]")
        if not holm_complete:
            print("\n[!] only one model has a workpoint cell. This is "
                  "PRE-SPECIFIED SINGLE-MODEL EXPLORATORY TRANSFER: the raw p "
                  "is UNADJUSTED and must not be cited as corrected, and this "
                  "is not the two-model panel.")

    json.dump({"protocol": PROTOCOL, "task": task,
               "gold_sha256": gmeta["gold_sha256"],
               "questions_sha256": gmeta["questions_sha256"],
               "revision": gmeta["revision"],
               "gate_interval": [GATE_LO, GATE_HI],
               "majority_class_rate": gmeta["majority_class_rate"],
               "holm_family_m": 2, "holm_complete": holm_complete,
               "p_adj": adj, "results": res,
               "blind_validation": False,
               "note": ("BBH gold is public; this is fixed-workpoint transfer, "
                        "not blind validation. alpha comes from the frozen "
                        "GSM8K record. Stage-0 diagnostics are not a gate.")},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
