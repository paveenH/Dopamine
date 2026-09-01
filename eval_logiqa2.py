#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA 2.0 P4 evaluation. The ONLY script that reads the gold labels.

Primary: dAcc = Acc(alpha) - Acc(0) per model, MAIN parsing (LAST match),
paired per item. Exact two-sided McNemar with discordant counts; paired
difference in pp with a question-level bootstrap 95% CI (B=10000, seed 0).
Holm over the TWO MODELS (m=2), judged only when both are complete.

Secondary, excluded from Holm, labelled descriptive: sensitivity accuracy
(FIRST), no_marker rate, first/last disagreement, answer_first,
pre_marker_chars, first_marker_pos (auxiliary -- confounded by tail length),
multi_marker and degenerate rate, generation length, truncation rate.

Commitment/timing fields are themselves outcomes of alpha, so stratifying
accuracy on them is post-treatment stratification -- consistent-with evidence,
never mediation (p4-amend-05).

A single model is allowed: it is scored, and Holm is explicitly WITHHELD with
raw p labelled unadjusted (running Holm over a partial family would report an
m=1 adjustment under an m=2 label).

Fails closed on: the wrong alpha set, a cell that is not 300
rows, sample_ids not covering 0..299, unverified steering_fires, a generation
file that claims accuracy was already computed, a formal digest differing from
the frozen value or between files, and an existing output.
"""

import argparse, json, os, sys, random
from math import comb

def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)

EXPECTED = {"llama3": {0, -6}, "qwen2.5": {0, 8}}
N = 300
FORMAL_DIGEST = "4d4b25e071a2a6dd"
B_BOOT, SEED = 10000, 0


def mcnemar_exact(a, b):
    """a,b are 0/1 vectors. b01 = a wrong -> b right."""
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def boot_ci(a, b, B=B_BOOT, seed=SEED):
    """Question-level paired bootstrap on the per-item difference, in pp."""
    rng = random.Random(seed)
    n = len(a)
    d = [(y - x) * 100.0 for x, y in zip(a, b)]
    out = []
    for _ in range(B):
        s = sum(d[rng.randrange(n)] for _ in range(n))
        out.append(s / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def med(xs):
    xs = sorted(xs)
    return None if not xs else xs[len(xs) // 2]


def holm(pairs):
    s = sorted(pairs, key=lambda t: t[1]); m = len(s); out = {}; run = 0.0
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(run, (m - i) * p)); run = adj; out[k] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--formal_file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    gold_blob = json.load(open(a.formal_file, encoding="utf-8"))
    if not str(gold_blob["meta"].get("digest", "")).startswith(FORMAL_DIGEST):
        die("formal file digest != frozen value")
    gold = {it["sample_id"]: it["answer_letter"] for it in gold_blob["data"]}
    if sorted(gold) != list(range(N)):
        die("gold does not cover 0..299")

    files = {}
    for p in a.generations:
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]; model = m.get("model")
        if model not in EXPECTED:
            die(f"{p}: unknown model {model!r}")
        if model in files:
            die(f"model {model!r} supplied twice")
        if m.get("accuracy_computed") is not False:
            die(f"{p}: generation file claims accuracy was already computed")
        if not str(m.get("formal_digest", "")).startswith(FORMAL_DIGEST):
            die(f"{p}: formal_digest != frozen value")
        files[model] = d

    missing = set(EXPECTED) - set(files)
    complete = not missing
    if missing:
        print(f"[!] missing model(s) {sorted(missing)} -- Holm WITHHELD")

    res = {}
    for model, d in files.items():
        cells = {}
        for tag, c in d["cells"].items():
            if len(c["rows"]) != N:
                die(f"{model}:{tag} has {len(c['rows'])} rows, expected {N}")
            ids = [r["sample_id"] for r in c["rows"]]
            if sorted(ids) != list(range(N)):
                die(f"{model}:{tag} sample_ids do not cover 0..{N-1}")
            exp = 0 if c["alpha"] == 0 else c["L"] * N
            if c.get("steering_fires") != exp:
                die(f"{model}:{tag} steering_fires {c.get('steering_fires')} "
                    f"!= {exp}; intervention unverified")
            cells[c["alpha"]] = {r["sample_id"]: r for r in c["rows"]}
        if set(cells) != EXPECTED[model]:
            die(f"{model} has alphas {sorted(cells)}, expected "
                f"{sorted(EXPECTED[model])}")

        base, steer = 0, [x for x in EXPECTED[model] if x != 0][0]
        # aligned BY sample_id, never by row order
        acc0  = [1 if cells[base][i]["last_match"]  == gold[i] else 0 for i in range(N)]
        accA  = [1 if cells[steer][i]["last_match"] == gold[i] else 0 for i in range(N)]
        f0    = [1 if cells[base][i]["first_match"] == gold[i] else 0 for i in range(N)]
        fA    = [1 if cells[steer][i]["first_match"]== gold[i] else 0 for i in range(N)]

        b01, b10, p = mcnemar_exact(acc0, accA)
        lo, hi = boot_ci(acc0, accA)
        def rate(al, fn): return sum(fn(cells[al][i]) for i in range(N)) / N

        res[model] = {
            "alpha": steer,
            "acc_base": sum(acc0)/N, "acc_steer": sum(accA)/N,
            "dAcc_pp": (sum(accA)-sum(acc0))/N*100,
            "discordant_0to1": b01, "discordant_1to0": b10, "p_raw": p,
            "ci95_pp": [lo, hi],
            "sensitivity_first": {"acc_base": sum(f0)/N, "acc_steer": sum(fA)/N,
                                  "dAcc_pp": (sum(fA)-sum(f0))/N*100},
            "descriptive": {
                str(al): {
                    "no_marker_rate":  rate(al, lambda r: r["n_matches"] == 0),
                    "first_last_disagree": rate(al, lambda r: r["n_matches"] > 1
                                                and not r["agree_first_last"]),
                    "answer_first":    rate(al, lambda r: bool(r["answer_first"])),
                    "degenerate":      rate(al, lambda r: bool(r["degenerate"])),
                    "multi_marker":    rate(al, lambda r: r["n_matches"] > 1),
                    "budget_exhausted": rate(al, lambda r:
                                             r["stop_reason"] == "budget_exhausted"),
                    # medians; pre_marker_chars is None when no marker exists,
                    # so its denominator is the scorable subset -- reported with
                    # its own n rather than silently imputed
                    "gen_tokens_med": med([cells[al][i]["generated_token_count"]
                                           for i in range(N)]),
                    "pre_marker_chars_med": med([cells[al][i]["pre_marker_chars"]
                                                 for i in range(N)
                                                 if cells[al][i]["pre_marker_chars"]
                                                 is not None]),
                    "pre_marker_n": sum(1 for i in range(N)
                                        if cells[al][i]["pre_marker_chars"] is not None),
                    # AUXILIARY: confounded by tail length -- a longer repetition
                    # tail lowers this independently of anything before the marker
                    "first_marker_pos_med": med([cells[al][i]["first_marker_pos"]
                                                 for i in range(N)
                                                 if cells[al][i]["first_marker_pos"]
                                                 is not None]),
                } for al in sorted(cells)
            },
        }

    adj = holm([(m, r["p_raw"]) for m, r in res.items()]) if complete else None

    print(f"\n{'model':9s} {'a':>3} {'acc0':>7} {'acc_a':>7} {'dAcc':>8} "
          f"{'0>1':>4} {'1>0':>4} {'p':>9} {'p_adj':>9}  CI95")
    for m, r in sorted(res.items()):
        pa = f"{adj[m]:9.4f}" if adj else "  WITHHELD"
        print(f"{m:9s} {r['alpha']:>3} {r['acc_base']:7.4f} {r['acc_steer']:7.4f} "
              f"{r['dAcc_pp']:+8.2f} {r['discordant_0to1']:4d} "
              f"{r['discordant_1to0']:4d} {r['p_raw']:9.4f} {pa}  "
              f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]")
    if not complete:
        print("\n[!] raw p is UNADJUSTED and must not be cited as corrected.")

    print(f"\n{'model':9s} {'a':>3} {'no_mk':>6} {'ans1st':>7} {'degen':>6} "
          f"{'multi':>6} {'exhaust':>8} {'tok_med':>8} {'preCh':>7} {'n':>4} "
          f"{'pos*':>6}   (descriptive, outside Holm)")
    for m, r in sorted(res.items()):
        for al, dd in r["descriptive"].items():
            pos = dd["first_marker_pos_med"]
            print(f"{m:9s} {al:>3} {dd['no_marker_rate']:6.3f} "
                  f"{dd['answer_first']:7.3f} {dd['degenerate']:6.3f} "
                  f"{dd['multi_marker']:6.3f} {dd['budget_exhausted']:8.3f} "
                  f"{dd['gen_tokens_med']:8d} "
                  f"{dd['pre_marker_chars_med']:7} {dd['pre_marker_n']:4d} "
                  f"{pos if pos is None else round(pos,3):>6}")
    print("  * first_marker_pos is AUXILIARY: confounded by tail length. "
          "preCh/n are over the scorable subset.")

    json.dump({"protocol": "logiqa2-p4-v0",
               "amendments": "p4-amend-02,03,04,05,06",
               "holm_family_m": 2, "holm_complete": complete,
               "results": res,
               "p_adj": adj,
               "note": ("descriptive fields are preflight-informed secondary, "
                        "outside Holm; they are outcomes of alpha, so "
                        "stratifying accuracy on them is post-treatment")},
              open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
