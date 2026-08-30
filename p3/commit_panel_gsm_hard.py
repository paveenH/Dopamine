#!/usr/bin/env python3
"""GSM-Hard commitment behaviour panel -- EXPLORATORY (post-unseal).

Protocol p3-supp-v1 Part 2, No-CoT half. Offline, python3.10, no GPU.

STATUS: EXPLORATORY, PERMANENTLY. The ten No-CoT cells were unsealed on
2026-08-30, so every number here is post-hoc however principled. It DESCRIBES a
mechanism consistent with the closed P3 result; it does not test one, and it
cannot modify P3's verdict, wording or boundaries.

FROZEN DEFINITIONS ARE IMPORTED, NEVER REIMPLEMENTED. The three-way commit
partition, is_loop, posN and early_candidate are the P1 definitions from
thinking_curve/extract_metrics.py -- keying the partition on marker PRESENCE
(an earlier version tested n_bare >= 4 and misfiled samples carrying 1-3
unparseable markers as "never committed", the opposite of what they are).

POST-TREATMENT STRATIFICATION. Commitment features are themselves outcomes of
alpha, so accuracy split by them is consistent-with evidence, never mediation.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "thinking_curve"))

from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k
from extract_metrics import per_sample

CELLS = {"llama3": ("llama3/gsm_hard", "gsm_hard_8B_11_20.json", (-8, -6, -4, 0, 4)),
         "qwen2.5": ("qwen2.5/gsm_hard", "gsm_hard_7B_16_22.json", (-4, 0, 4, 6, 8))}
GOLD = os.path.join(ROOT, "gsm_hard_p3_gold.SEALED.json")


def load(model):
    sub, fname, doses = CELLS[model]
    gold = {int(s["sample_id"]): str(s["gold"])
            for s in json.load(open(GOLD, encoding="utf-8"))["data"]}
    out = {}
    for a in doses:
        p = os.path.join(ROOT, sub, f"mdf_{a}".replace("-", "neg"), fname)
        d = json.load(open(p, encoding="utf-8"))
        rows = []
        for s in d["data"]:
            i = int(s["sample_id"])
            hits = all_hash(s["generated"])
            g = norm_gsm8k(gold[i])
            if hits:
                corr = int(norm_gsm8k(hits[0]) == g)
            else:
                fb = fallback_gsm8k(s["generated"])
                corr = int(fb is not None and norm_gsm8k(fb) == g)
            # per_sample needs a `correct` key; it is used for nothing here
            # except being carried through.
            rows.append(per_sample({"generated": s["generated"],
                                    "question": s["question"],
                                    "correct": corr,
                                    "x_prefill": 0.0, "x_decode": []}))
            rows[-1]["correct"] = bool(corr)
        out[a] = rows
    return out


def med(v):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def panel(model):
    data = load(model)
    doses = CELLS[model][2]
    print(f"\n{'='*104}")
    print(f"{model}  GSM-Hard No-CoT commitment panel   [EXPLORATORY -- post-unseal]")
    print("=" * 104)
    hdr = (f"{'a':>4} {'acc':>6} {'early%':>7} {'cmt%':>6} {'mu%':>6} {'loop%':>6} "
           f"{'nomk%':>6} {'posN_med':>9} {'nmark_med':>10} {'post%':>6} "
           f"{'chars':>6} {'ec_acc':>7} {'nec_acc':>8}")
    print(hdr)
    print("-" * 104)
    for a in doses:
        r = data[a]
        n = len(r)
        acc = sum(x["correct"] for x in r) / n
        early = sum(x["early_candidate"] for x in r) / n
        cs = [x["commit_state"] for x in r]
        cmt = cs.count("committed") / n
        loop = sum(x["is_loop"] for x in r) / n
        mu = cs.count("marker_unparsed") / n - loop
        nomk = cs.count("no_marker") / n
        pos = med([x["posN"] for x in r])
        nm = med([x["n_markers"] for x in r])
        # post-commit share of the generation, committed samples only
        post = med([1.0 - x["posN"] for x in r if x["posN"] == x["posN"]])
        ch = med([x["gen_chars"] for x in r])
        ec = [x["correct"] for x in r if x["early_candidate"]]
        ne = [x["correct"] for x in r if not x["early_candidate"]]
        f = lambda v: (sum(v) / len(v)) if v else float("nan")
        print(f"{a:>+4d} {acc:>6.3f} {100*early:>6.1f}% {100*cmt:>5.1f}% "
              f"{100*mu:>5.1f}% {100*loop:>5.1f}% {100*nomk:>5.1f}% "
              f"{pos:>9.4f} {nm:>10.1f} {100*post:>5.1f}% {ch:>6.0f} "
              f"{f(ec):>7.3f} {f(ne):>8.3f}")
    return data


if __name__ == "__main__":
    for m in ("llama3", "qwen2.5"):
        panel(m)
    print("\nEXPLORATORY. Commitment features are outcomes of alpha, so any "
          "accuracy split by them is\nconsistent-with evidence, never mediation. "
          "Does not modify the closed P3 result.")
