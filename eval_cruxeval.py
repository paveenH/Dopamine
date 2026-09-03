#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O scoring. The ONLY script that reads gold. Protocol `cruxeval-p4c-v0`.

FIXED-WORKPOINT TRANSFER. alpha is read from the frozen GSM8K record
(llama -6, qwen +8) and is never re-searched; this script cannot select a dose.

    PRIMARY family, and only this:
        llama  -6 vs 0
        qwen   +8 vs 0
        paired exact two-sided McNemar + item-level paired bootstrap 95% CI
        Holm m=2, judged ONLY when both models are complete.

    If only one model completes, its contrast is reported as SINGLE-MODEL
    EXPLORATORY TRANSFER: Holm is WITHHELD and the raw p is labelled
    unadjusted. Running Holm at m=1 under an m=2 label is the error this
    prevents.

    DIAGNOSTICS, outside the Holm family, p UNADJUSTED:
        neighbour  llama -4 / qwen +6   -- is the workpoint locally stable?
        reverse    llama +4 / qwen -6   -- does the direction ordering continue?
    They are reported whether or not they are significant, and they MUST NOT
    redefine the workpoint. Four points are not a dose-response curve: they can
    show an ordering continues or breaks, but cannot locate a peak, establish
    an inverted-U, or license calling any dose an overshoot point.

SCORING -- and why it is NOT the official executor
--------------------------------------------------
Official CRUXEval-O runs `exec(f"{code}\\nassert {gold} == {candidate}")`, which
interpolates MODEL-GENERATED TEXT into an executed expression. This protocol
does not do that on the server or on a normal local machine, and
`reliability_guard` is not a security sandbox and must not be described as one.

Instead both sides are parsed with `ast.literal_eval` -- which cannot execute
code by construction -- and compared as Python objects, so semantically equal
spellings both pass ([1,2] == [1, 2]). Verified at freeze: 800/800 CRUXEval
gold values are pure literals, so on this benchmark the two agree wherever the
candidate is itself a literal.

The one recorded gap: a candidate that is a non-literal EXPRESSION evaluating
to the right value (`[1]*3`, `list(range(3))`) is scored INCORRECT here and
would be scored correct officially. The rate of such candidates is reported per
cell as `nonliteral_rate`, so the size of the gap is visible rather than
assumed.

first_acc (the FIRST '####') is MAIN, matching GSM8K/GSM-Hard/BBH production.
last_acc is a tail-revision SENSITIVITY readout and is never the headline.

There is NO ACCURACY GATE. A low baseline is recorded as a limitation on the
reading, not used to cancel the test. Hard stops are technical only.

@author: paveenhuang
"""

import argparse
import ast
import json
import os
import random
import re
import sys
from math import comb

PROTOCOL = "cruxeval-p4c-v0"
N = 300
B_BOOT, SEED = 10000, 0

# frozen GSM8K workpoints -- read, never searched
WORKPOINT = {"llama3": -6, "qwen2.5": 8}
NEIGHBOUR = {"llama3": -4, "qwen2.5": 6}
REVERSE = {"llama3": 4, "qwen2.5": -6}

MARKER_RE = re.compile(r"####[ \t]*(.*)")

# Special tokens that a decoder may leave in the returned text. They are a
# GENERATION ARTIFACT, not part of the model's answer: Qwen's preflight emitted
# `[1, 1, 1, 1]<|endoftext|>`, an entirely correct answer that failed to parse
# purely because the EOS text was still attached. Stripping them is a decoding
# fix, not a scoring concession -- it changes no answer, only removes a token
# the tokenizer should not have surfaced.
EOS_TEXT = ("<|endoftext|>", "<|eot_id|>", "<|im_end|>", "<|end_of_text|>",
            "</s>")


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


def extract(text: str, which: str = "first"):
    """Return the payload after a '####' marker, or None if there is none.

    FIRST is MAIN (GSM8K/GSM-Hard/BBH production convention); LAST is the
    tail-revision sensitivity readout. The payload is the remainder of the
    marker's LINE -- verified at freeze that no gold contains a newline.

    TWO GENERATION ARTIFACTS ARE REMOVED, and neither is a scoring concession:

    1. A trailing EOS token text (see EOS_TEXT). The decoder surfaced it; the
       model did not write it as part of its answer.

    2. A TRAILING `####`. Llama's preflight writes `#### <literal> ####` and
       then loops `#### x  #### x  #### x`, so the line remainder carries the
       next marker. Without this the payload happens to parse ANYWAY, because
       `#` starts a Python comment and `ast.literal_eval` silently truncates
       there -- i.e. the parser would be right BY COINCIDENCE, not by design,
       and that coincidence is unsafe: 5 of the 300 gold values contain `#`
       (e.g. sample_755 `'ph>t#A#BiEcDefW#ON#iiNCU'`). Cutting at the marker
       EXPLICITLY keeps the in-quotes `#` intact -- which is correct, since a
       `####` inside a string literal is part of the answer -- while removing
       the marker the model appended after it.

    Nothing else is stripped. Prose after the payload (Qwen's
    `#### 'ohesteo' The function f removes ...`) is NOT rescued: that is the
    model failing to obey the frozen format, which is a result to report, not
    a parser to widen.
    """
    ms = MARKER_RE.findall(text)
    if not ms:
        return None
    p = (ms[0] if which == "first" else ms[-1]).strip()
    for t in EOS_TEXT:
        if p.endswith(t):
            p = p[: -len(t)].rstrip()
    # cut a marker the model appended AFTER its answer, never one inside a
    # string literal (an unterminated quote before it means we are inside one)
    i = p.find("####")
    while i != -1:
        head = p[:i]
        if head.count("'") % 2 == 0 and head.count('"') % 2 == 0:
            p = head.rstrip()
            break
        i = p.find("####", i + 4)
    return p.strip()


def as_literal(s):
    """(ok, value). Never executes: literal_eval cannot call or import."""
    if s is None:
        return False, None
    try:
        return True, ast.literal_eval(s)
    except Exception:
        return False, None


def correct(pred_text, gold_value):
    ok, v = as_literal(pred_text)
    if not ok:
        return 0
    try:
        return 1 if v == gold_value else 0
    except Exception:
        # exotic __eq__ cannot arise from a literal, but a comparison that
        # raises must score incorrect rather than crash the run
        return 0


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
    """Item-level paired bootstrap on the per-item difference, in pp. Each item
    is an independent question here, so the bootstrap unit is the item; there
    is no clustering structure."""
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


def contrast(acc0, accA):
    b01, b10, p = mcnemar_exact(acc0, accA)
    lo, hi = boot_ci(acc0, accA)
    return {"acc_base": sum(acc0) / N, "acc_steer": sum(accA) / N,
            "dAcc_pp": (sum(accA) - sum(acc0)) / N * 100,
            "discordant_0to1": b01, "discordant_1to0": b10,
            "p_raw": p, "ci95_pp": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True,
                    help="cell JSONs written by get_answer_cruxeval.py")
    ap.add_argument("--gold_file", required=True,
                    help="the gold-bearing cruxeval_p4c_formal.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    gblob = json.load(open(a.gold_file, encoding="utf-8"))
    gmeta = gblob["meta"]
    if gmeta.get("protocol") != PROTOCOL:
        die(f"gold file protocol {gmeta.get('protocol')!r} != {PROTOCOL!r}")
    if sorted(r["sample_id"] for r in gblob["data"]) != list(range(N)):
        die(f"gold does not cover 0..{N-1}")

    # HARD STOP 6: every gold must parse, or it is not scoreable at all.
    gold, unparsed = {}, []
    for r in gblob["data"]:
        ok, v = as_literal(r["gold"])
        if not ok:
            unparsed.append((r["sample_id"], r["gold"][:60]))
        gold[r["sample_id"]] = v
    if unparsed:
        die(f"{len(unparsed)} gold value(s) do not ast.literal_eval, e.g. "
            f"{unparsed[:3]}; the scorer compares parsed Python objects")

    # model -> alpha -> {sample_id: row}
    cells, cmeta = {}, {}
    for p in a.generations:
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        if m.get("protocol") != PROTOCOL:
            die(f"{p}: protocol {m.get('protocol')!r} != {PROTOCOL!r}")
        if m.get("accuracy_computed") is not False:
            die(f"{p}: generation file claims accuracy was already computed")
        # A preflight cell is FORMAT ONLY (8 items). Scoring one would turn a
        # format check into an accuracy reading, which the protocol forbids.
        if m.get("preflight"):
            die(f"{p}: this is a PREFLIGHT cell. Preflight is format-only and "
                "its accuracy must not be viewed; it can never enter a result.")
        if m.get("questions_sha256") != gmeta["questions_sha256"]:
            die(f"{p}: questions_sha256 differs from the gold file; the cell "
                "was generated from a different sample")
        if m.get("cot") is not False or m.get("few_shot") is not False:
            die(f"{p}: cot/few_shot must both be False in this protocol")
        rows = d["data"]
        if len(rows) != N:
            die(f"{p}: {len(rows)} rows, expected {N}")
        if sorted(r["sample_id"] for r in rows) != list(range(N)):
            die(f"{p}: sample_ids do not cover 0..{N-1}")
        exp = 0 if m["alpha"] == 0 else m["L"] * N
        if m.get("steering_fires") != exp:
            die(f"{p}: steering_fires {m.get('steering_fires')} != {exp}; "
                "intervention unverified")
        mdl, al = m["model"], m["alpha"]
        if mdl not in WORKPOINT:
            die(f"{p}: unknown model {mdl!r}")
        frozen = {0, WORKPOINT[mdl], NEIGHBOUR[mdl], REVERSE[mdl]}
        if al not in frozen:
            die(f"{p}: alpha {al} is not in {mdl}'s frozen matrix "
                f"{sorted(frozen)}; this protocol does not search doses")
        if al in cells.get(mdl, {}):
            die(f"{mdl} alpha={al} supplied twice")
        cells.setdefault(mdl, {})[al] = {r["sample_id"]: r for r in rows}
        cmeta.setdefault(mdl, {})[al] = m

    res = {}
    for mdl, byalpha in sorted(cells.items()):
        if 0 not in byalpha:
            die(f"{mdl}: no alpha=0 cell; it is the baseline for every contrast")
        r = {"alphas": sorted(byalpha), "cells": {}}

        def score(al, which):
            return [correct(extract(byalpha[al][i]["generated"], which), gold[i])
                    for i in range(N)]

        acc = {al: score(al, "first") for al in byalpha}
        accL = {al: score(al, "last") for al in byalpha}

        # per-cell descriptive panel -- provenance, never a gate
        for al in sorted(byalpha):
            texts = [byalpha[al][i]["generated"] for i in range(N)]
            firsts = [extract(t, "first") for t in texts]
            parsed = [as_literal(f)[0] for f in firsts]
            r["cells"][str(al)] = {
                "role": cmeta[mdl][al].get("role"),
                "first_acc": sum(acc[al]) / N,
                "last_acc": sum(accL[al]) / N,
                "no_marker_rate": sum(1 for f in firsts if f is None) / N,
                # marker present but the payload is not a literal: the size of
                # the gap to the official exec-based metric
                "nonliteral_rate": sum(1 for f, ok in zip(firsts, parsed)
                                       if f is not None and not ok) / N,
                "answer_first_rate": sum(
                    1 for i in range(N) if byalpha[al][i]["answer_first"]) / N,
                "degenerate_tail_rate": sum(
                    1 for i in range(N) if byalpha[al][i]["degenerate_tail"]) / N,
                "multi_marker_rate": sum(
                    1 for i in range(N) if byalpha[al][i]["n_markers"] > 1) / N,
                "gen_chars_med": med([len(t) for t in texts]),
                "provenance": cmeta[mdl][al].get("provenance"),
            }

        wp = WORKPOINT[mdl]
        if wp in byalpha:
            t = contrast(acc[0], acc[wp])
            t.update({"alpha": wp,
                      "alpha_source": "frozen GSM8K workpoint, not searched",
                      "sensitivity_last": {
                          "acc_base": sum(accL[0]) / N,
                          "acc_steer": sum(accL[wp]) / N,
                          "dAcc_pp": (sum(accL[wp]) - sum(accL[0])) / N * 100}})
            r["transfer"] = t

        for kind, table in (("neighbour", NEIGHBOUR), ("reverse", REVERSE)):
            al = table[mdl]
            if al in byalpha:
                d_ = contrast(acc[0], acc[al])
                d_["alpha"] = al
                d_["scope"] = (
                    f"{kind} DIAGNOSTIC. Outside the Holm family; p is "
                    "UNADJUSTED. It MUST NOT redefine the workpoint, which "
                    "stays read from the frozen GSM8K record. Four sampled "
                    "doses are not a dose-response curve: they can show an "
                    "ordering continues or breaks, but cannot locate a peak, "
                    "establish an inverted-U, or license calling any dose an "
                    "overshoot point.")
                r[f"{kind}_diagnostic"] = d_

        if "transfer" in r and "reverse_diagnostic" in r:
            a0 = sum(acc[0]) / N
            aw = r["transfer"]["acc_steer"]
            ar = r["reverse_diagnostic"]["acc_steer"]
            r["reverse_diagnostic"]["ordering"] = {
                "acc_workpoint": aw, "acc_zero": a0, "acc_reverse": ar,
                "continues": bool(aw > a0 > ar),
                "note": ("`continues` compares POINT ESTIMATES only and is not "
                         "a test; read it beside both CIs."),
            }
        res[mdl] = r

    # ---------------- report ----------------
    print(f"\n=== P4c CRUXEval-O  n={N}  (NO accuracy gate -- a low baseline is "
          f"a limitation on the reading, not a cancelled test)")
    print(f"    majority-class gold rate {gmeta['majority_class_rate']:.4f} "
          f"(trivial constant guess; gates nothing)")

    print(f"\n--- per-cell panel  (first_acc is MAIN; last_acc is sensitivity)")
    print(f"{'model':9s} {'a':>3} {'role':>10} {'first':>7} {'last':>7} "
          f"{'nomk':>6} {'nonlit':>7} {'ansfst':>7} {'degen':>6} {'multi':>6} {'chars':>6}")
    for mdl, r in sorted(res.items()):
        for al in sorted(r["cells"], key=lambda x: int(x)):
            c = r["cells"][al]
            print(f"{mdl:9s} {al:>3} {str(c['role']):>10} {c['first_acc']:7.4f} "
                  f"{c['last_acc']:7.4f} {c['no_marker_rate']:6.3f} "
                  f"{c['nonliteral_rate']:7.3f} {c['answer_first_rate']:7.3f} "
                  f"{c['degenerate_tail_rate']:6.3f} {c['multi_marker_rate']:6.3f} "
                  f"{c['gen_chars_med']:6d}")
    print("    nonlit = marker present but payload is not a Python literal. "
          "Scored INCORRECT here;")
    print("             the official exec-based metric would accept a "
          "correctly-evaluating expression.")

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

    for kind, title in (("neighbour", "NEIGHBOUR (workpoint local stability)"),
                        ("reverse", "REVERSE (direction ordering)")):
        blk = {m: r for m, r in res.items() if f"{kind}_diagnostic" in r}
        if not blk:
            continue
        print(f"\n=== {title} DIAGNOSTIC -- NOT in the Holm family; p is UNADJUSTED")
        print("    Does NOT redefine the workpoint, which stays read from the "
              "frozen GSM8K record.")
        print(f"{'model':9s} {'a':>3} {'acc0':>7} {'acc_a':>7} {'dAcc':>8} "
              f"{'p_raw':>9}  {'CI95':>18}")
        for m, r in sorted(blk.items()):
            v = r[f"{kind}_diagnostic"]
            line = (f"{m:9s} {v['alpha']:>3} {v['acc_base']:7.4f} "
                    f"{v['acc_steer']:7.4f} {v['dAcc_pp']:+8.2f} "
                    f"{v['p_raw']:9.4f}  "
                    f"[{v['ci95_pp'][0]:+6.2f}, {v['ci95_pp'][1]:+6.2f}]")
            o_ = v.get("ordering")
            if o_:
                line += ("  CONTINUES" if o_["continues"] else "  BREAKS")
                line += (f"  {v['alpha']:+d}:{o_['acc_reverse']:.3f} "
                         f"0:{o_['acc_zero']:.3f} wp:{o_['acc_workpoint']:.3f}")
            print(line)

    json.dump({"protocol": PROTOCOL, "task": "cruxeval_o", "n": N,
               "gold_sha256": gmeta["gold_sha256"],
               "questions_sha256": gmeta["questions_sha256"],
               "revision": gmeta["revision"],
               "majority_class_rate": gmeta["majority_class_rate"],
               "accuracy_gate": None,
               "holm_family_m": 2, "holm_complete": holm_complete,
               "p_adj": adj, "results": res,
               "blind_validation": False,
               "scoring": ("ast.literal_eval both sides, Python object "
                           "equality. NOT the official exec-based pass@1; see "
                           "nonliteral_rate for the size of the gap. Model "
                           "output is never executed."),
               "note": ("CRUXEval gold is public; this is fixed-workpoint "
                        "transfer, not blind validation. alpha comes from the "
                        "frozen GSM8K record. P4c changes reasoning type AND "
                        "answer space at once, so it cannot isolate which "
                        "decides the outcome.")},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
