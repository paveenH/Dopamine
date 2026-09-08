#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O chat-template four-point dose sweep scoring. llama3 ONLY.

Scores generation files written by get_answer_cruxeval_chat_sweep.py under
EITHER protocol:
    --protocol cruxeval-o-cot-chat-sweep-v1     (CoT)
    --protocol cruxeval-o-nocot-chat-sweep-v1   (No-CoT)

This is a SEPARATE, NEW scorer. It does NOT modify eval_cruxeval.py -- that
script's default cruxeval-p4c-v0 behaviour (Holm m=2, WORKPOINT/NEIGHBOUR/
REVERSE) and its --protocol cruxeval-o-cot-chat-v1 single-comparison (m=1)
behaviour are both completely untouched by this file's existence. Scoring
primitives (extract/as_literal/correct/mcnemar_exact/boot_ci/holm/med) are
IMPORTED from eval_cruxeval.py rather than reimplemented, so both scorers
apply the exact same FIRST-marker parsing/ast.literal_eval convention.

STATISTICAL DESIGN (own, independent of eval_cruxeval.py's WORKPOINT tables)
-----------------------------------------------------------------------------
Within ONE cell family (one --protocol, hence one of {CoT, No-CoT}):
    three non-zero doses {-6,-4,+4}, each vs this family's OWN alpha=0
    paired exact two-sided McNemar + item-level paired bootstrap 95% CI
    Holm m=3 over the three doses of THIS family only

CoT and No-CoT are NEVER pooled into one Holm family -- each --protocol
invocation scores and corrects independently. Comparing the two families'
numbers (chat-CoT vs chat-No-CoT, or either vs bare) is DESCRIPTIVE ONLY: no
cross-family significance test is computed or reported by this script.

WORKPOINT DEFINITION for this sweep (deliberately narrower than
eval_cruxeval.py's fixed-workpoint-transfer language, which reads alpha from
the frozen GSM8K record): a dose counts as an "effective task-specific
workpoint" under THIS chat interface ONLY IF BOTH of:
    (a) its point estimate is a POSITIVE change vs this family's own alpha=0
        (dAcc_pp > 0), AND
    (b) its Holm-adjusted p (m=3, within this family) is < 0.05
A dose failing either condition is reported (every non-zero dose is always
reported, whether or not it passes) but is NOT called a workpoint. This
mirrors, but is stricter than, the ZebraLogic-Easy / ProofWriter-OWA
conventions in this repo (a dose that is numerically highest but not
Holm-significant is explicitly NOT an "established workpoint").

first_acc (the FIRST '####') is MAIN; last_acc is a tail-revision sensitivity
readout, never the headline. Same scoring caveat as eval_cruxeval.py: this is
ast.literal_eval-based object comparison, NOT the official exec-based
CRUXEval-O pass@1; nonliteral_rate reports the size of that gap.

There is NO ACCURACY GATE. A low baseline is a limitation on the reading, not
a cancelled test. Hard stops are technical only.

@author: paveenhuang
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cruxeval import (                                            # noqa: E402
    N, PROTOCOL as BARE_PROTOCOL, EOS_TEXT,                            # noqa: F401
    extract, as_literal, correct, mcnemar_exact, boot_ci, holm, med, die,
)

MODEL = "llama3"
PROTOCOLS = {
    "cruxeval-o-cot-chat-sweep-v1": {"cot": True},
    "cruxeval-o-nocot-chat-sweep-v1": {"cot": False},
}
EXPECTED_ALPHAS = {0, -6, -4, 4}
BAND = (11, 20)


def contrast(acc0, accA):
    b01, b10, p = mcnemar_exact(acc0, accA)
    lo, hi = boot_ci(acc0, accA)
    return {"acc_base": sum(acc0) / N, "acc_steer": sum(accA) / N,
            "dAcc_pp": (sum(accA) - sum(acc0)) / N * 100,
            "discordant_0to1": b01, "discordant_1to0": b10,
            "p_raw": p, "ci95_pp": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    ap.add_argument("--generations", nargs="+", required=True,
                    help="the FOUR cell JSONs written by "
                         "get_answer_cruxeval_chat_sweep.py for this "
                         "protocol (alpha in {-6,-4,0,+4})")
    ap.add_argument("--gold_file", required=True,
                    help="the gold-bearing cruxeval_p4c_formal.json (SAME "
                         "gold every CRUXEval-O scorer reads)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    expected_cot = PROTOCOLS[a.protocol]["cot"]

    gblob = json.load(open(a.gold_file, encoding="utf-8"))
    gmeta = gblob["meta"]
    if gmeta.get("protocol") != BARE_PROTOCOL:
        die(f"gold file protocol {gmeta.get('protocol')!r} != "
            f"{BARE_PROTOCOL!r} (the gold file's own tag never changes; "
            "only the generation-file protocol varies by --protocol)")
    if sorted(r["sample_id"] for r in gblob["data"]) != list(range(N)):
        die(f"gold does not cover 0..{N-1}")

    gold, unparsed = {}, []
    for r in gblob["data"]:
        ok, v = as_literal(r["gold"])
        if not ok:
            unparsed.append((r["sample_id"], r["gold"][:60]))
        gold[r["sample_id"]] = v
    if unparsed:
        die(f"{len(unparsed)} gold value(s) do not ast.literal_eval, e.g. "
            f"{unparsed[:3]}; the scorer compares parsed Python objects")

    byalpha, cmeta = {}, {}
    for p in a.generations:
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        if m.get("protocol") != a.protocol:
            die(f"{p}: protocol {m.get('protocol')!r} != {a.protocol!r}")
        if m.get("model") != MODEL:
            die(f"{p}: model {m.get('model')!r} != {MODEL!r}; this sweep is "
                "llama3 only")
        if m.get("accuracy_computed") is not False:
            die(f"{p}: generation file claims accuracy was already computed")
        if m.get("preflight"):
            die(f"{p}: this is a PREFLIGHT cell. Preflight is format-only "
                "and its accuracy must not be viewed; it can never enter a "
                "result.")
        if m.get("questions_sha256") != gmeta["questions_sha256"]:
            die(f"{p}: questions_sha256 differs from the gold file; the "
                "cell was generated from a different sample")
        if m.get("few_shot") is not False:
            die(f"{p}: few_shot must be False")
        if m.get("cot") is not expected_cot:
            die(f"{p}: cot={m.get('cot')!r}, expected {expected_cot!r} for "
                f"protocol {a.protocol!r}")
        if m.get("chat_template_applied") is not True:
            die(f"{p}: chat_template_applied is not True -- this scorer is "
                "for the chat-wrapped sweep, not a bare cell")
        if (m.get("layer_start"), m.get("layer_end")) != BAND:
            die(f"{p}: band {(m.get('layer_start'), m.get('layer_end'))} != "
                f"the frozen llama3 band {BAND}")
        al = m.get("alpha")
        if al not in EXPECTED_ALPHAS:
            die(f"{p}: alpha {al} is not in the frozen four-point matrix "
                f"{sorted(EXPECTED_ALPHAS)}; this scorer does not accept an "
                "out-of-family dose")
        rows = d["data"]
        if len(rows) != N:
            die(f"{p}: {len(rows)} rows, expected {N}")
        if sorted(r["sample_id"] for r in rows) != list(range(N)):
            die(f"{p}: sample_ids do not cover 0..{N-1}")
        exp = 0 if al == 0 else m["L"] * N
        if m.get("steering_fires") != exp:
            die(f"{p}: steering_fires {m.get('steering_fires')} != {exp}; "
                "intervention unverified")
        if al in byalpha:
            die(f"alpha={al} supplied twice")
        for r in rows:
            if "generated_token_count" not in r or "truncated" not in r:
                die(f"{p}: sample_id={r.get('sample_id')} is missing "
                    "generated_token_count/truncated -- this scorer "
                    "requires return_metadata=True generation")
        byalpha[al] = {r["sample_id"]: r for r in rows}
        cmeta[al] = m

    missing = EXPECTED_ALPHAS - set(byalpha)
    extra = set(byalpha) - EXPECTED_ALPHAS
    if missing or extra:
        die(f"protocol {a.protocol!r} requires EXACTLY alpha set "
            f"{sorted(EXPECTED_ALPHAS)}; got {sorted(byalpha)} "
            f"(missing {sorted(missing)}, extra {sorted(extra)})")

    def score(al, which):
        return [correct(extract(byalpha[al][i]["generated"], which), gold[i])
                for i in range(N)]

    acc = {al: score(al, "first") for al in byalpha}
    accL = {al: score(al, "last") for al in byalpha}

    cells = {}
    for al in sorted(byalpha):
        texts = [byalpha[al][i]["generated"] for i in range(N)]
        toks = [byalpha[al][i]["generated_token_count"] for i in range(N)]
        n_trunc = sum(1 for i in range(N) if byalpha[al][i]["truncated"])
        firsts = [extract(t, "first") for t in texts]
        parsed = [as_literal(f)[0] for f in firsts]
        cells[str(al)] = {
            "first_acc": sum(acc[al]) / N,
            "last_acc": sum(accL[al]) / N,
            "no_marker_rate": sum(1 for f in firsts if f is None) / N,
            "nonliteral_rate": sum(1 for f, ok in zip(firsts, parsed)
                                   if f is not None and not ok) / N,
            "answer_first_rate": sum(
                1 for i in range(N) if byalpha[al][i]["answer_first"]) / N,
            "degenerate_tail_rate": sum(
                1 for i in range(N) if byalpha[al][i]["degenerate_tail"]) / N,
            "multi_marker_rate": sum(
                1 for i in range(N) if byalpha[al][i]["n_markers"] > 1) / N,
            "truncation_rate": n_trunc / N,
            "gen_chars_med": med([len(t) for t in texts]),
            "gen_tokens_med": med(toks),
            "provenance": cmeta[al].get("provenance"),
        }

    non_zero = sorted(al for al in byalpha if al != 0)
    contrasts = {al: contrast(acc[0], acc[al]) for al in non_zero}
    for al in non_zero:
        contrasts[al]["sensitivity_last"] = {
            "acc_base": sum(accL[0]) / N,
            "acc_steer": sum(accL[al]) / N,
            "dAcc_pp": (sum(accL[al]) - sum(accL[0])) / N * 100,
        }

    adj = holm([(al, contrasts[al]["p_raw"]) for al in non_zero])
    for al in non_zero:
        contrasts[al]["p_adj"] = adj[al]
        contrasts[al]["is_workpoint"] = bool(
            contrasts[al]["dAcc_pp"] > 0 and adj[al] < 0.05)

    # ---------------- report ----------------
    print(f"\n=== CRUXEval-O chat four-point sweep  protocol={a.protocol}  "
          f"model={MODEL}  n={N}  (NO accuracy gate)")
    print(f"    majority-class gold rate {gmeta['majority_class_rate']:.4f} "
          "(trivial constant guess; gates nothing)")

    print(f"\n--- per-cell panel  (first_acc is MAIN; last_acc is sensitivity)")
    print(f"{'a':>3} {'first':>7} {'last':>7} {'nomk':>6} {'nonlit':>7} "
          f"{'ansfst':>7} {'degen':>6} {'multi':>6} {'trunc':>6} {'chars':>6} "
          f"{'toks':>6}")
    for al in sorted(cells, key=lambda x: int(x)):
        c = cells[al]
        print(f"{al:>3} {c['first_acc']:7.4f} {c['last_acc']:7.4f} "
              f"{c['no_marker_rate']:6.3f} {c['nonliteral_rate']:7.3f} "
              f"{c['answer_first_rate']:7.3f} {c['degenerate_tail_rate']:6.3f} "
              f"{c['multi_marker_rate']:6.3f} {c['truncation_rate']:6.3f} "
              f"{c['gen_chars_med']:6d} {c['gen_tokens_med']:6d}")
    print("    nonlit = marker present but payload is not a Python literal. "
          "Scored INCORRECT here;")
    print("             the official exec-based metric would accept a "
          "correctly-evaluating expression.")

    print(f"\n=== DOSE CONTRASTS vs this family's own alpha=0  "
          f"(Holm m=3, WITHIN protocol={a.protocol} ONLY)")
    print(f"{'a':>3} {'acc0':>7} {'acc_a':>7} {'dAcc':>8} {'0>1':>4} "
          f"{'1>0':>4} {'p':>9} {'p_adj':>9}  {'workpoint?':>10}  CI95")
    for al in non_zero:
        t = contrasts[al]
        print(f"{al:>3} {t['acc_base']:7.4f} {t['acc_steer']:7.4f} "
              f"{t['dAcc_pp']:+8.2f} {t['discordant_0to1']:4d} "
              f"{t['discordant_1to0']:4d} {t['p_raw']:9.4f} "
              f"{t['p_adj']:9.4f}  {str(t['is_workpoint']):>10}  "
              f"[{t['ci95_pp'][0]:+.2f}, {t['ci95_pp'][1]:+.2f}]")
    workpoints = [al for al in non_zero if contrasts[al]["is_workpoint"]]
    if workpoints:
        print(f"\n[RESULT] task-specific workpoint(s) under this chat "
              f"interface (positive AND Holm p<0.05, m=3): {workpoints}")
    else:
        print("\n[RESULT] NO dose satisfied BOTH the positive-direction and "
              "Holm-significance criteria; no task-specific workpoint "
              "detected among {-6,-4,+4} under this chat interface.")
    print("\n[!] This Holm family (m=3) is WITHIN this ONE protocol only. "
          "CoT and No-CoT sweeps are scored by separate invocations and are "
          "NEVER pooled into one Holm family. Any comparison across "
          "protocols (chat-CoT vs chat-No-CoT, or either vs the bare "
          "cruxeval-p4c-v0 / cot-transfer-followup-v0 / cruxeval-o-cot-"
          "chat-v1 results) is DESCRIPTIVE ONLY -- no cross-protocol "
          "significance test is computed here.")

    json.dump({"protocol": a.protocol, "task": "cruxeval_o", "model": MODEL,
               "n": N, "cot": expected_cot,
               "gold_sha256": gmeta["gold_sha256"],
               "questions_sha256": gmeta["questions_sha256"],
               "revision": gmeta["revision"],
               "majority_class_rate": gmeta["majority_class_rate"],
               "accuracy_gate": None,
               "holm_family_m": 3, "holm_scope": "within this protocol only",
               "cells": cells, "dose_contrasts": contrasts,
               "workpoints": workpoints,
               "blind_validation": False,
               "scoring": ("ast.literal_eval both sides, Python object "
                           "equality. NOT the official exec-based pass@1; "
                           "see nonliteral_rate for the size of the gap. "
                           "Model output is never executed."),
               "workpoint_definition": (
                   "a dose qualifies as an effective task-specific "
                   "workpoint under this chat interface ONLY IF its point "
                   "estimate is a positive change (dAcc_pp > 0) vs this "
                   "family's own alpha=0 AND its Holm-adjusted p (m=3, "
                   "within this family) is < 0.05. A numerically highest "
                   "but non-significant dose is NOT an established "
                   "workpoint."),
               "note": ("EXPLORATORY four-point chat-interface dose sweep. "
                        "Does not replace or redefine the m=1 single "
                        "comparison in cruxeval-o-cot-chat-v1, nor the bare "
                        "cruxeval-p4c-v0 / cot-transfer-followup-v0 "
                        "results. Cross-protocol (CoT vs No-CoT, chat vs "
                        "bare) comparisons are descriptive only.")},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
