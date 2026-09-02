#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBH P4b behavioural recompute. Protocol bbh-p4b-v0, amendment p4b-amend-01.

EVERYTHING HERE IS EXPLORATORY AND OUTSIDE THE HOLM FAMILY.

The accuracy result is produced by eval_bbh_numeric.py and is NOT recomputed as
a headline here; this script recomputes it only so the exploratory pairings
below are internally consistent, and asserts it reproduces the frozen cells.

The readouts:

  early_candidate   -- the FROZEN earlycand-v1 detector, IMPORTED from
                       early_candidate_detector.has_early_candidate. It is
                       never re-tuned here: re-tuning would fork the definition
                       against every stored GSM8K and MATH number. Its blind
                       audit on this task passed 30/30 (precision 1.000,
                       recall 1.000) but the alpha=0 base rate is at CEILING
                       (llama .952, qwen 1.000), so only a DECREASE is
                       measurable and a flat rate must NOT be read as
                       "alpha does not move answer-formation timing".

  bare_first_line   -- the first non-empty line is a BARE integer and nothing
                       else. A stricter, purely morphological companion to the
                       detector. It is NOT a synonym: an output can drop the
                       bare digit and still state the answer first in prose
                       ("I have a total of 6 ..."), which is why the regime is
                       described as MIXED rather than "reasoning before marker".

  multi_marker      -- two or more '####' occurrences. Resubmission.
  degenerate_tail   -- the final 40-char block recurs >=4x in the whole text.
                       The strict GSM8K detector, deliberately: a permissive
                       n-gram proxy read 80-86% on GSM8K and was all false
                       positives.
  cap_hit           -- generated length within 2% of the observed per-cell max.
                       Length is in CHARACTERS, not tokens.
  corpus_continuation
                    -- the generation drifts into training-corpus boilerplate
                       ("You are an AI assistant ..."). It sits AFTER the
                       marker, so it does not affect first-caliber extraction.

  pre_marker_chars / posN
                    -- FORMAT DIAGNOSTICS ONLY, never timing evidence. They are
                       contaminated by the "early empty '####', first parseable
                       marker later" mixture, so a FALL in posN does not mean
                       the answer moved later. The audited early_candidate flag
                       plus raw-text morphology carry the timing reading.

  paired_by_earlycand_transition
                    -- EXPLORATORY. Groups are selected on the OUTCOME of the
                       manipulation (the post-alpha early_candidate flag), i.e.
                       POST-TREATMENT SELECTION. It cannot be read as a
                       mediation or causal decomposition, it does not enter
                       Holm, and its p values are unadjusted.

@author: paveenhuang
"""

import argparse
import json
import os
import re
import statistics as st
import sys
from math import comb

from utils import extract_gsm8k_answer, normalize_gsm8k

HASH_NUM = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
BARE_LINE = re.compile(r"^\s*(-?\d+)\s*$")
CORPUS = re.compile(r"You are an AI assistant|User will you give you")

WORKPOINT = {"llama3": -6, "qwen2.5": 8}
REVERSE = {"llama3": 4, "qwen2.5": -6}


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    sys.exit(2)


def is_degenerate(text: str) -> bool:
    """Frozen GSM8K definition: final 40-char block recurs >=4x."""
    t = (text or "").strip()
    if len(t) < 40:
        return False
    tail = t[-40:]
    return t.count(tail) >= 4


def mcnemar_exact(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * p)


def per_sample(rows, gold):
    out = []
    for r in rows:
        sid = r["sample_id"]
        t = r.get("generated") or ""
        lines = [x for x in t.split("\n") if x.strip()]
        m = HASH_NUM.search(t)
        pred = extract_gsm8k_answer(t)
        ok = pred is not None and normalize_gsm8k(str(pred)) == normalize_gsm8k(str(gold[sid]))
        out.append(
            dict(
                sample_id=sid,
                text=t,
                chars=len(t),
                correct=ok,
                n_marker=t.count("####"),
                no_marker=("####" not in t),
                multi_marker=(t.count("####") >= 2),
                bare_first_line=bool(lines and BARE_LINE.match(lines[0])),
                degenerate=is_degenerate(t),
                corpus_continuation=bool(CORPUS.search(t)),
                pre_marker_chars=(m.start() if m else None),
                posN=(m.start() / max(len(t), 1) if m else None),
            )
        )
    return out


def rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--gold_file", required=True)
    ap.add_argument("--detector_dir", default=None,
                    help="dir holding the frozen early_candidate_detector.py")
    ap.add_argument("--expect_acc", default=None,
                    help="frozen eval JSON; per-cell first_acc must reproduce it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    if a.detector_dir:
        sys.path.insert(0, a.detector_dir)
    try:
        from early_candidate_detector import has_early_candidate
    except ImportError:
        die("cannot import the FROZEN early_candidate_detector; pass "
            "--detector_dir pointing at the directory holding it. It is "
            "never reimplemented here.")

    gj = json.load(open(a.gold_file))
    gold = {g["sample_id"]: g["gold"] for g in gj["data"]}
    qsha = gj.get("meta", {}).get("questions_sha256")

    cells = {}
    for p in a.generations:
        d = json.load(open(p))
        meta = d.get("meta", {})
        mdl, al = meta.get("model"), meta.get("alpha")
        if meta.get("protocol") != "bbh-p4b-v0":
            die(f"{p}: protocol {meta.get('protocol')!r}, expected bbh-p4b-v0")
        if meta.get("questions_sha256") != qsha:
            die(f"{p}: questions_sha256 differs from gold")
        if al not in (0, WORKPOINT.get(mdl), REVERSE.get(mdl)):
            die(f"{p}: alpha {al} is not 0, {mdl}'s frozen workpoint, or its "
                f"frozen reverse diagnostic dose")
        rows = d["data"]
        ids = [r["sample_id"] for r in rows]
        if len(set(ids)) != len(ids):
            die(f"{p}: duplicate sample_id")
        if set(ids) != set(gold):
            die(f"{p}: sample_id set differs from gold")
        recs = per_sample(rows, gold)
        for r in recs:
            r["early_candidate"] = bool(has_early_candidate(r["text"]))
        cells[(mdl, al)] = recs

    exp = json.load(open(a.expect_acc)) if a.expect_acc else None

    out = {
        "protocol": "bbh-p4b-v0",
        "amendments": "p4b-amend-01",
        "task": gj.get("meta", {}).get("task", "object_counting"),
        "questions_sha256": qsha,
        "scope": (
            "EXPLORATORY. Every readout here is outside the Holm family and "
            "carries no adjusted p. early_candidate is an OUTCOME of alpha, so "
            "stratifying accuracy on it is POST-TREATMENT stratification: "
            "consistent-with evidence, never mediation."
        ),
        "detector": (
            "earlycand-v1, IMPORTED frozen and never re-tuned. Blind audit on "
            "this task passed 30/30 (precision 1.000, recall 1.000), but the "
            "alpha=0 base rate is at CEILING (llama .952, qwen 1.000), so only "
            "a DECREASE is measurable; a flat rate must NOT be read as 'alpha "
            "does not move answer-formation timing'."
        ),
        "posN_caveat": (
            "pre_marker_chars and posN are FORMAT DIAGNOSTICS ONLY. They mix "
            "'early empty ####, first parseable marker later' cases, so a FALL "
            "in posN does not mean the answer moved later and must not be cited "
            "as timing evidence."
        ),
        "cells": {},
        "paired_by_earlycand_transition": {},
    }

    for (mdl, al), recs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        acc = sum(r["correct"] for r in recs) / len(recs)
        if exp:
            e = exp["results"][mdl]
            frozen = (e["stage0"]["first_acc"] if al == 0
                      else e["transfer"]["acc_steer"] if al == WORKPOINT[mdl]
                      else e["reverse_diagnostic"]["acc_steer"])
            if abs(acc - frozen) > 1e-9:
                die(f"{mdl} a={al}: first_acc {acc} does not reproduce the "
                    f"frozen eval value {frozen}")
        pm = [r["pre_marker_chars"] for r in recs if r["pre_marker_chars"] is not None]
        pn = [r["posN"] for r in recs if r["posN"] is not None]
        ch = [r["chars"] for r in recs]
        cmax = max(ch)
        out["cells"][f"{mdl}|a{al}"] = dict(
            model=mdl, alpha=al, n=len(recs), first_acc=round(acc, 6),
            role=("baseline" if al == 0
                  else "workpoint" if al == WORKPOINT[mdl] else "reverse_diagnostic"),
            early_candidate_rate=round(rate(recs, "early_candidate"), 4),
            bare_first_line_rate=round(rate(recs, "bare_first_line"), 4),
            multi_marker_rate=round(rate(recs, "multi_marker"), 4),
            no_marker_rate=round(rate(recs, "no_marker"), 4),
            degenerate_tail_rate=round(rate(recs, "degenerate"), 4),
            corpus_continuation_rate=round(rate(recs, "corpus_continuation"), 4),
            chars_median=st.median(ch),
            chars_p90=sorted(ch)[int(0.9 * len(ch))],
            cap_hit_rate=round(sum(1 for x in ch if x >= 0.98 * cmax) / len(ch), 4),
            pre_marker_chars_median=(st.median(pm) if pm else None),
            posN_median=(round(st.median(pn), 4) if pn else None),
        )

    for mdl, wp in WORKPOINT.items():
        if (mdl, 0) not in cells or (mdl, wp) not in cells:
            continue
        b = {r["sample_id"]: r for r in cells[(mdl, 0)]}
        s = {r["sample_id"]: r for r in cells[(mdl, wp)]}
        grp = {}
        for sid in b:
            key = ("stayed_early" if s[sid]["early_candidate"] else "turned_off")
            grp.setdefault(key, []).append(sid)
        blk = {}
        for key, ids in grp.items():
            a01 = sum(1 for i in ids if not b[i]["correct"] and s[i]["correct"])
            a10 = sum(1 for i in ids if b[i]["correct"] and not s[i]["correct"])
            blk[key] = dict(
                n=len(ids),
                acc_base=round(sum(b[i]["correct"] for i in ids) / len(ids), 4),
                acc_steer=round(sum(s[i]["correct"] for i in ids) / len(ids), 4),
                dAcc_pp=round(100 * (sum(s[i]["correct"] for i in ids)
                                     - sum(b[i]["correct"] for i in ids)) / len(ids), 2),
                gained=a01, lost=a10,
                p_raw_unadjusted=round(mcnemar_exact(a01, a10), 6),
            )
        out["paired_by_earlycand_transition"][mdl] = dict(
            alpha=wp,
            groups=blk,
            selection=(
                "POST-TREATMENT SELECTION: groups are defined by the "
                "early_candidate flag AFTER steering, i.e. by an outcome of the "
                "manipulation. Not a mediation analysis, not a causal "
                "decomposition, not in Holm; p values are UNADJUSTED."
            ),
            no_control_group=(
                f"alpha=0 early_candidate is "
                f"{out['cells'][f'{mdl}|a0']['early_candidate_rate']}, so the "
                "'turned_off' group has no matched alpha=0 counterpart and the "
                "between-group difference cannot be attributed to alpha."
            ),
        )

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    hdr = (f"{'model':9}{'a':>4}{'role':>20}{'acc':>8}{'earlyc':>8}{'bare1':>7}"
           f"{'multi':>7}{'degen':>7}{'corpus':>8}{'cap':>7}{'chars':>8}")
    print("=== BBH BEHAVIOUR (EXPLORATORY; not in Holm)")
    print(hdr)
    for k, c in out["cells"].items():
        print(f"{c['model']:9}{c['alpha']:>4}{c['role']:>20}{c['first_acc']:>8.3f}"
              f"{c['early_candidate_rate']*100:>8.1f}{c['bare_first_line_rate']*100:>7.1f}"
              f"{c['multi_marker_rate']*100:>7.1f}{c['degenerate_tail_rate']*100:>7.1f}"
              f"{c['corpus_continuation_rate']*100:>8.1f}{c['cap_hit_rate']*100:>7.1f}"
              f"{c['chars_median']:>8.0f}")
    print("\n=== EARLY-CANDIDATE TRANSITION (EXPLORATORY, post-treatment selection)")
    for mdl, blk in out["paired_by_earlycand_transition"].items():
        print(f"  {mdl} alpha={blk['alpha']}")
        for key, g in blk["groups"].items():
            print(f"    {key:14} n={g['n']:>4}  acc {g['acc_base']:.3f} -> "
                  f"{g['acc_steer']:.3f}  ({g['dAcc_pp']:+.2f} pp)  "
                  f"gained {g['gained']} lost {g['lost']}  p_raw {g['p_raw_unadjusted']:.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
