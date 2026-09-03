#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explicit-CoT transfer follow-up evaluation (`cot-transfer-followup-v0`).

POST-HOC EXPLORATORY FOLLOW-UP. Reads gold. Does not rewrite, rescale, or
supersede any frozen No-CoT P4/P4b/P4c result -- it only reads the No-CoT
generation files (already scored and frozen by `eval_logiqa2.py` /
`eval_bbh_numeric.py` / `eval_cruxeval.py`) to recompute PER-ITEM correctness
under each task's own already-frozen MAIN parser, so the CoT-vs-No-CoT DiD in
S4.2 of `docs/PREREG_COT_TRANSFER_FOLLOWUP.md` can be a genuine item-level
paired bootstrap. It never recomputes or reports the No-CoT AGGREGATE result
differently from what is already frozen in
docs/p4_logiqa2_evaluation.json / docs/bbh_p4b_object_counting_result.json /
docs/p4c_cruxeval_evaluation.json -- this script's own per-item recomputation
of the No-CoT accuracy is checked against those frozen aggregates and is a
HARD STOP if it disagrees, so a parser or file mismatch cannot silently
manufacture a different "No-CoT" number for the interaction term.

Usage (one task per invocation; --task selects the parser/marker convention):

    python eval_cot_transfer_followup.py --task logiqa2 \
      --generations cot_llama3.json cot_qwen2.5.json \
      --formal_file logiqa2_p4_formal.json \
      --nocot_generations formal_llama3.json formal_qwen2.5.json \
      --nocot_evaluation docs/p4_logiqa2_evaluation.json \
      --out docs/cot_followup_logiqa2_evaluation.json

    python eval_cot_transfer_followup.py --task bbh --bbh_task object_counting \
      --generations bbh_l3.json bbh_q.json \
      --gold_file bbh_p4b_object_counting.json \
      --nocot_generations nocot_mdf0_l.json nocot_mdf_neg6_l.json ... \
      --nocot_evaluation docs/bbh_p4b_object_counting_result.json \
      --out docs/cot_followup_bbh_object_counting_evaluation.json

    python eval_cot_transfer_followup.py --task cruxeval \
      --generations cot_l.json cot_q.json \
      --gold_file cruxeval_p4c_formal.json \
      --nocot_generations nocot_l_mdf0.json nocot_l_mdf_neg6.json ... \
      --nocot_evaluation docs/p4c_cruxeval_evaluation.json \
      --out docs/cot_followup_cruxeval_evaluation.json

PRIMARY (S4.1): per task, per model, CoT workpoint vs that task's OWN CoT
alpha=0. Exact two-sided McNemar, item-level paired bootstrap 95% CI
(B=10000, seed 0). This script reports its OWN task's up-to-2 rows; run it
once per task and combine the (up to 6) rows externally for the m=6 Holm
family -- see `combine_cot_followup_holm.py`. Doing Holm inside a single-task
invocation would silently apply m=2 under an m=6 label; that combiner ALSO
withholds Holm entirely unless all 6 pre-registered rows are present, since
correcting at a smaller realized m is anti-conservative, not conservative.

NOTE on LogiQA's historical metadata: `get_answer_logiqa2.py` (the verified
`logiqa2-p4-v0` No-CoT runner) never wrote a `"cot"` key at all, because it is
a No-CoT-only script. `load_logiqa()` interprets a missing `cot` field as
False ONLY when `meta.protocol == "logiqa2-p4-v0"` -- a missing field from any
other source is NOT inferred and is left to fail the explicit cot==False /
cot==True check downstream.

SECONDARY (S4.2, descriptive, never Holm-adjusted, never a significance test):
DiD = [Acc(CoT,wp) - Acc(CoT,0)] - [Acc(NoCoT,wp) - Acc(NoCoT,0)], via a
question-level joint paired bootstrap over all four accuracies per item.
"""

import argparse
import ast
import json
import os
import random
import re
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import extract_gsm8k_answer, normalize_gsm8k          # noqa: E402

PROTOCOL = "cot-transfer-followup-v0"
B_BOOT, SEED = 10000, 0

WORKPOINT = {"llama3": -6, "qwen2.5": 8}

ANSWER_RE_LOGIQA = re.compile(r"Final answer:\s*([A-D])\b")
MARKER_RE_CRUX = re.compile(r"####[ \t]*(.*)")
EOS_TEXT = ("<|endoftext|>", "<|eot_id|>", "<|im_end|>", "<|end_of_text|>",
            "</s>")


def die(m):
    print(f"[FATAL] {m}", file=sys.stderr)
    raise SystemExit(2)


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
    rng = random.Random(seed)
    n = len(a)
    d = [(y - x) * 100.0 for x, y in zip(a, b)]
    out = []
    for _ in range(B):
        out.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def did_boot_ci(acc_cot0, acc_cotA, acc_nc0, acc_ncA, B=B_BOOT, seed=SEED):
    """Question-level JOINT paired bootstrap on the DiD:
    DiD = (cotA-cot0) - (ncA-nc0), resampling one sample_id at a time and
    reading all four accuracies for that same item, per
    PREREG_COT_TRANSFER_FOLLOWUP.md S4.2."""
    rng = random.Random(seed)
    n = len(acc_cot0)
    per_item = [
        (acc_cotA[i] - acc_cot0[i]) - (acc_ncA[i] - acc_nc0[i])
        for i in range(n)
    ]
    point = sum(per_item) / n * 100.0
    out = []
    for _ in range(B):
        out.append(sum(per_item[rng.randrange(n)] for _ in range(n)) / n * 100.0)
    out.sort()
    return point, out[int(.025 * B)], out[int(.975 * B)]


# ---------------------------------------------------------------- LogiQA 2.0
def logiqa_main_correct(text, gold_letter):
    ms = ANSWER_RE_LOGIQA.findall(text)
    if not ms:
        return 0
    return 1 if ms[-1] == gold_letter else 0  # MAIN = LAST


def load_logiqa(gen_files, formal_file):
    gold_blob = json.load(open(formal_file, encoding="utf-8"))
    gold = {it["sample_id"]: it["answer_letter"] for it in gold_blob["data"]}
    n = len(gold)
    if sorted(gold) != list(range(n)):
        die("logiqa gold does not cover 0..N-1")

    out = {}
    for p in gen_files:
        d = json.load(open(p, encoding="utf-8"))
        model = d["meta"]["model"]
        cot = d["meta"].get("cot")
        if cot is None:
            # get_answer_logiqa2.py (the verified logiqa2-p4-v0 No-CoT runner)
            # never wrote a "cot" key at all -- it is a No-CoT-only script, so
            # the field's absence there always means No-CoT, never "unknown".
            # This inference is scoped to that ONE verified protocol string;
            # a missing "cot" field from anything else is NOT interpreted and
            # falls through to the explicit want_cot check below, which dies.
            if d["meta"].get("protocol") == "logiqa2-p4-v0":
                cot = False
        byalpha = {}
        for tag, c in d["cells"].items():
            rows = {r["sample_id"]: r for r in c["rows"]}
            acc = [logiqa_main_correct(rows[i]["generated"], gold[i])
                   for i in range(n)]
            byalpha[c["alpha"]] = acc
        out[model] = {"cot": cot, "byalpha": byalpha}
    return out, n


# ------------------------------------------------------------------- BBH
def last_hash_answer_bbh(text):
    ms = re.findall(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if ms:
        return ms[-1].replace(",", "")
    return extract_gsm8k_answer(text)


def load_bbh(gen_files, gold_file):
    gblob = json.load(open(gold_file, encoding="utf-8"))
    gold = {r["sample_id"]: r["gold"] for r in gblob["data"]}
    n = len(gold)
    if sorted(gold) != list(range(n)):
        die("bbh gold does not cover 0..N-1")

    out = {}
    for p in gen_files:
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        model = m["model"]
        alpha = m["alpha"]
        cot = m.get("cot")
        rows = {r["sample_id"]: r for r in d["data"]}
        acc = [1 if normalize_gsm8k(extract_gsm8k_answer(rows[i]["generated"]))
               == normalize_gsm8k(gold[i]) else 0 for i in range(n)]
        out.setdefault(model, {"cot": cot, "byalpha": {}})
        if out[model]["cot"] != cot:
            die(f"{p}: mixed cot=True/False generation files supplied for "
                f"model {model!r} in one call")
        out[model]["byalpha"][alpha] = acc
    return out, n


# --------------------------------------------------------------- CRUXEval-O
def as_literal(s):
    if s is None:
        return False, None
    try:
        return True, ast.literal_eval(s)
    except Exception:
        return False, None


def crux_extract(text, which="first"):
    ms = MARKER_RE_CRUX.findall(text)
    if not ms:
        return None
    p = (ms[0] if which == "first" else ms[-1]).strip()
    for t in EOS_TEXT:
        if p.endswith(t):
            p = p[: -len(t)].rstrip()
    i = p.find("####")
    while i != -1:
        head = p[:i]
        if head.count("'") % 2 == 0 and head.count('"') % 2 == 0:
            p = head.rstrip()
            break
        i = p.find("####", i + 4)
    return p.strip()


def crux_correct(text, gold_value):
    ok, v = as_literal(crux_extract(text, "first"))  # FIRST = MAIN
    if not ok:
        return 0
    try:
        return 1 if v == gold_value else 0
    except Exception:
        return 0


def load_cruxeval(gen_files, gold_file):
    gblob = json.load(open(gold_file, encoding="utf-8"))
    gold, unparsed = {}, []
    for r in gblob["data"]:
        ok, v = as_literal(r["gold"])
        if not ok:
            unparsed.append(r["sample_id"])
        gold[r["sample_id"]] = v
    if unparsed:
        die(f"{len(unparsed)} gold value(s) do not ast.literal_eval")
    n = len(gold)
    if sorted(gold) != list(range(n)):
        die("cruxeval gold does not cover 0..N-1")

    out = {}
    for p in gen_files:
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        if m.get("preflight"):
            die(f"{p}: this is a PREFLIGHT cell and must not be scored")
        model = m["model"]
        alpha = m["alpha"]
        cot = m.get("cot")
        rows = {r["sample_id"]: r for r in d["data"]}
        acc = [crux_correct(rows[i]["generated"], gold[i]) for i in range(n)]
        out.setdefault(model, {"cot": cot, "byalpha": {}})
        if out[model]["cot"] != cot:
            die(f"{p}: mixed cot=True/False generation files supplied for "
                f"model {model!r} in one call")
        out[model]["byalpha"][alpha] = acc
    return out, n


LOADERS = {"logiqa2": load_logiqa, "bbh": load_bbh, "cruxeval": load_cruxeval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(LOADERS))
    ap.add_argument("--bbh_task", choices=["object_counting",
                                           "multistep_arithmetic_two"])
    ap.add_argument("--generations", nargs="+", required=True,
                    help="the CoT generation files (one per model)")
    ap.add_argument("--formal_file",
                    help="logiqa2 only: the gold-bearing formal file")
    ap.add_argument("--gold_file",
                    help="bbh/cruxeval: the gold-bearing file")
    ap.add_argument("--nocot_generations", nargs="+", required=True,
                    help="the already-frozen No-CoT generation files, ALL "
                         "cells, both models -- used only to recompute "
                         "per-item accuracy for the S4.2 interaction term")
    ap.add_argument("--nocot_evaluation", required=True,
                    help="the frozen No-CoT eval JSON, used as a consistency "
                         "check on the recomputed No-CoT accuracy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out):
        die(f"{a.out} exists; refusing to overwrite")

    loader = LOADERS[a.task]
    if a.task == "logiqa2":
        if not a.formal_file:
            die("--formal_file is required for --task logiqa2")
        cot_data, n = loader(a.generations, a.formal_file)
        nocot_data, n2 = loader(a.nocot_generations, a.formal_file)
    else:
        if not a.gold_file:
            die(f"--gold_file is required for --task {a.task}")
        cot_data, n = loader(a.generations, a.gold_file)
        nocot_data, n2 = loader(a.nocot_generations, a.gold_file)
    if n != n2:
        die("cot and nocot item counts disagree")

    for d, label in ((cot_data, "cot"), (nocot_data, "nocot")):
        for model, rec in d.items():
            want_cot = (label == "cot")
            if rec["cot"] is not want_cot:
                die(f"{label} file for model {model!r} has cot={rec['cot']!r}, "
                    f"expected {want_cot}")

    # ---- consistency check: recomputed No-CoT accuracy must match the
    # already-frozen aggregate exactly (within float rounding), or this
    # script is reading a different sample / parser than the frozen result.
    frozen = json.load(open(a.nocot_evaluation, encoding="utf-8"))
    for model in WORKPOINT:
        if model not in nocot_data:
            continue
        wp = WORKPOINT[model]
        acc0 = nocot_data[model]["byalpha"].get(0)
        accA = nocot_data[model]["byalpha"].get(wp)
        if acc0 is None or accA is None:
            continue
        got0, gotA = sum(acc0) / n, sum(accA) / n
        if a.task == "logiqa2":
            want0 = frozen["results"][model]["acc_base"]
            wantA = frozen["results"][model]["acc_steer"]
        else:
            want0 = frozen["results"][model]["transfer"]["acc_base"]
            wantA = frozen["results"][model]["transfer"]["acc_steer"]
        if abs(got0 - want0) > 1e-6:
            die(f"{model}: recomputed No-CoT alpha=0 accuracy {got0} != frozen "
                f"{want0}; this script is not reading the same sample/parser "
                f"as {a.nocot_evaluation}")
        if abs(gotA - wantA) > 1e-6:
            die(f"{model}: recomputed No-CoT alpha={wp} accuracy {gotA} != "
                f"frozen {wantA}; this script is not reading the same "
                f"sample/parser as {a.nocot_evaluation}")
    print("[cot-followup] recomputed No-CoT accuracy matches the frozen "
          "evaluation exactly, for every model with a supplied cell")

    # ---------------- S4.1 primary: CoT workpoint vs CoT alpha=0 ----------
    rows = {}
    for model, rec in cot_data.items():
        wp = WORKPOINT.get(model)
        if wp is None or 0 not in rec["byalpha"] or wp not in rec["byalpha"]:
            continue
        acc0, accA = rec["byalpha"][0], rec["byalpha"][wp]
        b01, b10, p = mcnemar_exact(acc0, accA)
        lo, hi = boot_ci(acc0, accA)
        rows[model] = {
            "task": a.task, "bbh_task": a.bbh_task, "model": model,
            "alpha": wp, "acc_cot0": sum(acc0) / n, "acc_cotA": sum(accA) / n,
            "dAcc_pp": (sum(accA) - sum(acc0)) / n * 100,
            "discordant_0to1": b01, "discordant_1to0": b10, "p_raw": p,
            "ci95_pp": [lo, hi],
        }

    # ---------------- S4.2 secondary: CoT x steering interaction ----------
    did = {}
    for model in rows:
        wp = WORKPOINT[model]
        if (model not in nocot_data or 0 not in nocot_data[model]["byalpha"]
                or wp not in nocot_data[model]["byalpha"]):
            continue
        point, lo, hi = did_boot_ci(
            cot_data[model]["byalpha"][0], cot_data[model]["byalpha"][wp],
            nocot_data[model]["byalpha"][0], nocot_data[model]["byalpha"][wp])
        did[model] = {
            "did_pp": point, "ci95_pp": [lo, hi],
            "detected": not (lo <= 0 <= hi),
            "note": ("descriptive only, never Holm-adjusted, never a "
                     "significance test. A CI containing 0 is reported as "
                     "'not detected', never an equivalence claim. Given the "
                     "three confounded mechanisms in "
                     "PREREG_COT_TRANSFER_FOLLOWUP.md S0, a nonzero DiD is "
                     "reported only as 'the steering effect differs between "
                     "CoT and No-CoT', never attributed to a single "
                     "mechanism."),
        }

    print(f"\n=== {a.task}{' / ' + a.bbh_task if a.bbh_task else ''}  "
          f"S4.1 PRIMARY (CoT workpoint vs that task's OWN CoT alpha=0)")
    print("    Part of a 6-comparison Holm family across all three tasks x "
          "two models -- combine externally, do not apply Holm here.")
    print(f"{'model':9s} {'a':>3} {'acc_cot0':>9} {'acc_cotA':>9} {'dAcc':>8} "
          f"{'0>1':>4} {'1>0':>4} {'p_raw':>9}  CI95")
    for m, r in sorted(rows.items()):
        print(f"{m:9s} {r['alpha']:>3} {r['acc_cot0']:9.4f} {r['acc_cotA']:9.4f} "
              f"{r['dAcc_pp']:+8.2f} {r['discordant_0to1']:4d} "
              f"{r['discordant_1to0']:4d} {r['p_raw']:9.4f}  "
              f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]")

    if did:
        print(f"\n=== {a.task}  S4.2 SECONDARY: CoT x steering interaction "
              f"(descriptive, outside Holm, NOT a significance test)")
        for m, r in sorted(did.items()):
            tag = "detected" if r["detected"] else "not detected"
            print(f"{m:9s} DiD={r['did_pp']:+.2f}pp  "
                  f"CI95=[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]  "
                  f"({tag})")

    json.dump({
        "protocol": PROTOCOL, "task": a.task, "bbh_task": a.bbh_task,
        "exploratory_followup": True,
        "note": ("post-hoc exploratory follow-up to the frozen No-CoT "
                 "P4/P4b/P4c result; does not replace it. S4.1 rows are part "
                 "of a 6-comparison Holm family combined across all three "
                 "tasks -- see combine_cot_followup_holm.py. S4.2 DiD is "
                 "descriptive, never Holm-adjusted, never a significance "
                 "test, and does not isolate reasoning-length vs "
                 "self-conditioning vs steering-propagation-depth."),
        "primary_rows": rows,
        "interaction": did,
    }, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
