"""Table 6 / Fig 5 data: fixed-workpoint transfer and task boundaries under one test standard.

Every row compares a fixed GSM8K workpoint with alpha=0 on the same items:
  - accuracy from each task's frozen scorer, recomputed per item and checked
    against the published cell accuracy (exits on any mismatch);
  - exact two-sided McNemar on the discordant pairs;
  - paired item bootstrap 95% CI (B=10000, seed 0, percentile);
  - Holm adjustment within a declared family (two options for transfer;
    boundary rows: one family per condition).
LogiQA gold is rebuilt by salted item key from the public test split; the
recomputed No-CoT cells must equal docs/p4_logiqa2_evaluation.json.

Run: ~/.venvs/rsn-analysis/bin/python TMLR/scripts/compute_transfer_table.py
"""
import csv, glob, json, math, sys
from pathlib import Path

import numpy as np

RA = Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RA)); sys.path.insert(0, str(REPO))
import analyze_first_last_acc as A
import eval_cot_transfer_followup as E

AD = RA / "AdaResult"
OUT = REPO / "TMLR" / "data" / "transfer_boundary.csv"
GOLD_GH = RA / "gsm_hard_p3_gold.SEALED.json"

# (model, task): (alpha, baseline file, workpoint file, published first_acc % at 0 / at alpha)
TRANSFER = {
    ("Llama", "GSM-Hard"): (-6, "llama3/gsm_hard/mdf_0/gsm_hard_8B_11_20.json",
                            "llama3/gsm_hard/mdf_neg6/gsm_hard_8B_11_20.json", (18.00, 24.33)),
    ("Qwen2.5", "GSM-Hard"): (8, "qwen2.5/gsm_hard/bare/mdf_0/gsm_hard_7B_16_22.json",
                              "qwen2.5/gsm_hard/bare/mdf_8/gsm_hard_7B_16_22.json", (34.00, 50.33)),
    ("Qwen3", "GSM-Hard"): (10, "qwen3/gsm_hard/mdf_0/gsm_hard_qwen3_bare_8B_17_26.json",
                            "qwen3/gsm_hard/mdf_10/gsm_hard_qwen3_bare_8B_17_26.json", (20.33, 31.67)),
    ("Granite", "GSM-Hard"): (-6, "granite/gsm_hard/bare/mdf_0/gsm_hard_granite33_bare_8B_11_20.json",
                              "granite/gsm_hard/bare/mdf_-6/gsm_hard_granite33_bare_8B_11_20.json", None),
    ("Llama", "MATH"): (-6, "llama3/math/mdf_0/math_8B_11_20.json",
                        "llama3/math/mdf_neg6/math_8B_11_20.json", (36.67, 43.33)),
    ("Qwen2.5", "MATH"): (8, "qwen2.5/math/bare/mdf_0/math_7B_16_22.json",
                          "qwen2.5/math/bare/mdf_8/math_7B_16_22.json", None),
    ("Qwen3", "MATH"): (10, "qwen3/math/mdf_0/math_qwen3_bare_8B_17_26.json",
                        "qwen3/math/mdf_10/math_qwen3_bare_8B_17_26.json", (27.33, 35.33)),
    ("Granite", "MATH"): (-6, "granite/math/bare/mdf_0/math_granite33_bare_8B_11_20.json",
                          "granite/math/bare/mdf_-6/math_granite33_bare_8B_11_20.json", None),
}
# published differences (pp) for rows whose cell accuracies are not listed above
PUB_DIFF = {("Granite", "GSM-Hard"): 2.33, ("Qwen2.5", "MATH"): 2.67, ("Granite", "MATH"): 7.33}

BBH_GOLD = RA / "benchmark" / "bbh_p4b_object_counting.json"
CRUX_GOLD = RA / "benchmark" / "cruxeval_p4c_formal.json"
# LogiQA gold rebuilt by data_logiqa2.item_key from the public test split; No-CoT is checked against the frozen file
LQ_GOLD = RA / "benchmark" / "logiqa2_p4_gold_by_key.json"
BOUNDARY = {  # (task, model, condition): (alpha, loader key, baseline glob, workpoint glob, published acc % 0 / alpha)
    ("BBH", "Llama", "No-CoT"): (-6, "bbh", "llama3/bbh/object_counting/mdf_0/*.json",
                                 "llama3/bbh/object_counting/mdf_neg6/*.json", (41.60, 40.80)),
    ("BBH", "Qwen2.5", "No-CoT"): (8, "bbh", "qwen2.5/bbh/object_counting/mdf_0/*.json",
                                   "qwen2.5/bbh/object_counting/mdf_8/*.json", (55.20, 57.60)),
    ("BBH", "Llama", "CoT"): (-6, "bbh", "llama3/bbh/object_counting/mdf_0_cot/*.json",
                              "llama3/bbh/object_counting/mdf_neg6_cot/*.json", (40.80, 56.80)),
    ("BBH", "Qwen2.5", "CoT"): (8, "bbh", "qwen2.5/bbh/object_counting/mdf_0_cot/*.json",
                                "qwen2.5/bbh/object_counting/mdf_8_cot/*.json", (52.80, 66.80)),
    ("CRUXEval-O", "Llama", "No-CoT"): (-6, "crux", "llama3/cruxeval/mdf_0/*.json",
                                        "llama3/cruxeval/mdf_neg6/*.json", (34.67, 31.00)),
    ("CRUXEval-O", "Qwen2.5", "No-CoT"): (8, "crux", "qwen2.5/cruxeval/mdf_0/*.json",
                                          "qwen2.5/cruxeval/mdf_8/*.json", (29.33, 37.67)),
    ("CRUXEval-O", "Llama", "CoT"): (-6, "crux", "llama3/cruxeval/mdf_0_cot/*.json",
                                     "llama3/cruxeval/mdf_neg6_cot/*.json", (34.67, 34.00)),
    ("CRUXEval-O", "Qwen2.5", "CoT"): (8, "crux", "qwen2.5/cruxeval/mdf_0_cot/*.json",
                                       "qwen2.5/cruxeval/mdf_8_cot/*.json", (34.67, 54.00)),
}


def die(m):
    sys.exit(f"[x] {m}")


def load_rows(rel):
    d = json.load(open(AD / rel, encoding="utf-8"))
    rows = d["data"]
    if len(rows) != 300:
        die(f"{rel}: {len(rows)} rows")
    return rows


def gsm_hard_items(rows):
    g = {int(s["sample_id"]): s["gold"] for s in json.load(open(GOLD_GH, encoding="utf-8"))["data"]}
    return [(r["generated"], g[int(r["sample_id"])]) for r in rows], [int(r["sample_id"]) for r in rows]


def math_items(rows):
    out = []
    for r in rows:
        gold = r.get("gold_answer")
        if gold is None:  # Qwen3 / Granite cells: gold = LAST box of the full solution
            b = A.all_boxed(r["gold_solution"])
            if not b:
                die("MATH gold without box")
            gold = b[-1]
        out.append((r.get("generated_neutral", r["generated"]), gold))
    return out, [r["question"] for r in rows]


def score(task, items):
    ext, norm, fb = ((A.all_hash, A.norm_gsm8k, A.fallback_gsm8k) if task == "GSM-Hard"
                     else (A.all_boxed, A.norm_math, A.fallback_math))
    res = []
    for text, gold in items:
        m = ext(text)
        pred = m[0] if m else fb(text)
        res.append(int(norm(pred) == norm(gold)))
    return res


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot(x0, x1, B=10000, seed=0):
    d = np.asarray(x1) - np.asarray(x0)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1) * 100
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def contrast(x0, x1):
    n = len(x0)
    w2c = sum(1 for a, b in zip(x0, x1) if not a and b)
    c2w = sum(1 for a, b in zip(x0, x1) if a and not b)
    lo, hi = boot(x0, x1)
    return dict(n=n, acc0=100 * sum(x0) / n, acc1=100 * sum(x1) / n, diff=100 * (sum(x1) - sum(x0)) / n,
                w2c=w2c, c2w=c2w, ci_lo=lo, ci_hi=hi, p=mcnemar(w2c, c2w))


def check(name, got, pub):
    if pub is None:
        return
    for g, p in zip(got, pub):
        if abs(g - p) > 0.006:
            die(f"{name}: recomputed {got} vs published {pub}")


def main():
    if OUT.exists() and "--overwrite" not in sys.argv:
        die(f"{OUT} exists; pass --overwrite")
    rows = []
    for (model, task), (alpha, f0, f1, pub) in TRANSFER.items():
        r0, r1 = load_rows(f0), load_rows(f1)
        if task == "GSM-Hard":
            i0, k0 = gsm_hard_items(r0); i1, k1 = gsm_hard_items(r1)
        else:
            i0, k0 = math_items(r0); i1, k1 = math_items(r1)
        if k0 != k1:
            die(f"{model} {task}: item order differs")
        c = contrast(score(task, i0), score(task, i1))
        check(f"{model} {task}", (round(c["acc0"], 2), round(c["acc1"], 2)), pub)
        if (model, task) in PUB_DIFF and abs(round(c["diff"], 2) - PUB_DIFF[(model, task)]) > 0.006:
            die(f"{model} {task}: diff {c['diff']:.2f} vs published {PUB_DIFF[(model, task)]}")
        rows.append(dict(section="transfer", task=task, model=model, condition="No-CoT", alpha=alpha,
                         source="recomputed", **c))

    gold = {"bbh": BBH_GOLD, "crux": CRUX_GOLD}
    loader = {"bbh": E.load_bbh, "crux": E.load_cruxeval}
    for (task, model, cond), (alpha, key, g0, g1, pub) in BOUNDARY.items():
        f0, f1 = glob.glob(str(AD / g0)), glob.glob(str(AD / g1))
        if len(f0) != 1 or len(f1) != 1:
            die(f"{task} {model} {cond}: files {f0} {f1}")
        out, _ = loader[key](f0 + f1, str(gold[key]))
        (mk,) = out
        by = out[mk]["byalpha"]
        c = contrast(by[0], by[alpha])
        check(f"{task} {model} {cond}", (round(c["acc0"], 2), round(c["acc1"], 2)), pub)
        rows.append(dict(section="boundary", task=task, model=model, condition=cond, alpha=alpha,
                         source="recomputed", **c))

    lq = json.load(open(REPO / "docs" / "p4_logiqa2_evaluation.json"))["results"]
    gk = json.load(open(LQ_GOLD))["gold_by_key"]
    for mk, model in (("llama3", "Llama"), ("qwen2.5", "Qwen2.5")):
        for cond, fname in (("No-CoT", "formal"), ("CoT", "cot")):
            cells = json.load(open(AD / mk / "logiqa_reason" / f"{fname}_{mk}.json"))["cells"]
            by = {}
            for c in cells.values():
                rs = sorted(c["rows"], key=lambda r: r["sample_id"])
                if [r["sample_id"] for r in rs] != list(range(300)):
                    die(f"LogiQA {mk} {cond}: sample ids")
                by[c["alpha"]] = ([E.logiqa_main_correct(r["generated"], gk[r["key"]]) for r in rs],
                                  [r["key"] for r in rs])
            alpha = lq[mk]["alpha"]
            if by[0][1] != by[alpha][1]:
                die(f"LogiQA {mk} {cond}: item order differs")
            c = contrast(by[0][0], by[alpha][0])
            if cond == "No-CoT":
                check(f"LogiQA {model}", (round(c["acc0"], 2), round(c["acc1"], 2)),
                      (round(100 * lq[mk]["acc_base"], 2), round(100 * lq[mk]["acc_steer"], 2)))
            rows.append(dict(section="boundary", task="LogiQA 2.0", model=model, condition=cond, alpha=alpha,
                             source="recomputed", **c))

    # Holm families
    tr = [r for r in rows if r["section"] == "transfer"]
    for r, a in zip(tr, holm([r["p"] for r in tr])):
        r["holm_all8"] = a
    for task in ("GSM-Hard", "MATH"):
        sub = [r for r in tr if r["task"] == task]
        for r, a in zip(sub, holm([r["p"] for r in sub])):
            r["holm_per_task4"] = a
    for cond in ("No-CoT", "CoT"):
        sub = [r for r in rows if r["section"] == "boundary" and r["condition"] == cond]
        for r, a in zip(sub, holm([r["p"] for r in sub])):
            r["holm_boundary"] = a
            r["holm_boundary_m"] = len(sub)

    fields = ["section", "task", "model", "condition", "alpha", "n", "acc0", "acc1", "diff", "w2c", "c2w",
              "ci_lo", "ci_hi", "p", "holm_all8", "holm_per_task4", "holm_boundary", "holm_boundary_m", "source"]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if k == "p" or k.startswith("holm_") and isinstance(v, float)
                            else round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})
    for r in rows:
        print(f"{r['section']:9s} {r['task']:11s} {r['model']:8s} {r['condition']:6s} {r['alpha']:+3d} "
              f"{r['acc0']:6.2f} -> {r['acc1']:6.2f}  {r['diff']:+6.2f} [{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] "
              f"{r['w2c']:3d}/{r['c2w']:<3d} p={r['p']:.3g} all8={r.get('holm_all8', float('nan')):.3g} "
              f"task4={r.get('holm_per_task4', float('nan')):.3g} bnd={r.get('holm_boundary', float('nan')):.3g}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
