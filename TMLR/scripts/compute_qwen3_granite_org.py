"""Qwen3 / Granite GSM8K Bare organization metrics + paired tests vs alpha=0.

Reuses the frozen offline implementation (RoleAnswer/unified_behavior_gsm8k_math.compute_row:
first_acc, EC = earlycand-v1, reason_first on the candidate-covered subset). Every existing
alpha cell under AdaResult/{qwen3,granite}/gsm8k/bare*/ is read. Paired exact McNemar vs
alpha=0 within each (model, budget); Holm over all non-zero doses of that sweep.

Run: ~/.venvs/rsn-analysis/bin/python TMLR/scripts/compute_qwen3_granite_org.py TMLR/data/gsm8k_org_qwen3_granite.csv
"""
import csv
import sys
from math import comb
from pathlib import Path

RA = Path("/Users/paveenhuang/Documents/RSNResult/RoleAnswer")
sys.path.insert(0, str(RA))
from unified_behavior_gsm8k_math import compute_row, load_cell  # noqa: E402
from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k  # noqa: E402

SWEEPS = [  # (model, budget, dir, filename)
    ("qwen3", "768", "qwen3/gsm8k/bare", "gsm8k_qwen3_bare_8B_answers_17_26.json"),
    ("qwen3", "2048", "qwen3/gsm8k/bare_2048", "gsm8k_qwen3_bare_8B_answers_17_26.json"),
    ("granite", "768", "granite/gsm8k/bare", "gsm8k_granite33_bare_8B_answers_11_20.json"),
]
KEYS = ["first_acc", "last_acc", "valid_submission_rate", "early_candidate_rate",
        "candidate_coverage", "reason_first_rate", "pre_candidate_chars_median"]


def first_correct(data):
    out = []
    for s in data:
        text = s.get("generated_neutral", "")
        marks = all_hash(text)
        pred = marks[0] if marks else fallback_gsm8k(text)
        out.append(int(norm_gsm8k(pred) == norm_gsm8k(s["answer"])))
    return out


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


rows = []
for model, budget, d, fname in SWEEPS:
    cells = {}
    for p in sorted((RA / "AdaResult" / d).glob(f"mdf_*/{fname}")):
        cells[int(p.parent.name.split("_", 1)[1])] = load_cell(p)
    base = first_correct(cells[0])
    sweep = []
    for a in sorted(cells):
        data = cells[a]
        assert [s["question"] for s in data] == [s["question"] for s in cells[0]], (model, a)
        r = compute_row(model, "gsm8k", False, a, data)
        fc = first_correct(data)
        assert round(sum(fc) / len(fc) * 100, 2) == r["first_acc"], (model, a)
        b = sum(x == 0 and y == 1 for x, y in zip(base, fc))
        c = sum(x == 1 and y == 0 for x, y in zip(base, fc))
        row = {"model": model, "budget": budget, "alpha": a, "n": r["n"], **{k: r[k] for k in KEYS},
               "fixed": b if a else "", "broke": c if a else "", "p_raw": mcnemar(b, c) if a else ""}
        sweep.append(row)
    nz = [r for r in sweep if r["alpha"] != 0]
    for r, pa in zip(nz, holm([r["p_raw"] for r in nz])):
        r["p_holm"], r["holm_m"] = pa, len(nz)
    rows += sweep

fields = list(rows[0]) + ["p_holm", "holm_m"]
with open(sys.argv[1], "w") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
           if k in ("model", "budget", "alpha", "first_acc", "early_candidate_rate", "reason_first_rate",
                    "fixed", "broke", "p_raw", "p_holm", "holm_m")})
