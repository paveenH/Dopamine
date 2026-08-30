#!/usr/bin/env python3
"""P3 supplement STAGE 2 -- apply the frozen P2 predictors to the four CoT
cells and FREEZE the per-cell commitment scores. Protocol p3-supp-v1
(docs/PREREG_P3_SUPPLEMENT.md, tag p3-supp-frozen; two-stage rule added by
docs/p3_supp_amendment_01.json).

WHAT THIS STEP IS FOR. Stage 1 froze the predicted DIRECTION of dAcc before any
CoT cell existed. Stage 2 seals the commitment side against accuracy the same
way P3's prediction step did: the score is computed and written here, and CoT
accuracy is computed only afterwards. Running the accuracy first would not
destroy a blind dataset (the gold was unsealed on 2026-08-30 and the questions
are the same 300), but it would destroy the LOCKED character of the condition
test, which is the only evidential claim this supplement makes.

THIS IS NOT A BLIND DATASET VALIDATION, and the artifact says so in its own
fields. Same questions, same gold, already unsealed. What is prospective is the
CONDITION: these four cells did not exist when the direction was frozen.

NOTHING IS REFITTED. The same frozen P2 coefficients, standardization and GSM8K
posN median as P3, applied through the same apply_frozen() copy, verified
character-for-character against the P2 original. The predictor was fitted on
768-token No-CoT GSM8K output; CoT generations are longer and differently
shaped, so the SCORE here is a descriptive commitment readout, not a calibrated
accuracy estimate. That was already true of P3 and is more true here.

LABEL FIREWALL, unchanged from P3: build_features() takes a STRING, so gold is
unreachable rather than merely unused, and the cells are re-checked for label
fields rather than trusting the upstream contains_labels flag.
"""
import hashlib, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "p2"))

from p2_features import build_features, COMMIT_STATES

# COPIED VERBATIM FROM p2/run_p2b_predict.py AND VERIFIED BELOW, exactly as in
# p3/run_p3_predict.py. Three copies of one function is deliberate: the P2
# module body refuses to execute once its own frozen file exists, so it cannot
# be imported, and every copy is checked against the original at run time.
def apply_frozen(art, rows):
    cols, med = art["features"], art["preprocessing"]["posN_impute_median"]
    mu = np.array(art["preprocessing"]["standardize_mean"])
    sd = np.array(art["preprocessing"]["standardize_sd"])
    X = np.empty((len(rows), len(cols)))
    for j, c in enumerate(cols):
        X[:, j] = [(med if r["posN"] is None else r["posN"]) if c == "posN" else r[c]
                   for r in rows]
    z = ((X - mu) / sd) @ np.array(art["coef"]) + art["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def _assert_frozen_helper():
    """Fail closed if this file's copy has drifted from the P2 original."""
    src = open(os.path.join(ROOT, "p2", "run_p2b_predict.py"), encoding="utf-8").read()
    mine = open(os.path.abspath(__file__), encoding="utf-8").read()
    pat = r"^def apply_frozen\(.*?(?=\n\n\n|\nreport =)"
    a = re.search(pat, src, re.S | re.M)
    b = re.search(pat, mine, re.S | re.M)
    if a is None or b is None:
        sys.exit("FAIL: cannot locate apply_frozen in P2 source or in this file")
    if a.group(0).strip() != b.group(0).strip():
        sys.exit("FAIL: apply_frozen has drifted from the frozen P2 definition")
    print("frozen helper verified identical to p2/run_p2b_predict.py")


OUT = os.path.join(HERE, "p3_supp_commit.json")
if os.path.exists(OUT):
    sys.exit(f"FAIL: {OUT} exists; refusing to overwrite a frozen file")

STAGE1 = os.path.join(HERE, "p3_supp_predictions.json")
if not os.path.exists(STAGE1):
    sys.exit("FAIL: p3_supp_predictions.json missing -- stage 1 must be frozen first")

# The four cells, fixed by the frozen protocol. alpha is the GSM8K workpoint
# already used in P3; it is NOT re-searched here and no CoT dose curve is run.
CELLS = {
    "llama": ("llama3/gsm_hard", "gsm_hard_8B_11_20.json", 9, (0, -6)),
    "qwen":  ("qwen2.5/gsm_hard", "gsm_hard_7B_16_22.json", 6, (0, 8)),
}
FORBIDDEN = ("answer", "gold", "gold_answer", "correct", "target", "accuracy")
MAX_NEW_TOKENS = 768
# >= 767, not == 768: the stored text is re-tokenized offline so the boundary is
# approximate. See docs/p3_supp_amendment_01.json addition_3.
CAP_THRESHOLD = 767


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_cell(model, alpha, tok):
    sub, fname, n_layers, _ = CELLS[model]
    tag = f"mdf_{alpha}".replace("-", "neg") + "_cot"
    p = os.path.join(ROOT, sub, tag, fname)
    if not os.path.exists(p):
        sys.exit(f"FAIL: missing cell {p}")
    d = json.load(open(p, encoding="utf-8"))
    meta, data = d["meta"], d["data"]

    # Supplement-specific provenance. A cell carrying p3-v1 would claim
    # authorisation from the CLOSED blind validation; cot=true alone does not
    # record which protocol authorised the run (amendment 01, correction 1).
    if meta.get("protocol") != "p3-supp-v1":
        sys.exit(f"FAIL [{model} a={alpha}]: protocol is {meta.get('protocol')!r}, "
                 f"expected 'p3-supp-v1'")
    if meta.get("cot") is not True:
        sys.exit(f"FAIL [{model} a={alpha}]: cot is not True")
    if meta.get("contains_labels") is not False:
        sys.exit(f"FAIL [{model} a={alpha}]: contains_labels is not False")
    leak = sorted({k for s in data for k in s if k.lower() in FORBIDDEN})
    if leak:
        sys.exit(f"FAIL [{model} a={alpha}]: label fields present: {leak}")
    if meta["alpha"] != alpha:
        sys.exit(f"FAIL [{model} a={alpha}]: meta alpha is {meta['alpha']}")
    if len(data) != 300:
        sys.exit(f"FAIL [{model} a={alpha}]: n={len(data)}, expected 300")
    ids = [s["sample_id"] for s in data]
    if len(set(ids)) != len(ids):
        sys.exit(f"FAIL [{model} a={alpha}]: duplicate sample_id")
    exp = 0 if alpha == 0 else n_layers * len(data)
    if meta["steering_fires"] != exp:
        sys.exit(f"FAIL [{model} a={alpha}]: steering_fires "
                 f"{meta['steering_fires']}, expected {exp}")
    # Held identical to the P3 No-CoT cells: the primary metric is a paired
    # contrast between the two cells of one model, and the fixed-budget reading
    # requires every cell to share one generation budget.
    if meta["max_new_tokens"] != MAX_NEW_TOKENS or meta["temperature"] != 0.0:
        sys.exit(f"FAIL [{model} a={alpha}]: generation params differ from the "
                 f"frozen main line")

    rows = [dict(build_features(s["generated"], "gsm8k"),
                 alpha=alpha, sample_id=s["sample_id"]) for s in data]
    assert all("y" not in r and "correct" not in r for r in rows)
    cs = {st: sum(r["commit_state"] == st for r in rows) for st in COMMIT_STATES}
    if sum(cs.values()) != len(rows):
        sys.exit(f"FAIL [{model} a={alpha}]: commit-state partition incomplete")

    ntok = [len(tok(s["generated"], add_special_tokens=False)["input_ids"])
            for s in data]
    cap = sum(t >= CAP_THRESHOLD for t in ntok) / len(ntok)
    return rows, cs, meta["questions_sha256"], cap, float(np.median(ntok))


report = {
    "protocol": "p3-supp-v1",
    "stage": 2,
    "status": "frozen AFTER generation, BEFORE any CoT accuracy was computed",
    "not_a_blind_dataset_validation": (
        "Same 300 questions and same gold, unsealed 2026-08-30. What is "
        "prospective is the CONDITION (CoT), not the data."),
    "what_this_seals": (
        "the commitment side of the four CoT cells, so the locked condition "
        "test cannot be re-specified after accuracy is seen"),
    "score_is_not_calibrated": (
        "The predictor was fitted on 768-token No-CoT GSM8K output. CoT "
        "generations are longer and differently shaped, so the per-cell score "
        "is a descriptive commitment readout, NOT an accuracy estimate."),
    "stage_1_sha256": sha(STAGE1),
    "predictor_sha256": {},
    "cell_sha256": {},
    "models": {},
}

_assert_frozen_helper()
from transformers import AutoTokenizer
TOKS = {"llama": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen": "Qwen/Qwen2.5-7B-Instruct"}

print("=" * 78)
print("P3 supplement stage 2: commitment scores for the four CoT cells")
print("COT ACCURACY IS NOT READ IN THIS FILE")
print("=" * 78)

for model in ("llama", "qwen"):
    ap = os.path.join(ROOT, "p2", f"p2_predictor_{model}.json")
    art = json.load(open(ap))
    report["predictor_sha256"][model] = sha(ap)
    sub, fname, _, doses = CELLS[model]
    tok = AutoTokenizer.from_pretrained(TOKS[model])

    cells, digests = {}, set()
    for a in doses:
        rows, cs, dig, cap, medtok = load_cell(model, a, tok)
        digests.add(dig)
        p = apply_frozen(art, rows)
        cells[a] = {
            "n": len(rows),
            "commit_score": float(p.mean()),
            "commit_state": cs,
            "early_candidate_rate": float(np.mean([r["early_candidate"] for r in rows])),
            "posN_median": (None if all(r["posN"] is None for r in rows) else
                            float(np.median([r["posN"] for r in rows
                                             if r["posN"] is not None]))),
            "cap_hit_rate": round(cap, 4),
            "median_decode_tokens": medtok,
        }
        tag = f"mdf_{a}".replace("-", "neg") + "_cot"
        report["cell_sha256"][f"{model}_a{a}_cot"] = sha(
            os.path.join(ROOT, sub, tag, fname))

    if len(digests) != 1:
        sys.exit(f"FAIL [{model}]: cells do not share one questions digest")
    # Must equal the P3 No-CoT digest: same 300 questions, different condition.
    dig = digests.pop()
    report["models"][model] = {
        "alpha": doses[1],
        "alpha_source": ("frozen GSM8K workpoint, already used in P3; NOT "
                         "re-searched and NOT derived from any CoT data"),
        "questions_sha256": dig,
        "cells": {str(a): cells[a] for a in doses},
        "commit_score_delta": cells[doses[1]]["commit_score"] - cells[0]["commit_score"],
    }

    print(f"\n{model.upper()}  predictor {sha(ap)[:16]}  questions {dig[:16]}")
    for a in doses:
        c = cells[a]
        pn = "n/a" if c["posN_median"] is None else f"{c['posN_median']:.4f}"
        print(f"  a={a:>+3d}  score {c['commit_score']:.4f}   "
              f"early-cand {c['early_candidate_rate']:.3f}   "
              f"posN {pn}   "
              f"cap {c['cap_hit_rate']:.3f}   med-tok {c['median_decode_tokens']:.0f}")
    print(f"  commit-score delta (a={doses[1]:+d} vs 0): "
          f"{report['models'][model]['commit_score_delta']:+.4f}")

json.dump(report, open(OUT, "w"), indent=2)
print(f"\nFROZEN -> {os.path.basename(OUT)}  sha256 {sha(OUT)[:16]}")
print("CoT accuracy may be computed only AFTER this file exists.")
