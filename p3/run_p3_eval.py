#!/usr/bin/env python3
"""P3 blind validation -- UNSEAL the gold and evaluate the frozen predictions.

Protocol p3-v1 + amendment p3-amend-05. Offline, python3.10, no GPU.

THIS IS THE IRREVERSIBLE STEP. It refuses to start unless p3_predictions.json
already exists, because reading gold before the predictions are frozen would
destroy the blind character of the test permanently -- no later re-freeze
repairs it.

NOTHING IS TUNED HERE. The predictor, the features, the marker adapter, the
selected workpoints and the success criteria were all frozen before this file
could see a single label. A disappointing result is a result.

TWO INDEPENDENT READOUTS, REPORTED SEPARATELY AND NEVER POOLED:
  (A) five-point ordering  -- can the predictor RANK doses? direction, selected
      workpoint, regret against the observed near-optimal set, Spearman rho,
      overshoot/rollover. Omnibus over five doses, Holm across the two models.
  (B) fixed-workpoint transfer (p3-amend-05) -- does the alpha already
      established on GSM8K (llama -6, qwen +8) beat alpha=0 on a NEW task? ONE
      paired McNemar per model. This is the question a practitioner faces.

ACCURACY CONVENTION. first_acc is the MAIN outcome, using the frozen offline
extractor imported from analyze_first_last_acc -- the same definition as GSM8K
production. last_acc is reported as a sensitivity readout only and never as the
headline. The two known unpatched extractor gaps (leading-zero normalisation,
empty first marker) are left unpatched: patching them would move every frozen
GSM8K and MATH number and needs its own re-freeze.

PREDICTED vs OBSERVED NEAR-OPTIMAL ARE DIFFERENT OBJECTS. The predicted set
comes from the frozen score rule; the observed set is the doses whose paired
per-question accuracy difference from the empirical best is not detectable. The
verdict is whether the PREDICTED workpoint lands in the OBSERVED set -- being
close in predicted score is not sufficient.
"""
import argparse, hashlib, json, os, sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PRED = os.path.join(HERE, "p3_predictions.json")
DOSES = {"llama": (-8, -6, -4, 0, 4), "qwen": (-4, 0, 4, 6, 8)}
CELLS = {"llama": ("llama3/gsm_hard", "gsm_hard_8B_11_20.json"),
         "qwen":  ("qwen2.5/gsm_hard", "gsm_hard_7B_16_22.json")}
# Frozen in p3_amendment_05.json from the GSM8K record, NOT from any GSM-Hard
# predicted score.
FIXED_WORKPOINT = {"llama": -6, "qwen": 8}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------- statistics
def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b (only-A-correct) and
    c (only-B-correct). Binomial(n, 0.5); returns 1.0 when there is nothing
    discordant rather than dividing by zero."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n))


def holm(pvals):
    """Holm-Bonferroni. Returns adjusted p in the ORDER GIVEN."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        out[i] = run
    return out


def spearman(x, y):
    """Rank correlation with average ranks for ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


# ---------------------------------------------------------------- data
def load_gold(path):
    d = json.load(open(path, encoding="utf-8"))
    gold = {int(s["sample_id"]): s for s in d["data"]}
    return d.get("meta", {}), gold


def score_cells(model, gold, gold_digest):
    """Per-dose per-question correctness. Fails closed on anything that would
    silently misalign the paired tests."""
    from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k
    sub, fname = CELLS[model]
    per_dose, meta_out = {}, {}
    for a in DOSES[model]:
        tag = f"mdf_{a}".replace("-", "neg")
        p = os.path.join(ROOT, sub, tag, fname)
        if not os.path.exists(p):
            sys.exit(f"FAIL: missing cell {p}")
        d = json.load(open(p, encoding="utf-8"))
        meta, data = d["meta"], d["data"]
        if meta["questions_sha256"] != gold_digest:
            sys.exit(f"FAIL [{model} a={a}]: questions digest {meta['questions_sha256'][:16]} "
                     f"!= gold digest {gold_digest[:16]}")
        ids = [int(s["sample_id"]) for s in data]
        if len(set(ids)) != len(ids):
            sys.exit(f"FAIL [{model} a={a}]: duplicate sample_id")
        missing = sorted(set(gold) - set(ids))
        extra = sorted(set(ids) - set(gold))
        if missing or extra:
            sys.exit(f"FAIL [{model} a={a}]: sample_id mismatch vs gold "
                     f"(missing {missing[:5]}, extra {extra[:5]})")

        first, last = {}, {}
        for s in data:
            i = int(s["sample_id"])
            g = norm_gsm8k(str(gold[i]["gold"]))
            hits = all_hash(s["generated"])
            if hits:
                first[i] = int(norm_gsm8k(hits[0]) == g)
                last[i] = int(norm_gsm8k(hits[-1]) == g)
            else:
                fb = fallback_gsm8k(s["generated"])
                v = int(fb is not None and norm_gsm8k(fb) == g)
                first[i] = last[i] = v
        per_dose[a] = {"first": first, "last": last}
        meta_out[a] = {"n": len(data), "steering_fires": meta["steering_fires"]}
    return per_dose, meta_out


# ---------------------------------------------------------------- readouts
def evaluate_model(model, pred, per_dose, ids):
    doses = list(DOSES[model])
    acc = {a: sum(per_dose[a]["first"].values()) / len(ids) for a in doses}
    acc_last = {a: sum(per_dose[a]["last"].values()) / len(ids) for a in doses}

    # (A) five-point ordering ------------------------------------------------
    pairs, praw = list(combinations(doses, 2)), []
    for x, y in pairs:
        fx, fy = per_dose[x]["first"], per_dose[y]["first"]
        b = sum(1 for i in ids if fx[i] and not fy[i])
        c = sum(1 for i in ids if fy[i] and not fx[i])
        praw.append(mcnemar_exact(b, c))
    padj = holm(praw)
    omnibus = min(padj) if padj else 1.0

    obs_best = max(doses, key=lambda a: (acc[a], -abs(a), a))
    observed_near = []
    for a in doses:
        fa, fb_ = per_dose[a]["first"], per_dose[obs_best]["first"]
        b = sum(1 for i in ids if fb_[i] and not fa[i])
        c = sum(1 for i in ids if fa[i] and not fb_[i])
        if a == obs_best or mcnemar_exact(b, c) > 0.05:
            observed_near.append(a)

    sel = pred["predicted_best_alpha"]
    pdir = pred["predicted_direction_from_alpha0"]
    odir = "positive" if obs_best > 0 else "negative" if obs_best < 0 else "none"

    ps = pred["dose_scores"]
    rho = spearman([ps[str(a)] for a in doses], [acc[a] for a in doses])

    # (B) fixed-workpoint transfer (p3-amend-05) -----------------------------
    w = FIXED_WORKPOINT[model]
    fw, f0 = per_dose[w]["first"], per_dose[0]["first"]
    b = sum(1 for i in ids if fw[i] and not f0[i])
    c = sum(1 for i in ids if f0[i] and not fw[i])

    return {
        "n": len(ids),
        "accuracy_first": {int(a): acc[a] for a in doses},
        "accuracy_last_sensitivity": {int(a): acc_last[a] for a in doses},
        "five_point": {
            "readable_dose_curve": omnibus <= 0.05,
            "omnibus_min_holm_p": omnibus,
            "pairwise": [{"pair": [x, y], "p_raw": pr, "p_holm": pa}
                         for (x, y), pr, pa in zip(pairs, praw, padj)],
            "observed_best_alpha": obs_best,
            "observed_near_optimal": observed_near,
            "predicted_best_alpha": sel,
            "predicted_direction": pdir,
            "observed_direction": odir,
            "direction_correct": pdir == odir,
            "selected_in_observed_near_optimal": sel in observed_near,
            "regret_pp": 100.0 * (acc[obs_best] - acc[sel]),
            "spearman_rho": rho,
            "predicted_rollover_onset": pred.get("predicted_plateau_or_overshoot_onset"),
        },
        "fixed_workpoint": {
            "alpha": w,
            "source": "frozen GSM8K record (p3_amendment_05)",
            "acc_alpha": acc[w],
            "acc_alpha0": acc[0],
            "diff_pp": 100.0 * (acc[w] - acc[0]),
            "discordant_only_alpha": b,
            "discordant_only_alpha0": c,
            "mcnemar_p": mcnemar_exact(b, c),
            "improves": acc[w] > acc[0],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="path to gsm_hard_p3_gold.SEALED.json")
    ap.add_argument("--out", default=os.path.join(HERE, "p3_evaluation.json"))
    args = ap.parse_args()

    if not os.path.exists(PRED):
        sys.exit("FAIL: p3_predictions.json does not exist. Freeze the predictions "
                 "BEFORE unsealing gold -- this order cannot be repaired afterwards.")
    if os.path.exists(args.out):
        sys.exit(f"FAIL: {args.out} exists; refusing to overwrite an evaluation")

    pred = json.load(open(PRED, encoding="utf-8"))
    gmeta, gold = load_gold(args.gold)
    gdig = gmeta.get("questions_sha256")

    print("=" * 78)
    print("P3 blind validation -- GOLD UNSEALED")
    print(f"  predictions {sha(PRED)[:16]}   gold {sha(args.gold)[:16]}   n_gold {len(gold)}")
    print("=" * 78)

    report = {"protocol": "p3-v1+amend05",
              "predictions_sha256": sha(PRED), "gold_sha256": sha(args.gold),
              "models": {}}

    for model in ("llama", "qwen"):
        pm = pred["models"][model]
        if gdig and pm["questions_sha256"] != gdig:
            sys.exit(f"FAIL [{model}]: prediction digest != gold digest")
        per_dose, cmeta = score_cells(model, gold, pm["questions_sha256"])
        ids = sorted(gold)
        r = evaluate_model(model, pm, per_dose, ids)
        r["cell_meta"] = {int(a): cmeta[a] for a in cmeta}
        report["models"][model] = r

        fp, fw = r["five_point"], r["fixed_workpoint"]
        print(f"\n{model.upper()}  n={r['n']}")
        print("  accuracy (first_acc, MAIN):")
        for a in DOSES[model]:
            mark = ""
            if a == fp["observed_best_alpha"]:
                mark += "  <- observed best"
            if a == fp["predicted_best_alpha"]:
                mark += "  <- PREDICTED"
            print(f"    a={a:>+3d}  {r['accuracy_first'][a]:.4f}{mark}")
        print(f"  [A] readable curve   : {fp['readable_dose_curve']} "
              f"(min Holm p {fp['omnibus_min_holm_p']:.3g})")
        print(f"      direction        : predicted {fp['predicted_direction']} / "
              f"observed {fp['observed_direction']}  -> {fp['direction_correct']}")
        print(f"      observed near-opt: {fp['observed_near_optimal']}")
        print(f"      selected in set  : {fp['selected_in_observed_near_optimal']}")
        print(f"      regret           : {fp['regret_pp']:.2f} pp")
        print(f"      Spearman rho     : {fp['spearman_rho']:+.3f}")
        print(f"  [B] fixed workpoint a={fw['alpha']:+d} vs a=0 : "
              f"{fw['acc_alpha']:.4f} vs {fw['acc_alpha0']:.4f} "
              f"({fw['diff_pp']:+.2f} pp)")
        print(f"      discordant {fw['discordant_only_alpha']}/{fw['discordant_only_alpha0']}"
              f"  McNemar p={fw['mcnemar_p']:.4g}  improves={fw['improves']}")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nWROTE {os.path.basename(args.out)}  sha256 {sha(args.out)[:16]}")


if __name__ == "__main__":
    main()
