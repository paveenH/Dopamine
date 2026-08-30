#!/usr/bin/env python3
"""P3 blind validation -- apply the FROZEN P2 predictors to GSM-Hard and FREEZE
the predictions. Protocol p3-v1 (docs/PREREG_P3.md, tag p3-prereg-v1).

THIS IS THE STEP THAT MAKES P3 BLIND. P2's transfer test was retrospective:
MATH accuracy already existed on disk when the predictor was frozen. Here the
GSM-Hard gold has never been read by anyone, so freezing this file BEFORE
unsealing it is what separates a genuine blind validation from another
retrospective one. Reading gold first destroys that PERMANENTLY -- no later
re-freeze repairs it.

LABEL FIREWALL, STRUCTURAL NOT CONVENTIONAL. The generation JSONs carry only
{sample_id, question, generated}; the gold lives in a SEPARATE sealed file this
script never opens and never takes a path to. build_features() receives a
STRING, so a label is unreachable rather than merely unused -- the same design
as P2 section 3.5. The blind runner already asserted contains_labels=false, and
this script re-checks it rather than trusting the upstream flag.

NOTHING IS REFITTED. Frozen coefficients, frozen standardization, frozen GSM8K
posN median, all six features retained -- including the ones that are
near-degenerate on this task. apply_frozen() and near_optimal() are IMPORTED
from the P2 script, not reimplemented, so the two transfer tests cannot drift
apart.

PRIMARY READOUT IS ORDERING. Dose ordering / direction / selected workpoint --
NOT whether the predicted probability equals GSM-Hard accuracy. GSM-Hard's base
rate is far below GSM8K's (the numbers are enlarged past what an 8B model
reliably computes), so calibration drift is EXPECTED and is not a failure.

BOTH MODELS ARE FULL SCOPE HERE, unlike P2. Llama was local-direction-only on
MATH because only three doses existed; GSM-Hard was generated with five doses
per model by design, so peak / overshoot / regret are testable on both.
"""
import hashlib, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "p2"))

from p2_features import build_features, COMMIT_STATES

# THE TWO FROZEN HELPERS ARE COPIED VERBATIM AND VERIFIED, NOT RE-DERIVED.
# run_p2b_predict.py cannot be imported (its module body refuses to run once
# p2b_predictions.json exists), and slicing source text out of it at runtime is
# unverifiable -- a first attempt at that silently extracted the wrong region.
# So the bodies live here and _assert_frozen_helpers() compares them against the
# P2 source character-for-character: if either is ever edited there, P3 fails
# closed rather than quietly scoring with different arithmetic.
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


def near_optimal(scores, best, tol):
    """Doses whose predicted score is within tol of the best -- the PREDICTED
    near-optimal region. tol is a fixed fraction of the observed spread, declared
    here before any accuracy is seen."""
    return sorted(a for a, s in scores.items() if s >= scores[best] - tol)


def _assert_frozen_helpers():
    """Fail closed if this file's copies have drifted from the P2 originals."""
    src = open(os.path.join(ROOT, "p2", "run_p2b_predict.py"), encoding="utf-8").read()
    mine = open(os.path.abspath(__file__), encoding="utf-8").read()
    for name in ("apply_frozen", "near_optimal"):
        pat = rf"^def {name}\(.*?(?=\n\n\n|\nreport =)"
        a = re.search(pat, src, re.S | re.M)
        b = re.search(pat, mine, re.S | re.M)
        if a is None or b is None:
            sys.exit(f"FAIL: cannot locate {name} in P2 source or in this file")
        if a.group(0).strip() != b.group(0).strip():
            sys.exit(f"FAIL: {name} has drifted from the frozen P2 definition")
    print("frozen helpers verified identical to p2/run_p2b_predict.py")


OUT = os.path.join(HERE, "p3_predictions.json")
if os.path.exists(OUT):
    sys.exit(f"FAIL: {OUT} exists; refusing to overwrite a frozen prediction file")

# Frozen in docs/PREREG_P3.md before generation. Both models are full scope.
CELLS = {
    "llama": ("llama3/gsm_hard", "gsm_hard_8B_11_20.json", 9),
    "qwen":  ("qwen2.5/gsm_hard", "gsm_hard_7B_16_22.json", 6),
}
DOSES = {"llama": (-8, -6, -4, 0, 4), "qwen": (-4, 0, 4, 6, 8)}
FORBIDDEN = ("answer", "gold", "gold_answer", "correct", "target", "accuracy")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_cells(model):
    """Read the ten generation JSONs. Fails closed on anything that would make
    the dose curve incomparable: a wrong n, a wrong fire count, a label field,
    a duplicate sample_id, or a questions digest that differs between cells."""
    sub, fname, n_layers = CELLS[model]
    rows, audits, digests = [], {}, set()
    for a in DOSES[model]:
        tag = f"mdf_{a}".replace("-", "neg")
        p = os.path.join(ROOT, sub, tag, fname)
        if not os.path.exists(p):
            sys.exit(f"FAIL: missing cell {p}")
        d = json.load(open(p, encoding="utf-8"))
        meta, data = d["meta"], d["data"]

        if meta.get("contains_labels") is not False:
            sys.exit(f"FAIL [{model} a={a}]: contains_labels is not False")
        leak = sorted({k for s in data for k in s if k.lower() in FORBIDDEN})
        if leak:
            sys.exit(f"FAIL [{model} a={a}]: label fields present: {leak}")
        if meta["alpha"] != a:
            sys.exit(f"FAIL [{model} a={a}]: meta alpha is {meta['alpha']}")
        if len(data) != 300:
            sys.exit(f"FAIL [{model} a={a}]: n={len(data)}, expected 300")
        ids = [s["sample_id"] for s in data]
        if len(set(ids)) != len(ids):
            sys.exit(f"FAIL [{model} a={a}]: duplicate sample_id")
        # steering_fires = L * B * tail(1); alpha=0 registers an all-zero mask
        exp = 0 if a == 0 else n_layers * len(data)
        if meta["steering_fires"] != exp:
            sys.exit(f"FAIL [{model} a={a}]: steering_fires "
                     f"{meta['steering_fires']}, expected {exp}")
        if meta["max_new_tokens"] != 768 or meta["temperature"] != 0.0:
            sys.exit(f"FAIL [{model} a={a}]: generation params differ from the "
                     f"frozen GSM8K main line")
        digests.add(meta["questions_sha256"])

        cell = [dict(build_features(s["generated"], "gsm8k"),
                     alpha=a, sample_id=s["sample_id"]) for s in data]
        assert all("y" not in r and "correct" not in r for r in cell)
        rows.extend(cell)
        cs = {st: sum(r["commit_state"] == st for r in cell) for st in COMMIT_STATES}
        if sum(cs.values()) != len(cell):
            sys.exit(f"FAIL [{model} a={a}]: commit-state partition incomplete")
        audits[a] = dict(n=len(cell), **cs)

    if len(digests) != 1:
        sys.exit(f"FAIL [{model}]: cells do not share one questions digest: {digests}")
    return rows, audits, digests.pop()


report = {"protocol": "p3-v1",
          "note": ("Frozen BEFORE the GSM-Hard gold was unsealed. Primary readout "
                   "is dose ordering/direction/selection, not calibration."),
          "predictor_sha256": {}, "models": {}}

_assert_frozen_helpers()
print("=" * 78)
print("P3 blind validation: apply frozen P2 predictors to GSM-Hard")
print("GSM-HARD ACCURACY IS NOT READ IN THIS FILE -- gold stays SEALED")
print("=" * 78)

for model in ("llama", "qwen"):
    ap = os.path.join(ROOT, "p2", f"p2_predictor_{model}.json")
    art = json.load(open(ap))
    report["predictor_sha256"][model] = sha(ap)

    rows, audits, digest = load_cells(model)
    p = apply_frozen(art, rows)
    alphas = np.array([r["alpha"] for r in rows])
    scores = {int(a): float(p[alphas == a].mean()) for a in sorted(set(alphas))}

    # Identical selection rule to P2B, imported not rewritten.
    best = max(scores, key=lambda a: (scores[a], -abs(a), a))
    spread = max(scores.values()) - min(scores.values())
    tol = 0.1 * spread
    direction = "positive" if best > 0 else "negative" if best < 0 else "none"

    seq = sorted(a for a in scores if (a >= 0 if best > 0 else a <= 0))
    seq = seq if best > 0 else seq[::-1]
    onset = None
    for i in range(1, len(seq)):
        if scores[seq[i]] <= scores[seq[i - 1]]:
            onset = seq[i]
            break

    report["models"][model] = {
        "scope": "full",
        "questions_sha256": digest,
        "dose_scores": scores,
        "predicted_direction_from_alpha0": direction,
        "predicted_best_alpha": best,
        "predicted_near_optimal": near_optimal(scores, best, tol),
        "near_optimal_tol": tol,
        "predicted_ordering": sorted(scores, key=lambda a: -scores[a]),
        "predicted_plateau_or_overshoot_onset": onset,
        "n_per_cell": {int(a): audits[a]["n"] for a in audits},
        "commit_state_audit": {int(a): audits[a] for a in audits},
    }

    print(f"\n{model.upper()}  predictor {sha(ap)[:16]}  questions {digest[:16]}")
    print("  dose scores (mean predicted correctness):")
    for a in sorted(scores):
        print(f"    a={a:>+3d}  {scores[a]:.4f}"
              + ("   <- predicted best" if a == best else ""))
    print(f"  direction from a=0 : {direction}")
    print(f"  predicted best     : {best:+d}")
    print(f"  near-optimal set   : {report['models'][model]['predicted_near_optimal']}"
          f" (tol {tol:.4f})")
    print(f"  plateau/overshoot  : {onset}")

json.dump(report, open(OUT, "w"), indent=2)
print(f"\nFROZEN -> {os.path.basename(OUT)}  sha256 {sha(OUT)[:16]}")
print("GSM-Hard gold may be unsealed only AFTER this file exists.")
