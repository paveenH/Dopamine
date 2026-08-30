#!/usr/bin/env python3
"""P3 supplement -- evaluate the CoT condition transfer.

Protocol p3-supp-v1 (docs/PREREG_P3_SUPPLEMENT.md, tag p3-supp-frozen) with
amendments 01-03. Offline, python3.10, no GPU.

REFUSES TO START unless p3_supp_commit.json exists. Stage 1 froze the predicted
DIRECTION before any CoT cell existed; stage 2 froze the commitment scores
before accuracy. This file is the only place a CoT label is read.

WHAT THIS IS, EXACTLY. A LOCKED PROSPECTIVE test of the CONDITION, not a blind
dataset validation: same 300 questions, same gold, unsealed 2026-08-30. What
did not exist at freeze time is the CoT condition itself. Do not describe the
result as a second blind test, and do not describe it as merely exploratory --
both misstate the evidential level.

TWO EVIDENTIAL TIERS, NEVER POOLED:
  PRIMARY   dAcc = Acc(CoT+alpha) - Acc(CoT), paired per question, exact
            McNemar, Holm across the two models (m=2). This is the locked
            prediction: stage 1 said "positive" for both.
  DESCRIPTIVE  the CoT x alpha interaction. Its No-CoT half is ALREADY
            UNSEALED, so it is not a locked prediction. Reported with a CI,
            EXCLUDED from the Holm family, and its four possible readings were
            pre-registered so the result cannot pick its own frame.

FIXED-BUDGET ESTIMAND. Every cell ran at max_new_tokens=768. The result is a
CoT condition transfer AT THAT BUDGET, not a statement about unconstrained
reasoning. Llama's cap-hit is high enough that this caveat is load-bearing, so
cap_hit is reported beside every accuracy and amendment 03's staged trigger is
evaluated here rather than argued afterwards.

ACCURACY CONVENTION, identical to P3: first_acc MAIN via the frozen offline
extractor imported from analyze_first_last_acc; last_acc sensitivity only.
"""
import hashlib, json, os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

COMMIT = os.path.join(HERE, "p3_supp_commit.json")
if not os.path.exists(COMMIT):
    sys.exit("FAIL: p3_supp_commit.json missing -- stage 2 must be frozen "
             "BEFORE any CoT accuracy is computed. Run freeze_p3_supp_commit.py.")

OUT = os.path.join(HERE, "p3_supp_evaluation.json")
GOLD = os.path.join(ROOT, "gsm_hard_p3_gold.SEALED.json")
CELLS = {"llama": ("llama3/gsm_hard", "gsm_hard_8B_11_20.json", -6),
         "qwen":  ("qwen2.5/gsm_hard", "gsm_hard_7B_16_22.json", 8)}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        out[i] = run
    return out


def boot_ci(diffs, B=10000, seed=0):
    """Paired question-level bootstrap, as declared in stage 1."""
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(B):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]


def score(model, alpha, cot, gold, gold_digest):
    """Per-question correctness for one cell. Fails closed on misalignment."""
    from analyze_first_last_acc import all_hash, norm_gsm8k, fallback_gsm8k
    sub, fname, _ = CELLS[model]
    tag = f"mdf_{alpha}".replace("-", "neg") + ("_cot" if cot else "")
    p = os.path.join(ROOT, sub, tag, fname)
    if not os.path.exists(p):
        sys.exit(f"FAIL: missing cell {p}")
    d = json.load(open(p, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    want = "p3-supp-v1" if cot else "p3-v1"
    if meta.get("protocol") != want:
        sys.exit(f"FAIL [{model} a={alpha} cot={cot}]: protocol "
                 f"{meta.get('protocol')!r}, expected {want!r}")
    if bool(meta.get("cot", False)) != cot:
        sys.exit(f"FAIL [{model} a={alpha} cot={cot}]: cot flag disagrees")
    if meta["questions_sha256"] != gold_digest:
        sys.exit(f"FAIL [{model} a={alpha} cot={cot}]: questions digest != gold")
    ids = [int(s["sample_id"]) for s in data]
    if len(set(ids)) != len(ids):
        sys.exit(f"FAIL [{model} a={alpha} cot={cot}]: duplicate sample_id")
    if sorted(ids) != sorted(gold):
        sys.exit(f"FAIL [{model} a={alpha} cot={cot}]: sample_id mismatch vs gold")

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
    return first, last


gd = json.load(open(GOLD, encoding="utf-8"))
gold = {int(s["sample_id"]): s for s in gd["data"]}
gold_digest = gd["meta"]["questions_sha256"]
commit = json.load(open(COMMIT, encoding="utf-8"))

print("=" * 78)
print("P3 supplement: CoT condition transfer")
print("LOCKED on the CONDITION -- NOT a blind dataset validation")
print(f"stage-2 commitment freeze: {sha(COMMIT)[:16]}")
print("=" * 78)

report = {
    "protocol": "p3-supp-v1",
    "evidential_status": (
        "LOCKED PROSPECTIVE test of the CONDITION. Same 300 questions and same "
        "gold as P3, unsealed 2026-08-30; the CoT cells did not exist when the "
        "direction was frozen. NOT a second blind dataset validation."),
    "estimand": ("CoT condition transfer at a FIXED 768-token generation "
                 "budget. Not a statement about unconstrained reasoning."),
    "does_not_modify": "docs/p3_result_20260830.json; tag p3-result-unsealed",
    "stage1_sha256": commit["stage_1_sha256"],
    "stage2_sha256": sha(COMMIT),
    "gold_sha256": sha(GOLD),
    "models": {}, "holm_primary": {},
}

praw, order = [], []
for model, (sub, fname, alpha) in CELLS.items():
    c0f, c0l = score(model, 0, True, gold, gold_digest)
    caf, cal = score(model, alpha, True, gold, gold_digest)
    n0f, _ = score(model, 0, False, gold, gold_digest)
    naf, _ = score(model, alpha, False, gold, gold_digest)
    ids = sorted(gold)

    acc = lambda d: sum(d.values()) / len(ids)
    b = sum(1 for i in ids if caf[i] and not c0f[i])
    c = sum(1 for i in ids if c0f[i] and not caf[i])
    p = mcnemar_exact(b, c)
    praw.append(p); order.append(model)

    d_cot = [caf[i] - c0f[i] for i in ids]
    lo, hi = boot_ci(d_cot)
    d_nocot = [naf[i] - n0f[i] for i in ids]
    inter = [a - b_ for a, b_ in zip(d_cot, d_nocot)]
    ilo, ihi = boot_ci(inter)

    cc = commit["models"][model]["cells"]
    report["models"][model] = {
        "alpha": alpha,
        "predicted_direction": "positive",
        "primary": {
            "acc_cot_alpha": acc(caf), "acc_cot_0": acc(c0f),
            "dAcc_pp": (acc(caf) - acc(c0f)) * 100,
            "discordant_b_only_alpha": b, "discordant_c_only_0": c,
            "mcnemar_p": p,
            "boot_ci_pp": [lo * 100, hi * 100],
            "direction_matches_prediction": acc(caf) > acc(c0f),
        },
        "sensitivity_last_acc": {"cot_alpha": acc(cal), "cot_0": acc(c0l)},
        "descriptive_interaction": {
            "excluded_from_holm": True,
            "why": ("its No-CoT half was already unsealed, so it is not a "
                    "locked prediction"),
            "dAcc_cot_pp": (acc(caf) - acc(c0f)) * 100,
            "dAcc_nocot_pp": (acc(naf) - acc(n0f)) * 100,
            "interaction_pp": (sum(inter) / len(ids)) * 100,
            "boot_ci_pp": [ilo * 100, ihi * 100],
        },
        "budget": {
            "max_new_tokens": 768,
            "cap_hit_cot_0": cc["0"]["cap_hit_rate"],
            "cap_hit_cot_alpha": cc[str(alpha)]["cap_hit_rate"],
            "median_decode_cot_0": cc["0"]["median_decode_tokens"],
            "median_decode_cot_alpha": cc[str(alpha)]["median_decode_tokens"],
        },
        "commitment_frozen_stage2": {
            "commit_score_0": cc["0"]["commit_score"],
            "commit_score_alpha": cc[str(alpha)]["commit_score"],
            "early_candidate_0": cc["0"]["early_candidate_rate"],
            "early_candidate_alpha": cc[str(alpha)]["early_candidate_rate"],
            "posN_median_0": cc["0"]["posN_median"],
            "posN_median_alpha": cc[str(alpha)]["posN_median"],
        },
    }

padj = holm(praw)
for m, pa in zip(order, padj):
    report["holm_primary"][m] = pa
    report["models"][m]["primary"]["mcnemar_p_holm"] = pa

for m in order:
    r = report["models"][m]
    pr, it, bu, cm = (r["primary"], r["descriptive_interaction"],
                      r["budget"], r["commitment_frozen_stage2"])
    print(f"\n{m.upper()}  CoT alpha={r['alpha']:+d}   [PRIMARY]")
    print(f"  Acc(CoT+a) {pr['acc_cot_alpha']:.4f}   Acc(CoT) {pr['acc_cot_0']:.4f}"
          f"   dAcc {pr['dAcc_pp']:+.2f} pp")
    print(f"  discordant {pr['discordant_b_only_alpha']}/{pr['discordant_c_only_0']}"
          f"   McNemar p={pr['mcnemar_p']:.3g}   Holm p={pr['mcnemar_p_holm']:.3g}")
    print(f"  bootstrap 95% CI  [{pr['boot_ci_pp'][0]:+.2f}, {pr['boot_ci_pp'][1]:+.2f}] pp")
    print(f"  predicted positive -> "
          f"{'MATCHED' if pr['direction_matches_prediction'] else 'NOT MATCHED'}")
    print(f"  [DESCRIPTIVE] interaction {it['interaction_pp']:+.2f} pp "
          f"[{it['boot_ci_pp'][0]:+.2f}, {it['boot_ci_pp'][1]:+.2f}]"
          f"   (No-CoT dAcc {it['dAcc_nocot_pp']:+.2f} pp)")
    print(f"  [BUDGET] cap-hit {bu['cap_hit_cot_0']:.3f} -> {bu['cap_hit_cot_alpha']:.3f}"
          f"   med-tok {bu['median_decode_cot_0']:.0f} -> {bu['median_decode_cot_alpha']:.0f}")
    print(f"  [COMMIT, frozen stage 2] score {cm['commit_score_0']:.4f} -> "
          f"{cm['commit_score_alpha']:.4f}   early-cand "
          f"{cm['early_candidate_0']:.3f} -> {cm['early_candidate_alpha']:.3f}")

json.dump(report, open(OUT, "w"), indent=2)
print(f"\n-> {os.path.basename(OUT)}  sha256 {sha(OUT)[:16]}")
