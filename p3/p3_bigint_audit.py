#!/usr/bin/env python3
"""P3 large-integer audit + exact-integer normalizer (protocol p3-v1 section 2.4).

Run AFTER download, BEFORE any model is run.

WHY. `norm_gsm8k` routes through float(), so integers above 2^53 corrupt
silently: '12345678901234567890' -> '12345678901234567168'. GSM-Hard
deliberately substitutes large values, so the boundary may genuinely be hit.
Continuing to score with a method known to be wrong is not acceptable; but
patching the frozen normalizer would move every stored GSM8K and MATH number,
so the fix is ADDITIVE and used for this validation only.

`norm_exact` must reproduce `norm_gsm8k` EXACTLY on existing GSM8K data. That
equivalence is the acceptance test -- an unverified replacement would silently
redefine accuracy for the whole project.

Large-integer questions are NEITHER deleted NOR separately excluded.
"""
import argparse, glob, json, os, sys
from decimal import Decimal, InvalidOperation
# This file lives in the REPO (so it syncs to the server), but norm_gsm8k lives
# in the offline workspace, which is deliberately not in git. Locate it via
# ROLEANSWER (env) or --roleanswer, and fail closed with an actionable message
# rather than a bare ModuleNotFoundError.
def _load_norm_gsm8k(explicit=None):
    for cand in [explicit, os.environ.get("ROLEANSWER"),
                 os.path.expanduser("~/Documents/RSNResult/RoleAnswer")]:
        if cand and os.path.isfile(os.path.join(cand, "analyze_first_last_acc.py")):
            sys.path.insert(0, cand)
            from analyze_first_last_acc import norm_gsm8k as _n
            return _n, cand
    sys.exit("FAIL: cannot find analyze_first_last_acc.py. Set ROLEANSWER=<path to "
             "RoleAnswer/> or pass --roleanswer. It is not in this repo by design.")

LIMIT = 2 ** 53
norm_gsm8k = None   # bound in main() once the workspace is located


def norm_exact(a):
    """Exact for arbitrary-size integer VALUES; else identical to norm_gsm8k.

    Handling `int(s)` alone is NOT enough, and this was a real defect: it only
    accepts a bare integer literal, so '9007199254740993.0' and
    '9.007199254740993e15' fell through to norm_gsm8k, went back through
    float(), and were corrupted to ...992 -- the exact values the audit flags.
    GSM-Hard's `target` is float64, so the '.0' form is the LIKELY surface form,
    which would have made the repair a no-op precisely where it is needed.

    Decimal parses every one of those literals exactly. A Decimal that is
    integral becomes an exact int; anything else keeps the frozen behaviour.
    """
    s = str(a).strip().replace(",", "")
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return norm_gsm8k(s)          # non-numeric keeps frozen behaviour
    if d == d.to_integral_value():
        return str(int(d))            # exact: no float anywhere
    return norm_gsm8k(s)              # true non-integers keep frozen behaviour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", help=("the LABEL-FREE questions JSON; the 2^53 "
                                         "count is read from its metadata. Never "
                                         "point this at the sealed gold file."))
    ap.add_argument("--roleanswer", help="path to the offline RoleAnswer/ workspace")
    ap.add_argument("--gsm8k_root", default="llama3/gsm8k",
                    help="existing GSM8K results, to prove norm_exact == norm_gsm8k")
    a = ap.parse_args()
    global norm_gsm8k
    norm_gsm8k, ra = _load_norm_gsm8k(a.roleanswer)
    if not os.path.isabs(a.gsm8k_root):
        a.gsm8k_root = os.path.join(ra, a.gsm8k_root)
    print(f"[0] norm_gsm8k loaded from {ra}")

    # [1] VERDICT equivalence on existing GSM8K -- the acceptance test.
    #
    # The criterion is that no scoring VERDICT changes, NOT that no string
    # differs. A normalizer that repairs a float64 corruption will by
    # construction differ on the corrupted string; demanding byte-identity
    # would reject the fix for doing its job. What must never change is
    # whether an answer counts as correct, because that would silently
    # redefine every stored GSM8K and MATH number.
    files = sorted(glob.glob(os.path.join(a.gsm8k_root, "mdf_*", "*answers*.json")))
    if not files:
        sys.exit(f"FAIL: no GSM8K result files under {a.gsm8k_root}; the equivalence "
                 "test would pass VACUOUSLY, which is worse than failing")
    n, flips, repairs = 0, [], []
    for p in files:
        for s in json.load(open(p, encoding="utf-8"))["data"]:
            gold = s.get("answer", s.get("gold_answer"))
            if gold is None:
                continue
            for k, v in s.items():
                if not k.startswith("pred_answer") or v is None:
                    continue
                n += 1
                old_ok = norm_gsm8k(v) == norm_gsm8k(gold)
                new_ok = norm_exact(v) == norm_exact(gold)
                if old_ok != new_ok:
                    flips.append((p, v, gold, old_ok, new_ok))
                elif norm_gsm8k(v) != norm_exact(v):
                    repairs.append((v, norm_gsm8k(v), norm_exact(v)))
    if n == 0:
        sys.exit("FAIL: 0 values compared; a guard never exercised is not a guard")
    print(f"[1] verdict equivalence on GSM8K: {n} predictions from {len(files)} files")
    print(f"    scoring verdicts changed : {len(flips)}")
    print(f"    float64 corruptions repaired (verdict unchanged): {len(repairs)}")
    for r in repairs[:5]:
        print(f"      {r[0]}  frozen->{r[1]}  exact->{r[2]}")
    for f in flips[:10]:
        print("    FLIP", f)
    if flips:
        sys.exit("FAIL: norm_exact changes a stored scoring verdict; do NOT use it")
    print("    no stored verdict changes   OK")

    # [2] REMOVED. The GSM-Hard gold audit runs inside data_gsm_hard.py, in
    # memory, and publishes only `n_gold_exceeding_2_53` + `bigint_audit_digest`
    # into BOTH output files' metadata. Re-auditing from a file would mean
    # opening the sealed gold, which is the one irreversible mistake in P3.
    # Read the count from the questions file's metadata instead -- no unsealing.
    if a.questions:
        m = json.load(open(a.questions, encoding="utf-8"))["meta"]
        n_big = m["n_gold_exceeding_2_53"]
        print(f"[2] gold answers exceeding 2^53 (from questions metadata): {n_big}")
        print("    -> use the ORIGINAL extractor unchanged" if not n_big else
              "    -> norm_exact is REQUIRED for this validation (protocol 2.4)")
    else:
        print("[2] pass --questions <gsm_hard_p3_questions.json> to read the "
              "audit count from metadata (never open the sealed gold file)")


if __name__ == "__main__":
    main()
