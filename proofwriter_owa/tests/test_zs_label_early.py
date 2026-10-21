#!/usr/bin/env python3
"""label_early_candidate.py vs the original pw-label-early-v1 script on the
2400 stored 1-shot chat replies (8 cells x 300). Stdlib only. The data lives in
the offline RoleAnswer workspace; if absent the equivalence part is reported as
SKIPPED (not passed) and the exit code is 0 only when RSN_REQUIRE_DATA is unset.
    python3 proofwriter_owa/tests/test_zs_label_early.py"""
import csv, importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import label_early_candidate as lec

WORK = "/Users/paveenhuang/Documents/RSNResult/RoleAnswer"
REF = f"{WORK}/proofwriter_chat_early_candidate_v1"
fails = []
def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c: fails.append(m)

for t in ["True", "\n#### False\nreason", "Answer: Unknown", "The answer is True.",
          "False, because the negation follows."]:
    check(lec.candidate(t) is not None, f"positive: {t!r}")
for t in ["", "We need to check if it is true.", "True or False?", "1. True",
          "Anne is round.", "Reasoning\n#### True"]:
    check(lec.candidate(t) is None, f"negative: {t!r}")
check(lec.candidate("The answer is True. " + "x" * 60) is None, "cap enforced")

if not os.path.exists(f"{REF}/per_sample.csv"):
    print("SKIPPED equivalence on 2400 stored replies (data not found)")
    sys.exit(1 if os.environ.get("RSN_REQUIRE_DATA") else 0)

spec = importlib.util.spec_from_file_location("ref_pw_early", f"{REF}/analyze_proofwriter_chat_early_candidate.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
csv_rows = {(r["model"], int(r["alpha"]), int(r["sample_id"])): r
            for r in csv.DictReader(open(f"{REF}/per_sample.csv"))}
n = mism_fn = mism_csv = 0
for model, alphas, fn in [("llama3", [-6, -4, 0, 4], "proofwriter_owa_8B_11_20.json"),
                          ("qwen2.5", [-6, 0, 6, 8], "proofwriter_owa_7B_16_22.json")]:
    for a in alphas:
        tok = f"neg{-a}" if a < 0 else str(a)
        d = json.load(open(f"{WORK}/AdarResult/{model}/proofwriter_owa/formal_chat_v1_mdf_{tok}/{fn}"))
        for r in d["data"]:
            t = r["generated"]; n += 1
            new = (lec.candidate(t), lec.candidate(t, 40), lec.candidate(t, 80))
            old = (ref.candidate(t), ref.candidate(t, 40), ref.candidate(t, 80))
            mism_fn += new != old
            c = csv_rows[(model, a, r["sample_id"])]
            mism_csv += ((new[0] is not None) != (c["early_candidate"] == "True")
                         or (new[0] or "") != (c["candidate_label"] or "")
                         or (new[1] is not None) != (c["ec40"] == "True")
                         or (new[2] is not None) != (c["ec80"] == "True"))
check(n == 2400, f"replies compared: {n}")
check(mism_fn == 0, f"tag-for-tag equal to the original candidate() (cap 60/40/80): {mism_fn} mismatches")
check(mism_csv == 0, f"equal to the stored per_sample.csv flags and labels: {mism_csv} mismatches")
sys.exit(1 if fails else 0)
