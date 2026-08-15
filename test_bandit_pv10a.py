#!/usr/bin/env python3.10
"""PV10-A invariants: the fixed-budget control differs from PV10-B in EXACTLY
one respect -- COMMIT is withheld until the budget is spent.

Anything else differing would break the A-vs-B attribution the control exists
to support, so those are asserted as equalities rather than assumed.
"""
import sys

import bandit_pv10 as p10
import bandit_pv10a as p10a

fails = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok: fails.append(msg)

D = ["C", "A", "D", "B"]
C = {"A": (1, 3), "B": (0, 1), "C": (2, 4), "D": (1, 2)}
H = ["C", "A", "D", "B", "C", "A", "C", "D", "C", "A"]

print("[the one intended difference]")
mid = p10a.build_decision_prompt(D, C, H, n=10)
check("COMMIT" not in mid, "COMMIT is NOT offered while budget remains")
check("SAMPLE" in mid, "SAMPLE is offered")
check("COMMIT" in p10.build_decision_prompt(D, C, H, n=10),
      "PV10-B DOES offer COMMIT at the same n (the contrast is real)")

print("\n[everything else is shared]")
check(mid.endswith(p10.REASON_ANCHOR), "same Reason anchor")
check(not mid.endswith("  "), "no double space before the anchor")
term_a = p10a.build_decision_prompt(D, C, H, n=p10.TOTAL_BUDGET)
term_b = p10.build_decision_prompt(D, C, H, n=p10.TOTAL_BUDGET)
check(term_a == term_b, "terminal prompt is BYTE-IDENTICAL to PV10-B's")
check("OPTIONS" in mid and "empirical rate" in mid, "same OPTIONS block")
check("CHOICE HISTORY" in mid, "same history block")
check(f"Samples used: 10" in mid, "budget lines reflect n")
check("Samples remaining: 90" in mid, "remaining budget is correct")

print("\n[no early-stopping hint]")
low = mid.lower()
for phrase in ["as few samples", "stop early", "when you are confident",
               "commit to that button"]:
    check(phrase not in low, f"does not hint at stopping: {phrase!r}")

print("\n[parser is shared, and still refuses COMMIT-less states correctly]")
ok = p10.parse_policy("Reason: x\nPolicy: SAMPLE Button C", D, terminal=False)
check(ok.valid and ok.action == "SAMPLE", "SAMPLE parses under the shared parser")
term = p10.parse_policy("Reason: x\nPolicy: COMMIT Button C", D, terminal=True)
check(term.valid and term.action == "COMMIT", "terminal COMMIT parses")

print("\n[identity]")
check(p10a.PROTOCOL_VERSION != p10.PROTOCOL_VERSION, "distinct protocol version")
tag_a = p10a.interface_tag(4, [0, 1, 2])
tag_b = p10.interface_tag(4, [0, 1, 2])
check(tag_a != tag_b, "distinct interface tag (cannot resume into PV10-B)")
check(tag_a.startswith("pv10a_p10a"), f"tag names the protocol: {tag_a}")

print("\n[bounds]")
try:
    p10a.build_decision_prompt(D, C, H, n=p10.TOTAL_BUDGET + 1)
    fails.append("n > budget must raise")
except ValueError:
    check(True, "n > budget raises")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print("\n[tokenizer: REAL Llama-3.1]")
    for n in (4, 50, 99):
        ids = tok(p10a.build_decision_prompt(D, C, H, n=n))["input_ids"]
        check(ids[-1] == p10.EXPECTED_WHITESPACE_TOKEN_ID,
              f"n={n}: prompt ends on token 220 ({len(ids)} tokens)")
except Exception as e:
    print(f"\n[tokenizer] SKIPPED ({type(e).__name__})")

print()
if fails:
    print("FAIL"); [print("  -", f) for f in fails]; sys.exit(1)
print("all PV10-A invariants hold")
