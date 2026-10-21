#!/usr/bin/env python3
"""Zero-shot prompt freeze + marker-registry checks. Stdlib only.
    python3 proofwriter_owa/tests/test_zs_prompt.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import prompt_zeroshot as pz
from prompt import SYSTEM_RULES
from answer_parser import MARKER_FAMILIES, get_marker_family, parse_final_answer_v2

fails = []
def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c: fails.append(m)

TH, Q = "Anne is big. If someone is big then they are round.", "Anne is round."
no, co = pz.build_user_prompt(TH, Q, False), pz.build_user_prompt(TH, Q, True)

SPEC_NOCOT = (SYSTEM_RULES + "\nTheory: " + TH + "\nQuestion: " + Q + "\n\n"
  "Include exactly one answer line using one of these three forms:\n"
  "#### True\n#### False\n#### Unknown\n")
check(no == SPEC_NOCOT, "No-CoT prompt equals the frozen text exactly")
_i = no.index("Include exactly one")
check(co == no[:_i] + "Let's think step by step.\n" + no[_i:], "CoT = No-CoT with one line inserted before 'Include exactly one'")
_l = co.splitlines(); _n = no.splitlines(); _k = _l.index("Let's think step by step.")
check(_l[:_k] + _l[_k + 1:] == _n and _l[_k + 1].startswith("Include exactly one"), "line-wise diff is that single line, placed right before the answer-line instruction")
check(co.endswith("#### Unknown\n") and no.endswith("#### Unknown\n"), "both end at the last answer form")
for bad in ("Reason briefly", "at most 5 steps", "After reasoning", "worked example",
            "Answer:", "Stop immediately", "stop", "Do not write code", "restate the theory", "DIFFERENT theories", "Now answer the theory", "repeat previous"):
    check(bad not in no and bad not in co, f"no residue of old wording: {bad!r}")
check("step by step" not in no, "No-CoT carries no reasoning instruction")
check(no.count("Theory:") == 1 and no.count("Question:") == 1, "single Theory/Question, no exemplar")
check(TH in no and Q in no, "theory/question verbatim")
check(set(pz.TEMPLATE_IDS.values()) <= set(MARKER_FAMILIES), "template ids registered")
for tid in pz.TEMPLATE_IDS.values():
    f = get_marker_family(tid)
    check(f["marker_family"] == "v2" and f["parse_final_answer"] is parse_final_answer_v2,
          f"{tid} -> v2 family (same parser object)")
check(pz.NOCOT_TEMPLATE_ID != "proofwriter-owa-cot-v2" and pz.COT_TEMPLATE_ID != "proofwriter-owa-cot-v2",
      "does not reuse the old CoT template id")
try:
    get_marker_family("proofwriter-owa-zs-unregistered"); check(False, "unknown id must raise")
except ValueError:
    check(True, "unregistered id still fails closed")
try:
    pz.condition_to_cot("x"); check(False, "bad condition must raise")
except ValueError:
    check(True, "bad condition raises")
sys.exit(1 if fails else 0)
