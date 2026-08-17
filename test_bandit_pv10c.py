#!/usr/bin/env python3.10
"""PV10-C invariants. No GPU. Exits non-zero on failure.

PV10-C differs from PV10-B in EXACTLY ONE WAY: the competitor cue in the
sampling clause. Everything asserted here is a way that claim could be false.
"""
import sys

import bandit_pv10 as p10
import bandit_pv10c as p10c

FAILED = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


CO = {"A": (1, 2), "B": (0, 1), "C": (3, 4), "D": (1, 1)}
HIST = ["A", "B", "C", "D", "C"]
ORDER = ["A", "B", "C", "D"]


def bprompt(n):
    return p10.build_decision_prompt(ORDER, CO, HIST, n)


def cprompt(n):
    return p10c.build_decision_prompt(ORDER, CO, HIST, n)


# ── 1. the cue is present while sampling, and names no arm ──────────────────
s = cprompt(5)
check("cue present in sampling prompt", "strongest alternative" in s)
check("cue differs from PV10-B", s != bprompt(5))
tail = p10c._COMPETITOR_CUE
check("cue names NO specific button",
      not any(f"Button {x}" in tail for x in "ABCDE"),
      "naming an arm would make competitor_named a transcription")

# ── 2. the terminal prompt is byte-identical to PV10-B ──────────────────────
check("terminal prompt byte-identical to PV10-B", cprompt(100) == bprompt(100))
check("cue does NOT leak into terminal prompt",
      "strongest alternative" not in cprompt(100))

# ── 3. anchor: prompt still ends at `Reason: ` with ONE trailing space ───────
for n in (0, 1, 5, 50, 99, 100):
    check(f"n={n} ends at REASON_ANCHOR", cprompt(n).endswith(p10c.REASON_ANCHOR))
check("anchor has exactly one trailing space",
      cprompt(5).endswith("Reason: ") and not cprompt(5).endswith("Reason:  "))

# ── 4. patching is restored -- importing C must not mutate B ────────────────
before = p10._SAMPLING_CLAUSE
cprompt(5)
check("bandit_pv10._SAMPLING_CLAUSE restored after build",
      p10._SAMPLING_CLAUSE == before)
check("PV10-B prompt unaffected by C being imported",
      "strongest alternative" not in bprompt(5))


class _Boom(Exception):
    pass


try:
    p10c.build_decision_prompt(ORDER, CO, None, 5)   # format_history raises
except Exception:
    pass
check("clause restored even when build raises",
      p10._SAMPLING_CLAUSE == before)

# ── 5. shared objects are the SAME objects, not copies ──────────────────────
for name in ("parse_policy", "apply_stop_boundary", "STOP_STRINGS",
             "ARM_LABELS", "REASON_ANCHOR", "TOTAL_BUDGET"):
    check(f"{name} shared with bandit_pv10",
          getattr(p10c, name) is getattr(p10, name))
check("POLICY_PARSER_VERSION unchanged (v2)",
      p10c.POLICY_PARSER_VERSION == p10.POLICY_PARSER_VERSION == "pv10-strict-v2")
check("PROTOCOL_VERSION still pv10", p10c.PROTOCOL_VERSION == "pv10")

# ── 6. the Stage-1 version DIFFERS, so a resume key cannot collide ──────────
check("STAGE1_INSTRUCTION_VERSION differs from PV10-B",
      p10c.STAGE1_INSTRUCTION_VERSION != p10.STAGE1_INSTRUCTION_VERSION,
      f"{p10.STAGE1_INSTRUCTION_VERSION} -> {p10c.STAGE1_INSTRUCTION_VERSION}")

# ── 7. the parser is NOT relaxed by the cue ────────────────────────────────
for bad, why in [
    ("Reason: x\nPolicy: SAMPLE Button C because it is best", "prose after arm"),
    ("Reason: x\nPolicy: SAMPLE Button C\nPolicy: COMMIT Button A", "conflicting"),
    ("Reason: x\nI will sample C", "no directive"),
    ("Reason: x\nPolicy: SAMPLE Button Z", "arm out of set"),
]:
    check(f"still invalid: {why}", not p10c.parse_policy(bad, set("ABCD")).valid)
ok = p10c.parse_policy("Reason: x\nPolicy: SAMPLE Button C", set("ABCD"))
check("well-formed policy still valid", ok.valid and ok.arm == "C")

# ── 8. fire arithmetic is unchanged: the cue adds tokens, not model calls ───
# A full-budget episode makes model_calls = TOTAL_BUDGET - forced_init + 1
# = 100 - 4 + 1 = 97, so 97 x 9 x 1 = 873 -- NOT PV9's 900. The cue adds
# prompt tokens, never a model call, so this must be unchanged.
CALLS = p10.TOTAL_BUDGET - 4 + 1
for alpha in (-4.0, 4.0):
    check(f"full-budget expected_fires == 873 at alpha={alpha}",
          p10.expected_fires(alpha, CALLS) == 873,
          "cue is prompt-side only")
check("alpha=0 registers no hook (fires == 0)",
      p10.expected_fires(0.0, CALLS) == 0)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all PV10-C invariants hold")
