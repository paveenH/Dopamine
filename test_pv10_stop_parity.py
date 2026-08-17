#!/usr/bin/env python3.10
"""PV10 v2: stop-boundary semantics and alpha=0 / +-4 stop PARITY.

Regression test for the PV10-A interface failure (2026-08-17). Two distinct
defects are covered, and both were real:

  1. STOP PARITY. `llms.generate()` had no `stop_strings` parameter, while
     `regenerate()` did. alpha=0 registers no hook and therefore takes the
     `generate` path, so an alpha=0 cell ran with NO stop marker while its own
     +-4 cells stopped on "#". The two arms of one experiment had different
     generation boundaries -- a confound, not a behaviour.

  2. RUNTIME/PARSER CONTRACT. HF halts only AFTER emitting the marker, so "#"
     stayed in the returned text and `_POLICY_RE` (anchored on `$`) read the
     line as `malformed`. 47 of PV10-A's 58 terminating rounds died this way.

These are unit-level checks: they verify the harness contract, NOT that a real
model will stay in format. Only a GPU run can show that.
"""
import sys

import bandit_pv10 as p10
import bandit_pv10a as p10a
import bandit_pv10_episode as ep

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


ARMS = ["A", "B", "C", "D"]

print("apply_stop_boundary: marker excluded, content before it preserved")
check("bare trailing marker removed",
      p10.apply_stop_boundary("Policy: SAMPLE Button C #")
      == "Policy: SAMPLE Button C ")
check("comment removed with marker",
      p10.apply_stop_boundary("Policy: SAMPLE Button C # Note: x")
      == "Policy: SAMPLE Button C ")
check("no marker -> unchanged",
      p10.apply_stop_boundary("Policy: SAMPLE Button C")
      == "Policy: SAMPLE Button C")
check("cuts at FIRST marker",
      p10.apply_stop_boundary("a # b # c") == "a ")
check("empty/None safe",
      p10.apply_stop_boundary("") == "" and p10.apply_stop_boundary(None) == "")

print("\nrescued: the two PV10-A failure shapes now parse")
for name, text in [("bare trailing #", "Reason: x\nPolicy: SAMPLE Button C #"),
                   ("# Note comment",
                    "Reason: x\nPolicy: SAMPLE Button C # Note: one button")]:
    r = p10.parse_policy(p10.apply_stop_boundary(text), ARMS)
    check(f"{name} -> valid SAMPLE C",
          r.valid and r.action == "SAMPLE" and r.arm == "C",
          f"got valid={r.valid} kind={r.invalid_kind}")

print("\nSTRICTNESS UNCHANGED: truncation must not recover a bad round")
strict = [
    ("prose after arm", "Reason: x\nPolicy: SAMPLE Button C because it is best",
     "malformed"),
    ("conflicting before marker",
     "Reason: x\nPolicy: SAMPLE Button C\nPolicy: COMMIT Button A\n#",
     "conflicting"),
    ("no directive before marker",
     "Reason: x\nPolicy: thinking\n# Policy: SAMPLE Button C", "malformed"),
    ("arm out of set", "Reason: x\nPolicy: SAMPLE Button E #", "arm_out_of_set"),
    ("nothing at all", "I am not sure what to do", "no_policy"),
]
for name, text, kind in strict:
    r = p10.parse_policy(p10.apply_stop_boundary(text), ARMS)
    check(f"{name} still invalid ({kind})",
          (not r.valid) and r.invalid_kind == kind,
          f"got valid={r.valid} kind={r.invalid_kind}")

r = p10.parse_policy(
    p10.apply_stop_boundary("Reason: x\nPolicy: SAMPLE Button C\n"
                            "Policy: SAMPLE Button C #"), ARMS)
check("agreeing repeat still valid (not a conflict)", r.valid and r.arm == "C")

r = p10.parse_policy(
    p10.apply_stop_boundary("Reason: x\nPolicy: SAMPLE Button C #"),
    ARMS, terminal=True)
check("SAMPLE at terminal still invalid",
      (not r.valid) and r.invalid_kind == "sample_at_terminal",
      f"got kind={r.invalid_kind}")

print("\nSTOP PARITY: both generation paths receive the same stop_strings")


class StrictVC:
    """Rejects an omitted stop_strings on EITHER path.

    Deliberately stricter than the shared FakeVC, whose `**kw` swallowed the
    missing argument -- which is exactly why the parity gap went unnoticed.
    """

    def __init__(self, text, layers=9):
        self.tokenizer = None
        self.text = text
        self.layers = layers
        self._fires = 0
        self.stop_seen = []

    def _rec(self, stop_strings, path):
        if not stop_strings:
            raise AssertionError(f"{path}() called without stop_strings")
        self.stop_seen.append((path, tuple(stop_strings)))
        return [self.text]

    def generate(self, inputs, max_new_tokens=128, temperature=0.0,
                 stop_strings=None, **kw):
        if "diff_matrices" in kw:
            raise TypeError("generate() takes no diff_matrices")
        return self._rec(stop_strings, "generate")

    def regenerate(self, inputs, diff_matrices=None, prefill_only=True,
                   prefill_tail_len=1, max_new_tokens=128, temperature=0.0,
                   stop_strings=None, **kw):
        if diff_matrices is None:
            raise ValueError("regenerate requires diff_matrices")
        self._fires += self.layers * len(inputs) * prefill_tail_len
        return self._rec(stop_strings, "regenerate")

    def steering_fire_count(self, reset=False):
        v = self._fires
        if reset:
            self._fires = 0
        return v


import inspect

sig = inspect.signature(ep.run_pv10_episode)
print(f"  (run_pv10_episode params: {', '.join(list(sig.parameters)[:6])} ...)")

src = inspect.getsource(ep.run_pv10_episode)
gen_call = src.split("vc.generate(")[1].split(")")[0] if "vc.generate(" in src else ""
check("unsteered path passes stop_strings",
      "stop_strings" in gen_call, f"got: {gen_call!r}")
regen_call = src.split("vc.regenerate(")[1].split("stop_strings")[0] \
    if "vc.regenerate(" in src else ""
check("steered path still passes stop_strings", "stop_strings" in src)
check("both paths use p10.STOP_STRINGS",
      src.count("stop_strings=list(p10.STOP_STRINGS)") == 2,
      f"count={src.count('stop_strings=list(p10.STOP_STRINGS)')}")
check("episode parses the STOPPED text, not raw",
      "parse_policy(stopped" in src)
check("raw_generation retained for audit", '"raw_generation": raw' in src)

print("\nllms.generate accepts stop_strings and forwards a tokenizer")
import llms
gsig = inspect.signature(llms.VicundaModel.generate)
check("generate() has stop_strings param", "stop_strings" in gsig.parameters)
check("stop_strings defaults to None (existing callers byte-identical)",
      gsig.parameters["stop_strings"].default is None)
gsrc = inspect.getsource(llms.VicundaModel.generate)
check("passes tokenizer with stop_strings (HF requirement)",
      'gen_kwargs["tokenizer"]' in gsrc)
check("only applied when set", "if stop_strings:" in gsrc)

print("\nversion + PV10-B invariants")
check("parser version bumped to v2",
      p10.POLICY_PARSER_VERSION == "pv10-strict-v2",
      p10.POLICY_PARSER_VERSION)
check("interface_tag carries the new parser version",
      "pv10-strict-v2" in p10.interface_tag(4, [0, 1]))
check("PV10-A tag distinct from PV10-B",
      p10a.interface_tag(4, [0, 1]) != p10.interface_tag(4, [0, 1]))
check("PV10-A tag also carries v2",
      "pv10-strict-v2" in p10a.interface_tag(4, [0, 1]))
check("_POLICY_RE unrelaxed (still anchored, still [.!] only)",
      p10._POLICY_RE.pattern ==
      r"policy\s*[:：]\s*(SAMPLE|COMMIT)\s+Button\s+([A-E])\s*[.!]?\s*$")

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("stop-parity + boundary contract OK")
