#!/usr/bin/env python3.10
"""PV10-C: the two checks that prevent a whole batch being voided. No GPU.

Deliberately minimal. Not a smoke test, not a parser re-test (v2 is covered by
test_pv10_stop_parity.py), and not a fires check (the runner verifies fires
fail-closed on the first seed). Only the two mechanical failures that would
silently invalidate every episode:

  1. `Reason: ` must still end on token 220. The cue adds tokens UPSTREAM of
     the anchor; if the tail slides (pv7's `.rstrip()` trap moved it to `:`=25)
     the RSN injection lands somewhere else and the intervention is not what
     it claims to be.
  2. C's TERMINAL prompt must be byte-identical to B's. If it is not, a B-vs-C
     A0 difference is no longer attributable to the competitor cue alone.
"""
import sys

import bandit_pv10 as p10
import bandit_pv10c as p10c

CO = {"A": (1, 2), "B": (0, 1), "C": (3, 4), "D": (1, 1)}
HIST = ["A", "B", "C", "D", "C"]
ORDER = ["A", "B", "C", "D"]

failed = []

# ── 1. anchor still ends on token 220 ───────────────────────────────────────
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
for n in (0, 5, 99, p10.TOTAL_BUDGET):
    ids = tk(p10c.build_decision_prompt(ORDER, CO, HIST, n),
             add_special_tokens=False)["input_ids"]
    ok = ids[-1] == 220
    print(f"{'ok  ' if ok else 'FAIL'}  n={n:3d} tail token {ids[-1]} (want 220)")
    if not ok:
        failed.append(f"anchor n={n}")

# ── 2. terminal prompt byte-identical to PV10-B ─────────────────────────────
T = p10.TOTAL_BUDGET
ok = (p10c.build_decision_prompt(ORDER, CO, HIST, T)
      == p10.build_decision_prompt(ORDER, CO, HIST, T))
print(f"{'ok  ' if ok else 'FAIL'}  terminal prompt byte-identical to PV10-B")
if not ok:
    failed.append("terminal drift")

ok = "strongest alternative" in p10c.build_decision_prompt(ORDER, CO, HIST, 5)
print(f"{'ok  ' if ok else 'FAIL'}  cue present while sampling")
if not ok:
    failed.append("cue missing")



# ── 3. the C prompt actually REACHES the runner ─────────────────────────────
# The seam is `prompt_module`; a driver that forgets to pass it would run
# PV10-B under a PV10-C directory name and the whole cell would be mislabeled.
import bandit_pv10_episode as ep
from test_bandit_pv10_episode import FakeVC, ENV, _orders

seen = {"B": 0, "C": 0}


def script(i, prompt):
    if "strongest alternative" in prompt:
        seen["C"] += 1
    else:
        seen["B"] += 1
    return "Reason: ok\nPolicy: SAMPLE Button A"


for variant, pm in (("B", p10), ("C", p10c)):
    seen["B"] = seen["C"] = 0
    vc = FakeVC(script)
    ep.run_pv10_episode(vc, seed=0, env=ENV, orders=_orders(), diff_mtx=None,
                        alpha=0.0, total_budget=8, max_new_tokens=32,
                        n_steered_layers=9, interface_tag=f"t_{variant}",
                        prompt_module=pm)
    # the terminal round is byte-identical to B in BOTH variants, so C is
    # expected to show exactly one B-shaped (terminal) prompt.
    ok = (seen["C"] > 0 and seen["B"] == 1) if variant == "C" else \
         (seen["C"] == 0 and seen["B"] > 0)
    print(f"{'ok  ' if ok else 'FAIL'}  variant {variant} reaches runner "
          f"(cue prompts={seen['C']}, plain={seen['B']})")
    if not ok:
        failed.append(f"runner variant {variant}")

# ── 4. B and C cannot resume into each other ────────────────────────────────
import run_bandit_pv10_episodes as R

SEEDS = [0, 1, 2, 3, 4, 5, 8, 11, 14, 19, 22, 23, 26, 31, 32, 46, 48, 50, 53, 57]
kb = R.resume_key(0.0, 11, 20, SEEDS, "M", p10)
kc = R.resume_key(0.0, 11, 20, SEEDS, "M", p10c)
ok = kb != kc
print(f"{'ok  ' if ok else 'FAIL'}  B/C resume keys distinct")
if not ok:
    failed.append("resume key collision")

ok = kb == R.resume_key(0.0, 11, 20, SEEDS, "M")
print(f"{'ok  ' if ok else 'FAIL'}  B key unchanged (stored B cells resume)")
if not ok:
    failed.append("B key drift")

print()
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
print("PV10-C: prompt reaches runner; B/C cells are separate")
