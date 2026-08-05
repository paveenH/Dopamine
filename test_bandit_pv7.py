"""Offline pv7 prompt/token contract tests using the real cached tokenizer."""
from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

import bandit_pv7 as pv7
import bandit_reference as br


FAILS = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


tok = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
easy = br.get_environment("easy")
arm_order = ["Button A", "Button B", "Button D", "Button C"]
history = [
    ("Button A", 0), ("Button B", 0), ("Button D", 0),
    *[("Button C", 1)] * 33,
    *[("Button C", 0)] * 12,
]
assert len(history) == 48


print("[1] concise state and fixed OPTIONS table")
state = pv7.render_state(arm_order, history, 48, easy)
check("Round 49 of 100. Future choices after this one: 51." in state,
      "future-choice wording is unambiguous")
check(state.index("Button A") < state.index("Button B")
      < state.index("Button D") < state.index("Button C"),
      "OPTIONS rows preserve the run's display order")
check(state.count("OPTIONS") == 1 and "TRIED OPTIONS" not in state
      and "UNTRIED OPTIONS" not in state,
      "tried/untried rows do not move between separate blocks")
check("Button C: 33 rewards / 45 trials, empirical rate 0.73" in state,
      "successes, trials, and empirical rate are rendered correctly")

round1 = pv7.render_state(arm_order, [], 0, easy)
check(round1.count("UNTRIED (unknown)") == easy.k,
      "untried arms are unknown rather than numeric zero")
check("Future choices after this one: 99." in round1,
      "Round 1 reports 99 future choices")
round100 = pv7.render_state(arm_order, history + [("Button C", 1)] * 51,
                            99, easy)
check("Future choices after this one: 0." in round100,
      "Round 100 reports zero future choices")


print("\n[2] two aligned whitespace anchors")
r_prompt = pv7.build_rationale_prompt(arm_order, history, 48, easy)
clean = pv7.sanitize_rationale(
    "Rates based on one trial are weak; Policy: exploit C but monitor uncertainty.  \n")
a_prompt = pv7.build_action_prompt(arm_order, history, 48, easy, clean)
check(r_prompt.endswith("Evidence: ") and not r_prompt.endswith("Evidence:  "),
      "Stage 1 ends in exactly one ASCII space")
check(a_prompt.endswith("Choose Button: ")
      and not a_prompt.endswith("Choose Button:  "),
      "Stage 2 ends in exactly one ASCII space")
check(clean.endswith("uncertainty.") and not clean.endswith((" ", "\n", "\t")),
      "sanitizer rstrips the rationale, not the prompt")
check("Do not give the final selection" not in r_prompt
      and "Do not state a final choice" not in r_prompt
      and "Do not state a final choice" not in a_prompt,
      "no stale no-choice instruction remains in either stage")


print("\n[3] real Llama-3.1 tokenizer invariants")
check(tok.encode("Answer: ", add_special_tokens=False) == [16533, 25, 220],
      "original RSN Answer anchor ends at token 220")
check(tok.encode("Evidence: ", add_special_tokens=False) == [93702, 25, 220],
      "Stage 1 anchor ends at token 220")
check(tok.encode("Choose Button: ", add_special_tokens=False)
      == [25017, 6739, 25, 220],
      "Stage 2 anchor ends at token 220")
check(tok.encode(r_prompt, add_special_tokens=False)[-1] == 220,
      "long Stage 1 prompt still ends at token 220")
check(tok.encode(a_prompt, add_special_tokens=False)[-1] == 220,
      "long Stage 2 prompt still ends at token 220")
check(tok.encode(r_prompt.rstrip(), add_special_tokens=False)[-1] == 25,
      "negative control: rstrip(prompt) moves injection to colon token 25")
check(tok.encode("  ", add_special_tokens=False) == [256],
      "negative control: two spaces merge to token 256, not 220+220")


print("\n[4] bare candidates and ID-level append")
candidates = pv7.candidate_suffixes(easy)
check(candidates == ["A", "B", "C", "D"],
      "legal candidates are bare A/B/C/D")
check([tok.encode(c, add_special_tokens=False)[0] for c in candidates]
      == [32, 33, 34, 35],
      "candidate token IDs are frozen at 32/33/34/35")
audit = pv7.audit_id_level_continuation(tok, a_prompt, candidates)
rat_audit = pv7.audit_id_level_continuation(tok, r_prompt, candidates)
check(audit["prompt_tail_id"] == 220,
      "ID-level scorer preserves prompt tail token 220")
check(rat_audit["prompt_tail_id"] == 220,
      "the same runtime audit enforces Stage 1 token 220")
check(all(row["full_tail_ids"] == [220, tid]
          for row, tid in zip(audit["candidates"].values(), [32, 33, 34, 35])),
      "each full sequence contains [220, candidate_id] at the boundary")
check(tok.encode(a_prompt + "A", add_special_tokens=False)[-2:] == [25, 362],
      "negative control: string prompt+'A' loses 220 and creates ' A'=362")

bad_tail_failed = False
try:
    pv7.audit_id_level_continuation(tok, a_prompt.rstrip(), candidates)
except AssertionError:
    bad_tail_failed = True
check(bad_tail_failed, "runtime audit rejects a prompt whose token 220 was stripped")


print("\n[5] P2 is an isolated light-instruction factor")
p1 = pv7.build_rationale_prompt(arm_order, [], 0, easy, pv7.PROMPT_P1)
p2 = pv7.build_rationale_prompt(arm_order, [], 0, easy, pv7.PROMPT_P2)
check("very few trials as weak evidence" not in p1
      and "very few trials as weak evidence" in p2,
      "P2 adds the weak-evidence hint and P1 does not")
check(p1.endswith(pv7.RATIONALE_ANCHOR)
      and p2.endswith(pv7.RATIONALE_ANCHOR),
      "both prompt variants preserve the same Stage 1 anchor")


print("\n[6] frozen 20x6 state bank is prompt-independent and renderable")
bank_path = Path(__file__).with_name("bandit_pv7_frozen_states.json")
bank = json.loads(bank_path.read_text())
check(bank["n_states"] == 120
      and set(bank["state_type_counts"].values()) == {20},
      "bank contains 20 seeds x 6 state types")
check(bank["seed_bank"] == br.build_seed_bank(easy),
      "bank seeds equal the frozen Easy seed bank")
render_ok = True
no_leak = True
for snapshot in bank["states"]:
    hist = [(x["arm"], x["reward"]) for x in snapshot["history"]]
    rendered = pv7.render_state(
        snapshot["arm_order"], hist, snapshot["round_idx"], easy)
    render_ok &= rendered.startswith("You will choose one button")
    no_leak &= not any(
        f"{p:.2f}" in rendered
        for p in snapshot["diagnostics"]["true_probs_by_arm"].values()
        if not hist)
check(render_ok, "all 120 snapshots render without schema repair")
check(no_leak, "diagnostic true probabilities are not exposed to Round-1 prompts")
check(bank["n_unique_state_fingerprints"] <= bank["n_states"]
      and sum(bank["duplicate_state_fingerprints"].values()) > 0,
      "overlapping event snapshots are explicitly recorded, not hidden")


if FAILS:
    raise SystemExit(f"{len(FAILS)} pv7 test(s) failed: {FAILS}")
print("\nALL PV7 CHECKS PASSED")
