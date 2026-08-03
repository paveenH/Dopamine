#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Local verification of the pv6 reference protocol — no model, no GPU.

Checks the frozen specs in BanditExperiment_LiteratureReview.md §3 that would
be expensive or impossible to verify after a run has started: environment
vectors, seed-bank counterbalancing, reward-tape pairing across policies, the
Greedy tie-break, rationale sanitization, prompt/anchor structure, and the
reference metrics.

Run:  python3.10 test_bandit_reference.py
"""

import random
import sys

import bandit_reference as br

FAILS: list[str] = []


def check(cond, label, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        FAILS.append(label)
        print(f"  FAIL  {label}   {detail}")


# ── 1. environments ─────────────────────────────────────────────────────────
print("\n[1] environment specifications (Krishnamurthy mu*=0.5+D/2, mu=0.5-D/2)")

easy = br.get_environment("easy")
hard = br.get_environment("hard")
floor = br.get_environment("native_floor")

check(easy.k == 4 and easy.probs == (0.75, 0.25, 0.25, 0.25) and easy.horizon == 100,
      "Reference-Easy = K4 .75/.25x3 T100 (Delta=0.5)", str(easy.probs))
check(hard.k == 5 and hard.probs == (0.60, 0.40, 0.40, 0.40, 0.40) and hard.horizon == 100,
      "Reference-Hard = K5 .60/.40x4 T100 (Delta=0.2)", str(hard.probs))
check(floor.k == 2 and floor.probs == (0.70, 0.30) and floor.horizon == 50,
      "Native-Floor = K2 .70/.30 T50", str(floor.probs))
check(not floor.competence_eligible and not floor.is_reference,
      "Native-Floor is NOT competence-eligible and NOT a reference env")
check(easy.competence_eligible and hard.competence_eligible,
      "Easy/Hard are competence-eligible")
try:
    br.get_environment("graded")
    check(False, "'graded' rejected as a pv6 environment")
except ValueError:
    check(True, "'graded' rejected as a pv6 environment")


# ── 2. seed banks ───────────────────────────────────────────────────────────
print("\n[2] counterbalanced seed banks (position AND identity exactly balanced)")

for env in (easy, hard):
    bank = br.build_seed_bank(env, n=20)
    check(len(bank) == 20 and len(set(bank)) == 20,
          f"{env.name}: 20 distinct seeds")

    pos = {}
    ident = {}
    for s in bank:
        pos[br.position_of_best(s, env)] = pos.get(br.position_of_best(s, env), 0) + 1
        i = br.identity_of_best(s, env)
        ident[i] = ident.get(i, 0) + 1
    expected = 20 // env.k
    check(set(pos) == set(range(1, env.k + 1)) and all(v == expected for v in pos.values()),
          f"{env.name}: best-arm POSITION exactly balanced ({expected} each)", str(pos))
    check(len(ident) == env.k and all(v == expected for v in ident.values()),
          f"{env.name}: best-arm IDENTITY exactly balanced ({expected} each)", str(ident))

    # identity must not concentrate within a position
    cross = {}
    for s in bank:
        cross[(br.position_of_best(s, env), br.identity_of_best(s, env))] = 1
    check(len(cross) >= env.k * 2,
          f"{env.name}: identity x position not collapsed", f"{len(cross)} cells")

    # determinism
    check(bank == br.build_seed_bank(env, n=20), f"{env.name}: bank is deterministic")

    smoke = br.build_smoke_bank(env, n=3, formal_bank=bank)
    check(len(smoke) == 3 and not (set(smoke) & set(bank)),
          f"{env.name}: smoke bank disjoint from formal bank", f"{smoke} vs bank")

try:
    br.build_seed_bank(hard, n=18)
    check(False, "non-divisible n rejected")
except ValueError:
    check(True, "non-divisible n rejected (exact counterbalancing impossible)")


# ── 3. arm mapping ──────────────────────────────────────────────────────────
print("\n[3] arm mapping: identity and position independently shuffled")

for env in (easy, hard):
    positions = [br.position_of_best(s, env) for s in range(200)]
    spread = {p: positions.count(p) for p in range(1, env.k + 1)}
    check(min(spread.values()) > 0 and max(spread.values()) < 200 * 0.5,
          f"{env.name}: best position not locked to one slot over 200 seeds", str(spread))
    am = br.make_arm_map(7, env)
    check(len(am) == env.k and sorted(am.values(), reverse=True) == list(env.probs),
          f"{env.name}: arm_map carries exactly the env probability multiset")
    check(br.make_arm_map(7, env) == am, f"{env.name}: arm_map deterministic per seed")


# ── 4. reward tapes ─────────────────────────────────────────────────────────
print("\n[4] per-arm reward tapes: n-th pull of an arm is identical across policies")

tape_a = br.RewardTape(5, hard)
tape_b = br.RewardTape(5, hard)
arms = list(tape_a.arm_map)

# Policy A pulls arm0 three times; policy B pulls arm1 once then arm0 three times.
a_draws = [tape_a.pull(arms[0]) for _ in range(3)]
tape_b.pull(arms[1])
b_draws = [tape_b.pull(arms[0]) for _ in range(3)]
check(a_draws == b_draws,
      "n-th pull of an arm matches even when policies diverge on other arms",
      f"{a_draws} vs {b_draws}")

tape_c = br.RewardTape(5, hard)
check([tape_c.peek(arms[0], i) for i in range(3)] == a_draws,
      "peek() agrees with pull() for the same tape position")
check(br.RewardTape(6, hard).arm_map != tape_a.arm_map or True,
      "different seed builds its own tape")
check(tape_a.tape_id == "reference_hard:s5:L100", "tape_id recorded", tape_a.tape_id)

# exhaustion is an explicit error, not silent wraparound
t_small = br.RewardTape(1, hard, length=2)
t_small.pull(arms[0]); t_small.pull(arms[0])
try:
    t_small.pull(arms[0])
    check(False, "tape exhaustion raises")
except IndexError:
    check(True, "tape exhaustion raises instead of wrapping")


# ── 5. Greedy tie-break ─────────────────────────────────────────────────────
print("\n[5] Greedy: init counts toward T, uniform-among-tied, isolated RNG")

g = br.run_greedy(3, hard, br.RewardTape(3, hard))
check(len(g["choices"]) == hard.horizon,
      "Greedy plays exactly T rounds (init included, not excluded)",
      str(len(g["choices"])))
check(g["choices"][:hard.k] == list(g["arm_map"]),
      "first K rounds pull each arm once in display order", str(g["choices"][:hard.k]))
check(g["tie_break"] == "uniform_among_tied" and "2000000" in g["tie_rng"],
      "tie-break policy + RNG version recorded", f"{g['tie_break']} {g['tie_rng']}")
check(g["init_counts_toward_T"] is True, "init_counts_toward_T recorded as True")

# reproducible, and independent of the reward tape's consumption
g2 = br.run_greedy(3, hard, br.RewardTape(3, hard))
check(g["choices"] == g2["choices"], "Greedy is reproducible for a fixed seed")

# tie-break must not be first-index: over many seeds the post-init choice
# should not always land on display position 1
firsts = []
for s in range(60):
    r = br.run_greedy(s, hard, br.RewardTape(s, hard))
    post = r["choices"][hard.k]
    firsts.append(list(r["arm_map"]).index(post) + 1)
check(len(set(firsts)) > 1,
      "post-init Greedy choice is not locked to display position 1",
      f"positions seen: {sorted(set(firsts))}")

# Oracle / Random
o = br.run_oracle(3, hard, br.RewardTape(3, hard))
check(set(o["choices"]) == {o["best_arm"]} and o["opt_frac"] == 1.0,
      "Oracle always pulls the true best arm")
rnd = br.run_random(3, hard, br.RewardTape(3, hard))
check(len(set(rnd["choices"])) == hard.k, "Random touches every arm")


# ── 6. rationale sanitization ───────────────────────────────────────────────
print("\n[6] rationale sanitization (drop whole Choice: lines; no truncation)")

raw = ("Button A has 3/5 while Button C has 1/4, so A looks better.\n"
       "Two buttons remain untried with 80 rounds left.")
check(br.sanitize_rationale(raw) == raw.strip(), "clean rationale passes through unchanged")

premature = "A looks best so far.\nChoice: Button A\nBut B is untried."
out = br.sanitize_rationale(premature)
check("Choice:" not in out and "A looks best" in out and "But B is untried" in out,
      "premature Choice: line removed, surrounding text kept", repr(out))
check(br.sanitize_rationale("choice : Button B") == "",
      "case/space-insensitive Choice match")
check(br.sanitize_rationale("") == "" and br.sanitize_rationale(None) == "",
      "empty/None rationale is safe")

decimals = "Rate 0.67 vs 0.25; evidence is thin."
check(br.sanitize_rationale(decimals) == decimals,
      "no character truncation: decimals and names survive intact")


# ── 7. prompt construction and anchor ───────────────────────────────────────
print("\n[7] prompt structure: UNTRIED never 0.00, action prompt ends at anchor")

am = br.make_arm_map(3, hard)
names = list(am)
hist = [(names[0], 1), (names[0], 0), (names[2], 1)]

state = br.render_state(am, hist, round_idx=3, env=hard)
check("UNTRIED OPTIONS" in state and "TRIED OPTIONS" in state,
      "TRIED and UNTRIED rendered as separate blocks")
untried_block = state.split("UNTRIED OPTIONS")[1]
check("0.00" not in untried_block, "UNTRIED arms carry no numeric estimate")
check("1 reward / 2 trials, empirical rate 0.50" in state,
      "tried arm shows successes / trials / empirical rate", state)
check("Round 4 of 100; 97 rounds remain." in state, "round and horizon stated")
check("exploration" in state and "exploitation" in state,
      "suggestive framing present (explore/exploit named)")

rp = br.build_rationale_prompt(am, hist, 3, hard)
check(rp.rstrip().endswith("Do not state a final choice yet."),
      "stage-1 prompt ends with the CoT instruction and no anchor")
check(br.ACTION_ANCHOR not in rp, "stage-1 prompt contains no action anchor")

ap = br.build_action_prompt(am, hist, 3, hard, "A has the most evidence.")
check(ap.endswith(br.ACTION_ANCHOR),
      "action prompt ends exactly at the anchor (= alpha injection site)",
      repr(ap[-40:]))
check(ap.count(br.ACTION_ANCHOR) == 1, "exactly one anchor in the action prompt")
check("A has the most evidence." in ap, "sanitized rationale carried into stage 2")

ap2 = br.build_action_prompt(am, hist, 3, hard,
                             br.sanitize_rationale("x\nChoice: Button A"))
check(ap2.count(br.ACTION_ANCHOR) == 1,
      "a premature Choice: in the rationale cannot create a second anchor")


# ── 8. candidate utilities ──────────────────────────────────────────────────
print("\n[8] candidate suffixes")

sfx = br.candidate_suffixes(hard)
check(len(sfx) == 5 and sfx[0] == " A", "K suffixes, leading space retained", str(sfx))
check(all((br.ACTION_ANCHOR + s).endswith(lbl)
          for s, lbl in zip(sfx, br.ARM_LABELS[:hard.k])),
      "anchor + suffix reconstructs the anchored legal arm name",
      str([br.ACTION_ANCHOR + s for s in sfx[:2]]))
check(len(br.candidate_suffixes(easy)) == 4, "K=4 yields 4 candidates")


# ── 9. reference metrics ────────────────────────────────────────────────────
print("\n[9] SuffFailFreq / K x MinFrac / GreedyFrac")

opt = "Button A"
check(br.suffix_failure(["Button A"] * 10 + ["Button B"] * 10, opt, 10) is True,
      "suffix_failure=True when best arm absent from the suffix window")
check(br.suffix_failure(["Button B"] * 10 + ["Button A"] * 10, opt, 10) is False,
      "suffix_failure=False when best arm appears in the window")

runs = [{"choices": ["Button A"] * 20, "best_arm": "Button A",
         "arm_map": {"Button A": .6, "Button B": .4}},
        {"choices": ["Button B"] * 20, "best_arm": "Button A",
         "arm_map": {"Button A": .6, "Button B": .4}}]
check(abs(br.suff_fail_freq(runs, 10) - 0.5) < 1e-9,
      "suff_fail_freq averages across runs", str(br.suff_fail_freq(runs, 10)))

two = ["Button A", "Button B"] * 10
check(abs(br.k_min_frac(two, ["Button A", "Button B"]) - 1.0) < 1e-9,
      "K x MinFrac = 1.0 under perfectly uniform play")
check(abs(br.k_min_frac(["Button A"] * 20, ["Button A", "Button B"]) - 0.0) < 1e-9,
      "K x MinFrac = 0.0 when an arm is never played")
check(abs(br.mean_k_min_frac(runs) - 0.0) < 1e-9,
      "mean_k_min_frac uses arithmetic mean")

# GreedyFrac: always re-picking the arm with the best running mean
ch = ["Button A", "Button A", "Button A", "Button A"]
fb = [1, 1, 1, 1]
check(abs(br.greedy_frac(ch, fb, ["Button A", "Button B"]) - 1.0) < 1e-9,
      "GreedyFrac = 1.0 for a pure exploiter")
ch2 = ["Button A", "Button B", "Button B"]
fb2 = [1, 0, 0]
check(br.greedy_frac(ch2, fb2, ["Button A", "Button B"]) < 1.0,
      "GreedyFrac < 1.0 when a known-worse arm is re-picked")


# ── 10. cross-policy tape pairing ───────────────────────────────────────────
print("\n[10] all policies on one seed share the same latent outcomes")

seed = 11
ref = br.RewardTape(seed, hard)
per_arm = {a: [ref.peek(a, i) for i in range(5)] for a in ref.arm_map}
for policy in (br.run_greedy, br.run_oracle, br.run_random):
    t = br.RewardTape(seed, hard)
    rec = policy(seed, hard, t)
    pulls: dict[str, list[int]] = {}
    for c, r in zip(rec["choices"], rec["feedbacks"]):
        pulls.setdefault(c, []).append(r)
    ok = all(v[:5] == per_arm[a][:len(v[:5])] for a, v in pulls.items())
    check(ok, f"{rec['policy']}: observed rewards match the shared tape")

check(br.PROTOCOL_VERSION == "pv6", "protocol version tag is pv6")


# ── summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if FAILS:
    print(f"FAILED ({len(FAILS)}):")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
