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

import json
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

    # identity must not concentrate within a position. Balancing the two
    # MARGINALS alone does not guarantee this — a greedy first-come fill hit
    # exact marginals while covering only 15/25 cells with 3x repeats. The
    # bank is built from a Latin rectangle instead, so the cross table is as
    # even as the arithmetic allows: no cell may exceed ceil(n / k^2).
    rep = br.bank_report(bank, env)
    cap = -(-20 // (env.k ** 2))
    check(rep["max_cell_repeat"] <= cap,
          f"{env.name}: no (position,identity) cell exceeds ceil(n/k^2)={cap}",
          f"max repeat {rep['max_cell_repeat']}")
    check(rep["n_cross_cells_used"] == min(20, env.k ** 2),
          f"{env.name}: cross coverage is maximal "
          f"({rep['n_cross_cells_used']}/{min(20, env.k ** 2)} attainable)")
    check(rep["position_balanced"] and rep["identity_balanced"],
          f"{env.name}: bank_report confirms both marginals balanced")

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
# Different seeds must produce genuinely different tapes, not merely be
# labelled differently. Compare the actual latent draws for a common arm.
# (The earlier version of this check ended in `or True`, so it could never
# fail — it asserted nothing.)
t5 = br.RewardTape(5, hard)
t6 = br.RewardTape(6, hard)
shared = set(t5.arm_map) & set(t6.arm_map)
seq5 = [[t5.peek(a, i) for i in range(40)] for a in sorted(shared)]
seq6 = [[t6.peek(a, i) for i in range(40)] for a in sorted(shared)]
check(seq5 != seq6, "different seeds yield different latent draw sequences")
check([t5.peek(a, i) for a in sorted(shared) for i in range(40)]
      == [br.RewardTape(5, hard).peek(a, i) for a in sorted(shared) for i in range(40)],
      "same seed rebuilds a byte-identical tape")
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

# MID-LINE Choice: must also be removed. An earlier version anchored the
# regex to ^, so "I conclude Choice: Button A" survived into the action
# prompt and created a SECOND anchor, moving the alpha injection site off the
# real decision token.
for mid in ("I conclude Choice: Button A",
            "Therefore, Choice: Button B is best",
            "  ...so Choice:Button C"):
    check(br.sanitize_rationale(mid) == "",
          f"mid-line Choice: removed -> {mid[:28]!r}",
          repr(br.sanitize_rationale(mid)))
check(br.sanitize_rationale("A is good.\nI conclude Choice: Button A") == "A is good.",
      "mid-line Choice: line dropped, preceding line kept")
check("choice" not in br.sanitize_rationale(
          "Choice: Button A\nmid Choice: Button B\nclean line").lower(),
      "no Choice: marker survives sanitization in any position")
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


# ── [11] bootstrap + frozen manifest ────────────────────────────────────────
print("\n[11] bootstrap CIs and the frozen baseline manifest")

vals = [0.0] * 15 + [1.0] * 5          # a 0.25 frequency, like SuffFail
ci1 = br._bootstrap_ci(vals, lambda v: sum(v) / len(v))
ci2 = br._bootstrap_ci(vals, lambda v: sum(v) / len(v))
check(ci1 == ci2, "bootstrap is deterministic under the frozen seed")
check(abs(ci1["point"] - 0.25) < 1e-12, "bootstrap point estimate is the mean")
check(ci1["lo"] <= ci1["point"] <= ci1["hi"], "point estimate lies inside the CI")
check(ci1["lo"] >= 0.0 and ci1["hi"] <= 1.0,
      "frequency CI stays within [0,1]")

deg = br._bootstrap_ci([0.4] * 20, lambda v: sum(v) / len(v))
check(deg["lo"] == deg["hi"] and abs(deg["lo"] - 0.4) < 1e-12,
      "zero-variance sample gives a degenerate CI")

# paired bootstrap: the seed-set guard is the load-bearing part
runs_g = [br.run_greedy(s, easy, br.RewardTape(s, easy)) for s in (0, 1, 2, 3)]
runs_r = [br.run_random(s, easy, br.RewardTape(s, easy)) for s in (0, 1, 2, 3)]
pc = br.paired_bootstrap_ci(runs_g, runs_r, lambda r: r["late_opt_frac"])
check(pc["paired"] is True and pc["seeds"] == [0, 1, 2, 3],
      "paired bootstrap records the seeds it paired on")
check(pc["point"] > 0, "greedy beats random on late_opt_frac (paired)")

try:
    br.paired_bootstrap_ci(runs_g, runs_r[:3], lambda r: r["late_opt_frac"])
    check(False, "paired bootstrap rejects mismatched seed sets")
except ValueError:
    check(True, "paired bootstrap rejects mismatched seed sets")

# a duplicate seed must FAIL, not be silently deduped down to a smaller n
_dup_a = runs_g + [runs_g[0]]
_dup_b = runs_r + [runs_r[0]]
for _lbl, _a, _b in (("list a", _dup_a, runs_r + [runs_r[0]]),
                     ("list b", runs_g + [runs_g[0]], _dup_b)):
    try:
        br.paired_bootstrap_ci(_a, _b, lambda r: r["late_opt_frac"])
        check(False, f"paired bootstrap rejects duplicate seeds in {_lbl}")
    except ValueError as _e:
        check("duplicate seeds" in str(_e),
              f"paired bootstrap rejects duplicate seeds in {_lbl}")

# and the duplicate check must fire BEFORE the seed-set check, otherwise a
# doubled list still compares equal as a SET and slips through
try:
    br.paired_bootstrap_ci(runs_g + [runs_g[0]], runs_r + [runs_r[0]],
                           lambda r: r["late_opt_frac"])
    check(False, "duplicate check precedes the set-equality check")
except ValueError as _e:
    check("duplicate seeds" in str(_e),
          "duplicate check precedes the set-equality check")

man1 = br.build_baseline_manifest()
man2 = br.build_baseline_manifest()
check(man1 == man2, "baseline manifest is reproducible")
check(man1["protocol"] == "pv6", "manifest carries the protocol version")
for _k in ("easy", "hard"):
    _e = man1["environments"][_k]
    _rep = _e["bank_report"]
    check(_rep["position_balanced"] and _rep["identity_balanced"],
          f"manifest {_k}: bank report attests both marginals balanced")
    check(not (set(_e["seed_bank"]) & set(_e["smoke_bank"])),
          f"manifest {_k}: smoke bank is disjoint from the formal bank")
    check(set(_e["policies"]) == {"random", "greedy", "oracle"},
          f"manifest {_k}: all three algorithmic baselines are frozen")
    check(_e["policies"]["oracle"]["late_opt_frac"]["point"] == 1.0,
          f"manifest {_k}: oracle always plays the best arm")
    check(_e["policies"]["greedy"]["suff_fail_freq_half"]["point"]
          > _e["policies"]["random"]["suff_fail_freq_half"]["point"],
          f"manifest {_k}: greedy locks in more than random (gate rule 1 basis)")
    check(_e["policies"]["random"]["k_min_frac_full"]["point"]
          > _e["policies"]["greedy"]["k_min_frac_full"]["point"],
          f"manifest {_k}: random flails more than greedy (gate rule 2 basis)")

# The launcher hardcodes the seed banks so it is self-describing. That is a
# COPY, and a copy can drift — a launcher running a different bank than the
# manifest would silently break counterbalancing while every artifact still
# claimed the frozen bank. Check it here rather than by eye.
import os
import re as _re
_sh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "run_bandit_reference.sh")
if os.path.exists(_sh_path):
    _sh = open(_sh_path).read()

    def _grab(name):
        m = _re.search(rf'^{name}="([^"]*)"', _sh, _re.M)
        return [int(x) for x in m.group(1).split()] if m else None

    for _k, _bv, _sv in (("easy", "BANK_EASY", "SMOKE_SEEDS_EASY"),
                         ("hard", "BANK_HARD", "SMOKE_SEEDS_HARD")):
        _e = man1["environments"][_k]
        check(_grab(_bv) == _e["seed_bank"],
              f"launcher {_bv} matches the frozen manifest bank")
        check(_grab(_sv) == _e["smoke_bank"],
              f"launcher {_sv} matches the frozen manifest smoke bank")
        check(not (set(_grab(_bv) or []) & set(_grab(_sv) or [])),
              f"launcher {_k}: smoke seeds are disjoint from the formal bank")

# the manifest must survive a JSON round-trip, since that is how it is cited
_rt = json.loads(json.dumps(man1, sort_keys=True))
check(_rt["environments"]["easy"]["seed_bank"]
      == man1["environments"]["easy"]["seed_bank"],
      "manifest seed bank survives a JSON round-trip")


# ── summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if FAILS:
    print(f"FAILED ({len(FAILS)}):")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
