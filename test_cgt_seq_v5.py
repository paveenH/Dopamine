"""
CGT-Sequential v5 protocol invariants — offline, no GPU, no server.

v5 is the Qwen label/position de-confounding calibration. It touches a driver
whose v1-v4 results are FROZEN, so the load-bearing check is not "does v5 work"
but "did adding v5 change anything else". Exits non-zero on failure.

Deliberately minimal: only the failures that would void a batch or silently
invalidate a frozen result. Not a parser re-test, not a model test.

    python3.10 test_cgt_seq_v5.py
"""
import collections
import re
import subprocess
import sys

import get_answer_cgt_seq as seq

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS.append(msg)


# ── 1. v1-v4 byte-identity vs the last frozen commit ────────────────────────
# The prompt builders are the only thing a stored v1-v4 result depends on, so
# this is checked against git rather than a hand-copied baseline that can drift.
print("[1] v1-v4 prompt builders unchanged vs git HEAD~ ...")
try:
    old_src = subprocess.run(
        ["git", "show", "HEAD:get_answer_cgt_seq.py"],
        capture_output=True, text=True, check=True).stdout
    import types
    old = types.ModuleType("seq_frozen")
    old.__dict__["__file__"] = "<git>"
    exec(compile(old_src, "<git:get_answer_cgt_seq.py>", "exec"), old.__dict__)
    diffs = 0
    for pv in ("v1", "v2", "v3", "v4"):
        for pres in ("asc", "desc"):
            if old.build_seq_system_prompt(pres, pv) != seq.build_seq_system_prompt(pres, pv):
                diffs += 1
        for bl, rd in [(9, 1), (1, 9), (6, 4), (4, 6)]:
            for reset in (True, False):
                for fb in ("", "Outcome from previous round: x."):
                    a = old.build_color_user_turn(3, 120, bl, rd, reset,
                                                  outcome_feedback=fb, prompt_ver=pv)
                    b = seq.build_color_user_turn(3, 120, bl, rd, reset,
                                                  outcome_feedback=fb, prompt_ver=pv)
                    if a != b:
                        diffs += 1
        for step, npct in [(1, 25), (3, 75), (5, None)]:
            if old.build_bet_user_turn("blue", 50, step, 5, next_pct=npct, prompt_ver=pv) != \
               seq.build_bet_user_turn("blue", 50, step, 5, next_pct=npct, prompt_ver=pv):
                diffs += 1
    check(diffs == 0, f"v1-v4 builders byte-identical ({diffs} diffs)")
    box_diffs = sum(1 for s in range(8)
                    if old.make_box_sequence(s) != seq.make_box_sequence(s))
    check(box_diffs == 0,
          f"make_box_sequence unchanged -- v5's order rng must not consume the "
          f"main stream ({box_diffs} diffs)")
except subprocess.CalledProcessError:
    print("  SKIP (not a git checkout / file not in HEAD)")


# ── 2. strict balance, not expectation ─────────────────────────────────────
print("[2] per-round order balance is STRICT ...")
bad_run = bad_phase = 0
for s in range(20):
    o = seq.make_order_sequence(s)
    if len(o) != seq.TOTAL_INTERACTIONS:
        bad_run += 1
    c = collections.Counter(o)
    if c["blue"] != c["red"]:
        bad_run += 1
    for ph in range(seq.N_PHASES):
        blk = o[ph * seq.ROUND_INTERACTIONS:(ph + 1) * seq.ROUND_INTERACTIONS]
        cb = collections.Counter(blk)
        if cb["blue"] != cb["red"]:
            bad_phase += 1
check(bad_run == 0, "every run is exactly half/half over 20 seeds")
check(bad_phase == 0, "every phase is exactly half/half (keeps strata populated)")
check(len({tuple(seq.make_order_sequence(s)) for s in range(20)}) == 20,
      "20 seeds give 20 distinct order sequences")
check(len(set(seq.make_order_sequence(0))) == 2, "order varies within a run")


# ── 3. all order-bearing surfaces agree, per round ─────────────────────────
print("[3] option order is consistent across surfaces ...")
QPAT = re.compile(r"choose your colour, (\w+) or (\w+)\.")
for first in ("blue", "red"):
    sysp = seq.build_seq_system_prompt("desc", "v5", first_option=first)
    q = seq.build_color_user_turn(1, 100, 9, 1, True, prompt_ver="v5",
                                  first_option=first)
    lines = sysp.split("reply with exactly one word:\n")[1].split("\n")
    exp = ("Blue", "Red") if first == "blue" else ("Red", "Blue")
    check((lines[0], lines[2]) == exp, f"system output list follows first_option={first}")
    check(QPAT.search(q).groups() == exp, f"question wording follows first_option={first}")


# ── 4. the example is neutral, and no stray ordered pair survives ──────────
print("[4] worked example is colour-neutral ...")
ex = re.search(r"\(e\.g\.[^)]*\)", seq.SEQ_SYSTEM_TEMPLATE_V5).group(0)
check("blue" not in ex.lower() and "red" not in ex.lower(),
      f"example names no colour: {ex[:60]}...")
check("%" in ex, "example still states the share=probability rule")
# The scene sentence is the ONE deliberate exception (it introduces the colour
# words and is not an option presentation). Anything else is a regression.
body = seq.SEQ_SYSTEM_TEMPLATE_V5.split("reply with exactly one word:")[0]
ordered = re.findall(r"\b(?:blue or red|red or blue|Blue or Red|Red or Blue)\b", body)
check(ordered == [], f"no fixed 'X or Y' colour pair outside the option list: {ordered}")


# ── 5. v5 inherits v4's bet turn, and resume keys cannot collide ───────────
print("[5] v5 inherits the v4 bet turn; resume key is versioned ...")
same = all(seq.build_bet_user_turn("blue", 50, st, 5, next_pct=np_, prompt_ver="v4")
           == seq.build_bet_user_turn("blue", 50, st, 5, next_pct=np_, prompt_ver="v5")
           for st, np_ in [(1, 25), (3, 75), (5, None)])
check(same, "v5 bet turn == v4 bet turn (only the colour step changed)")
check(hasattr(seq, "ORDER_VERSION") and isinstance(seq.ORDER_VERSION, str),
      f"ORDER_VERSION exists ({getattr(seq, 'ORDER_VERSION', None)!r})")
src = open("get_answer_cgt_seq.py").read()
check('f"ord{ORDER_VERSION}"' in src and 'if args.prompt_ver == "v5" else []' in src,
      "ORDER_VERSION enters iface for v5 ONLY (stored v1-v4 keys unchanged)")
check("prompt_template_first_red" in src,
      "v5 metadata stores BOTH orders (a single template would misattest)")


# ── 6. full episode: order reaches the prompts and the records ────────────
print("[6] end-to-end episode via a fake model ...")


class _Tok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<|{t['role']}|>{t['content']}" for t in msgs) + "<|assistant|>"


class _VC:
    tokenizer = _Tok()

    def __init__(self):
        self.colour_calls = []

    def regenerate(self, inputs, diff_matrices=None, **kw):
        p = inputs[0]
        last_user = p.split("<|user|>")[-1].split("<|assistant|>")[0]
        if "choose your colour" in last_user:
            self.colour_calls.append((p.split("<|user|>")[0], last_user))
            return ["Blue"]
        return ["Accept"]


vc = _VC()
out = seq.run_episode(vc, diff_mtx=[0], seed=0, presentation="desc", use_chat=True,
                      max_new_tokens=8, temperature=1.0, top_p=0.9,
                      save_all_raw=True, prompt_ver="v5", anchor="default")
recs = out["records"]
check(len(recs) == seq.TOTAL_INTERACTIONS, f"{len(recs)} rounds recorded")
c = collections.Counter(r["first_option"] for r in recs)
check(c["blue"] == c["red"], f"records are balanced: {dict(c)}")
check(all("first_option" in r and "chose_first" in r for r in recs),
      "every record carries first_option and chose_first (attribution fields)")
bad_q = bad_s = 0
for r, (sysblk, user) in zip(recs, vc.colour_calls):
    exp = ("Blue", "Red") if r["first_option"] == "blue" else ("Red", "Blue")
    if QPAT.search(user).groups() != exp:
        bad_q += 1
    lst = sysblk.split("reply with exactly one word:")[1].split("\n")
    if (lst[1], lst[3]) != exp:
        bad_s += 1
check(bad_q == 0, f"question order matches first_option in all rounds ({bad_q} bad)")
check(bad_s == 0, f"system list order matches first_option in all rounds ({bad_s} bad)")

# v1-v4 must still report a constant first_option, so old analyses are unaffected
out4 = seq.run_episode(_VC(), diff_mtx=[0], seed=0, presentation="desc", use_chat=True,
                       max_new_tokens=8, temperature=1.0, top_p=0.9,
                       save_all_raw=True, prompt_ver="v4", anchor="default")
check({r["first_option"] for r in out4["records"]} == {"blue"},
      "v4 records a constant first_option='blue' (order was never balanced)")


print()
if FAILS:
    print(f"FAILED ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL CGT-SEQ v5 CHECKS PASSED")
