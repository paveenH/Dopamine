#!/usr/bin/env python3.10
"""PV10 protocol invariants. Exits non-zero on failure. No GPU, no server.

Covers the four things that would silently corrupt a PV10 cell:
  * the injection anchor moving off token 220
  * display_order and initial_pull_order becoming coupled
  * the parser recovering an action the model did not commit to
  * the resume key failing to separate two different configurations

The tokenizer checks use the REAL Llama-3.1 tokenizer from the local HF cache.
They are skipped (loudly) if it is unavailable, because a fake tokenizer that
agrees with our expectations would defeat the purpose.
"""

from __future__ import annotations

import sys

import bandit_pv10 as p10

FAILURES: list[str] = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def _counts(order, trials=1, succ=0):
    return {a: (succ, trials) for a in order}


# ───────────────────────────── prompt surface ───────────────────────────────

def test_prompt_shape():
    print("\n[prompt surface]")
    o = p10.make_orders(0, 4)
    counts = {"A": (1, 3), "B": (0, 2), "C": (2, 4), "D": (1, 1)}
    hist = list(o.initial_pull_order) + ["A", "C"]
    pr = p10.build_decision_prompt(o.display_order, counts, hist, n=6)

    check(pr.endswith(p10.REASON_ANCHOR), "prompt ends at the Reason anchor")
    check(not pr.endswith("  "), "prompt does not end in a double space")
    check(pr.count("Policy: SAMPLE") >= 1 and pr.count("Policy: COMMIT") >= 1,
          "sampling prompt offers both SAMPLE and COMMIT")
    check("Sampling rewards are observations, not points." in pr,
          "states that rewards are observations, not points")
    check("Samples used: 6" in pr and "Samples remaining: 94" in pr,
          "budget lines reflect n")

    # OPTIONS must follow display order, not alphabetical order.
    rows = [ln for ln in pr.splitlines() if ln.startswith("- Button ")]
    shown = [ln.split()[2].rstrip(":") for ln in rows]
    check(tuple(shown) == o.display_order,
          f"OPTIONS follows display_order {o.display_order}")

    # No confidence threshold and no minimize-samples instruction: the stopping
    # time must read the model's own subjective threshold.
    low = pr.lower()
    check("as few" not in low and "as possible" not in low,
          "no 'use as few samples as possible' instruction")
    check("95%" not in pr and "confidence level" not in low,
          "no experimenter-supplied confidence threshold")


def test_terminal_prompt():
    print("\n[terminal prompt]")
    o = p10.make_orders(3, 4)
    counts = _counts(o.display_order, trials=25, succ=12)
    pr = p10.build_decision_prompt(o.display_order, counts,
                                   list(o.display_order), n=p10.TOTAL_BUDGET)

    check(pr.endswith(p10.REASON_ANCHOR), "terminal prompt uses the SAME anchor")
    check("No samples remain." in pr, "terminal prompt withdraws sampling")
    check("Policy: SAMPLE" not in pr, "terminal prompt offers only COMMIT")
    check("Samples remaining: 0" in pr, "terminal prompt shows 0 remaining")

    try:
        p10.build_decision_prompt(o.display_order, counts, [], n=101)
        check(False, "n > budget must raise")
    except ValueError:
        check(True, "n > budget raises")


def test_history_and_options():
    print("\n[history / options]")
    o = p10.make_orders(7, 4)
    # The forced-init pulls must appear in initial_pull_order, so history and
    # OPTIONS can never disagree about what was sampled.
    h = p10.format_history(list(o.initial_pull_order))
    check(f"[{' '.join(o.initial_pull_order)}]" in h,
          "history lists forced init in initial_pull_order")
    check("(none)" in p10.format_history([]), "empty history is explicit")

    opts = p10.format_options(o.display_order, {a: (0, 0) for a in o.display_order})
    check("n/a" in opts, "zero-trial arm shows n/a, not a fabricated rate")


# ───────────────────────────── frozen orders ────────────────────────────────

def test_orders_independent():
    print("\n[orders]")
    seeds = list(range(200))
    same = sum(p10.make_orders(s, 4).display_order
               == p10.make_orders(s, 4).initial_pull_order for s in seeds)
    # If the two orders shared an RNG stream, one would be a deterministic
    # function of the other. Under independence, P(equal) = 1/24 -> ~8 of 200.
    check(same < 30, f"display and initial-pull orders are not coupled "
                     f"({same}/200 coincide, chance ~8)")

    o1 = p10.make_orders(42, 4)
    o2 = p10.make_orders(42, 4)
    check(o1 == o2, "orders are deterministic in the seed")

    n_disp = len({p10.make_orders(s, 4).display_order for s in seeds})
    n_init = len({p10.make_orders(s, 4).initial_pull_order for s in seeds})
    check(n_disp > 12 and n_init > 12,
          f"both orders explore the permutation space ({n_disp}, {n_init} of 24)")

    o = p10.make_orders(5, 4)
    check(sorted(o.display_order) == sorted(o.initial_pull_order) ==
          list("ABCD"), "both orders permute the same arm set")


def test_assign_orders_counterbalanced():
    print("\n[cell-level counterbalance]")
    from bandit_reference import build_seed_bank, get_environment, make_arm_map
    env = get_environment("easy")
    seeds = build_seed_bank(env, n=20)

    o1 = p10.assign_orders(seeds, 4)
    o2 = p10.assign_orders(list(reversed(seeds)), 4)
    check(o1 == o2, "orders are deterministic in the SORTED seed list")

    rep = p10.order_balance_report(seeds, 4,
                                   arm_map_fn=lambda s: make_arm_map(s, env))
    check(rep["n_seeds"] == 20, "report covers the whole bank")

    # n=20, k=4 -> five complete Latin squares -> exactly 5 per cell.
    for arm, rows in rep["label_by_display_row"].items():
        check(rows == [5, 5, 5, 5],
              f"label {arm} x display row exactly balanced {rows}")
    for arm, pos in rep["label_by_initial_pull_position"].items():
        check(pos == [5, 5, 5, 5],
              f"label {arm} x initial-pull position exactly balanced {pos}")

    # The counterbalancing must not have re-coupled the two orders.
    same = sum(o1[s].display_order == o1[s].initial_pull_order for s in seeds)
    check(same <= 3, f"display and initial-pull stay independent "
                     f"({same}/20 coincide, chance ~0.8)")

    # true-best balance comes from the pv6 bank, and must survive unchanged.
    check(set(rep["true_best_by_label"].values()) == {5},
          f"true best is balanced across labels {rep['true_best_by_label']}")
    check(max(rep["true_best_by_display_row"]) <= 6,
          f"true best is spread across display rows "
          f"{rep['true_best_by_display_row']}")


# ───────────────────────────── strict parser ────────────────────────────────

ARMS = ("A", "B", "C", "D")


def test_parser_valid():
    print("\n[parser: valid]")
    r = p10.parse_policy("Button C leads.\nPolicy: SAMPLE Button C", ARMS)
    check(r.valid and r.action == "SAMPLE" and r.arm == "C", "parses SAMPLE")
    check(r.native_ends_after_policy, "clean stop sets native_ends_after_policy")

    r = p10.parse_policy("Enough evidence.\nPolicy: COMMIT Button A", ARMS)
    check(r.valid and r.action == "COMMIT" and r.arm == "A", "parses COMMIT")

    r = p10.parse_policy("x\npolicy:  commit   button   d", ARMS)
    check(r.valid and r.action == "COMMIT" and r.arm == "D",
          "case and spacing tolerant")

    r = p10.parse_policy("x\nPolicy: COMMIT Button B.", ARMS)
    check(r.valid and r.arm == "B", "trailing period allowed")

    # Text after a well-formed Policy is REPORTED, not voided.
    r = p10.parse_policy("x\nPolicy: SAMPLE Button A\n#tag spam", ARMS)
    check(r.valid, "trailing text does not void a well-formed policy")
    check(not r.native_ends_after_policy,
          "trailing text clears native_ends_after_policy")

    # A repeated but AGREEING directive is unambiguous.
    r = p10.parse_policy("Policy: SAMPLE Button A\nPolicy: SAMPLE Button A", ARMS)
    check(r.valid and r.arm == "A", "agreeing repeat is valid")


def test_parser_fail_closed():
    print("\n[parser: fail-closed]")
    cases = [
        ("I like Button A a lot.", "no_policy", "no policy marker"),
        ("Policy: I will try Button A", "malformed", "prose instead of directive"),
        ("Policy: SAMPLE A", "malformed", "missing the word Button"),
        ("Policy: EXPLORE Button A", "malformed", "unknown action verb"),
        ("Policy: SAMPLE Button A\nPolicy: COMMIT Button B",
         "conflicting", "two disagreeing directives"),
        ("Policy: SAMPLE Button E", "arm_out_of_set", "arm outside the set"),
    ]
    for text, kind, desc in cases:
        r = p10.parse_policy(text, ARMS)
        check(not r.valid and r.invalid_kind == kind,
              f"invalid [{kind}]: {desc}")
        check(r.arm is None and r.action is None,
              f"  ... and recovers NO action for: {desc}")

    r = p10.parse_policy("", ARMS)
    check(not r.valid and r.invalid_kind == "no_policy", "empty text is invalid")

    # Explaining instead of committing must stay a visible failure, never be
    # silently recovered into a choice.
    r = p10.parse_policy("Policy: SAMPLE Button A because it looks best", ARMS)
    check(not r.valid, "policy wrapped in prose is invalid, not recovered")


def test_parser_terminal():
    print("\n[parser: terminal]")
    r = p10.parse_policy("done\nPolicy: COMMIT Button B", ARMS, terminal=True)
    check(r.valid and r.action == "COMMIT", "COMMIT valid at terminal")

    r = p10.parse_policy("more\nPolicy: SAMPLE Button B", ARMS, terminal=True)
    check(not r.valid and r.invalid_kind == "sample_at_terminal",
          "SAMPLE at terminal is invalid")
    check(r.arm is None, "  ... and is not silently converted to COMMIT")


# ───────────────────────────── resume key ───────────────────────────────────

def test_format_flags():
    print("\n[format flags]")
    r = p10.parse_policy("x\nPolicy: COMMIT Button B", ARMS)
    check(r.valid and r.format_exact and not r.trailing_period_tolerated,
          "clean line is format_exact")

    r = p10.parse_policy("x\nPolicy: COMMIT Button B.", ARMS)
    check(r.valid and r.arm == "B", "trailing period still valid")
    check(r.trailing_period_tolerated and not r.format_exact,
          "trailing period is RECORDED, not silent")


def test_fires_arithmetic():
    print("\n[steering fires]")
    # alpha=0 registers no hook at all -- 0 fires, not "steered by zero".
    check(p10.expected_fires(0.0, 97) == 0, "alpha=0 gives exactly 0 fires")

    # tau - 3 at K=4: the four forced pulls consume samples, not generations.
    check(p10.model_calls_for(100, 4) == 97, "tau=100 -> 97 model calls")
    check(p10.model_calls_for(4, 4) == 1, "immediate commit -> 1 model call")
    check(p10.model_calls_for(20, 4) == 17, "tau=20 -> 17 model calls")

    check(p10.expected_fires(4.0, p10.model_calls_for(100, 4)) == 873,
          "full-budget episode = 873 fires (NOT PV9's 900)")
    check(p10.expected_fires(-4.0, p10.model_calls_for(20, 4)) == 153,
          "tau=20 -> 9 x 17 = 153 fires")

    # The 9 comes from the half-open range(10,19) for the 11-20 band.
    from utils import decoder_layer_range
    check(len(list(decoder_layer_range(11, 20))) == 9,
          "layers 11-20 is 9 steered layers (half-open)")

    # K must not silently keep K=4 arithmetic.
    check(p10.model_calls_for(20, 5) == 16, "K=5 changes the call count")


def test_termination_reasons():
    print("\n[termination]")
    check(set(p10.TERMINATION_REASONS) ==
          {"autonomous_commit", "forced_commit", "invalid_policy"},
          "exactly three mutually exclusive termination reasons")
    check("invalid_policy" in p10.TERMINATION_REASONS,
          "invalid ends the episode (it is a termination reason, not a retry)")


def test_interface_tag():
    print("\n[resume key]")
    a = p10.interface_tag(4, [1, 2, 3])
    check(a == p10.interface_tag(4, [3, 2, 1]), "order-insensitive in seeds")
    check(a != p10.interface_tag(4, [1, 2, 4]),
          "seed CONTENT changes the key (not just the count)")
    check(a != p10.interface_tag(5, [1, 2, 3]), "k changes the key")

    import subprocess
    # PYTHONHASHSEED is randomized per process; a builtin hash() here would give
    # the same cell a different key on every run.
    code = ("import bandit_pv10 as p; print(p.interface_tag(4,[1,2,3]))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True).stdout.strip() for _ in range(3)}
    check(len(outs) == 1 and outs.pop() == a,
          "key is stable across processes (not PYTHONHASHSEED-dependent)")


# ─────────────────────── real-tokenizer invariants ──────────────────────────

def test_tokenizer():
    print("\n[tokenizer: REAL Llama-3.1]")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception as e:
        print(f"  SKIP tokenizer checks: {type(e).__name__}: {e}")
        FAILURES.append("tokenizer unavailable -- anchor invariant UNVERIFIED")
        return

    ids = tok.encode(p10.REASON_ANCHOR, add_special_tokens=False)
    check(ids[-1] == p10.EXPECTED_WHITESPACE_TOKEN_ID,
          f"'Reason: ' ends on token {p10.EXPECTED_WHITESPACE_TOKEN_ID} "
          f"(measured, not inherited from PV9's 'Evidence: ')")

    # The rstrip trap: dropping the trailing space moves the site to ':'.
    stripped = tok.encode(p10.REASON_ANCHOR.rstrip(), add_special_tokens=False)
    check(stripped[-1] != p10.EXPECTED_WHITESPACE_TOKEN_ID,
          f"rstrip moves the tail off 220 (to {stripped[-1]}) -- negative control")

    # A double space is ONE token 256, not two 220s.
    dbl = tok.encode(p10.REASON_ANCHOR + " ", add_special_tokens=False)
    check(dbl[-1] != p10.EXPECTED_WHITESPACE_TOKEN_ID,
          f"double space is token {dbl[-1]}, not 220 -- negative control")

    o = p10.make_orders(0, 4)
    for n, hist in ((4, list(o.initial_pull_order)),
                    (50, list(o.initial_pull_order) + ["A"] * 46),
                    (p10.TOTAL_BUDGET, list(o.initial_pull_order) + ["B"] * 96)):
        pr = p10.build_decision_prompt(
            o.display_order, _counts(o.display_order, trials=9, succ=4),
            hist, n=n)
        info = p10.audit_prompt_tokens(tok, pr)
        check(info["tail_token_id"] == p10.EXPECTED_WHITESPACE_TOKEN_ID,
              f"n={n}: real prompt ends on token 220 ({info['n_tokens']} tokens)")

    # The audit must RAISE on a violated prompt, not warn.
    try:
        p10.audit_prompt_tokens(tok, "Reason:")
        check(False, "audit raises on a stripped anchor")
    except AssertionError:
        check(True, "audit raises on a stripped anchor")


def main():
    test_prompt_shape()
    test_terminal_prompt()
    test_history_and_options()
    test_orders_independent()
    test_assign_orders_counterbalanced()
    test_parser_valid()
    test_parser_fail_closed()
    test_parser_terminal()
    test_format_flags()
    test_fires_arithmetic()
    test_termination_reasons()
    test_interface_tag()
    test_tokenizer()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all PV10 invariants hold")


if __name__ == "__main__":
    main()
