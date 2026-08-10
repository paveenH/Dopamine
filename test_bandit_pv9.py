#!/usr/bin/env python3
"""PV9 checks. Standalone; exits non-zero on failure. No GPU.

Uses the REAL Llama-3.1 tokenizer from the local HF cache, because the anchor
invariant under test (Stage-1 prompt tail == token 220) is a property of that
tokenizer and cannot be faked.

The FakeVC follows the pv6/pv7 convention: it mirrors the real VicundaModel
contract, so `generate` takes no diff_matrices and `regenerate` raises on
diff_matrices=None like llms.py:821. It models INTENT, not arithmetic -- the
site counter reimplements the formula rather than executing llms.py hooks, so
it verifies the runner asks for the right thing, never that llms.py computes
it correctly.
"""
from __future__ import annotations

import json
import subprocess
import sys

import bandit_reference as br
import bandit_pv7 as p7
import bandit_pv9 as p9
import bandit_pv9_episode as p9ep
from test_bandit_pv7_episode import FakeVC

FAILS: list[str] = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


def main() -> int:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception as exc:                                    # noqa: BLE001
        print(f"cannot load the Llama-3.1 tokenizer ({exc})")
        return 2

    easy = br.get_environment("easy")
    nt = br.get_environment("neartie")

    # ── 1. the neartie environment ──────────────────────────────────────────
    print("[1] neartie environment")
    check(nt.k == 4 and nt.horizon == 100, "K=4, T=100")
    check(nt.probs == (0.60, 0.55, 0.25, 0.25), "probs .60/.55/.25/.25")
    check(nt.probs[0] - nt.probs[1] == 0.05 or
          abs(nt.probs[0] - nt.probs[1] - 0.05) < 1e-9, "top gap is 0.05")
    # competence_eligible=False is the whole reason this environment is safe
    # to add: at gap .05 with ~25 pulls/arm the empirical SE (~.10) is DOUBLE
    # the gap, so a high SuffFail measures the environment, not the policy.
    check(nt.competence_eligible is False, "NOT competence eligible")
    check(nt.is_reference is False, "not a Krishnamurthy reference instance")
    check(br.build_seed_bank(nt) == br.build_seed_bank(easy),
          "shares easy's seed bank (arm_map depends only on seed and k)")
    rep = br.bank_report(br.build_seed_bank(nt), nt)
    check(rep["position_balanced"] and rep["identity_balanced"],
          "seed bank stays counterbalanced")

    # pv6's frozen basis must be untouched by the new environment.
    r = subprocess.run([sys.executable, "freeze_bandit_baseline.py", "--check"],
                       capture_output=True, text=True)
    check(r.returncode == 0, "pv6 frozen manifest still reproduces")
    r = subprocess.run([sys.executable, "freeze_pv9_baseline.py", "--check"],
                       capture_output=True, text=True)
    check(r.returncode == 0, "PV9 manifest reproduces and easy matches pv6")

    # ── 2. Stage-1 prompt: the four modifications ───────────────────────────
    print("\n[2] Stage-1 prompt content")
    am = br.make_arm_map(0, easy)
    order = list(am)
    hist = [(order[0], 1), (order[0], 0), (order[2], 1)]
    s1 = p9.build_rationale_prompt(order, hist, 3, easy)

    check("Your score so far: 2 points." in s1, "self-relevant score line")
    check(p9.score_so_far(hist) == 2, "score = total reward")
    check(p9.score_so_far([(order[0], 1)]) == 1 and
          "1 point." in p9.build_rationale_prompt(order, [(order[0], 1)], 1, easy),
          "singular 'point' at a score of 1")
    check(s1.count(p9._UNTRIED_CUE) == 2, "cue on both untried arms")
    # The cue must never appear on a row that has been pulled -- that is what
    # makes it a discovery scaffold rather than a general exploration nudge,
    # and it is why PV9 cannot drive one-shot-zero revisits.
    for line in s1.splitlines():
        if line.startswith("- ") and "UNTRIED" not in line:
            check(p9._UNTRIED_CUE not in line, f"no cue on tried row: {line[:28]}")
    check("fixed but unknown probability" in s1 and
          "may differ across buttons" in s1, "explicit Bernoulli structure")
    check("50 words" in s1, "50-word instruction")
    check("CHOICE HISTORY" in s1, "history block inherited from pv8")
    # Structure, never strategy: PV9 states a benefit direction for UNTRIED
    # arms only. Anything resembling "few trials are weak evidence" is P2's
    # failed hint and must not have crept back in.
    check("weak evidence" not in s1, "no P2-style decision hint")

    # ── 3. anchor / token invariants ────────────────────────────────────────
    print("\n[3] token invariants")
    for label, env, rnd in (("easy", easy, 3), ("neartie", nt, 0),
                            ("late", easy, 61)):
        h = hist if rnd == 3 else (
            [] if rnd == 0 else [(order[i % 4], i % 2) for i in range(rnd)])
        o = list(br.make_arm_map(0, env))
        pr = p9.build_rationale_prompt(o, h, rnd, env)
        ids = tok(pr, return_tensors="pt")["input_ids"][0].tolist()
        check(ids[-1] == p7.EXPECTED_WHITESPACE_TOKEN_ID,
              f"{label}: Stage-1 tail is token 220")
        check(pr.endswith(p7.RATIONALE_ANCHOR), f"{label}: ends at Evidence anchor")
    # rstrip moves the tail off 220 -- an assertion, not a comment.
    bad = p9.build_rationale_prompt(order, hist, 3, easy).rstrip()
    check(tok(bad, return_tensors="pt")["input_ids"][0].tolist()[-1] != 220,
          "rstrip destroys the anchor (negative control)")

    # ── 4. the OPTIONS core is identical across stages ──────────────────────
    print("\n[4] Stage-1 / Stage-2 isolation")
    s2 = p7.build_action_prompt(order, hist, 3, easy, "Evidence: x\nPolicy: y")
    for token, label in ((p9._UNTRIED_CUE, "untried cue"),
                         ("Your score so far", "score line"),
                         ("CHOICE HISTORY", "history block")):
        check(token not in s2, f"{label} absent from Stage 2")
    # Only the cue may differ. Counts, order and rates must match, or the
    # information-isolation argument collapses into an uncontrolled display
    # difference -- pv6's option-display drift with extra steps.
    def rows(text):
        return [ln for ln in text.splitlines() if ln.startswith("- Button ")]
    r1 = [ln.replace(f". {p9._UNTRIED_CUE}", "") for ln in rows(s1)]
    check(r1 == rows(s2), "OPTIONS rows identical once the cue is removed")

    # ── 5. stop strings ─────────────────────────────────────────────────────
    print("\n[5] stop strings")
    check(p9.STOP_STRINGS == ("#",), "only '#' (blank line was rejected)")
    # Measured on the stored pv8 alpha=0 cell: 1999/2000 rationales contain a
    # blank line and in ALL of them it precedes the Policy line, so a "\n\n"
    # stop would delete the policy in essentially every round.
    check("\n\n" not in p9.STOP_STRINGS, "no blank-line stop")
    for s in p9.STOP_STRINGS:
        check(s not in p9._P9_INSTRUCTION and s not in p9._TASK_HEADER_PV9,
              f"stop marker {s!r} cannot occur in the prompt (CGT failure mode)")
    real = ("Evidence: A is well supported.\n\n"
            "Policy: EXPLOIT Button A because it leads.  #bandit #rl")
    cut, marker = p9ep.apply_stop(real)
    check(marker == "#" and "Policy: EXPLOIT Button A" in cut,
          "hashtag spray cut, Policy line kept")
    check(p9ep.apply_stop("clean")[1] is None, "no marker on clean text")

    print("\n[5b] stop_reason vocabulary (judged on RAW text)")
    for text, want in (
            ("Evidence: a\n\nPolicy: EXPLOIT Button A because x.", "native_clean"),
            ("Evidence: a\n\nPolicy: EXPLOIT Button A.  #t #u", "stop_marker_applied"),
            ("Evidence: a\n\nPolicy: EXPLOIT Button A.\nRound 5 of 100.",
             "continued_after_policy"),
            ("Evidence: a is fine.", "no_policy_line"),
            ("   ", "empty")):
        check(p9.stop_reason(text) == want,
              f"stop_reason -> {want} ({p9.stop_reason(text)})")

    # ── 6. episode runner ───────────────────────────────────────────────────
    print("\n[6] episode runner")
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, bandit_pv9_episode; "
         "print('bandit_pv6_episode' in sys.modules)"],
        capture_output=True, text=True)
    check(probe.stdout.strip() == "False", "importing PV9 does not drag in pv6")

    small = br.Environment(name="reference_easy", k=4,
                           probs=easy.probs, horizon=5,
                           is_reference=True, competence_eligible=True)
    reply = ("Evidence: B untried.\n"
             "Policy: EXPLORE Button B because unknown.  #tag #tag2")

    vc = FakeVC(tok, reply=reply)
    rec = p9ep.run_pv9_episode(vc, None, seed=0, env=small, attest=True)
    check(rec["protocol"] == "pv9", "record tagged pv9")
    check(vc.generate_calls == 5 and vc.regenerate_calls == 0,
          "alpha=0 uses generate only -- no hook registered anywhere")
    check(rec["steering_fires"] == {"rationale": 0, "action": 0},
          "alpha=0 fires nothing")
    check(all("#" not in r["rationale_clean"] for r in rec["rounds"]),
          "stop rule strips the spray from what Stage 2 reads")
    check(all("Policy: EXPLORE Button B" in r["rationale_clean"]
              for r in rec["rounds"]), "Policy line survives the stop rule")
    # THE RAW RECORD MUST NOT BE TRUNCATED. An earlier implementation wrapped
    # generate/regenerate, so the loop stored ALREADY-CUT text as
    # `rationale_raw`: the tail became unrecoverable and every round reported
    # a clean native stop whether or not the model actually terminated. That
    # is the single measurement PV9 exists to make, so it is asserted here.
    check(all("#tag" in r["rationale_raw"] for r in rec["rounds"]),
          "rationale_raw keeps the untruncated continuation")
    check(all(r["rationale_stopped"] != r["rationale_raw"]
              for r in rec["rounds"]), "rationale_stopped stored separately")
    check(all(r["stop_marker"] == "#" for r in rec["rounds"]),
          "stop_marker records which marker fired")
    check(rec["stop_reason_counts"].get("stop_marker_applied") == 5,
          f"a sprayed reply is NOT reported as native: {rec['stop_reason_counts']}")
    # The shim must be restored, or a later episode in the same process would
    # silently keep truncating.
    check(p7.extract_evidence_policy_block.__module__ == "bandit_pv7",
          "extractor restored after the episode")

    # A natively terminating reply must read `native_clean` -- otherwise the
    # metric could not tell the two cases apart in either direction.
    vc_n = FakeVC(tok, reply="Evidence: B untried.\nPolicy: EXPLORE Button B ok.")
    rec_n = p9ep.run_pv9_episode(vc_n, None, seed=0, env=small)
    check(rec_n["stop_reason_counts"].get("native_clean") == 5,
          f"native termination reads native_clean: {rec_n['stop_reason_counts']}")
    check(all(r["stop_marker"] is None for r in rec_n["rounds"]),
          "no marker recorded when none fired")

    vc2 = FakeVC(tok, reply=reply)
    diff = [None] * 32
    rec2 = p9ep.run_pv9_episode(vc2, diff, seed=0, env=small,
                                rationale_alpha=-4.0, attest=True)
    check(vc2.regenerate_calls == 5 and vc2.generate_calls == 0,
          "rationale_alpha!=0 routes Stage 1 through regenerate")
    exp = p9ep.expected_fires(-4.0, 0.0, 9, small.k, small.horizon)
    check(rec2["steering_fires"] == exp,
          f"fires {rec2['steering_fires']} == expected {exp}")
    check(rec2["steering_fires"]["action"] == 0,
          "Stage 2 unsteered when only rationale_alpha is set")
    # THE STOP RULE MUST NOT BE ALPHA-CORRELATED. The unsteered path calls
    # `generate` and the steered path calls `regenerate`; if only one were
    # wrapped, the stop rule would become a generation setting that varies
    # with alpha, and no cross-alpha comparison would be readable. A dry run
    # caught exactly this -- the steered cells kept the hashtag spray while
    # alpha=0 did not.
    check(all("#" not in r["rationale_clean"] for r in rec2["rounds"]),
          "steered path gets the same post-processing")
    check(all("#tag" in r["rationale_raw"] for r in rec2["rounds"]),
          "steered rationale_raw is untruncated too")
    check(rec2["stop_reason_counts"] == rec["stop_reason_counts"],
          f"identical stop_reason across alphas: {rec2['stop_reason_counts']}")
    check([r["rationale_clean"] for r in rec2["rounds"]] ==
          [r["rationale_clean"] for r in rec["rounds"]],
          "same reply text yields byte-identical clean text on both paths")
    check(p7.extract_evidence_policy_block.__module__ == "bandit_pv7",
          "extractor restored after the steered episode")
    # Same reply text under both alphas, so any trajectory difference would be
    # the harness, not the model.
    check([r["action"] for r in rec["rounds"]] ==
          [r["action"] for r in rec2["rounds"]],
          "identical replies give identical trajectories across alphas")

    # ── 7. resume key ───────────────────────────────────────────────────────
    print("\n[7] resume key")
    k = p9ep.resume_key
    base = k("easy", -4, 0, 11, 20, [0, 1, 2])
    check(base.startswith("pv9_easy"), "protocol and env in the key")
    check(base != k("neartie", -4, 0, 11, 20, [0, 1, 2]),
          "easy and neartie differ despite a shared seed bank")
    check(base != k("easy", 0, 0, 11, 20, [0, 1, 2]), "alpha in the key")
    check(base == k("easy", -4, 0, 11, 20, [2, 1, 0]), "seed order-insensitive")
    check(base != k("easy", -4, 0, 11, 20, [6, 12, 13]),
          "seed CONTENT in the key, not just the count")
    check(base != k("easy", -4, 0, 11, 20, [0, 1, 2],
                    model_config={"mask_sha256": "x"}), "model config in the key")
    for seg in ("scscore-v1", "cucue-v1", "ststop-hash-v1"):
        check(seg in base, f"{seg} in the key")

    # ── 8. gate wrapper reads the PV9 basis, not pv6's ──────────────────────
    print("\n[8] gate manifest binding")
    import evaluate_competence_gate as gate
    import evaluate_competence_gate_pv9 as gpv9
    before = gate.MANIFEST
    # The frozen evaluator hardcodes the pv6 file, which has NO neartie block,
    # so without the rebind every neartie evaluation dies on
    # "reference_neartie not in the frozen manifest".
    man = json.load(open(before))
    check("neartie" not in man["environments"],
          "pv6 manifest has no neartie (why the rebind is required)")
    with gpv9.pv9_manifest():
        inner = gate.MANIFEST
        man9 = json.load(open(inner))
        check(inner == gpv9.PV9_MANIFEST, "manifest rebound inside the scope")
        check("neartie" in man9["environments"], "PV9 manifest has neartie")
        check(man9["environments"]["easy"] == man["environments"]["easy"],
              "easy basis identical across manifests (rebind is safe)")
    check(gate.MANIFEST == before,
          "manifest restored -- a later pv6/pv7/pv8 eval is unaffected")
    gpv9.assert_easy_basis_agrees()
    check(True, "assert_easy_basis_agrees passes")
    # NearTie baselines must not drift: they are the only comparison basis a
    # non-eligible environment has.
    ntb = man9["environments"]["neartie"]["policies"]
    check(abs(ntb["greedy"]["suff_fail_freq_half"]["point"] - 0.55) < 1e-9,
          f"neartie Greedy SuffFail frozen at .550 "
          f"({ntb['greedy']['suff_fail_freq_half']['point']})")
    check(abs(ntb["greedy"]["late_opt_frac"]["point"] - 0.36) < 1e-9,
          f"neartie Greedy late_opt frozen at .360 "
          f"({ntb['greedy']['late_opt_frac']['point']})")
    check(abs(ntb["random"]["k_min_frac_full"]["point"] - 0.794) < 1e-9,
          "neartie Random KxMinFrac frozen at .794")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
