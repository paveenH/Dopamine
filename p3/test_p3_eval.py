#!/usr/bin/env python3
"""Synthetic-fixture tests for run_p3_eval.py. NO real gold is read.

Every guard is MUTATION-TESTED: the test builds the specific defect and asserts
the evaluator rejects it. A guard never shown to fire on a real defect is not a
guard. The statistics are checked against hand-computed values, not against the
evaluator's own output.
"""
import json, os, shutil, subprocess, sys, tempfile
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_p3_eval as ev

N_OK = N_FAIL = 0


def ok(cond, label):
    global N_OK, N_FAIL
    if cond:
        N_OK += 1
        print(f"  ok    {label}")
    else:
        N_FAIL += 1
        print(f"  FAIL  {label}")


# ---------------------------------------------------------------- statistics
print("statistics")
ok(abs(ev.mcnemar_exact(0, 0) - 1.0) < 1e-12, "McNemar with no discordant pairs is 1.0")
# b=1,c=0,n=1: 2*sum(comb(1,0))/2 = 1.0
ok(abs(ev.mcnemar_exact(1, 0) - 1.0) < 1e-12, "McNemar 1/0 = 1.0 (hand-computed)")
# b=10,c=0,n=10: 2*1/1024
ok(abs(ev.mcnemar_exact(10, 0) - 2.0 / 1024) < 1e-12, "McNemar 10/0 = 2/1024 (hand)")
ok(abs(ev.mcnemar_exact(3, 7) - ev.mcnemar_exact(7, 3)) < 1e-15, "McNemar is symmetric")
ok(ev.mcnemar_exact(20, 2) < 0.001, "McNemar detects a lopsided split")

h = ev.holm([0.01, 0.04, 0.03])
ok(abs(h[0] - 0.03) < 1e-12, "Holm smallest p x3")
ok(h[1] >= h[2], "Holm is monotone in the original order")
ok(ev.holm([0.5])[0] == 0.5, "Holm with one test is identity")

ok(abs(ev.spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-12, "Spearman perfect +1")
ok(abs(ev.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-12, "Spearman perfect -1")
ok(abs(ev.spearman([1, 1, 2], [1, 1, 2]) - 1.0) < 1e-12, "Spearman handles ties")


# ---------------------------------------------------------------- fixtures
def build(tmp, model, accs, gold_digest="D" * 64, ids=None, dup=False,
          drop=None, wrong_digest_at=None):
    """Write synthetic cells whose per-question correctness is CONTROLLED:
    the first k questions are correct, so accuracy is exactly k/n."""
    sub, fname = ev.CELLS[model]
    n = 20
    ids = list(range(n)) if ids is None else ids
    gold = {"meta": {"questions_sha256": gold_digest},
            "data": [{"sample_id": i, "gold": "42"} for i in ids]}
    gp = os.path.join(tmp, "gold.json")
    json.dump(gold, open(gp, "w"))

    for a, acc in accs.items():
        k = round(acc * n)
        rows = []
        for pos, i in enumerate(ids):
            val = "42" if pos < k else "99"
            rows.append({"sample_id": i, "question": "q",
                         "generated": f"reasoning here #### {val}"})
        if dup and a == 0:
            rows[1]["sample_id"] = rows[0]["sample_id"]
        if drop is not None and a == drop:
            rows = rows[:-1]
        dig = gold_digest if a != wrong_digest_at else "E" * 64
        d = {"meta": {"alpha": a, "questions_sha256": dig, "steering_fires": 0},
             "data": rows}
        p = os.path.join(tmp, sub, f"mdf_{a}".replace("-", "neg"))
        os.makedirs(p, exist_ok=True)
        json.dump(d, open(os.path.join(p, fname), "w"))
    return gp


def run(tmp, gp, pred):
    """Run the evaluator in a sandbox ROOT and return (rc, stdout+stderr)."""
    pp = os.path.join(tmp, "p3_predictions.json")
    json.dump(pred, open(pp, "w"))
    code = (
        "import sys,os,json;"
        f"sys.path.insert(0,{os.path.dirname(HERE)!r});"
        f"sys.path.insert(0,{HERE!r});"
        "import run_p3_eval as ev;"
        f"ev.ROOT={tmp!r};ev.HERE={tmp!r};ev.PRED={pp!r};"
        f"sys.argv=['x','--gold',{gp!r},'--out',{os.path.join(tmp,'out.json')!r}];"
        "ev.main()"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


BASE_PRED = {"models": {
    "llama": {"questions_sha256": "D" * 64, "predicted_best_alpha": -6,
              "predicted_direction_from_alpha0": "negative",
              "predicted_plateau_or_overshoot_onset": -8,
              "dose_scores": {"-8": .55, "-6": .69, "-4": .68, "0": .63, "4": .57}},
    "qwen": {"questions_sha256": "D" * 64, "predicted_best_alpha": 8,
             "predicted_direction_from_alpha0": "positive",
             "predicted_plateau_or_overshoot_onset": 4,
             "dose_scores": {"-4": .69, "0": .70, "4": .68, "6": .72, "8": .86}},
}}
L_ACC = {-8: .25, -6: .75, -4: .70, 0: .50, 4: .35}
Q_ACC = {-4: .45, 0: .50, 4: .55, 6: .65, 8: .80}

print("\nhappy path")
tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC)
build(tmp, "qwen", Q_ACC)
rc, out = run(tmp, gp, BASE_PRED)
ok(rc == 0, "evaluator runs on a clean fixture")
res = json.load(open(os.path.join(tmp, "out.json"))) if rc == 0 else {}
if res:
    L = res["models"]["llama"]
    ok(abs(L["accuracy_first"]["-6"] - .75) < 1e-9, "accuracy matches the constructed value")
    ok(L["five_point"]["observed_best_alpha"] == -6, "observed best is -6")
    ok(L["five_point"]["direction_correct"] is True, "direction correct when it is")
    ok(abs(L["five_point"]["regret_pp"]) < 1e-9, "regret is 0 when selection == observed best")
    ok(L["five_point"]["selected_in_observed_near_optimal"] is True, "selected in observed set")
    ok(L["five_point"]["spearman_rho"] > 0.8, "Spearman positive when orderings agree")
    ok(L["fixed_workpoint"]["alpha"] == -6, "llama fixed workpoint is -6")
    ok(L["fixed_workpoint"]["improves"] is True, "fixed workpoint improves over a=0")
    ok(abs(L["fixed_workpoint"]["diff_pp"] - 25.0) < 1e-9, "diff_pp = 75-50 = 25")
    Q = res["models"]["qwen"]
    ok(Q["fixed_workpoint"]["alpha"] == 8, "qwen fixed workpoint is +8")
    ok(Q["five_point"]["observed_best_alpha"] == 8, "qwen observed best is +8")
shutil.rmtree(tmp)

print("\nwrong predictions are reported as wrong (not silently passed)")
tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", {-8: .70, -6: .30, -4: .35, 0: .50, 4: .75})  # best is +4
build(tmp, "qwen", Q_ACC)
rc, out = run(tmp, gp, BASE_PRED)
res = json.load(open(os.path.join(tmp, "out.json"))) if rc == 0 else {}
if res:
    L = res["models"]["llama"]
    ok(L["five_point"]["observed_best_alpha"] == 4, "observed best is +4 here")
    ok(L["five_point"]["direction_correct"] is False, "direction reported INCORRECT")
    ok(L["five_point"]["regret_pp"] > 40, "regret is large when the pick is bad")
    ok(L["five_point"]["spearman_rho"] < 0, "Spearman negative when orderings disagree")
    ok(L["fixed_workpoint"]["improves"] is False, "fixed workpoint reported as no improvement")
shutil.rmtree(tmp)

print("\nfail-closed guards (each defect built explicitly)")
tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC, wrong_digest_at=-4)
build(tmp, "qwen", Q_ACC)
rc, out = run(tmp, gp, BASE_PRED)
ok(rc != 0 and "digest" in out, "rejects a cell whose questions digest differs")
shutil.rmtree(tmp)

tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC, drop=-4)
build(tmp, "qwen", Q_ACC)
rc, out = run(tmp, gp, BASE_PRED)
ok(rc != 0 and "mismatch" in out, "rejects a cell missing a sample")
shutil.rmtree(tmp)

tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC, dup=True)
build(tmp, "qwen", Q_ACC)
rc, out = run(tmp, gp, BASE_PRED)
ok(rc != 0 and "duplicate" in out, "rejects a duplicate sample_id")
shutil.rmtree(tmp)

tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC)
build(tmp, "qwen", Q_ACC)
os.remove(os.path.join(tmp, *ev.CELLS["llama"][0].split("/"), "mdf_neg6",
                       ev.CELLS["llama"][1]))
rc, out = run(tmp, gp, BASE_PRED)
ok(rc != 0 and "missing cell" in out, "rejects a missing cell file")
shutil.rmtree(tmp)

print("\nordering: gold may not be read before predictions are frozen")
tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC)
build(tmp, "qwen", Q_ACC)
code = ("import sys;"
        f"sys.path.insert(0,{os.path.dirname(HERE)!r});sys.path.insert(0,{HERE!r});"
        "import run_p3_eval as ev;"
        f"ev.ROOT={tmp!r};ev.PRED={os.path.join(tmp,'nope.json')!r};"
        f"sys.argv=['x','--gold',{gp!r},'--out',{os.path.join(tmp,'o.json')!r}];ev.main()")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
ok(r.returncode != 0 and "BEFORE unsealing" in (r.stdout + r.stderr),
   "refuses to run without a frozen prediction file")
shutil.rmtree(tmp)

print("\nsample-id ORDER must not matter (alignment is by id, not position)")
tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC)
build(tmp, "qwen", Q_ACC)
# shuffle one cell's row order; correctness per id must be unchanged
p = os.path.join(tmp, *ev.CELLS["llama"][0].split("/"), "mdf_0", ev.CELLS["llama"][1])
d = json.load(open(p)); d["data"] = d["data"][::-1]; json.dump(d, open(p, "w"))
rc, out = run(tmp, gp, BASE_PRED)
res = json.load(open(os.path.join(tmp, "out.json"))) if rc == 0 else {}
ok(rc == 0 and res and abs(res["models"]["llama"]["accuracy_first"]["0"] - .50) < 1e-9,
   "row order does not change accuracy (id-keyed, verified by reversing a cell)")
shutil.rmtree(tmp)

print("\noverwrite guard")
tmp = tempfile.mkdtemp()
gp = build(tmp, "llama", L_ACC)
build(tmp, "qwen", Q_ACC)
open(os.path.join(tmp, "out.json"), "w").write("{}")
rc, out = run(tmp, gp, BASE_PRED)
ok(rc != 0 and "refusing to overwrite" in out, "refuses to overwrite an existing evaluation")
shutil.rmtree(tmp)

print("\n" + "=" * 60)
print(f"{N_OK}/{N_OK + N_FAIL} checks passed")
sys.exit(1 if N_FAIL else 0)
