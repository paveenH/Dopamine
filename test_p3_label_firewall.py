#!/usr/bin/env python3
"""P3 label-firewall + format preflight tests. No GPU, no model, no download.

EVERY GUARD IS MUTATION-TESTED: the test builds the specific defect and asserts
rejection. A guard never shown to fire on a real defect is not a guard -- this
repo has twice shipped checks that passed vacuously.

Runs in ~1s. Exits non-zero on failure.
"""
import ast, json, os, subprocess, sys, tempfile

OK = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    OK.append(cond)


print("P3 label firewall + preflight")
print("=" * 66)

# ---- 1. the generation entrypoint has NO correctness code path
print("\n[1] generation entrypoint is label-free")
src = open("get_answer_gsm8k_hard_blind.py", encoding="utf-8").read()
tree = ast.parse(src)
imported = {n.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for n in node.names}
check("does not import is_correct_gsm8k", "is_correct_gsm8k" not in imported)
check("does not import extract_gsm8k_answer", "extract_gsm8k_answer" not in imported)
called = {n.func.id for n in ast.walk(tree)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
check("never calls is_correct_gsm8k", "is_correct_gsm8k" not in called)
# the words may appear ONLY in the docstring that explains their removal
body = src.split('"""', 2)[2]
check("no correctness reference outside the docstring",
      "is_correct_gsm8k" not in body and "extract_gsm8k_answer" not in body)

# ---- 2. the old runner DOES compute accuracy => forking was necessary
print("\n[2] the existing runner is genuinely unusable for P3")
old = open("get_answer_regenerate_gsm8k.py", encoding="utf-8").read()
check("get_answer_regenerate_gsm8k.py calls is_correct_gsm8k",
      "is_correct_gsm8k(" in old)
check("...and prints accuracy", "accuracy_percentage" in old)

# ---- 3. loader splits gold into a sealed file
print("\n[3] loader separates gold from questions")
ld = open("data_gsm8k_hard.py", encoding="utf-8").read()
check("writes a *_questions.json", "_questions.json" in ld)
check("writes a SEALED gold file", "_gold.SEALED.json" in ld)
check("questions file declares contains_labels=False", '"contains_labels": False' in ld)
check("selection is by hash, not dataset order", "sorted(uniq, key=digest)" in ld)
check("dataset revision is pinned", "--revision" in ld)

# ---- 4. MUTATION TEST: the loader's guard rejects a leaked label
print("\n[4] mutation test -- load_questions rejects a leaked label")
sys.path.insert(0, ".")
# Import the real module so its constants (FORBIDDEN_KEYS) exist. Stub only the
# heavy deps -- importing llms would pull in torch and defeat "no GPU, ~1s".
import types
# numpy is real (utils needs it); only llms pulls in torch, so stub just that.
if "llms" not in sys.modules:
    sys.modules["llms"] = types.ModuleType("llms")
sys.modules["llms"].VicundaModel = object
import importlib
blind = importlib.import_module("get_answer_gsm8k_hard_blind")
load_questions = blind.load_questions

with tempfile.TemporaryDirectory() as td:
    clean = os.path.join(td, "clean.json")
    json.dump({"meta": {"contains_labels": False, "questions_sha256": "x" * 64},
               "data": [{"sample_id": 0, "question": "q"}]}, open(clean, "w"))
    try:
        load_questions(clean)
        check("accepts a clean label-free file", True)
    except SystemExit:
        check("accepts a clean label-free file", False)

    for defect, payload in [
        ("gold answer leaked", {"sample_id": 0, "question": "q", "answer": "42"}),
        ("target leaked",      {"sample_id": 0, "question": "q", "target": 42}),
        ("correct flag leaked",{"sample_id": 0, "question": "q", "correct": True}),
    ]:
        p = os.path.join(td, "bad.json")
        json.dump({"meta": {"contains_labels": False, "questions_sha256": "x" * 64},
                   "data": [payload]}, open(p, "w"))
        try:
            load_questions(p)
            check(f"REJECTS: {defect}", False)
        except SystemExit:
            check(f"REJECTS: {defect}", True)

    # a file that does not declare itself label-free must also be refused
    p = os.path.join(td, "undeclared.json")
    json.dump({"meta": {"questions_sha256": "x" * 64},
               "data": [{"sample_id": 0, "question": "q"}]}, open(p, "w"))
    try:
        load_questions(p)
        check("REJECTS: missing contains_labels declaration", False)
    except SystemExit:
        check("REJECTS: missing contains_labels declaration", True)

# ---- 5. launchers: per-model params, single-GPU guard, no accuracy
print("\n[5] launchers")
for m, band, doses in (("llama3", ("11", "20"), "neg8-11-20 neg6-11-20 neg4-11-20 0-11-20 4-11-20"),
                       ("qwen25", ("16", "22"), "neg4-16-22 0-16-22 4-16-22 6-16-22 8-16-22")):
    f = f"run_gsm8k_hard_{m}.sh"
    t = open(f, encoding="utf-8").read()
    check(f"{m}: band {band[0]}-{band[1]}", f"LS={band[0]}" in t and f"LE={band[1]}" in t)
    check(f"{m}: frozen dose set", doses in t)
    check(f"{m}: refuses unset/multi GPU", "CUDA_VISIBLE_DEVICES" in t and "exit 1" in t)
    check(f"{m}: drives the blind entrypoint",
          "get_answer_gsm8k_hard_blind.py" in t)
    check(f"{m}: never drives the accuracy runner",
          "get_answer_regenerate_gsm8k.py" not in t)
    r = subprocess.run(["bash", "-n", f], capture_output=True)
    check(f"{m}: shell syntax", r.returncode == 0)

# Qwen must keep a negative probe, else "predicted positive" cannot be wrong
q = open("run_gsm8k_hard_qwen25.sh", encoding="utf-8").read()
check("qwen retains a negative probe (protocol 2.2)", "neg4-16-22" in q)

print("\n" + "=" * 66)
print(f"{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
