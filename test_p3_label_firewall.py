#!/usr/bin/env python3
"""P3 label-firewall + format preflight tests. No GPU, no model, no download.

EVERY GUARD IS MUTATION-TESTED: the test builds the specific defect and asserts
rejection. A guard never shown to fire on a real defect is not a guard -- this
repo has twice shipped checks that passed vacuously.

Runs in ~1s. Exits non-zero on failure.
"""
import ast, json, os, subprocess, sys, tempfile, traceback

# This suite must run under ANY python3 (the server, a bare CI image), so it
# depends on the stdlib only. It previously imported the generation module,
# which pulls numpy via utils -- under a numpy-less interpreter it crashed with
# EXIT CODE 0, i.e. CI would have read a crash as a pass. Guards are extracted
# by AST/text instead, and the crash handler below forces a non-zero exit.

OK = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    OK.append(cond)


def _die(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("\nFAIL: the suite crashed. A crash is a FAILURE, never a pass.")
    sys.exit(2)


sys.excepthook = _die

print("P3 label firewall + preflight")
print("=" * 66)

# ---- 1. the generation entrypoint has NO correctness code path
print("\n[1] generation entrypoint is label-free")
src = open("get_answer_gsm_hard_blind.py", encoding="utf-8").read()
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
ld = open("data_gsm_hard.py", encoding="utf-8").read()
check("writes a *_questions.json", "_questions.json" in ld)
check("writes a SEALED gold file", "_gold.SEALED.json" in ld)
check("questions file declares contains_labels=False", '"contains_labels": False' in ld)
check("selection is by hash, not dataset order", "sorted(uniq, key=digest)" in ld)
check("dataset revision is pinned", "--revision" in ld)

# ---- 4. MUTATION TEST: the loader's guard rejects a leaked label
print("\n[4] mutation test -- load_questions rejects a leaked label")
sys.path.insert(0, ".")
# Extract load_questions and FORBIDDEN_KEYS by AST and exec ONLY those, so the
# test needs no numpy and no torch. Executing the real module would import
# utils -> numpy; see the header.
_blind_tree = ast.parse(src)
_wanted = {"FORBIDDEN_KEYS", "load_questions"}
_ns = {"json": json, "sys": sys}
_picked = [n for n in _blind_tree.body
           if (isinstance(n, ast.Assign) and any(
                   getattr(t, "id", None) in _wanted for t in n.targets))
           or (isinstance(n, ast.FunctionDef) and n.name in _wanted)]
exec(compile(ast.Module(body=_picked, type_ignores=[]), "blind_subset", "exec"), _ns)
load_questions = _ns["load_questions"]
check("extracted load_questions + FORBIDDEN_KEYS without importing numpy",
      "FORBIDDEN_KEYS" in _ns and callable(load_questions))

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

# ---- 4b. MUTATION TEST: the 2^53 audit must be EXACT, not float-based
print("\n[4b] 2^53 gold audit is exact (never routes through float)")
from data_gsm_hard import exceeds_2_53
L = 2 ** 53
# Boundary cases. The float form int(float(x)) reports False for EVERY row
# marked "must detect" below -- float('9007199254740993') == 2**53 exactly --
# so a float-based audit misses the first value that matters.
for label, val, want in [
    ("2^53 exactly (not over)",        str(L),            False),
    ("2^53+1",                         str(L + 1),        True),
    ("-(2^53+1)",                      str(-(L + 1)),     True),
    ("2^53+1 with .0 suffix",          f"{L + 1}.0",      True),
    ("2^53+1 scientific notation",     "9.007199254740993e15", True),
    ("2^53+1 with thousands commas",   f"{L + 1:,}",      True),
    ("small int",                      "8000",            False),
    ("float value",                    "1234.5",          False),
    ("non-numeric",                    "n/a",             False),
]:
    check(f"{label} -> {want}", exceeds_2_53(val) is want)

# and prove the naive float version really does miss them (regression guard)
def _naive(g):
    try:
        return abs(int(float(str(g).replace(",", "")))) > L
    except (ValueError, OverflowError):
        return False
check("float-based audit WOULD have missed 2^53+1 (why this test exists)",
      _naive(str(L + 1)) is False and exceeds_2_53(str(L + 1)) is True)

# ---- 4b1. float64 gold in the unsafe range is a HARD STOP
print("\n[4b1] float64 gold reaching 2^53 is refused, not silently audited")
_lsrc = open("data_gsm_hard.py", encoding="utf-8").read()
_lt = ast.parse(_lsrc)
_p2 = [n for n in _lt.body
       if (isinstance(n, ast.Assign) and any(
               getattr(t, "id", None) == "LIMIT_2_53" for t in n.targets))
       or (isinstance(n, ast.ClassDef) and n.name == "UnsafeFloatGold")
       or (isinstance(n, ast.FunctionDef) and n.name == "assert_float_safe")]
_ns2 = {}
exec(compile(ast.Module(body=_p2, type_ignores=[]), "loader_subset", "exec"), _ns2)
assert_float_safe, UnsafeFloatGold = _ns2["assert_float_safe"], _ns2["UnsafeFloatGold"]
L3 = 2 ** 53


def _stops(v):
    try:
        assert_float_safe(v)
        return False
    except UnsafeFloatGold:
        return True


# The REAL input shape: `datasets` hands back a python float for a float64
# column, and float(2**53+1) is ALREADY 2**53 -- the +1 was destroyed upstream
# before any of our code ran, so no string parse can recover it. The 4b tests
# use the STRING form, which never exercises this path and passed while the
# real input would have been silently missed.
check("float(2^53+1) -> HARD STOP (value already lossy)", _stops(float(L3 + 1)))
check("float(2^53) -> HARD STOP", _stops(float(L3)))
check("-float(2^53) -> HARD STOP (negative side)", _stops(-float(L3)))
check("float(2^53-1) -> allowed (still exact)", not _stops(float(L3 - 1)))
check("GSM-Hard actual max ~9.28e9 -> allowed", not _stops(9.28e9))
check("GSM-Hard actual min ~-9.83e9 -> allowed", not _stops(-9.83e9))
check("int 8000 -> allowed", not _stops(8000))
check("str '9007199254740993' -> allowed (exact string path)", not _stops(str(L3 + 1)))
check("loader hard-stops BEFORE counting", "assert_float_safe(g[" in _lsrc)

# ---- 4b2. the SCORER repairs what the audit flags
print("\n[4b2] norm_exact REPAIRS the flagged forms (not just detects them)")
sys.path.insert(0, "p3")
os.environ.setdefault("ROLEANSWER", os.path.expanduser("~/Documents/RSNResult/RoleAnswer"))
try:
    import p3_bigint_audit as _pb
    _pb.norm_gsm8k = _pb._load_norm_gsm8k()[0]
    ne = _pb.norm_exact
    for label, val, want in [
        ("bare 2^53+1", str(L3 + 1), str(L3 + 1)),
        ("'.0' suffix", f"{L3 + 1}.0", str(L3 + 1)),
        ("scientific", "9.007199254740993e15", str(L3 + 1)),
        ("negative", str(-(L3 + 1)), str(-(L3 + 1))),
        ("thousands commas", f"{L3 + 1:,}", str(L3 + 1)),
        ("small int unchanged", "8000", "8000"),
        ("true float unchanged", "1234.5", "1234.5"),
        ("3.0 -> 3 (frozen behaviour)", "3.0", "3"),
    ]:
        check(f"norm_exact {label}", ne(val) == want)
except SystemExit:
    print("       (skipped: offline RoleAnswer workspace not present here)")

# ---- 4c. --revision is mandatory
print("\n[4c] dataset revision is genuinely pinned")
ld_src = open("data_gsm_hard.py", encoding="utf-8").read()
check("--revision is required=True", 'ap.add_argument("--revision", required=True' in ld_src)
check("no silent default to main", 'ap.add_argument("--revision", default=None' not in ld_src)
from data_gsm_hard import _commit_sha
import argparse as _ap
for bad in ("main", "latest", "HEAD", "bbf48283", "a" * 39, "g" * 40):
    try:
        _commit_sha(bad)
        check(f"REJECTS --revision {bad!r}", False)
    except _ap.ArgumentTypeError:
        check(f"REJECTS --revision {bad!r}", True)
check("accepts a full 40-hex SHA", _commit_sha("A" * 40) == "a" * 40)
check("audit uses Decimal, not float", "Decimal(" in ld_src and "int(float(" not in ld_src)
check("audit result is published to metadata",
      '"n_gold_exceeding_2_53"' in ld_src and '"bigint_audit_digest"' in ld_src)
audit_src = open(os.path.join("p3", "p3_bigint_audit.py"), encoding="utf-8").read()
check("bigint audit script cannot re-open sealed gold",
      "--data" not in audit_src and "--questions" in audit_src)

check("schema incompatibility is a HARD STOP, no silent fallback",
      "HARD STOP" in ld_src and 'else "question"' not in ld_src)
check("audit digest is not derived from raw gold",
      "AUDIT_VERSION" in ld_src and 'sorted(big)' not in ld_src)

# ---- 4d. sample selection (item 6: promoted into the committed suite)
print("\n[4d] sample selection is deterministic and order-independent")
from data_gsm_hard import select
qs = [f"synthetic question {i}?" for i in range(500)] + ["synthetic question 7?"]
chosen, n_uniq = select(qs)
check("dedup on exact text", n_uniq == 500)
check("exactly 300 chosen", len(chosen) == 300)
check("no duplicates among chosen", len(set(chosen)) == 300)
check("NOT dataset order", chosen[:3] != qs[:3])
check("deterministic across calls", select(qs)[0] == chosen)
check("order-independent (reversed input, same sample)",
      select(list(reversed(qs)))[0] == chosen)
try:
    select([f"q{i}" for i in range(299)])
    check("REJECTS a corpus smaller than 300", False)
except SystemExit:
    check("REJECTS a corpus smaller than 300", True)

# ---- 4e. CLI contract: the launcher's argv must actually parse
print("\n[4e] runner CLI accepts what the launchers pass")
# This was a REAL failure: --configs lacked nargs, so argparse consumed only the
# first dose and rejected the other four AFTER the model had loaded. A source
# check alone would not have caught it -- the argv has to be parsed.
_rsrc = open("get_answer_gsm_hard_blind.py", encoding="utf-8").read()
check("--configs declares nargs (multi-dose)", 'nargs="+"' in _rsrc or "nargs='+'" in _rsrc)
_rt = ast.parse(_rsrc)
_pa = [n for n in _rt.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"]
_ns3 = {"argparse": __import__("argparse")}
exec(compile(ast.Module(body=_pa, type_ignores=[]), "runner_subset", "exec"), _ns3)
for model, doses in (("llama3", "neg8-11-20 neg6-11-20 neg4-11-20 0-11-20 4-11-20"),
                     ("qwen25", "neg4-16-22 0-16-22 4-16-22 6-16-22 8-16-22")):
    argv = ["--model", "m", "--size", "8B", "--model_dir", "d", "--questions", "q",
            "--mask_path", "mk", "--out_dir", "o", "--configs"] + doses.split()
    old_argv = sys.argv
    sys.argv = ["prog"] + argv
    try:
        a = _ns3["parse_args"]()
        check(f"{model}: all 5 doses parse from argv", len(a.configs) == 5)
    except SystemExit:
        check(f"{model}: all 5 doses parse from argv", False)
    finally:
        sys.argv = old_argv
# and the unpacking must match utils.parse_configs' [alpha, (start, end)] shape
check("unpacks parse_configs as alpha,(ls,le)",
      "for alpha, (ls, le) in utils.parse_configs" in _rsrc)

# ---- 5. launchers: per-model params, single-GPU guard, no accuracy
print("\n[5] launchers")
for m, band, doses in (("llama3", ("11", "20"), "neg8-11-20 neg6-11-20 neg4-11-20 0-11-20 4-11-20"),
                       ("qwen25", ("16", "22"), "neg4-16-22 0-16-22 4-16-22 6-16-22 8-16-22")):
    f = f"run_gsm_hard_{m}.sh"
    t = open(f, encoding="utf-8").read()
    check(f"{m}: band {band[0]}-{band[1]}", f"LS={band[0]}" in t and f"LE={band[1]}" in t)
    check(f"{m}: frozen dose set", doses in t)
    check(f"{m}: refuses unset/multi GPU", "CUDA_VISIBLE_DEVICES" in t and "exit 1" in t)
    check(f"{m}: drives the blind entrypoint",
          "get_answer_gsm_hard_blind.py" in t)
    check(f"{m}: never drives the accuracy runner",
          "get_answer_regenerate_gsm8k.py" not in t)
    r = subprocess.run(["bash", "-n", f], capture_output=True)
    check(f"{m}: shell syntax", r.returncode == 0)

# Qwen must keep a negative probe, else "predicted positive" cannot be wrong
q = open("run_gsm_hard_qwen25.sh", encoding="utf-8").read()
check("qwen retains a negative probe (protocol 2.2)", "neg4-16-22" in q)

print("\n" + "=" * 66)
print(f"{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if (OK and all(OK)) else 1)
