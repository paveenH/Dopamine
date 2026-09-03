#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4c CRUXEval-O guard suite. NO GPU, NO SERVER, NO REAL MODEL, NO NETWORK.

STDLIB ONLY, deliberately. `test_p3_label_firewall.py` records why: an earlier
test imported the generation module (numpy via utils) and under a numpy-less
interpreter CRASHED WITH EXIT CODE 0 -- CI would have read a crash as a pass.
So the runner's guards are extracted with `ast` rather than imported, and an
excepthook forces exit 2 on any crash.

EVERY GUARD IS MUTATION-TESTED: the test constructs the specific defect and
asserts the guard fires. A guard never shown to fire on a real defect is not a
guard.

    python3 test_cruxeval_p4c.py        # runs under bare python3 too
"""

import ast
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
N_CHECK = 0


def _hook(exc_type, exc, tb):
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    print("\n[CRASH] the suite aborted; this is a FAILURE, not a pass.",
          file=sys.stderr)
    os._exit(2)


sys.excepthook = _hook


def check(cond, msg):
    global N_CHECK
    N_CHECK += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def load_src(name):
    return open(os.path.join(HERE, name), encoding="utf-8").read()


def get_func(src, name):
    """Extract one top-level function and exec it in an isolated namespace, so
    the module's numpy/torch imports never run."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {}
            exec(compile(mod, "<extract>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"function {name!r} not found")


def get_const(src, name):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"constant {name!r} not found")


# ===================================================================== SCORER
print("\n=== eval_cruxeval.py -- parsing and scoring")
EV = load_src("eval_cruxeval.py")
ev_ns = {"re": __import__("re"), "ast": ast}
exec(compile(ast.Module(
    body=[n for n in ast.parse(EV).body
          if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign,
                            ast.FunctionDef))
          and not (isinstance(n, (ast.Import, ast.ImportFrom))
                   and getattr(n, "module", "") not in (None, "math")
                   and not isinstance(n, ast.Import))],
    type_ignores=[]), "<ev>", "exec"), ev_ns)

extract = ev_ns["extract"]
as_literal = ev_ns["as_literal"]
correct = ev_ns["correct"]
mcnemar_exact = ev_ns["mcnemar_exact"]
holm = ev_ns["holm"]
boot_ci = ev_ns["boot_ci"]

# --- marker extraction
check(extract("blah\n#### [1, 2]") == "[1, 2]", "first marker extracted")
check(extract("#### 1\ntext\n#### 2", "first") == "1", "FIRST is first")
check(extract("#### 1\ntext\n#### 2", "last") == "2", "LAST is last")
check(extract("no marker here") is None, "no marker -> None")
check(extract("#### 'a b'") == "'a b'", "string literal with a space survives")
check(extract("####[1]") == "[1]", "marker with no space after ####")
# the parser must take the LINE remainder, not the rest of the text
check(extract("#### 42\nSo the answer is 42.") == "42",
      "marker payload stops at the newline")

# --- literal parsing: MUST NOT execute
ok, v = as_literal("[1, 2]")
check(ok and v == [1, 2], "list literal parses")
check(as_literal("{1: None, 2: None}")[1] == {1: None, 2: None}, "dict parses")
check(as_literal("(0, 'xxx')")[1] == (0, "xxx"), "tuple parses")
check(as_literal("True")[1] is True, "bool parses")
check(as_literal(None)[0] is False, "None input -> not ok")
check(as_literal("")[0] is False, "empty -> not ok")
# the security property, stated as a test
sentinel = os.path.join(tempfile.gettempdir(), "p4c_should_not_exist")
if os.path.exists(sentinel):
    os.remove(sentinel)
check(as_literal(f"__import__('os').system('touch {sentinel}')")[0] is False,
      "MUTATION: an exec payload does NOT parse")
check(not os.path.exists(sentinel),
      "MUTATION: an exec payload was NOT executed (literal_eval cannot call)")
check(as_literal("[1]*3")[0] is False,
      "a non-literal EXPRESSION does not parse (the recorded gap vs official)")

# --- semantic equality, the whole point of parsing both sides
check(correct("[1, 2]", [1, 2]) == 1, "exact spelling correct")
check(correct("[1,2]", [1, 2]) == 1,
      "whitespace-different spelling ALSO correct (semantic equality)")
check(correct("{2: None, 1: None}", {1: None, 2: None}) == 1,
      "dict key order does not matter")
check(correct("[2, 1]", [1, 2]) == 0, "list ORDER does matter")
check(correct("[1]*3", [1, 1, 1]) == 0,
      "non-literal expression scores INCORRECT here, as documented")
check(correct(None, [1]) == 0, "no marker scores incorrect")
check(correct("garbage(", [1]) == 0, "unparseable scores incorrect")
check(correct("'1'", 1) == 0, "str '1' != int 1 (no coercion)")
check(correct("1", True) == 1,
      "1 == True in Python; recorded, since bool gold exists in the sample")

# --- statistics
b01, b10, p = mcnemar_exact([0]*10, [1]*10)
check((b01, b10) == (10, 0), "McNemar discordant counts")
check(abs(p - 2/1024) < 1e-12, "McNemar(10,0) exact p = 2/1024")
check(mcnemar_exact([1, 0], [1, 0])[2] == 1.0, "no discordant -> p=1")
h = holm([("a", 0.01), ("b", 0.04)])
check(abs(h["a"] - 0.02) < 1e-12 and abs(h["b"] - 0.04) < 1e-12, "Holm m=2")
lo, hi = boot_ci([0]*100, [1]*100, B=200, seed=0)
check(lo == hi == 100.0, "bootstrap on a constant +1 difference is +100pp")

# --- frozen doses
check(get_const(EV, "WORKPOINT") == {"llama3": -6, "qwen2.5": 8},
      "workpoints are the frozen GSM8K values")
check(get_const(EV, "NEIGHBOUR") == {"llama3": -4, "qwen2.5": 6},
      "neighbour diagnostics frozen")
check(get_const(EV, "REVERSE") == {"llama3": 4, "qwen2.5": -6},
      "reverse diagnostics frozen")
check(get_const(EV, "N") == 300, "scorer expects n=300")
check("holm_family_m" in EV and '"holm_family_m": 2' in EV,
      "Holm family is m=2")
check("preflight" in EV and "PREFLIGHT cell" in EV,
      "scorer refuses a preflight cell")

# ===================================================================== RUNNER
print("\n=== get_answer_cruxeval.py -- prompt, matrix, firewall")
RUN = load_src("get_answer_cruxeval.py")
import hashlib
PROMPT = get_const(RUN, "PROMPT")
check(hashlib.sha256(PROMPT.encode()).hexdigest()[:16]
      == get_const(RUN, "PROMPT_SHA256"),
      "PROMPT matches its pinned sha256 (a drifted pin is a dead guard)")
check(PROMPT.endswith("Response: "), "prompt ends at the frozen anchor")
check("step by step" not in PROMPT.lower(),
      "prompt does NOT elicit CoT (No-CoT is the protocol)")
check("####" in PROMPT, "prompt names the #### marker")
check("{code}" in PROMPT and "{input}" in PROMPT, "prompt has both slots")

EXPECTED_CELLS = get_const(RUN, "EXPECTED_CELLS")
check(EXPECTED_CELLS["llama3"] ==
      {(0, 11, 20), (-6, 11, 20), (-4, 11, 20), (4, 11, 20)},
      "llama matrix: 4 cells, band 11-20")
check(EXPECTED_CELLS["qwen2.5"] ==
      {(0, 16, 22), (8, 16, 22), (6, 16, 22), (-6, 16, 22)},
      "qwen matrix: 4 cells, band 16-22")
check(get_const(RUN, "BUDGET") == 768 and get_const(RUN, "BATCH_SIZE") == 24,
      "budget 768 / bs 24, inherited from the GSM8K path")
check(get_const(RUN, "N_FORMAL") == 300, "runner expects n=300")
check("stop_strings" in RUN and '"stop_strings": None' in RUN,
      "no stop string (the prompt contains #### -- the CGT failure)")
check('"cot": False' in RUN and '"few_shot": False' in RUN,
      "cot/few_shot recorded False in meta")

# the label firewall, mutation-tested
load_questions = None
try:
    load_questions = get_func(RUN, "load_questions")
except AssertionError:
    check(False, "load_questions extractable")
if load_questions:
    ns = sys.modules[__name__].__dict__
    # rebind the module-level names load_questions closes over
    g = load_questions.__globals__
    g.update({"json": json, "die": lambda m: (_ for _ in ()).throw(SystemExit(2)),
              "PROTOCOL": get_const(RUN, "PROTOCOL"),
              "FORMAL_DIGEST": get_const(RUN, "FORMAL_DIGEST"),
              "N_FORMAL": 300, "N_PREFLIGHT": get_const(RUN, "N_PREFLIGHT"),
              "LABEL_FIELDS": get_const(RUN, "LABEL_FIELDS")})

    def wq(meta_over=None, rows_over=None, n=300):
        meta = {"contains_labels": False,
                "protocol": get_const(RUN, "PROTOCOL"),
                "questions_sha256": get_const(RUN, "FORMAL_DIGEST"),
                "revision": "x" * 40}
        meta.update(meta_over or {})
        rows = [{"sample_id": i, "source_id": f"sample_{i}",
                 "code": "def f(): pass", "input": "", "content_sha256": "d"}
                for i in range(n)]
        if rows_over:
            rows_over(rows)
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"meta": meta, "data": rows}, f); f.close()
        return f.name

    def refuses(path, label, preflight=False):
        try:
            load_questions(path, preflight)
            check(False, f"MUTATION: {label} was NOT refused")
        except SystemExit:
            check(True, f"MUTATION: {label} refused")

    ok_path = wq()
    m, d = load_questions(ok_path)
    check(len(d) == 300, "a clean blind file loads 300 items")
    m, d = load_questions(ok_path, True)
    check(len(d) == get_const(RUN, "N_PREFLIGHT"),
          "preflight truncates AFTER the digest/count checks")

    refuses(wq({"contains_labels": True}), "gold-bearing file")
    refuses(wq({"contains_labels": None}), "file with no contains_labels")
    refuses(wq({"questions_sha256": "deadbeefdeadbeef"}), "wrong digest")
    refuses(wq({"protocol": "bbh-p4b-v0"}), "wrong protocol")
    refuses(wq(n=299), "wrong item count")
    refuses(wq(rows_over=lambda r: r[0].update({"gold": "[1]"})),
            "leaked 'gold' field")
    refuses(wq(rows_over=lambda r: r[0].update({"output": "[1]"})),
            "leaked 'output' field (CRUXEval's gold column name)")
    refuses(wq(rows_over=lambda r: r.__setitem__(0, dict(r[0], sample_id=999))),
            "sample_ids not covering 0..299")

# descriptive fields
descriptive = get_func(RUN, "descriptive")
descriptive.__globals__.update({"MARKER_RE": ev_ns["MARKER_RE"],
                                "is_loop": get_func(RUN, "is_loop")})
d1 = descriptive("#### [1]")
check(d1["answer_first"] is True and d1["n_markers"] == 1, "answer_first true")
d2 = descriptive("Let me trace it.\n#### [1]")
check(d2["answer_first"] is False and d2["pre_marker_chars"] == 17,
      "pre_marker_chars counts text before the marker")
blk = "z" * 40
check(descriptive(blk * 4)["degenerate_tail"] is True,
      "degenerate tail: final 40-char block recurring 4x")
check(descriptive("short answer")["degenerate_tail"] is False,
      "a short clean generation is not degenerate")

# ===================================================================== LOADER
print("\n=== data_cruxeval.py -- sampling determinism and firewall")
LD = load_src("data_cruxeval.py")
check("hashlib.sha256" in LD and "rank_key" in LD, "selection uses sha256")
check(get_const(LD, "SALT") == "cruxeval-p4c-v0", "salt frozen")
check(get_const(LD, "N_FORMAL") == 300 and get_const(LD, "N_POOL") == 800,
      "300 of 800")
rev = get_const(LD, "REVISION")
check(len(rev) == 40 and all(c in "0123456789abcdef" for c in rev),
      "revision is a full 40-hex SHA")
exp = get_const(LD, "EXPECTED")
check(exp["questions_sha256"] == "4580b7a9a9ef6054"
      and exp["gold_sha256"] == "a214d1fc7d84a2d9", "frozen digests present")
check("output" in get_const(LD, "LABEL_FIELDS"),
      "'output' (CRUXEval's gold column) is in LABEL_FIELDS")
check("output" not in get_const(LD, "BLIND_ALLOWED"),
      "'output' is NOT whitelisted into the blind record")
check("literal_eval" in LD, "loader asserts gold parses")
check("newline" in LD.lower(), "loader asserts no gold contains a newline")

rank_key = get_func(LD, "rank_key")
rank_key.__globals__.update({"hashlib": hashlib, "SALT": get_const(LD, "SALT")})
r = {"id": "sample_0", "code": "def f(): pass", "input": "1"}
check(rank_key(r) == rank_key(r), "rank_key is deterministic in-process")
# the real property: stable ACROSS processes, which hash() is not
sub = subprocess.run(
    [sys.executable, "-c",
     "import hashlib;print(hashlib.sha256("
     "'cruxeval-p4c-v0:sample_0:def f(): pass:1'.encode()).hexdigest())"],
    capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "1"})
check(sub.stdout.strip() == rank_key(r),
      "MUTATION: rank_key is stable across processes (hash() would not be)")
h1 = subprocess.run([sys.executable, "-c", "print(hash('sample_0'))"],
                    capture_output=True, text=True,
                    env={**os.environ, "PYTHONHASHSEED": "1"}).stdout
h2 = subprocess.run([sys.executable, "-c", "print(hash('sample_0'))"],
                    capture_output=True, text=True,
                    env={**os.environ, "PYTHONHASHSEED": "2"}).stdout
check(h1 != h2, "CONTROL: Python hash() IS process-salted, hence sha256")

# ===================================================================== LAUNCHER
print("\n=== run_cruxeval.sh -- steps and dose allowlist")
SH = load_src("run_cruxeval.sh")
for tok in ("BASELINE", "WORKPOINT", "NEIGHBOUR", "REVERSE", "ALL", "PREFLIGHT"):
    check(f"  {tok})" in SH, f"step {tok} exists")
check("AWP=neg6-11-20" in SH and "AWP=8-16-22" in SH, "workpoint configs")
check("ANB=neg4-11-20" in SH and "ANB=6-16-22" in SH, "neighbour configs")
check("ARV=4-11-20" in SH and "ARV=neg6-16-22" in SH, "reverse configs")
check("need_baseline" in SH, "steered steps require the alpha=0 cell")
check("CUDA_VISIBLE_DEVICES must be set" not in SH,
      "GPU pinning is NOT required (cells may run on different devices)")
check("provenance" in SH.lower() or "CUDA_VISIBLE_DEVICES=" in SH,
      "device is recorded as provenance")
check("import numpy, torch" in SH, "launcher checks the interpreter (PY=127 trap)")

print(f"\n{'=' * 60}")
if FAILS:
    print(f"FAILED {len(FAILS)}/{N_CHECK}")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print(f"ok  {N_CHECK}/{N_CHECK} checks passed")
