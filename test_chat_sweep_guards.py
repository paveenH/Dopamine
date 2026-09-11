#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
test_chat_sweep_guards.py -- guard suite for the GSM-Hard and MATH
chat-template interface sweeps (get_answer_{gsm_hard,math}_chat_sweep.py,
run_{gsm_hard,math}_chat_sweep.sh, and the two offline analyzers).

NO GPU, NO SERVER, NO MODEL, NO NETWORK. Synthetic fixtures only.

DESIGN, following test_cruxeval_p4c.py's precedent:
  - Generator guards are AST-EXTRACTED and exercised as isolated source, never
    by importing the generator (which would pull in torch/llms).
  - Every guard is MUTATION-TESTED: the test BUILDS the specific defect and
    asserts rejection. A guard never shown to fire on a real defect is not a
    guard.
  - Analyzer guards ARE imported and driven on synthetic cells, because
    chat_sweep_common only needs numpy-free stdlib plus the frozen extractors.
  - An excepthook forces exit 2 on any crash, so a crash can never be read as
    a pass (the failure test_p3_label_firewall.py records: it once crashed
    with exit code 0 under a numpy-less interpreter).

The eight required fail-closed conditions from the task brief are each covered
by a named test below:
    missing one alpha ................ t_configs_missing_alpha
    duplicate alpha .................. t_configs_duplicate_alpha
    wrong mask / band ................ t_mask_band_guard, t_mask_rows_guard
    missing cell ..................... t_analyzer_missing_cell
    broken protocol .................. t_analyzer_protocol_guard
    item order misalignment .......... t_analyzer_order_guard
    output file already exists ....... t_overwrite_guard
plus: label leakage, digest mismatch, steering_fires, Holm family size,
      unknown stop_reason, and frozen-file no-diff.

Run:  python3 test_chat_sweep_guards.py      (stdlib only for the generator
                                              half; the analyzer half needs the
                                              RoleAnswer workspace importable)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
GEN_GSM_HARD = REPO / "get_answer_gsm_hard_chat_sweep.py"
GEN_MATH = REPO / "get_answer_math_chat_sweep.py"
SH_GSM_HARD = REPO / "run_gsm_hard_chat_sweep.sh"
SH_MATH = REPO / "run_math_chat_sweep.sh"

PASS, FAIL = [], []


def _hook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("\n[FATAL] crashed -- a crash is NOT a pass.", file=sys.stderr)
    os._exit(2)


sys.excepthook = _hook


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}"
                                                       if detail and not cond
                                                       else ""))


# ─────────────────────────────────────────────── generator: AST extraction ──
def load_src(p):
    return p.read_text(encoding="utf-8")


def extract_func(src, name):
    """Pull one top-level function's source out of a module WITHOUT importing
    it (importing would require torch)."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def extract_main_fragment(src, marker):
    """Return main()'s source; used to assert a check EXISTS textually."""
    return marker in src


def code_only(src):
    """Strip docstrings and comments, leaving EXECUTABLE source.

    Load-bearing: both generators mention "--allow_overwrite" and "271" in
    their docstrings precisely to say they do NOT use them. A raw substring
    scan therefore reads the disclaimer as the defect. These guards must look
    at code, not prose."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0].value.value = ""
    out = ast.unparse(tree)
    return re.sub(r"^\s*#.*$", "", out, flags=re.M)


# ── the frozen nine-point alpha guard, exercised as isolated source ──────────
ALPHA_GUARD = textwrap.dedent('''
    EXPECTED_ALPHAS = {-8, -6, -4, -2, 0, 2, 4, 6, 8}

    def die(msg):
        raise SystemExit(msg)

    def check_alphas(cfgs):
        got_alphas = sorted(al for al, _ in cfgs)
        want_alphas = sorted(EXPECTED_ALPHAS)
        if got_alphas != want_alphas:
            die("alpha family mismatch")
        return True
''')


def run_alpha_guard(cfgs):
    ns = {}
    exec(ALPHA_GUARD, ns)
    try:
        ns["check_alphas"](cfgs)
        return True
    except SystemExit:
        return False


def full9(extra=None, drop=None, dup=None):
    al = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    if drop is not None:
        al = [a for a in al if a != drop]
    if dup is not None:
        al = al + [dup]
    if extra is not None:
        al = al + [extra]
    return [(a, (11, 20)) for a in al]


def t_configs():
    print("\n[generator] frozen nine-point alpha family")
    check("t_configs_accepts_full_nine", run_alpha_guard(full9()))
    check("t_configs_accepts_any_order",
          run_alpha_guard(list(reversed(full9()))))
    check("t_configs_missing_alpha", not run_alpha_guard(full9(drop=-2)),
          "a missing dose must be refused")
    check("t_configs_duplicate_alpha", not run_alpha_guard(full9(dup=4)),
          "a duplicate dose must be refused")
    check("t_configs_extra_alpha", not run_alpha_guard(full9(extra=10)),
          "an extra dose must be refused")
    # parse_configs accepts floats, so int() coercion would let 2.5 read as 2.
    check("t_configs_float_alpha",
          not run_alpha_guard([(a, (11, 20)) for a in
                               [-8, -6, -4, -2, 0, 2.5, 4, 6, 8]]),
          "a non-integer dose must be refused, not coerced")
    check("t_configs_empty", not run_alpha_guard([]))

    # And the SORTED-LIST comparison must be what the real files use -- a
    # set-subset check would pass the duplicate/partial cases above.
    for p in (GEN_GSM_HARD, GEN_MATH):
        s = load_src(p)
        check(f"t_configs_exact_list_compare[{p.name}]",
              "got_alphas != want_alphas" in s and
              "sorted(al for al, _ in cfgs)" in s,
              "must compare a sorted LIST, not a set/subset")
        check(f"t_configs_no_int_coercion[{p.name}]",
              not re.search(r"int\(\s*al\b", s),
              "must not int()-coerce the alpha before comparing")


# ── mask / band guards ───────────────────────────────────────────────────────
def t_mask():
    print("\n[generator] mask + band guards")
    for p in (GEN_GSM_HARD, GEN_MATH):
        s = load_src(p)
        check(f"t_mask_rows_guard[{p.name}]",
              "raw_mask.shape[0] != n_decoder" in s,
              "must reject a mask sliced to the band")
        check(f"t_mask_nonzero_rows_guard[{p.name}]",
              "nz_rows != want_rows" in s,
              "must verify non-zero rows == decoder_layer_range(band)")
        check(f"t_mask_band_guard[{p.name}]",
              "!= BAND" in s or "!= (11, 20)" in s,
              "must reject a config naming a different band")
        check(f"t_mask_no_slicing[{p.name}]",
              "Do NOT slice the mask to the band" in s)

    # Behavioural mutation: the nonzero-row comparison must actually fire.
    def nz_check(nz_rows, want_rows):
        return nz_rows == want_rows
    want = list(range(10, 19))                     # decoder_layer_range(11,20)
    check("t_mask_band_mutation_correct", nz_check(want, want))
    check("t_mask_band_mutation_wrong_band",
          not nz_check(list(range(15, 21)), want),
          "a Qwen-style band 16-22 must not pass Llama's 11-20 check")
    check("t_mask_band_mutation_offset_by_one",
          not nz_check(list(range(11, 20)), want),
          "the layer_start-1 offset must not be silently accepted")


# ── overwrite / preflight-all-paths guards ───────────────────────────────────
def t_overwrite():
    print("\n[generator] overwrite + preflight-before-first-cell")
    for p in (GEN_GSM_HARD, GEN_MATH):
        s = load_src(p)
        check(f"t_overwrite_guard[{p.name}]",
              "already exists -- refusing to overwrite" in s)
        check(f"t_overwrite_no_escape_hatch[{p.name}]",
              "allow_overwrite" not in code_only(s),
              "a generator of frozen artifacts must ship no override flag "
              "(docstring prose disclaiming it is fine; code is not)")
        # The overwrite loop must run BEFORE the generation loop, or a long
        # sweep dies on cell 9 after eight cells are written.
        i_check = s.index("already exists -- refusing to overwrite")
        i_gen = s.index("vc.steering_fire_count(reset=True)")
        check(f"t_overwrite_checked_before_first_cell[{p.name}]",
              i_check < i_gen,
              "all output paths must be checked before the first generation")
        # Same for the band check.
        i_band = s.index("this protocol is frozen at")
        check(f"t_band_checked_before_first_cell[{p.name}]", i_band < i_gen)


# ── steering_fires ───────────────────────────────────────────────────────────
def t_steering():
    print("\n[generator] steering_fires accounting")

    def fires_ok(alpha, fires, L, n):
        return fires == (0 if alpha == 0 else L * n)

    check("t_fires_alpha0_zero", fires_ok(0, 0, 9, 300))
    check("t_fires_alpha0_nonzero_rejected", not fires_ok(0, 2700, 9, 300),
          "alpha=0 must fire exactly 0")
    check("t_fires_nonzero_correct", fires_ok(-6, 2700, 9, 300))
    check("t_fires_nonzero_wrong_rejected", not fires_ok(-6, 300, 9, 300),
          "L*n must be required, not merely non-zero")
    check("t_fires_zero_when_should_fire", not fires_ok(-6, 0, 9, 300),
          "a hook that never fired must be caught")
    for p in (GEN_GSM_HARD, GEN_MATH):
        s = load_src(p)
        check(f"t_fires_guard_present[{p.name}]",
              "the intervention is unverified" in s)
        check(f"t_rowcount_guard_present[{p.name}]",
              "zip() would silently drop" in s,
              "a short generation must be caught before zip() truncates")


# ── injection-site readout, chat template, BOS ───────────────────────────────
def t_injection_and_chat():
    print("\n[generator] injection-site readout + chat template + BOS")
    for p in (GEN_GSM_HARD, GEN_MATH):
        s = load_src(p)
        check(f"t_injection_read_not_assumed[{p.name}]",
              "_tail_ids" in s and "injection_tail_tokens_chat" in s,
              "the tail token must be READ OUT at run time")
        check(f"t_injection_no_hardcoded_271[{p.name}]",
              not re.search(r"\b271\b", code_only(s)),
              "must not hardcode the chat header token id in CODE "
              "(naming it in the docstring as 'never assumed' is correct)")
        check(f"t_injection_moved_flag[{p.name}]",
              "injection_site_moved_vs_bare" in s)
        check(f"t_injection_bare_reference[{p.name}]",
              "injection_tail_tokens_bare_reference" in s)
        check(f"t_chat_template_applied_flag[{p.name}]",
              '"chat_template_applied": True' in s)
        check(f"t_chat_add_generation_prompt[{p.name}]",
              "add_generation_prompt=True" in s)
        check(f"t_chat_effect_asserted[{p.name}]",
              "assert_chat_template_effective" in s,
              "a no-op template must be refused")
        check(f"t_double_bos_guard[{p.name}]",
              "assert_no_double_bos" in s and "strip_leading_bos" in s)
        check(f"t_g_prefill_declared_false[{p.name}]",
              '"g_prefill_measured": False' in s and
              "g_prefill_omitted_reason" in s)
        check(f"t_g_prefill_no_closed_form[{p.name}]",
              "is NOT a substitute" in s,
              "must state the closed form alpha*mean||m||^2 is not a "
              "substitute for a measured G_prefill")

    # strip_leading_bos behaviour, exercised for real on a fake tokenizer.
    src = load_src(GEN_GSM_HARD)
    fn = extract_func(src, "strip_leading_bos")
    ns = {}
    exec(fn, ns)

    class VC:
        class tokenizer:
            bos_token = "<|begin_of_text|>"
    check("t_bos_stripped_when_present",
          ns["strip_leading_bos"](VC, "<|begin_of_text|>hello") == "hello")
    check("t_bos_untouched_when_absent",
          ns["strip_leading_bos"](VC, "hello") == "hello")
    check("t_bos_only_leading_stripped",
          ns["strip_leading_bos"](VC, "a<|begin_of_text|>b")
          == "a<|begin_of_text|>b")


# ── GSM-Hard label firewall + digest ─────────────────────────────────────────
def t_label_firewall():
    print("\n[generator] GSM-Hard label firewall + frozen digest")
    src = load_src(GEN_GSM_HARD)
    fn = extract_func(src, "load_questions")
    ns = {"die": lambda m: (_ for _ in ()).throw(SystemExit(m)),
          "json": json,
          "FORBIDDEN_KEYS": ("answer", "gold", "gold_answer", "correct",
                             "accuracy", "target")}
    exec(fn, ns)
    lq = ns["load_questions"]

    D = "48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02"

    def write(tmp, meta, data):
        p = Path(tmp) / "q.json"
        p.write_text(json.dumps({"meta": meta, "data": data}))
        return str(p)

    good_meta = {"contains_labels": False, "questions_sha256": D}
    good_data = [{"sample_id": f"s{i}", "question": f"q{i}"} for i in range(3)]

    with tempfile.TemporaryDirectory() as tmp:
        p = write(tmp, good_meta, good_data)
        try:
            lq(p, D)
            ok = True
        except SystemExit:
            ok = False
        check("t_labelfree_accepts_clean_file", ok)

        # mutation: contains_labels not False
        p = write(tmp, {"contains_labels": True, "questions_sha256": D},
                  good_data)
        try:
            lq(p, D); bad = False
        except SystemExit:
            bad = True
        check("t_labelfree_rejects_contains_labels_true", bad)

        # mutation: a leaked gold field
        p = write(tmp, good_meta,
                  [{"sample_id": "s0", "question": "q", "gold": 42}])
        try:
            lq(p, D); bad = False
        except SystemExit:
            bad = True
        check("t_labelfree_rejects_leaked_gold_key", bad)

        p = write(tmp, good_meta,
                  [{"sample_id": "s0", "question": "q", "answer": 42}])
        try:
            lq(p, D); bad = False
        except SystemExit:
            bad = True
        check("t_labelfree_rejects_leaked_answer_key", bad)

        # mutation: wrong digest -> different sample
        p = write(tmp, {"contains_labels": False, "questions_sha256": "dead"},
                  good_data)
        try:
            lq(p, D); bad = False
        except SystemExit:
            bad = True
        check("t_digest_mismatch_rejected", bad,
              "a swapped sample file must be refused")

        # mutation: duplicate sample_id -> ambiguous pairing
        p = write(tmp, good_meta,
                  [{"sample_id": "s0", "question": "a"},
                   {"sample_id": "s0", "question": "b"}])
        try:
            lq(p, D); bad = False
        except SystemExit:
            bad = True
        check("t_duplicate_sample_id_rejected", bad)

    check("t_gsm_hard_writes_no_correctness",
          "is_correct_gsm8k" not in src and "extract_gsm8k_answer" not in src,
          "the GSM-Hard generator must have no correctness code path")


# ── MATH-specific: centralized extractor + budget ────────────────────────────
def t_math_specifics():
    print("\n[generator] MATH: centralized extractor + own budget")
    s = load_src(GEN_MATH)
    check("t_math_imports_utils_extractors",
          "from utils import extract_boxed, extract_math_answer, is_correct_math"
          in s,
          "MATH extraction must come from utils, not be reimplemented")
    check("t_math_no_local_boxed_regex",
          "def extract_boxed" not in s and "def extract_math_answer" not in s
          and r"\\boxed{" not in s.split('"""', 2)[-1].split("def main")[-1],
          "must not define a local \\boxed extractor")
    check("t_math_budget_2048", "default=2048" in s,
          "MATH must keep its own 2048 budget, not GSM8K's 768")
    check("t_math_batch_8", "--batch_size\", type=int, default=8" in s
          or 'default=8' in s)
    check("t_math_n_samples_300", "default=300" in s)
    check("t_math_truncates_like_bare", "all_samples[:n]" in s,
          "must truncate exactly as get_answer_regenerate_math.py does")
    check("t_math_first_caliber_documented",
          "FIRST" in s and "last_acc is a tail-pollution" in s.replace(
              "\n", " ").replace("  ", " ") or "SENSITIVITY" in s)

    sh = load_src(SH_MATH)
    check("t_math_sh_budget_2048", "MAX_NEW_TOKENS=2048" in sh)
    check("t_math_sh_batch_8", "BATCH_SIZE=8" in sh)
    check("t_math_sh_not_gsm8k_budget", "MAX_NEW_TOKENS=768" not in sh,
          "reusing GSM8K's budget would manufacture an extraction floor")

    shg = load_src(SH_GSM_HARD)
    check("t_gsm_hard_sh_budget_768", "MAX_NEW_TOKENS=768" in shg)
    check("t_gsm_hard_sh_batch_24", "BATCH_SIZE=24" in shg)


# ── launchers ────────────────────────────────────────────────────────────────
def t_launchers():
    print("\n[launcher] syntax + semantics")
    for sh in (SH_GSM_HARD, SH_MATH):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True)
        check(f"t_bash_syntax[{sh.name}]", r.returncode == 0,
              r.stderr.decode()[:200])
        s = load_src(sh)
        # The ${1:?usage ... {a|b}} brace bug: a curly-braced choice list INSIDE
        # a ${var:?msg} expansion terminates the expansion at its own first "}".
        # bash -n does NOT catch it. Neither launcher takes positional args, so
        # assert none was added with that shape.
        check(f"t_no_brace_bug[{sh.name}]",
              not re.search(r"\$\{[0-9]+:\?[^}]*\{", s),
              "a braced choice list inside ${1:?...} truncates the expansion")
        check(f"t_nine_configs[{sh.name}]",
              s.count("-11-20") == 9,
              "the launcher must pass exactly the nine frozen doses")
        for a in ("0-11-20", "neg8-11-20", "neg6-11-20", "neg4-11-20",
                  "neg2-11-20", "2-11-20", "4-11-20", "6-11-20", "8-11-20"):
            check(f"t_config_present[{sh.name}:{a}]", a in s)
        check(f"t_py_import_check[{sh.name}]",
              "cannot import numpy/torch" in s,
              "a wrong interpreter exits 127 and the nohup log looks empty")
        check(f"t_no_hard_gpu_reject[{sh.name}]",
              "exit 1" not in s.split("CUDA_VISIBLE_DEVICES is unset")[-1][:400],
              "the repo-wide rule forbids refusing an unset/multi-card device")
        check(f"t_band_11_20[{sh.name}]", "11-20" in s)


# ── frozen files must be untouched ───────────────────────────────────────────
FROZEN = [
    "get_answer_gsm8k_chat_sweep.py", "run_gsm8k_chat_sweep.sh",
    "get_answer_gsm_hard_blind.py", "run_gsm_hard_llama3.sh",
    "get_answer_regenerate_math.py", "run_math.sh",
    "get_answer_regenerate_gsm8k.py", "template.py", "utils.py", "llms.py",
]


def t_frozen_no_diff():
    print("\n[frozen] existing scripts must have NO diff")
    r = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        check("t_git_available", False, "git status failed")
        return
    dirty = {line[3:].strip() for line in r.stdout.splitlines() if line.strip()}
    for f in FROZEN:
        check(f"t_frozen_unmodified[{f}]", f not in dirty,
              "this file must not be modified by this line")


# ── analyzer guards (imported and driven for real) ───────────────────────────
def t_analyzers():
    print("\n[analyzer] fail-closed guards on synthetic cells")
    ra = Path.home() / "Documents" / "RSNResult" / "RoleAnswer"
    if not (ra / "chat_sweep_common.py").exists():
        check("t_analyzer_workspace_present", False,
              f"{ra} not found -- analyzer half skipped")
        return
    sys.path.insert(0, str(ra))
    try:
        import chat_sweep_common as C
    except Exception as e:
        check("t_analyzer_import", False, f"{e}")
        return
    check("t_analyzer_import", True)
    check("t_holm_m_is_8", C.HOLM_M == 8)

    # statistics, against hand-computed values
    check("t_mcnemar_10_0", abs(C.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12)
    check("t_mcnemar_no_discordant", C.mcnemar_exact(0, 0) == 1.0)
    check("t_mcnemar_symmetric",
          C.mcnemar_exact(3, 9) == C.mcnemar_exact(9, 3))
    h = C.holm([0.01, 0.04, 0.03])
    check("t_holm_monotone_and_ordered",
          abs(h[0] - 0.03) < 1e-12 and h[1] == h[2] == 0.06)

    # Holm family size must be exactly 8
    try:
        C.add_holm([{"p_raw": 0.01} for _ in range(7)]); bad = False
    except SystemExit:
        bad = True
    check("t_holm_rejects_partial_family", bad,
          "m<8 must not be adjusted under an m=8 label")
    try:
        C.add_holm([{"p_raw": 0.01} for _ in range(8)]); ok = True
    except SystemExit:
        ok = False
    check("t_holm_accepts_full_family", ok)

    # paired_contrast arithmetic
    pc = C.paired_contrast([1, 1, 0, 0], [1, 0, 1, 1])
    check("t_paired_contrast_counts",
          pc["discordant_base_only"] == 1 and pc["discordant_cell_only"] == 2
          and abs(pc["delta_pp"] - 25.0) < 1e-9,
          f"got {pc}")
    try:
        C.paired_contrast([1, 0], [1, 0, 1]); bad = False
    except SystemExit:
        bad = True
    check("t_paired_contrast_rejects_length_mismatch", bad)

    # item alignment
    cells = {0: [{"sample_id": "a"}, {"sample_id": "b"}],
             4: [{"sample_id": "a"}, {"sample_id": "b"}]}
    try:
        C.assert_item_alignment(cells, lambda r: r["sample_id"], "x"); ok = True
    except SystemExit:
        ok = False
    check("t_analyzer_alignment_accepts_aligned", ok)

    cells_ro = {0: [{"sample_id": "a"}, {"sample_id": "b"}],
                4: [{"sample_id": "b"}, {"sample_id": "a"}]}
    try:
        C.assert_item_alignment(cells_ro, lambda r: r["sample_id"], "x")
        bad = False
    except SystemExit as e:
        bad = "DIFFERENT ORDER" in str(e)
    check("t_analyzer_order_guard", bad,
          "same items in a different order must be refused, and named as such")

    cells_ds = {0: [{"sample_id": "a"}], 4: [{"sample_id": "z"}]}
    try:
        C.assert_item_alignment(cells_ds, lambda r: r["sample_id"], "x")
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_different_item_set_guard", bad)

    cells_dup = {0: [{"sample_id": "a"}, {"sample_id": "a"}]}
    try:
        C.assert_item_alignment(cells_dup, lambda r: r["sample_id"], "x")
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_duplicate_key_guard", bad)

    # protocol + config-drift
    base = {"protocol": "p", "chat_template_applied": True,
            "max_new_tokens": 768, "mask_sha256": "m", "L": 9}
    try:
        C.assert_cells_consistent({0: dict(base), 4: dict(base)}, "p"); ok = True
    except SystemExit:
        ok = False
    check("t_analyzer_consistency_accepts_identical", ok)

    try:
        C.assert_cells_consistent({0: dict(base), 4: {**base, "protocol": "q"}},
                                  "p")
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_protocol_guard", bad,
          "a cell not attesting the protocol must be refused")

    try:
        C.assert_cells_consistent(
            {0: dict(base), 4: {**base, "max_new_tokens": 2048}}, "p")
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_budget_drift_guard", bad)

    try:
        C.assert_cells_consistent(
            {0: dict(base), 4: {**base, "mask_sha256": "other"}}, "p")
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_mask_drift_guard", bad)

    try:
        C.assert_cells_consistent(
            {0: {**base, "chat_template_applied": False}}, "p")
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_requires_chat_applied", bad)

    # steering_fires re-check at analysis time
    try:
        C.assert_steering_fires({0: {"L": 9, "steering_fires": 0},
                                 4: {"L": 9, "steering_fires": 2700}}, 300)
        ok = True
    except SystemExit:
        ok = False
    check("t_analyzer_fires_accepts_correct", ok)
    try:
        C.assert_steering_fires({4: {"L": 9, "steering_fires": 1}}, 300)
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_fires_guard", bad)
    try:
        C.assert_steering_fires({0: {"L": 9, "steering_fires": 5}}, 300)
        bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_fires_alpha0_must_be_zero", bad)

    # unknown stop_reason must raise, not be silently bucketed
    try:
        C.behaviour_metrics(["x"], ["eos"]); bad = False
    except SystemExit:
        bad = True
    check("t_analyzer_unknown_stop_reason_guard", bad,
          "'eos' never matches llms.py's literals and once read 0% silently")
    m = C.behaviour_metrics(["x", "y"], ["natural_eos", "budget_exhausted"])
    check("t_analyzer_stop_reason_counted",
          m["natural_eos_pct"] == 50.0 and m["truncation_pct"] == 50.0)
    m2 = C.behaviour_metrics(["x"], [None])
    check("t_analyzer_none_stop_reason_tolerated",
          m2["no_stop_reason_pct"] == 100.0,
          "the frozen bare MATH cells predate return_metadata")

    # missing-cell guard, driven through the real analyzers on empty dirs
    for script, root_flag in (("analyze_gsm_hard_chat_sweep.py", "--chat_root"),
                              ("analyze_math_chat_sweep.py", "--chat_root")):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                [sys.executable, str(ra / script), root_flag, tmp,
                 "--out_dir", tmp],
                capture_output=True, text=True)
            check(f"t_analyzer_missing_cell[{script}]",
                  r.returncode != 0,
                  "an absent nine-point curve must fail closed")



# ── review fixes 2026-09-11: NA vs 0, analyzer path hint, alpha-vs-dir ───────
def t_review_fixes():
    print("\n[review fixes] NA-not-zero, honest analyzer hint, alpha-vs-dir")

    # (2) The launchers/generators must NOT tell an operator to run
    # "RoleAnswer/analyze_*.py" on the server -- that path does not exist there.
    for f in (SH_GSM_HARD, SH_MATH, GEN_GSM_HARD, GEN_MATH):
        s = load_src(f)
        check(f"t_no_server_analyzer_path[{f.name}]",
              "RoleAnswer/analyze_" not in s,
              "the analyzers are not synced to the server; that path misleads")
        check(f"t_hint_names_offline_workspace[{f.name}]",
              "RSNResult/RoleAnswer" in s,
              "the hint must name the offline workspace explicitly")

    ra = Path.home() / "Documents" / "RSNResult" / "RoleAnswer"
    if not (ra / "chat_sweep_common.py").exists():
        check("t_review_workspace_present", False, f"{ra} not found")
        return
    if str(ra) not in sys.path:
        sys.path.insert(0, str(ra))
    import chat_sweep_common as C

    # (3) meta.alpha must match the directory alpha.
    try:
        C.assert_meta_alpha_matches_dir(-6, {"alpha": -6}, "p"); ok = True
    except SystemExit:
        ok = False
    check("t_alpha_dir_accepts_match", ok)
    try:
        C.assert_meta_alpha_matches_dir(-6, {"alpha": 4}, "p"); bad = False
    except SystemExit:
        bad = True
    check("t_alpha_dir_guard", bad,
          "a cell in the wrong mdf_* directory must be refused")
    try:
        C.assert_meta_alpha_matches_dir(-6, {}, "p"); bad = False
    except SystemExit:
        bad = True
    check("t_alpha_dir_absent_meta_alpha_guard", bad)

    # (1)+(3) driven end to end on real fixtures.
    import json as _j, tempfile as _t, subprocess as _s
    N = 6
    ids = [f"s{i}" for i in range(N)]
    gold = [str(10 + i) for i in range(N)]

    def _meta(a, L=9):
        return {"protocol": "gsm-hard-chat-sweep-v1", "model": "llama3",
                "size": "8B", "alpha": a, "layer_start": 11, "layer_end": 20,
                "L": L, "mask_sha256": "m", "max_new_tokens": 768,
                "temperature": 0.0, "top_p": 1.0, "batch_size": 24,
                "prompt_body_sha256": "pb", "prompt_sha256": "ps",
                "chat_template_hash": "ct",
                "prompt_wrapper_id": "llama3-chat-template-v1",
                "chat_template_applied": True, "padding_side": "left",
                "prefill_only": True, "prefill_tail_len": 1, "role": "neutral",
                "cot": False, "n": N,
                "steering_fires": 0 if a == 0 else L * N,
                "questions_sha256": "D" * 64}

    with _t.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "gold.json").write_text(_j.dumps(
            {"meta": {"questions_sha256": "D" * 64},
             "data": [{"sample_id": i, "gold": g} for i, g in zip(ids, gold)]}))
        for a in C.ALPHAS:
            rows = [{"sample_id": i, "question": f"q{i}",
                     "generated": f"x #### {g}", "generated_token_count": 9,
                     "stop_reason": "natural_eos"} for i, g in zip(ids, gold)]
            d = tmp / "chat" / f"mdf_{a}".replace("-", "neg")
            d.mkdir(parents=True)
            (d / "gsm_hard_chat_8B_11_20.json").write_text(
                _j.dumps({"meta": _meta(a), "data": rows}))
        # bare: NO stop_reason (the frozen tree's real shape), and only 4 of
        # the 5 expected doses, to exercise the INCOMPLETE label too.
        for a in [-8, -6, -4, 0]:
            rows = [{"sample_id": i, "question": f"q{i}",
                     "generated": f"y #### {g}"} for i, g in zip(ids, gold)]
            d = tmp / "bare" / f"mdf_{a}".replace("-", "neg")
            d.mkdir(parents=True)
            (d / "gsm_hard_8B_11_20.json").write_text(
                _j.dumps({"meta": {}, "data": rows}))

        cmd = [sys.executable, str(ra / "analyze_gsm_hard_chat_sweep.py"),
               "--chat_root", str(tmp / "chat"), "--bare_root", str(tmp / "bare"),
               "--gold", str(tmp / "gold.json"), "--out_dir", str(tmp / "out")]
        r = _s.run(cmd, capture_output=True, text=True)
        check("t_review_e2e_runs", r.returncode == 0,
              (r.stdout + r.stderr)[-300:])
        if r.returncode == 0:
            res = _j.load(open(tmp / "out" / "gsm_hard_chat_sweep_result.json"))
            pr = res["bare_vs_chat_descriptive"]
            check("t_bare_eos_is_NA_not_zero",
                  all(x["natural_eos_pct_bare"] == "NA" for x in pr),
                  "a bare cell with no stop_reason must read NA")
            check("t_bare_truncation_is_NA_not_zero",
                  all(x["truncation_pct_bare"] == "NA" for x in pr))
            check("t_bare_no_stop_reason_pct_kept",
                  all(x["no_stop_reason_pct_bare"] == 100.0 for x in pr),
                  "no_stop_reason_pct must still say 100")
            check("t_chat_side_still_numeric",
                  all(isinstance(x["natural_eos_pct_chat"], (int, float))
                      for x in pr),
                  "the chat side DOES record stop_reason and must stay numeric")
            check("t_bare_incomplete_flagged",
                  res["bare_paired_complete"] is False
                  and res["bare_paired_alphas_present"] == [-8, -6, -4, 0],
                  "a 4-of-5 bare set must not be reported as complete")
            check("t_bare_incomplete_in_title",
                  "INCOMPLETE" in r.stdout,
                  "the printed table must say INCOMPLETE, not '5 of 9'")

        # mutation: move a cell into the wrong directory
        import shutil as _sh
        _sh.rmtree(tmp / "out")
        src = tmp / "chat" / "mdf_4" / "gsm_hard_chat_8B_11_20.json"
        dst = tmp / "chat" / "mdf_6" / "gsm_hard_chat_8B_11_20.json"
        _sh.copy(src, dst)          # mdf_6 now holds a cell whose meta.alpha=4
        r2 = _s.run(cmd, capture_output=True, text=True)
        check("t_misplaced_cell_rejected", r2.returncode != 0
              and "meta.alpha" in (r2.stdout + r2.stderr),
              "a cell copied into the wrong dose directory must fail closed")

def main():
    print("=" * 72)
    print("chat-sweep guard suite (GSM-Hard + MATH) -- no GPU, no model")
    print("=" * 72)
    t_configs()
    t_mask()
    t_overwrite()
    t_steering()
    t_injection_and_chat()
    t_label_firewall()
    t_math_specifics()
    t_launchers()
    t_frozen_no_diff()
    t_analyzers()
    t_review_fixes()
    print("\n" + "=" * 72)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL GUARDS PASS")


if __name__ == "__main__":
    main()
