#!/usr/bin/env python3
"""End-to-end local checks for the zero-shot chat pipeline with SYNTHETIC cells
and fakes (no GPU, no model, no tokenizer download, stdlib only):
generation core (fires / atomic write / reuse validation), injection-site
report, the 6-cell evaluator and its refusals, strict scoring
and denominators.   python3 proofwriter_owa/tests/test_zs_pipeline.py"""
import contextlib, copy, io, json, os, re, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.path.dirname(HERE)
sys.path.insert(0, PW)
import zs_common as zc
import prompt_zeroshot as pz
import get_answer_proofwriter_owa_zeroshot as gen
import eval_proofwriter_owa_zeroshot as ev

fails = []
def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c: fails.append(m)

def raises(fn, exc=zc.ZSError):
    try: fn()
    except exc as e: return str(e)
    except SystemExit as e: return f"exit{e.code}"
    return None

TMP = tempfile.mkdtemp()
N = zc.N_ITEMS
LABELS = ["True", "False", "Unknown"]
samples = [{"sample_id": i, "key": f"k{i}", "dataset": "D3" if i % 2 else "D5",
            "official_theory_id": f"t{i}", "official_qid": f"q{i}",
            "theory_text": f"Theory {i}.", "question_text": f"Q {i}.",
            "content_sha256": "x"} for i in range(N)]
gold_label = {i: LABELS[i % 3] for i in range(N)}
bm = os.path.join(TMP, "blind.json"); gm = os.path.join(TMP, "gold.json")
json.dump({"meta": {"contains_labels": False, "manifest_sha256_16": "m16"}, "data": samples}, open(bm, "w"))
json.dump({"meta": {"contains_labels": True},
           "data": [dict(s, answer=gold_label[s["sample_id"]]) for s in samples]}, open(gm, "w"))
_, S = zc.load_blind_manifest(bm)

def text_for(i, wrong=False, style="clean"):
    lab = gold_label[i] if not wrong else LABELS[(i + 1) % 3]
    return {"clean": f"#### {lab}", "explained": f"#### {lab}\nBecause of the rules.",
            "resubmit": f"#### {lab}\n#### {lab}\n#### {lab}", "none": "I think it is " + lab.lower(),
            "reason": f"Step 1 ...\n#### {lab}"}[style]

PROV = {"host": "h", "gpu_names": ["A6000"], "sharded": False, "python": "3.10",
        "torch": "2", "transformers": "4", "numpy": "1", "platform": "linux",
        "cuda_visible_devices": "0"}
SITE = {"final_token_id": 271, "tail_token_ids": [1, 2, 271]}

def make_cell(model, cell, cond="nocot", n_wrong=60, style="clean", host="h", trunc=0,
              mutate=None, fires=None, path=None):
    alpha = zc.alpha_of(model, cell)
    meta = zc.expected_fixed_meta(model, cond, alpha)
    meta.update({"mask_path": "m.npy", "mask_sha256": "msha", "prompt_sha256": "psha",
                 "bare_prompt_sha256": zc.bare_prompt_sha256(S, cond),
                 "chat_template_hash": "cth", "manifest_sha256_16": "m16",
                 "injection_site": dict(SITE), "n_truncated": trunc,
                 "provenance": dict(PROV, host=host)})
    if fires is not None: meta["steering_fires"] = fires
    rows = []
    for i, s in enumerate(S):
        t = text_for(i, wrong=(i < n_wrong), style=style)
        tr = i < trunc
        rows.append({k: s[k] for k in ("sample_id", "key", "dataset", "official_theory_id", "official_qid")}
                    | {"generated": t, "generated_token_count": 12,
                       "stop_reason": "budget_exhausted" if tr else "natural_eos", "truncated": tr})
    d = {"meta": meta, "data": rows}
    if mutate: mutate(d)
    p = path or os.path.join(TMP, f"{model}_{cond}_{cell}_{len(os.listdir(TMP))}.json")
    json.dump(d, open(p, "w"))
    return p

# ── scoring / denominators ────────────────────────────────────────────────
rows = [{"sample_id": 0, "generated": "#### True", "generated_token_count": 5, "stop_reason": "natural_eos", "truncated": False},
        {"sample_id": 1, "generated": "False\n", "generated_token_count": 5, "stop_reason": "natural_eos", "truncated": False},
        {"sample_id": 2, "generated": "ok so #### Unknown is it", "generated_token_count": 5, "stop_reason": "natural_eos", "truncated": False},
        {"sample_id": 3, "generated": "#### False\n#### True", "generated_token_count": 5, "stop_reason": "natural_eos", "truncated": False}]
g = {0: "True", 1: "False", 2: "Unknown", 3: "False"}
st = zc.cell_stats(rows, g)
check(st["first_acc"] == {"n": 2, "denominator": 4, "rate": 0.5}, "first_acc: bare label & inline marker score incorrect, denominator = all items")
check(st["valid_submission"]["n"] == 2 and st["conditional_accuracy"] == {"n": 2, "denominator": 2, "rate": 1.0},
      "valid/conditional counts: bare label gets no fallback")
check(st["sensitivity_last_marker_acc"]["n"] == 1, "last-marker sensitivity separate (item 3 flips)")
check(st["early_candidate_label"]["n"] == 3 and st["early_candidate_label"]["denominator"] == 4,
      "label early-candidate counted over ALL items (bare 'False' counts even without a strict marker)")
check(st["early_candidate_label"]["without_valid_submission"]["n"] == 1, "early candidate without strict marker reported")
check(st["answer_first"]["denominator"] == 2, "answer_first denominator = valid-submission subset")
check(st["post_marker"]["counts"] == {"clean_stop": 1, "trailing_text": 0, "additional_marker": 1},
      "post-marker classes: clean_stop / additional_marker")
check(zc.post_marker_class("#### True\nBecause x") == "trailing_text", "explanation after marker = trailing_text")
check(zc.post_marker_class("#### True. ") == "clean_stop", "trailing period/whitespace = clean_stop")
check(st["post_marker"]["answer_first_clean_stop"]["n"] == 1, "answer-first compliance counted separately from marker-first")

def run_main(mod, argv):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try: rc = mod.main(argv)
        except SystemExit as e: rc = e.code
    return rc, err.getvalue()

# ── no gate: loop/truncation are reported split by validity, never gating ──
trunc = make_cell("llama3", "a0", trunc=60)
rows_t = json.load(open(trunc))["data"]
stt = zc.cell_stats(rows_t)
check(stt["truncation_by_validity"]["with_valid_submission"]["n"] == 60
      and stt["truncation_by_validity"]["without_valid_submission"]["denominator"] == 0,
      "truncation reported split by validity (60 valid+truncated, 0 invalid)")
lp_rows = json.load(open(make_cell("llama3", "a0")))["data"]
for r in lp_rows[:50]: r["generated"] = "#### True\n" + "xyz abc loop block here 1234567890. " * 12
stl = zc.cell_stats(lp_rows)
check(stl["loop_by_validity"]["with_valid_submission"]["n"] == 50 and stl["loop_by_validity"]["without_valid_submission"]["n"] == 0,
      "loop after a valid marker counted under 'with_valid_submission'")
check(not hasattr(zc, "GATE_THRESHOLDS") and not hasattr(zc, "gate_verdict") and not hasattr(gen, "load_gate")
      and not hasattr(ev, "run_gate") and not hasattr(ev, "check_gate_file"), "gate code is gone")
badcell = make_cell("qwen2.5", "wp", style="none")
check(zc.validate_cell(badcell, "qwen2.5", "nocot", 8, S)[1] is not None, "a cell with NO valid submissions is still accepted (no gate)")

# ── 4-cell evaluator ──────────────────────────────────────────────────────
def four(cond="nocot", kw=None):
    kw = kw or {}
    cells = {}
    for m in zc.MODEL_CONFIG:
        for c in zc.CELLS:
            nw = 60 if c == "a0" else 30
            cells[(m, c)] = make_cell(m, c, cond, n_wrong=nw, **kw.get((m, c), {}))
    return cells, None

def run_eval(cells, cond="nocot", out=None, drop=None, extra=None):
    order = [cells[k] for k in sorted(cells) if k != drop] + (extra or [])
    out = out or os.path.join(TMP, f"eval_{len(os.listdir(TMP))}.json")
    rc, err = run_main(ev, ["--condition", cond, "--manifest_blind", bm, "--gold", gm,
                            "--cells", *order, "--out", out])
    if os.environ.get("ZS_VERBOSE") and rc: print("      refusal:", err.strip()[:150])
    return rc, err, out

cells, _ = four()
rc, err, out = run_eval(cells)
R = json.load(open(out)) if rc == 0 else {}
check(rc == 0, f"evaluator accepts the complete, consistent set ({err.strip()[:80]})")
if R:
    rv = R["results"]["qwen2.5"]["reverse_vs_alpha0_EXPLORATORY"]
    check(R["results"]["qwen2.5"]["reverse_alpha"] == -6 and rv["alpha0_correct"] == 240 and "mcnemar_exact_p_unadjusted" in rv
          and R["results"]["llama3"]["reverse_alpha"] == 6, "reverse doses (qwen -6, llama +6) reported as exploratory unadjusted contrasts")
    check(R["holm_family"]["m"] == 2 and "holm_p_adj" not in rv, "Holm stays m=2 (the two workpoints); -6 outside it")
    w = R["results"]["llama3"]["workpoint_vs_alpha0"]
    check(w["alpha0_correct"] == 240 and w["workpoint_correct"] == 270 and w["wrong_to_right"] == 30 and w["right_to_wrong"] == 0,
          "paired counts: 240 -> 270, 30 gained / 0 lost")
    check(abs(w["delta_first_acc_pp"] - 10.0) < 1e-9, "delta = +10.00 pp")
    check(R["holm_family"]["m"] == 2 and w["holm_p_adj"] >= w["mcnemar_exact_p"], "Holm family m=2 and p_adj >= raw p")
    check(R["protocol"] == zc.PROTOCOL and R["n_shot"] == 0 and R["cot"] is False and R["condition"] == "nocot",
          "output records protocol / n_shot / cot / condition")
    check(len(R["conditions"]) == 6, "output lists its own 6 conditions")
rc, err, _ = run_eval(cells, out=out)
check(rc == 2 and "refusing to overwrite" in err, "evaluator refuses to overwrite")

# refusals
rc, err, _ = run_eval(cells, drop=("qwen2.5", "wp"))
check(rc == 2 and "missing" in err, "missing cell -> refused")
rc, err, _ = run_eval(cells, drop=("qwen2.5", "rev"))
check(rc == 2 and "missing" in err, "missing qwen -6 cell -> refused")
old_style = make_cell("llama3", "a0", mutate=lambda d: d["meta"].__setitem__("alpha", -4))
rc, err, _ = run_eval(cells, extra=[old_style])
check(rc == 2, "an out-of-family alpha (old -4 cell) -> refused")
def mut(field, val):
    def f(d): d["meta"][field] = val
    return f
for field, val in [("protocol", "proofwriter-owa-chat-v1"), ("n_shot", 1), ("cot", True),
                   ("prompt_template_id", "proofwriter-owa-cot-v2"), ("max_new_tokens", 2048),
                   ("batch_size", 24), ("prefill_tail_len", 4), ("temperature", 0.7),
                   ("chat_template_applied", False), ("steering_fires", 123), ("model_dir", "x/y"),
                   ("bare_prompt_sha256", "deadbeef")]:
    c2, _ = four("nocot", {("llama3", "wp"): {"mutate": mut(field, val)}})
    rc, err, _ = run_eval(c2)
    check(rc == 2, f"meta.{field} mismatch -> refused")
for k, kw in {"fires": {"fires": 2699}, "provenance host": {"host": "other"}}.items():
    c2, _ = four("nocot", {("qwen2.5", "wp"): kw})
    rc, err, _ = run_eval(c2)
    check(rc == 2, f"{k} wrong -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["meta"].__setitem__("mask_sha256", "other")}})
rc, err, _ = run_eval(c2); check(rc == 2 and "mask_sha256" in err, "mask hash differs between a model's cells -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["meta"]["provenance"].__setitem__("gpu_names", ["H100"])}})
rc, err, _ = run_eval(c2); check(rc == 2 and "gpu_names" in err, "different GPU model between cells -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["meta"]["injection_site"].__setitem__("final_token_id", 9)}})
rc, err, _ = run_eval(c2); check(rc == 2 and "injection_site" in err, "different injection site -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["data"].reverse()}})
rc, err, _ = run_eval(c2); check(rc == 2 and "order" in err, "sample order differs -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["data"].pop()}})
rc, err, _ = run_eval(c2); check(rc == 2, "short cell -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["data"][0].__setitem__("key", "zzz")}})
rc, err, _ = run_eval(c2); check(rc == 2, "identity (key) mismatch vs manifest -> refused")
c2, _ = four("nocot", {("llama3", "wp"): {"mutate": lambda d: d["data"][0].__setitem__("truncated", True)}})
rc, err, _ = run_eval(c2); check(rc == 2, "truncated flag disagreeing with stop_reason -> refused")
cells_cot, _ = four("cot")
rc, err, _ = run_eval(cells_cot, cond="nocot"); check(rc == 2, "CoT cells scored as No-CoT -> refused")
rc, err, _ = run_eval(cells_cot, cond="cot"); check(rc == 0, "CoT set scored as its own family")
# gold mismatch
gbad = json.load(open(gm)); gbad["data"][5]["official_qid"] = "bad"
gbp = os.path.join(TMP, "gold_bad.json"); json.dump(gbad, open(gbp, "w"))
rc, err = run_main(ev, ["--stage", "eval", "--manifest_blind", bm, "--gold", gbp,
                        "--cells", *[cells[k] for k in sorted(cells)],
                        "--out", os.path.join(TMP, "e_gb.json")])
check(rc == 2, "gold file identity mismatch -> refused")
# required-field completeness: both cells missing the same field must NOT pass
for fld in ("mask_sha256", "chat_template_hash", "injection_site", "prompt_sha256", "provenance"):
    def dropper(d, fld=fld): d["meta"].pop(fld, None)
    c2, _ = four("nocot", {("llama3", "a0"): {"mutate": dropper}, ("llama3", "wp"): {"mutate": dropper}}) if False else (None, None)
    ca = make_cell("llama3", "a0", mutate=dropper); cb = make_cell("llama3", "wp", mutate=dropper)
    ma = json.load(open(ca))["meta"]; mb = json.load(open(cb))["meta"]
    check("missing or empty" in (raises(lambda: zc.check_cross_cell([ma, mb], "llama3")) or ""),
          f"both cells missing {fld} -> cross-cell check refuses")
for fld, val in (("mask_sha256", ""), ("injection_site", {})):
    ma = json.load(open(make_cell("llama3", "a0")))["meta"]; ma[fld] = val; mb = json.loads(json.dumps(ma))
    check("missing or empty" in (raises(lambda: zc.check_cross_cell([ma, mb], "llama3")) or ""), f"both cells with empty {fld} -> refused")
ma = json.load(open(make_cell("llama3", "a0")))["meta"]; ma["provenance"]["gpu_names"] = []; mb = json.loads(json.dumps(ma))
check("gpu_names" in (raises(lambda: zc.check_cross_cell([ma, mb], "llama3")) or ""), "both cells with gpu_names=[] -> refused")
ma = json.load(open(make_cell("llama3", "a0")))["meta"]; ma["injection_site"] = {"x": 1}; mb = json.loads(json.dumps(ma))
check("final_token_id" in (raises(lambda: zc.check_cross_cell([ma, mb], "llama3")) or ""), "injection_site without final_token_id -> refused")

# ── generation core with fakes ────────────────────────────────────────────
class FakeMask(list):
    def __mul__(self, a): return FakeMask([[x * a for x in r] for r in self])
class FakeVC:
    def __init__(self, fires_per_site=1): self.n = 0; self.k = fires_per_site; self.calls = []
    def steering_fire_count(self, reset=False):
        v = self.n
        if reset: self.n = 0
        return v
    def regenerate(self, batch, **kw):
        self.calls.append(kw)
        if any(any(x != 0 for x in r) for r in kw["diff_matrices"]):
            self.n += self.k * len(batch) * 9            # L=9 llama sites
        return [{"text": "#### True", "generated_token_count": 3, "stop_reason": "natural_eos"} for _ in batch]
mask = FakeMask([[0.0, 0.0]] * 10 + [[1.0, 2.0]] * 9 + [[0.0, 0.0]] * 13)
def go(alpha, k=1, path=None):
    vc = FakeVC(k); path = path or os.path.join(TMP, f"g_{alpha}_{k}_{len(os.listdir(TMP))}.json")
    return vc, path, gen.generate_cell(vc, model="llama3", condition="nocot", alpha=alpha, samples=S,
        chat_prompts=["p"] * N, raw_mask=mask, mask_nz_rows=list(range(10, 19)), mask_sha="msha",
        mask_path="m.npy", site=dict(SITE), chat_template_hash="cth", prompt_sha="psha",
        manifest_meta={"manifest_sha256_16": "m16"}, out_path=path, prov=dict(PROV))
vc, p, meta = go(-6)
check(meta["steering_fires"] == 2700 and vc.calls[0]["max_new_tokens"] == 1024 and vc.calls[0]["temperature"] == 0.0
      and vc.calls[0]["top_p"] == 1.0 and vc.calls[0]["batch_size"] == 8 and len(vc.calls[0]["diff_matrices"]) == 32,
      "generation: -6 -> 2700 fires, 1024/greedy/top_p1/bs8, full-length diff matrix")
vc, p0, meta0 = go(0)
check(meta0["steering_fires"] == 0 and len(vc.calls[0]["diff_matrices"]) == 32, "alpha=0: same path, all-zero full matrix, 0 fires")
extra = {"mask_sha256": "msha", "prompt_sha256": "psha", "chat_template_hash": "cth", "manifest_sha256_16": "m16",
         "mask_path": "m.npy", "injection_site": dict(SITE)}
check(zc.validate_cell(p, "llama3", "nocot", -6, S, extra)[0]["n_shot"] == 0, "written cell passes the evaluator's validator")
check(raises(lambda: zc.validate_cell(p, "llama3", "nocot", -6, S, dict(extra, prompt_sha256="other"))) is not None,
      "resume: different prompt hash -> NOT reusable")
check(raises(lambda: zc.validate_cell(p, "llama3", "cot", -6, S, extra)) is not None, "resume: different condition -> NOT reusable")
check(raises(lambda: zc.validate_cell(p, "qwen2.5", "nocot", 8, S, extra)) is not None, "resume: different model -> NOT reusable")
before = os.path.getsize(p)
check(raises(lambda: go(-6, path=p)) is not None and os.path.getsize(p) == before, "atomic writer refuses to overwrite an existing cell")
check("fires" in (raises(lambda: go(-6, k=2)) or ""), "wrong fire count -> cell NOT written")
check(not [f for f in os.listdir(TMP) if f.startswith(".tmp_")], "no temp files left behind")
tr = os.path.join(TMP, "trunc.json"); open(tr, "w").write('{"meta": {')
check(raises(lambda: zc.validate_cell(tr, "llama3", "nocot", 0, S)) is not None, "truncated/corrupt file -> NOT reusable")

# ── injection-site report with fake tokenizers ────────────────────────────
class FakeTok:
    def __init__(self, bos, header, dup=False): self.bos_token_id = bos; self.bos_token = "<s>" if bos else None; self.header, self.dup = header, dup
    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) % 50 + 100 for c in text[-5:]] + self.header
        if self.bos_token_id is not None: ids = [self.bos_token_id] * (2 if self.dup else 1) + ids
        return {"input_ids": ids}
    def decode(self, ids): return "".join(f"<{i}>" for i in ids)
    def convert_ids_to_tokens(self, ids): return [f"t{i}" for i in ids]
rep = gen.injection_site_report(FakeTok(1, [8, 271]), ["abcde", "abcde"])
check(rep["final_token_id"] == 271 and rep["bos_count_per_prompt"] == [1], "injection site read from tokenizer; single BOS")
rep2 = gen.injection_site_report(FakeTok(None, [8, 198]), ["abcde"])
check(rep2["final_token_id"] == 198 and rep2["final_token_id"] != rep["final_token_id"] and rep2["bos_count_per_prompt"] is None,
      "the two models' final tokens are recorded independently (no equal-token assumption)")
check("double BOS" in (raises(lambda: gen.injection_site_report(FakeTok(1, [8, 271], dup=True), ["abcde"])) or ""), "double BOS -> hard stop")
class VarTok(FakeTok):
    def __call__(self, text, add_special_tokens=True):
        d = super().__call__(text, add_special_tokens); d["input_ids"] = d["input_ids"] + ([5] if text == "z" else []); return d
check("not uniform" in (raises(lambda: gen.injection_site_report(VarTok(1, [8, 271]), ["abcde", "z"])) or ""), "non-uniform final token -> hard stop")
check(gen.strip_leading_bos(FakeTok(1, []), "<s>hello") == "hello" and gen.strip_leading_bos(FakeTok(None, []), "<s>hello") == "<s>hello",
      "BOS text stripped only when the tokenizer has a BOS")

# ── static properties of the experiment definition ───────────────────────
src = open(os.path.join(PW, "get_answer_proofwriter_owa.py")).read()
m = re.search(r"MAX_NEW_TOKENS_FROZEN\s*=\s*(\d+)", src)
check(m and int(m.group(1)) == zc.GEN_FIXED["max_new_tokens"], "1024 token budget equals the frozen bare-line constant")
check({k: v["workpoint"] for k, v in zc.MODEL_CONFIG.items()} == {"llama3": -6, "qwen2.5": 8}, "workpoints = GSM8K record (-6 / +8)")
check({k: v["reverse"] for k, v in zc.MODEL_CONFIG.items()} == {"llama3": 6, "qwen2.5": -6}, "reverse doses = llama +6 / qwen -6")
check({k: v["L"] for k, v in zc.MODEL_CONFIG.items()} == {"llama3": 9, "qwen2.5": 6}, "layer counts 9 / 6")
check([zc.expected_fires(m_, a_) for m_, a_ in (("llama3", 0), ("llama3", -6), ("qwen2.5", 0), ("qwen2.5", 8))] == [0, 2700, 0, 1800],
      "expected fires 0 / 2700 / 0 / 1800")
for m_ in zc.MODEL_CONFIG:
    c = zc.MODEL_CONFIG[m_]
    check(c["layer_end"] - c["layer_start"] == c["L"], f"{m_} band length == L")
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
