#!/usr/bin/env python3
"""Local stdlib-only checks for the FinQA Chat 0-shot CoT pipeline (synthetic cells + fakes; no GPU, no model,
no tokenizer download, no numpy). NOT covered here (server only): real tokenizer / BOS / final token id,
mask non-zero-row check (needs numpy), the actual generation path and real steering_fires.
    python3 finqa/test_finqa_chat.py"""
import contextlib, copy, io, json, os, random, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import get_answer_finqa_chat as G
import eval_finqa_chat as E
import prompt_finqa as PF
import chat_cot_sat_lib as L

fails = []
def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c: fails.append(m)

def raises(fn, exc=(SystemExit, G.FQError, ValueError), label=""):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            fn()
    except exc:
        return True
    return False

DATA = os.path.join(os.path.dirname(HERE), "finqa", "dist", "finqa_formal.json")
meta, samples = G.load_samples(DATA)
check(len(samples) == 300 and meta["prompt_content_sha256_16"] == G.prompt_digest(samples), "real data file loads, digest recomputes")
raw = json.load(open(DATA))
TMP = tempfile.mkdtemp()
def dump(obj, name):
    p = os.path.join(TMP, name); json.dump(obj, open(p, "w")); return p

# ---- A. data-file refusals
bad = copy.deepcopy(raw); bad["data"][5]["table_linear"] += " x"
check(raises(lambda: G.load_samples(dump(bad, "a1.json"))), "edited table text rejected (digest)")
bad = copy.deepcopy(raw); bad["data"][3], bad["data"][4] = bad["data"][4], bad["data"][3]
check(raises(lambda: G.load_samples(dump(bad, "a2.json"))), "reordered samples rejected")
bad = copy.deepcopy(raw); bad["data"][9]["id"] = bad["data"][8]["id"]
check(raises(lambda: G.load_samples(dump(bad, "a3.json"))), "duplicate official id rejected")
bad = copy.deepcopy(raw); bad["data"] = bad["data"][:299]
check(raises(lambda: G.load_samples(dump(bad, "a4.json"))), "299 rows rejected")

# ---- B. prompt / chat wrapping with a fake tokenizer
check(PF.FINQA_TEMPLATE.count("Let's think step by step.") == 1 and "Answer: " in PF.FINQA_TEMPLATE, "reused prompt keeps the CoT line and Answer: once")
class FakeTok:
    bos_token, bos_token_id, padding_side = "<s>", 1, "left"
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert len(msgs) == 1 and msgs[0]["role"] == "user" and not tokenize and add_generation_prompt
        return "<s><|hdr|>user<|/hdr|>\n\n" + msgs[0]["content"].strip() + "<|eot|><|hdr|>assistant<|/hdr|>\n\n"
    def __call__(self, t, add_special_tokens=True):
        ids = [1] if t.startswith("<s>") else []
        ids += [ord(c) % 250 + 2 for c in t.replace("<s>", "")]
        return {"input_ids": ([1] + ids if add_special_tokens and not t.startswith("<s>") else ids)}
    def decode(self, ids): return "".join(chr(i) for i in ids)
tk = FakeTok()
users = [PF.build_finqa_prompt(s) for s in samples[:5]]
wr = G.wrap_prompts(tk, users)
check(all(not w.startswith("<s>") for w in wr) and all("assistant" in w for w in wr), "leading BOS stripped, assistant header present")
check(not raises(lambda: G.assert_no_double_bos(tk, wr)) and L.check_prompt_tokens(tk, wr)["bos_counts"] == [1], "exactly one BOS after tokenisation")
class DoubleTok(FakeTok):
    def __call__(self, t, add_special_tokens=True): return {"input_ids": [1, 1, 5]}
check(raises(lambda: G.assert_no_double_bos(DoubleTok(), wr)), "double BOS detected and refused")
check(raises(lambda: L.check_prompt_tokens(tk, wr[:2] + ["short"])), "non-uniform final token refused")

# ---- C. synthetic cells
gold = {s["sample_id"]: float(s["gold_value"]) for s in samples}
ALPHAS = {"llama3": [0, -6], "qwen2.5": [0, 8]}
def gen_text(i, a, mdl):
    rnd = random.Random(f"{mdl}{a}{i}")
    r = rnd.random()
    g = samples[i]["gold_raw"]
    if r < 0.45: return f"Step 1: compute.\nThe change is large.\n#### {g}"
    if r < 0.60: return f"{g}\n\nStep 1: recompute.\n#### {g}\n" + ("#### 1 " * 0)
    if r < 0.70: return "Let's think.\n" + ("repeat this tail please " * 20)
    if r < 0.85: return "Step 1: x.\n#### 99999"
    return f"Step 1.\n#### {g}\n" + ("I am done. " * 30)
def build_cell(mdl, a, mod=None):
    n = 300
    gens = []
    for i, s in enumerate(samples):
        t = gen_text(i, a, mdl)
        trunc = "repeat this tail" in t
        gens.append({"text": t, "generated_token_count": 512 if trunc else 90, "stop_reason": "budget_exhausted" if trunc else "natural_eos"})
    rows = G.make_rows(samples, gens)
    tok = {"bos_token_id": 1, "bos_counts": [1], "n_prompts": n, "final_token_id": 271, "final_token_text": "\n\n", "final_token_counts": {"271": n}}
    expect = {"mask_sha256": "m" * 64 if mdl == "llama3" else "q" * 64, "user_prompt_sha256": L.sha("\n".join(PF.build_finqa_prompt(s) for s in samples)),
              "wrapped_prompt_sha256": "w" + mdl, "chat_template_hash": "c" + mdl, "model_dir": G.MODELS[mdl]["model_dir"],
              "questions_file_sha256": G.file_sha256(DATA), "sample_order_sha256": G.order_sha(samples),
              "prompt_content_sha256_16": meta["prompt_content_sha256_16"], "mode": "formal", "final_token_id": 271}
    fires = G.expected_fires(mdl, a, n)
    prov = {"host": "h1", "cuda_visible_devices": "0", "param_devices": ["cuda:0"], "sharded": False, "cuda_device_names": ["GPU"],
            "library_versions": {"torch": "2", "transformers": "4", "numpy": "1"}, "code_version": "abc"}
    m = G.build_meta(mdl, a, n, meta, DATA, G.MODELS[mdl]["model_dir"], "/mask", list(range(G.MODELS[mdl]["band"][0] - 1, G.MODELS[mdl]["band"][1] - 1)),
                     tok, expect["chat_template_hash"], expect, fires, fires, prov, "left")
    d = {"meta": m, "data": rows}
    if mod: mod(d)
    return d
def write_cells(tag, mods=None, skip=(), extra=()):
    mods = mods or {}
    paths = []
    for mdl, al in ALPHAS.items():
        for a in al:
            if (mdl, a) in skip: continue
            paths.append(dump(build_cell(mdl, a, mods.get((mdl, a))), f"{tag}_{mdl}_{a}.json"))
    return paths
def run_eval(paths, out, **kw):
    args = ["--cells", *paths, "--gold_file", DATA, "--out", out]
    return E.main(args)

good = write_cells("good")
out = os.path.join(TMP, "good_eval.json")
with contextlib.redirect_stdout(io.StringIO()):
    rc = run_eval(good, out)
ev = json.load(open(out))
check(rc == 0 and ev["holm_family"]["m"] == 2, "four-cell evaluation runs; Holm m=2")
exp_first = {}
for mdl, al in ALPHAS.items():
    for a in al:
        c = sum(1 for i, s in enumerate(samples) if E.is_correct(E.first_legal_answer(gen_text(i, a, mdl))[1], gold[s["sample_id"]]))
        exp_first[(mdl, a)] = c
r = ev["results"]["llama3"]
check(r["cells"]["alpha0"]["first_correct"] == exp_first[("llama3", 0)] and r["cells"]["workpoint"]["first_correct"] == exp_first[("llama3", -6)], "first_acc matches hand count (denominator 300)")
check(abs(r["cells"]["alpha0"]["first_acc"] - exp_first[("llama3", 0)] / 300) < 1e-12, "first_acc denominator is 300")
c0 = r["cells"]["alpha0"]
check(c0["n_valid"] + sum(1 for i in range(300) if "####" not in gen_text(i, 0, "llama3") or E.first_legal_answer(gen_text(i, 0, "llama3"))[1] is None) == 300, "valid + no-answer = 300")
check(c0["valid_then_loop"] + c0["valid_no_loop"] == c0["n_valid"] and c0["no_answer_loop"] <= 300 - c0["n_valid"], "loop split by validity is consistent")
check(0 <= c0["conditional_accuracy"] <= 1 and abs(c0["conditional_accuracy"] * c0["n_valid"] - c0["first_correct"]) < 1e-9, "conditional accuracy = correct / valid")
w = r["workpoint_vs_alpha0"]
check(w["wrong_to_right"] - w["right_to_wrong"] == w["workpoint_correct"] - w["alpha0_correct"], "W->C minus C->W equals the net correct change")
check(len(w["bootstrap_ci95_pp"]) == 2 and w["bootstrap"]["B"] == 10000 and "holm_p_adj" in w and w["holm_p_adj"] >= w["mcnemar_exact_p"] - 1e-15, "bootstrap B=10000 seed 0, Holm adj >= raw")
check(raises(lambda: run_eval(good, out)), "existing --out is never overwritten")
a = E.classify_first_line
check([a("5"), a("$5.3 million"), a("-12.5%"), a("3.2 percent"), a("Answer: 94"), a("The answer is 12"), a("Question: what is it?"),
       a("Step 1: find x"), a("To find the change, we need to"), a("x" * 90), a(""), a("#### 5")] ==
      ["bare_number", "currency", "percent", "unit_number", "labeled_answer", "labeled_answer", "question_restatement", "reasoning_opener", "reasoning_opener", "long_line", "empty", "answer_marker"],
      "first-line classifier (bare / currency / percent / unit / labelled / restatement / opener)")
check("EXPLORATORY_first_line_audit" in r and r["EXPLORATORY_first_line_audit"]["alpha0"]["first_line_categories"], "first-line audit present without a detector")

# ---- D. evaluator refusals (each must abort with no output file)
def refused(name, mods=None, skip=(), mutate_paths=None):
    ps = write_cells(name, mods, skip)
    if mutate_paths: ps = mutate_paths(ps)
    o = os.path.join(TMP, name + "_out.json")
    ok = raises(lambda: run_eval(ps, o))
    return ok and not os.path.exists(o)
def setm(k, v): return lambda d: d["meta"].__setitem__(k, v)
check(refused("d_src", {("llama3", -6): setm("questions_file_sha256", "x" * 64)}), "wrong source file hash refused")
check(refused("d_content", {("qwen2.5", 8): setm("prompt_content_sha256_16", "0" * 16)}), "different prompt content digest refused")
def swap(d): d["data"][10]["question"], d["data"][11]["question"] = d["data"][11]["question"], d["data"][10]["question"]
check(refused("d_swap", {("llama3", 0): swap}), "mismatched question/id row refused")
def swapid(d): d["data"][10]["id"], d["data"][11]["id"] = d["data"][11]["id"], d["data"][10]["id"]
check(refused("d_swapid", {("qwen2.5", 0): swapid}), "swapped ids refused")
check(refused("d_meta", {("llama3", 0): lambda d: d["meta"].pop("injection_token")}), "missing injection_token metadata refused")
check(refused("d_chat", {("llama3", 0): lambda d: d["meta"].pop("chat_template_hash")}), "missing chat_template_hash refused")
check(refused("d_fire", {("llama3", -6): setm("steering_fires", 2699)}), "wrong steering_fires refused")
check(refused("d_fire0", {("llama3", 0): setm("steering_fires", 2700)}), "alpha=0 with non-zero fires refused")
check(refused("d_fireq", {("qwen2.5", 8): lambda d: (d["meta"].__setitem__("steering_fires", 2700), d["meta"].__setitem__("steering_fires_expected", 2700))}), "Qwen with Llama fire count refused")
check(refused("d_band", {("llama3", -6): setm("layers", [10, 19])}), "wrong layer band refused")
check(refused("d_maskcell", {("llama3", -6): setm("mask_sha256", "z" * 64)}), "different mask between a model's alpha cells refused")
check(refused("d_prov", {("qwen2.5", 8): lambda d: d["meta"]["provenance"].__setitem__("host", "other")}), "different host between cells refused")
check(refused("d_gpu", {("qwen2.5", 0): lambda d: d["meta"]["provenance"].__setitem__("cuda_device_names", []),
                         ("qwen2.5", 8): lambda d: d["meta"]["provenance"].__setitem__("cuda_device_names", [])}), "empty GPU names refused")
check(refused("d_proto", {("llama3", 0): setm("protocol", "finqa-v0")}), "wrong protocol refused")
check(refused("d_cot", {("llama3", 0): setm("cot", False)}), "non-CoT cell refused")
check(refused("d_missing", skip=[("qwen2.5", 8)]), "missing cell refused")
check(refused("d_dup", mutate_paths=lambda ps: ps + [ps[0]]), "duplicate cell refused")
check(refused("d_extra", {("llama3", -6): setm("alpha", -4)}), "non-registered alpha refused")
def baddiag(d): d["data"][0]["no_answer"] = not d["data"][0]["no_answer"]
check(refused("d_diag", {("llama3", 0): baddiag}), "stored diagnostics disagreeing with the text refused")
check(refused("d_rows", {("llama3", 0): lambda d: d["data"].pop()}), "299-row cell refused")
def stopr(d): d["data"][0]["stop_reason"] = "weird"
check(refused("d_stop", {("llama3", 0): stopr}), "unknown stop_reason refused")

# ---- E. resume / existing-cell verification
smp = samples
def vcheck(mod=None, mdl="llama3", a=-6):
    d = build_cell(mdl, a, mod)
    expect = {k: d["meta"].get(k) for k in ("mask_sha256", "user_prompt_sha256", "wrapped_prompt_sha256", "chat_template_hash", "model_dir",
              "questions_file_sha256", "sample_order_sha256", "prompt_content_sha256_16", "mode")}
    expect["final_token_id"] = 271
    p = dump(d, "v.json")
    return G.verify_existing_cell(p, mdl, a, smp, expect), p, expect
probs, p, expect = vcheck()
check(probs == [], "a complete, matching existing cell passes verification")
def bad_expect(key):
    e = dict(expect); e[key] = "different"
    return G.verify_existing_cell(p, "llama3", -6, smp, e)
for k in ("mask_sha256", "wrapped_prompt_sha256", "chat_template_hash", "questions_file_sha256", "model_dir", "mode"):
    check(bad_expect(k) != [], f"existing cell with different {k} refused")
check(vcheck(lambda d: d["data"].pop())[0] != [], "truncated existing cell refused")
check(vcheck(lambda d: d["meta"].__setitem__("steering_fires", 1))[0] != [], "existing cell with wrong fires refused")
check(vcheck(lambda d: d["data"][3].__setitem__("question", "other"))[0] != [], "existing cell with mismatched question refused")
check(vcheck(lambda d: d["data"][3].pop("generated"))[0] != [], "existing cell with a missing field refused")
badp = os.path.join(TMP, "corrupt.json"); open(badp, "w").write('{"meta": {"proto')
check(G.verify_existing_cell(badp, "llama3", -6, smp, expect) != [], "corrupted (half-written) JSON refused")
check(not os.path.exists(os.path.join(TMP, "x.tmp")) and L.atomic_write_json({"a": 1}, os.path.join(TMP, "x.json")) is None and not os.path.exists(os.path.join(TMP, "x.json.tmp")), "atomic write leaves no .tmp")
check(G.alphas_for("llama3") == [0, -6] and G.alphas_for("qwen2.5") == [0, 8], "only the four registered cells exist")
check((G.expected_fires("llama3", -6, 300), G.expected_fires("qwen2.5", 8, 300), G.expected_fires("llama3", 0, 300)) == (2700, 1800, 0), "expected fires 2700 / 1800 / 0")

shutil.rmtree(TMP, ignore_errors=True)
print("\n%d failures" % len(fails)); sys.exit(1 if fails else 0)
