#!/usr/bin/env python3
"""Stdlib-only tests for chat_cot_sat_lib (fake tokenizer). The runner itself needs torch/numpy/GPU and is NOT covered here."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_cot_sat_lib as L
from template import select_templates_gsm8k

ok = 0
def check(c, m):
    global ok
    if not c:
        print("FAIL:", m); sys.exit(1)
    ok += 1

no, co = L.template_pair(select_templates_gsm8k)
check(co.count(L.COT_LINE) == 1 and L.remove_cot_line(co) == no, "template pair")
# question that itself contains the CoT phrase: rfind must remove the TEMPLATE line, not the question's
q = "Say Let's think step by step.\nthen add 2 and 3."
p_no, p_co = L.render_pair(no, co, q)
check(q in p_co and L.remove_cot_line(p_co) == p_no, "question containing the phrase is untouched")
# mutations
try: L.template_pair(lambda **k: {"neutral": "x"} if not k["cot"] else {"neutral": "x\n" + L.COT_LINE + "\nQ Provide your final numeric answer"}); check(False, "mutant 1")
except ValueError: check(True, "")
bad_co = co.replace("Solve the following", "Please solve the following")
try: L.template_pair(lambda **k: {"neutral": no if not k["cot"] else bad_co}); check(False, "wording drift accepted")
except ValueError: check(True, "")

class Tok:
    bos_token_id = 1
    def __call__(self, t, add_special_tokens=True): return {"input_ids": [1] + [5] * 3 + [t.__len__() % 2 and 9 or 9]}
    def decode(self, ids): return "<h>"
check(L.check_prompt_tokens(Tok(), ["a", "bb"])["final_token_id"] == 9, "tail ok")
class TokDouble(Tok):
    def __call__(self, t, add_special_tokens=True): return {"input_ids": [1, 1, 5, 9]}
try: L.check_prompt_tokens(TokDouble(), ["a"]); check(False, "double BOS accepted")
except ValueError: check(True, "")
class TokMixed(Tok):
    def __call__(self, t, add_special_tokens=True): return {"input_ids": [1, 5, 9 if t == "a" else 8]}
try: L.check_prompt_tokens(TokMixed(), ["a", "b"]); check(False, "mixed tail accepted")
except ValueError: check(True, "")

check(L.alphas_for("llama3") == [0, -6] and L.alphas_for("qwen2.5") == [0, 8], "alpha family")
check(L.expected_fires("llama3", -6, 300) == 2700 and L.expected_fires("qwen2.5", 8, 300) == 1800
      and L.expected_fires("qwen2.5", 0, 300) == 0, "fires")
check(L.alpha_dir(-6) == "mdf_-6" and L.alpha_dir(0) == "mdf_0", "dir naming")

def cell(model="qwen2.5", task="gsm8k", alpha=8, n=3):
    tk = L.TASKS[task]["text_key"]
    meta = {"protocol": L.PROTOCOL, "model": model, "task": task, "alpha": alpha, "cot": True, "chat_template_applied": True,
            "layers": list(L.MODELS[model]["band"]), "mask_sha256": "m", "prompt_sha256": "p", "max_new_tokens": 768,
            "temperature": 0.0, "batch_size": 24, "prefill_tail_len": 1, "n": n,
            "steering_fires": L.expected_fires(model, alpha, n), "steering_fires_expected": L.expected_fires(model, alpha, n)}
    rows = [{tk: "x", "stop_reason": "natural_eos", "generated_token_count": 5} for _ in range(n)]
    return {"meta": meta, "data": rows}
def verify(d, **kw):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(d, f); p = f.name
    a = dict(model="qwen2.5", task="gsm8k", alpha=8, n=3, mask_sha256="m", prompt_sha256="p"); a.update(kw)
    r = L.verify_existing_cell(p, **a); os.unlink(p); return r
check(verify(cell()) == [], "good cell passes")
d = cell(); d["meta"]["steering_fires"] = 0; check(verify(d), "wrong fires fails")
d = cell(); d["data"].pop(); check(verify(d), "short rows fails")
d = cell(); d["meta"]["prompt_sha256"] = "q"; check(verify(d), "different prompts fails")
d = cell(); d["meta"]["mask_sha256"] = "z"; check(verify(d), "different mask fails")
d = cell(); d["data"][1]["stop_reason"] = None; check(verify(d), "bad stop reason fails")
d = cell(alpha=0); check(verify(d, alpha=0) == [], "alpha0 fires 0 passes")
d = cell(model="llama3", alpha=-6, n=300); check(verify(d, model="llama3", alpha=-6, n=300) == [], "llama 2700 passes")
print(f"all {ok} checks passed")
