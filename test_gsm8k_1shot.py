#!/usr/bin/env python3
"""Local, stdlib-only checks for gsm8k-base-1shot-v1 (no GPU, no network, no numpy).
Real tokenizer / injection / fires checks run on the server via `run_gsm8k_base_1shot.sh check`.
"""
import hashlib
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm8k_1shot as G
import prepare_gsm8k_1shot_demo as P

NEUTRAL = ("Solve the following math problem.\nQuestion: {context}\n"
           "Provide your final numeric answer after '####'.\nAnswer: ")
n_ok = 0


def ok(c, msg):
    global n_ok
    if not c:
        sys.exit(f"FAIL: {msg}")
    n_ok += 1


def expect_exit(fn, msg):
    try:
        fn()
    except SystemExit:
        return ok(True, msg)
    ok(False, msg + " (no exit)")


# --- demo selection: first non-overlapping row, source order, whitespace-normalised
rows = [("Q  a", "x #### 1"), ("Q b", "y #### 2"), ("Q c", "z #### 3")]
idx, q, a = P.pick_first_non_overlapping(rows, ["Q a", "Q b"])
ok(idx == 2 and q == "Q c", "selection skips overlapping rows (normalised) and keeps source order")
expect_exit(lambda: P.pick_first_non_overlapping(rows, ["Q a", "Q b", "Q c"]), "all-overlap dies")

# --- demo file + loader guards
tmp = tempfile.mkdtemp()
test = [{"question": "T1 {x}?", "answer": "18"}, {"question": "T2", "answer": "5"}]
tp = os.path.join(tmp, "test.json")
raw = json.dumps(test).encode()
open(tp, "wb").write(raw)
dq, da = "Demo q?", "Step one.\n<<2+2=4>>4\n#### 4"
demo = {"protocol": G.PROTOCOL, "question": dq, "answer": da,
        "content_sha256": P.content_hash(dq, da), "test_file_sha256": hashlib.sha256(raw).hexdigest()}
dp = os.path.join(tmp, "demo.json")
json.dump(demo, open(dp, "w"))
d = G.load_demo(dp, tp)
ok(d["question"] == dq, "loader accepts a consistent demo")
for field, val, msg in (("answer", "tampered #### 4", "content tamper"),
                        ("test_file_sha256", "0" * 64, "test file drift"),
                        ("protocol", "other", "wrong protocol")):
    bad = dict(demo); bad[field] = val
    json.dump(bad, open(dp, "w"))
    expect_exit(lambda: G.load_demo(dp, tp), f"loader dies on {msg}")
ov = dict(demo, question="T2", content_sha256=P.content_hash("T2", da))
json.dump(ov, open(dp, "w"))
expect_exit(lambda: G.load_demo(dp, tp), "loader dies on demo/test overlap")

# --- prompt assembly is exactly the specified text; braces in text are safe
prefix, prompts = G.build_prompts(NEUTRAL, demo, [t["question"] for t in test])
want0 = ("Solve the following math problem.\nQuestion: Demo q?\nProvide your final numeric answer after '####'.\n"
         "Answer: Step one.\n<<2+2=4>>4\n#### 4\n\n"
         "Solve the following math problem.\nQuestion: T1 {x}?\nProvide your final numeric answer after '####'.\nAnswer: ")
ok(prompts[0] == want0, "prompt text matches the specification byte for byte")
ok(all(p.endswith("Answer: ") and not p.endswith("  ") for p in prompts), "ends in one ASCII space")
ok(all(p.startswith(prefix) for p in prompts), "same demo prefix for every question")
ok(not any(t["answer"] in p for t, p in zip(test, prompts)), "test gold never in prompt")
ok(prompts[0].count("####") == 3, "demo marker kept once, not duplicated")
ok(G.prompts_sha256(prompts) == G.prompts_sha256(G.build_prompts(NEUTRAL, demo, [t["question"] for t in test])[1]),
   "prompt hash deterministic (same prompts for every alpha)")
expect_exit(lambda: G.build_prompts(NEUTRAL.rstrip(), demo, ["q"]), "missing trailing space is rejected")

# --- generator wiring with stubbed heavy modules (FakeVC returns only NEW text)
for m in ("numpy", "torch", "tqdm", "llms", "template", "utils"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["numpy"].ndarray = object
sys.modules["tqdm"].tqdm =lambda x, **k: x
sys.modules["llms"].VicundaModel = object
sys.modules["template"].select_templates_gsm8k = lambda **k: {"neutral": NEUTRAL}
u = sys.modules["utils"]
u.extract_gsm8k_answer = lambda t: t.split("####")[-1].strip() if "####" in t else ""
u.is_correct_gsm8k = lambda p, g: str(p) == str(g)
import importlib.util
spec = importlib.util.spec_from_file_location("gen", os.path.join(os.path.dirname(__file__), "get_answer_regenerate_gsm8k.py"))
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)


class Tok:
    bos_token_id = 1
    def __call__(self, p): return {"input_ids": [1] + [ord(c) % 50 + 3 for c in p]}
    def decode(self, ids): return " "


class FakeVC:
    tokenizer = Tok()
    def __init__(self): self.seen = []
    def regenerate(self, prompts, **k):
        self.seen += prompts
        return [{"text": "so #### 18", "generated_token_count": 5, "stop_reason": "natural_eos"} for _ in prompts]


json.dump(demo, open(dp, "w"))
gen.args = types.SimpleNamespace(fewshot_file="demo.json", max_new_tokens=8, temperature=0.0, top_p=0.9,
                                 batch_size=2, save_gen_meta=True)
gen.FEWSHOT["demo"] = G.load_demo(dp, tp)
vc = FakeVC()
samples = [dict(question=t["question"], answer=t["answer"]) for t in test]
out, acc = gen.run_gsm8k_regenerate(vc, samples, None, {"neutral": NEUTRAL}, ["neutral"], 2)
ok(vc.seen == prompts, "generator sends exactly the 1-shot prompts")
ok(out[0]["generated_neutral"] == "so #### 18" and "Demo q" not in out[0]["generated_neutral"],
   "stored output is the new generation only (demo not echoed in)")
ok(gen.FEWSHOT["prompts_sha256"] == G.prompts_sha256(prompts) and gen.FEWSHOT["tok_checked"], "hash + BOS check recorded")
ok(out[0]["correct_neutral"] and not out[1]["correct_neutral"], "scoring sees only generated text")
gen.args.fewshot_file = None
vc2 = FakeVC()
gen.run_gsm8k_regenerate(vc2, [dict(question="T2", answer="5")], None, {"neutral": NEUTRAL}, ["neutral"], 2)
ok(vc2.seen == [NEUTRAL.format(context="T2")], "default (no --fewshot_file) prompts unchanged")

print(f"all {n_ok} checks passed")
