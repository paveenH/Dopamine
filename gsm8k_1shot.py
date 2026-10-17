"""Prompt assembly for `gsm8k-base-1shot-v1` (shared by the generator and the check).

One fixed worked demonstration (see prepare_gsm8k_1shot_demo.py) followed by the
target question in the SAME neutral/plain template. Default-off in the runner: nothing
here is imported unless --fewshot_file is passed.
"""
import hashlib
import json

PROTOCOL = "gsm8k-base-1shot-v1"


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_demo(demo_path, test_path):
    """Load the frozen demo; fail closed if it, or the test file, drifted."""
    demo = json.load(open(demo_path, encoding="utf-8"))
    if demo.get("protocol") != PROTOCOL:
        raise SystemExit(f"[FATAL] demo protocol {demo.get('protocol')!r} != {PROTOCOL!r}")
    want = _sha(json.dumps({"question": demo["question"], "answer": demo["answer"]},
                           ensure_ascii=False, sort_keys=True))
    if want != demo["content_sha256"]:
        raise SystemExit("[FATAL] demo content hash does not match its content")
    raw = open(test_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != demo["test_file_sha256"]:
        raise SystemExit("[FATAL] test file differs from the one the demo was selected against")
    norm = lambda t: " ".join(t.split())
    if norm(demo["question"]) in {norm(s["question"]) for s in json.loads(raw)}:
        raise SystemExit("[FATAL] demo question overlaps the test set")
    return demo


def build_prompts(neutral_template, demo, questions):
    """neutral_template: the frozen plain No-CoT neutral template with {context}."""
    prefix = neutral_template.format(context=demo["question"]) + demo["answer"] + "\n\n"
    prompts = [prefix + neutral_template.format(context=q) for q in questions]
    for p in prompts:
        if not p.endswith("Answer: ") or p.endswith("Answer:  "):
            raise SystemExit("[FATAL] prompt does not end in exactly one ASCII space after 'Answer:'")
    return prefix, prompts


def prompts_sha256(prompts):
    return _sha("\n<<SEP>>\n".join(prompts))


def template_sha256(neutral_template, demo):
    return _sha(neutral_template + "\n<<DEMO>>\n" + demo["content_sha256"])
