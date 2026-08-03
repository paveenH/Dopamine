"""Offline protocol tests for pv6 (no GPU, no real model).

A FakeVC records every call, so the frozen steering semantics are checkable as
facts rather than as comments: the rationale pass must receive diff_matrices
None, the action pass must receive the real matrix, and the action prompt's
last token must be the anchor tail.
"""
import sys

import numpy as np

import bandit_reference as br
import bandit_pv6_episode as pv6

FAILS = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


class FakeTokenizer:
    """Whitespace-ish tokenizer: one id per word, plus a BOS."""
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 10)

    def encode(self, text, add_special_tokens=True):
        ids = [self._id(w) for w in text.split()]
        return ([128000] + ids) if add_special_tokens else ids

    def __call__(self, text, return_tensors=None, **kw):
        if isinstance(text, str):
            text = [text]
        import torch
        ids = [self.encode(t) for t in text]
        return {"input_ids": torch.tensor(ids)}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "<s>") for i in ids)

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True):
        return ("<|start_header_id|>user<|end_header_id|> "
                + msgs[0]["content"] + " <|eot_id|> <|assistant|> ")


class FakeVC:
    def __init__(self, k=4, prefer_index=0):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.k = k
        self.prefer_index = prefer_index

    def regenerate(self, inputs, diff_matrices=None, max_new_tokens=None,
                   **kw):
        self.calls.append({"stage": "rationale", "diff": diff_matrices,
                           "prompt": inputs[0],
                           "max_new_tokens": max_new_tokens})
        return ["I should try things. Choice: Button A"]

    def regenerate_logits_teacher_forcing(self, prompts, answer_token_ids,
                                          diff_matrices=None):
        self.calls.append({"stage": "action", "diff": diff_matrices,
                           "prompt": prompts[0],
                           "n_candidates": len(prompts)})
        out = []
        V = 200
        for i, ids in enumerate(answer_token_ids):
            arr = np.full((len(ids), V), -10.0, dtype=np.float32)
            for k, tid in enumerate(ids):
                arr[k, tid % V] = 5.0 if i == self.prefer_index else 0.0
            out.append(arr)
        return out


easy = br.get_environment("easy")

print("[1] frozen steering semantics")
vc = FakeVC()
fake_diff = [np.zeros((4096,), dtype=np.float32)]
rec = pv6.run_reference_episode(vc, fake_diff, seed=0, env=easy, attest=True)

rat = [c for c in vc.calls if c["stage"] == "rationale"]
act = [c for c in vc.calls if c["stage"] == "action"]
check(len(rat) == easy.horizon, "one rationale pass per round")
check(len(act) == easy.horizon, "one action pass per round")
check(all(c["diff"] is None for c in rat),
      "rationale pass receives NO alpha (diff_matrices=None)")
check(all(c["diff"] is fake_diff for c in act),
      "action pass receives the alpha matrix")
check(all(c["max_new_tokens"] == pv6.RATIONALE_MAX_TOKENS for c in rat),
      "rationale honours the frozen 64-token cap")
check(all(c["n_candidates"] == easy.k for c in act),
      "action scores exactly K candidates")

print("\n[2] action prompt ends at the anchor")
check(all(c["prompt"].endswith(br.ACTION_ANCHOR) for c in act),
      "every bare action prompt ends at ACTION_ANCHOR")
check(not any(br.ACTION_ANCHOR in c["prompt"] for c in rat),
      "rationale prompt never contains the action anchor")

print("\n[3] rationale sanitization is applied in-loop")
# FakeVC always emits a premature "Choice: Button A" line
check(all("Choice:" not in c["prompt"][:-len(br.ACTION_ANCHOR)] for c in act),
      "no leftover Choice: marker before the single action anchor")
check(all(c["prompt"].count(br.ACTION_ANCHOR) == 1 for c in act),
      "action prompt contains EXACTLY one anchor")
check(rec["rationales_raw"][0] != rec["rationales_clean"][0],
      "raw and clean rationale are both kept and differ")

print("\n[4] no invalid choices, no fallback")
check(rec["invalid_rate"] == 0.0, "invalid_rate is structurally 0")
check(all(c in br.ARM_LABELS[:easy.k] for c in rec["choices"]),
      "every choice is a legal arm label")
check(len(rec["choices"]) == easy.horizon, "episode ran the full horizon")
check(len(rec["candidate_scores"]) == easy.horizon,
      "a candidate score trace is stored per round")
check(set(rec["candidate_scores"][0]) == set(br.ARM_LABELS[:easy.k]),
      "score trace keys are the arm labels")

print("\n[5] scoring picks the argmax candidate")
vc2 = FakeVC(prefer_index=2)
rec2 = pv6.run_reference_episode(vc2, None, seed=0, env=easy)
want = br.ARM_LABELS[2]
check(all(c == want for c in rec2["choices"]),
      "argmax follows the highest-scoring candidate")

print("\n[6] rewards come from the shared per-arm tape")
tape = br.RewardTape(0, easy)
pulls = {}
ok = True
for c, r in zip(rec2["choices"], rec2["feedbacks"]):
    n = pulls.get(c, 0)
    if tape.peek(c, n) != r:
        ok = False
    pulls[c] = n + 1
check(ok, "feedback matches the tape's n-th draw for that arm")

print("\n[7] alpha=0 passes None, not a zero matrix")
check(all(c["diff"] is None for c in vc2.calls if c["stage"] == "action"),
      "alpha=0 action pass receives None")

print("\n[8] token attestation")
at = rec["attestation"]
check("candidate_tokenization" in at, "candidate tokenization audit recorded")
check(at["candidate_tokenization"]["anchor"] == br.ACTION_ANCHOR,
      "audit records the frozen anchor")
r0 = at["round_0"]
check(r0["tokens"]["prompt_endswith_anchor"],
      "attested action prompt ends with the anchor")
check(r0["tokens"]["injection_is_anchor_tail"],
      "injection token is the anchor tail, not a control token")
check("rationale_raw" in r0 and "rationale_clean" in r0,
      "attestation keeps both rationale forms")

print("\n[9] chat mode keeps the anchor last")
vc3 = FakeVC()
rec3 = pv6.run_reference_episode(vc3, fake_diff, seed=0, env=easy,
                                 use_chat=True, attest=True)
act3 = [c for c in vc3.calls if c["stage"] == "action"]
check(all(c["prompt"].endswith(br.ACTION_ANCHOR) for c in act3),
      "chat action prompt still ends at the anchor")
check(all("<|assistant|>" in c["prompt"] for c in act3),
      "chat template was actually applied")
t3 = rec3["attestation"]["round_0"]["tokens"]
check(t3["injection_is_anchor_tail"],
      "chat: injection token is the anchor tail, not a chat control token")
check(t3["use_chat"] is True, "chat: attestation records the interface")
check(all(c["prompt"].count(br.ACTION_ANCHOR) == 1 for c in act3),
      "chat: exactly one anchor in the action prompt")

print("\n[10] metrics apply unchanged to a model record")
check(0.0 <= rec2["opt_frac"] <= 1.0, "opt_frac computed for a model record")
check(isinstance(rec2["suffix_failure"], bool),
      "suffix_failure computed for a model record")
check(rec2["policy"] == "model", "record is tagged as a model policy")
check(rec2["protocol"] == "pv6", "record carries the pv6 protocol tag")

print("\n" + "=" * 60)
if FAILS:
    print(f"FAILED ({len(FAILS)}):")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
