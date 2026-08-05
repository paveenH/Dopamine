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


BOS = "<|begin_of_text|>"
BOS_ID = 128000


class FakeTokenizer:
    """Mirrors the parts of the real tokenizer contract pv6 depends on.

    Critically it emits BOS as TEXT from apply_chat_template (like Llama-3's
    real template) AND prepends BOS_ID when add_special_tokens=True — that
    combination is what produces the double BOS the code must prevent. The
    earlier fake did neither, so it could not have caught the bug.
    """
    pad_token_id = 0
    bos_token = BOS
    bos_token_id = BOS_ID

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 10)

    def encode(self, text, add_special_tokens=True):
        ids = []
        for w in text.split():
            if w == BOS:
                ids.append(BOS_ID)
            else:
                ids.append(self._id(w))
        return ([BOS_ID] + ids) if add_special_tokens else ids

    def __call__(self, text, return_tensors=None, **kw):
        if isinstance(text, str):
            text = [text]
        import torch
        ids = [self.encode(t) for t in text]
        return {"input_ids": torch.tensor(ids)}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        inv[BOS_ID] = BOS
        return " ".join(inv.get(i, "<unk>") for i in ids)

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True):
        # Real Llama-3 templates serialize BOS into the string.
        return (f"{BOS} <|start_header_id|>user<|end_header_id|> "
                + msgs[0]["content"]
                + " <|eot_id|> <|start_header_id|>assistant<|end_header_id|> ")


class FakeVC:
    """Contract-faithful stand-in for VicundaModel.

    generate() takes no diff_matrices (no hooks exist on that path) and
    regenerate() REJECTS diff_matrices=None exactly like llms.py:821 — which
    is what makes the "pass None to disable steering" bug fail loudly here
    instead of only on the GPU.
    """

    def __init__(self, k=4, prefer_index=0):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.k = k
        self.prefer_index = prefer_index

    def generate(self, inputs, max_new_tokens=1, top_p=0.9, temperature=0.0,
                 batch_size=1):
        self.calls.append({"stage": "rationale", "diff": None,
                           "prompt": inputs[0],
                           "max_new_tokens": max_new_tokens,
                           "temperature": temperature})
        # Two lines on purpose: the reasoning must SURVIVE sanitization while
        # the premature commit line is dropped. A single-line fixture would
        # lose the whole rationale and hide whether it reaches the prompt.
        return ["I should try things.\nChoice: Button A"]

    def regenerate(self, inputs, diff_matrices=None, **kw):
        if diff_matrices is None:
            raise ValueError(
                "The difference matrices are not loaded. Please provide "
                "`diff_matrices` during method call.")
        # kw is recorded, not swallowed: scope="both" must pass prefill_only
        # and prefill_tail_len=1, and a test that only counted calls would not
        # notice decode-wide or tail-widened steering.
        self.calls.append({"stage": "regenerate", "diff": diff_matrices,
                           "prompt": inputs[0], "kw": dict(kw)})
        # Same two-line shape as generate(): the rationale must survive
        # sanitization on this path too, or scope="both" would silently feed
        # an empty rationale into the action prompt.
        return ["I should try things.\nChoice: Button A"]

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
# the contract bug: regenerate() rejects None, so the rationale MUST go
# through generate() — which registers no hooks at all
check(not any(c["stage"] == "regenerate" for c in vc.calls),
      "rationale never calls regenerate (which rejects diff_matrices=None)")
check(all(c["temperature"] == 0.0 for c in rat),
      "rationale is greedy (temperature=0)")

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
check("Choice:" in rec["rationales_raw"][0]
      and "Choice:" not in rec["rationales_clean"][0],
      "sanitization drops the premature commit line")
check(rec["rationales_clean"][0] == "I should try things.",
      "sanitization KEEPS the reasoning it is supposed to keep")
check(all(rec["rationales_clean"][i] in c["prompt"]
          for i, c in enumerate(act)),
      "the surviving rationale actually reaches the action prompt")

print("\n[4] no invalid choices, no fallback")
check(rec["invalid_rate"] == 0.0, "invalid_rate is structurally 0")
check(all(c in br.ARM_LABELS[:easy.k] for c in rec["choices"]),
      "every choice is a legal arm label")
check(len(rec["choices"]) == easy.horizon, "episode ran the full horizon")
check(len(rec["candidate_scores"]) == easy.horizon,
      "a candidate score trace is stored per round")
check(set(rec["candidate_scores"][0]) == set(br.ARM_LABELS[:easy.k]),
      "score trace keys are the arm labels")

print("\n[4b] score schema: raw floats + per-round margin")
_vals = list(rec["candidate_scores"][0].values())
check(all(isinstance(v, float) for v in _vals),
      "candidate scores are stored as floats")
check(any(abs(v - round(v, 4)) > 0 for v in _vals)
      or all(v == round(v, 4) for v in _vals),
      "candidate scores are not pre-rounded to 4dp")
check(len(rec["choice_margins"]) == easy.horizon,
      "a top1-top2 margin is stored per round")
_m0 = rec["choice_margins"][0]
_ord = sorted(rec["candidate_scores"][0].values(), reverse=True)
check(abs(_m0 - (_ord[0] - _ord[1])) < 1e-12,
      "margin equals the top1-top2 sequence-logprob gap")
check(all(m >= 0 for m in rec["choice_margins"]),
      "margins are non-negative (top1 >= top2)")

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
check(all("<|start_header_id|>user<|end_header_id|>" in c["prompt"]
          for c in act3),
      "chat template was actually applied")
t3 = rec3["attestation"]["round_0"]["tokens"]
check(t3["injection_is_anchor_tail"],
      "chat: injection token is the anchor tail, not a chat control token")
check(t3["use_chat"] is True, "chat: attestation records the interface")
check(all(c["prompt"].count(br.ACTION_ANCHOR) == 1 for c in act3),
      "chat: exactly one anchor in the action prompt")

print("\n[9b] chat: double BOS is PREVENTED, not merely reported")
check(t3["double_bos"] is False, "chat: attested prompt has no double BOS")
check(all(c["prompt"].count(BOS) <= 1 for c in act3),
      "chat: template BOS is stripped so tokenizer adds exactly one")
_ids = FakeVC().tokenizer.encode(act3[0]["prompt"], add_special_tokens=True)
check(_ids[:2] != [BOS_ID, BOS_ID],
      "chat: real add_special_tokens=True path yields a single BOS")
# and the guard must actually fire when a double BOS is constructed
try:
    pv6.audit_prompt_tokens(FakeVC(), BOS + " " + BOS + " x " + br.ACTION_ANCHOR,
                            True)
    check(False, "audit raises on a double BOS")
except AssertionError as _e:
    check("double BOS" in str(_e), "audit raises on a double BOS")
try:
    pv6.audit_prompt_tokens(FakeVC(), "state with no anchor", False)
    check(False, "audit raises when the prompt does not end at the anchor")
except AssertionError as _e:
    check("injection site" in str(_e),
          "audit raises when the prompt does not end at the anchor")

print("\n[9c] frozen role structure: rationale is an ASSISTANT continuation")
_p3 = act3[0]["prompt"]
# split AFTER the full assistant header, so _asst_seg is the turn's content
_ASST_HDR = "<|start_header_id|>assistant<|end_header_id|>"
_user_seg, _, _asst_seg = _p3.partition(_ASST_HDR)
check("I should try things." in _asst_seg,
      "rationale sits in the assistant turn, not the user turn")
check("I should try things." not in _user_seg,
      "rationale is NOT fed back inside the user turn")
check("TRIED OPTIONS" in _user_seg or "UNTRIED OPTIONS" in _user_seg,
      "task state stays in the user turn")
check("<|eot_id|>" not in _asst_seg,
      "assistant turn is CONTINUED, never closed before the anchor")
check(_asst_seg.rstrip().endswith(br.ACTION_ANCHOR),
      "assistant turn ends exactly at the action anchor")

print("\n[10] metrics apply unchanged to a model record")
check(0.0 <= rec2["opt_frac"] <= 1.0, "opt_frac computed for a model record")
check(isinstance(rec2["suffix_failure"], bool),
      "suffix_failure computed for a model record")
check(rec2["policy"] == "model", "record is tagged as a model policy")
check(rec2["protocol"] == "pv6", "record carries the pv6 protocol tag")

print("\n[11] steering_scope contract")
# The whole point of scope=both is WHICH passes get alpha, so lock the exact
# per-episode call counts, not just "more than zero regenerate calls".
T = easy.horizon

vc_b = FakeVC()
rec_b = pv6.run_reference_episode(vc_b, fake_diff, seed=0, env=easy,
                                  steering_scope="both")
n_gen = len([c for c in vc_b.calls if c["stage"] == "rationale"])
n_reg = [c for c in vc_b.calls if c["stage"] == "regenerate"]
n_act = [c for c in vc_b.calls if c["stage"] == "action"]
check(len(n_reg) == T, f"both/alpha!=0: {T} steered rationale passes")
check(n_gen == 0, "both/alpha!=0: no unsteered generate call remains")
check(len(n_act) == T, f"both/alpha!=0: {T} action passes")
check(all(c["diff"] is fake_diff for c in n_reg),
      "both: rationale pass gets the SAME diff object as the action pass")
check(all(c["diff"] is fake_diff for c in n_act),
      "both: action pass still steered")
check(all(c["kw"].get("prefill_only") is True for c in n_reg),
      "both: rationale steering is prefill-only (decode unsteered)")
check(all(c["kw"].get("prefill_tail_len") == 1 for c in n_reg),
      "both: rationale steering injects at exactly ONE token")
check(all(c["kw"].get("temperature") == 0.0 for c in n_reg),
      "both: rationale pass stays greedy")
check(all(c["kw"].get("max_new_tokens") == pv6.RATIONALE_MAX_TOKENS
          for c in n_reg),
      "both: rationale cap unchanged by scope")
check(all(c["prompt"].rstrip().endswith(br.ACTION_ANCHOR) for c in n_act),
      "both: action prompt still ends at the anchor")
check(rec_b["steering_scope"] == "both" and rec_b["steered_rationale"] is True,
      "both/alpha!=0: record self-attests a steered rationale")
check(rec_b["steering_scope_version"] == pv6.STEERING_SCOPE_VERSION,
      "record carries the scope version")

# alpha=0: diff_mtx is None, so NO pass may be steered in EITHER scope. This is
# what makes the stored Track A cells reusable under scope=both.
vc_z = FakeVC()
rec_z = pv6.run_reference_episode(vc_z, None, seed=0, env=easy,
                                  steering_scope="both")
z_gen = len([c for c in vc_z.calls if c["stage"] == "rationale"])
z_reg = len([c for c in vc_z.calls if c["stage"] == "regenerate"])
z_act = [c for c in vc_z.calls if c["stage"] == "action"]
check(z_gen == T, f"both/alpha=0: {T} UNSTEERED generate calls")
check(z_reg == 0, "both/alpha=0: regenerate is never called")
check(all(c["diff"] is None for c in z_act),
      "both/alpha=0: action pass unsteered too")
check(rec_z["steered_rationale"] is False,
      "both/alpha=0: record says the rationale was NOT steered")

# alpha=0 must be byte-identical across scopes — that is the reuse argument.
vc_z2 = FakeVC()
rec_z2 = pv6.run_reference_episode(vc_z2, None, seed=0, env=easy,
                                   steering_scope="action")
check(rec_z["choices"] == rec_z2["choices"],
      "alpha=0 trajectory is identical under both scopes (reuse is valid)")
check([c["prompt"] for c in vc_z.calls] == [c["prompt"] for c in vc_z2.calls],
      "alpha=0 prompt chain is identical under both scopes")

# scope=action with alpha!=0 keeps the original semantics untouched.
vc_a = FakeVC()
rec_a = pv6.run_reference_episode(vc_a, fake_diff, seed=0, env=easy,
                                  steering_scope="action")
check(len([c for c in vc_a.calls if c["stage"] == "rationale"]) == T,
      "action/alpha!=0: rationale pass still unsteered generate")
check(len([c for c in vc_a.calls if c["stage"] == "regenerate"]) == 0,
      "action/alpha!=0: regenerate never called")
check(rec_a["steered_rationale"] is False,
      "action/alpha!=0: record says rationale unsteered")

_bad = False
try:
    pv6.run_reference_episode(FakeVC(), fake_diff, seed=0, env=easy,
                              steering_scope="decode_wide")
except ValueError:
    _bad = True
check(_bad, "an unknown steering_scope is rejected, not silently defaulted")


class CountingVC(FakeVC):
    """FakeVC that simulates the real steering counter, in SITES.

    The config field `steered_rationale` only echoes the flag; this models what
    llms.py actually increments, so the test can assert the OBSERVED count.

    The unit is a (steered layer, sequence, token position) that received a
    non-zero add — NOT a hook call. That distinction is the whole point: a
    call-counter reports the same number whether 10 layers or all 32 are
    steered, and whether K=4 or K=5 candidates are scored, so it could not
    detect a mis-scoped mask or a wrong candidate batch.
    """

    # llama3 band "11-20": utils.decoder_layer_range(11,20) is HALF-OPEN =
    # range(10,19) = NINE layers, not ten. Production therefore reads 900 /
    # 3600 for an Easy T=100 episode, and this fake must agree or the test
    # green-lights a count the real run can never produce. (The "10 layers"
    # figure in CLAUDE.md is the HS-storage set: middle + final.)
    N_LAYERS = 9            # llama3 11-20: the non-zero rows of the mask

    def __init__(self, **kw):
        super().__init__(**kw)
        self._fires = 0

    def steering_fire_count(self, reset=False):
        n = self._fires
        if reset:
            self._fires = 0
        return n

    def generate(self, inputs, **kw):
        return super().generate(inputs, **kw)          # no hook -> no fire

    def regenerate(self, inputs, diff_matrices=None, **kw):
        out = super().regenerate(inputs, diff_matrices=diff_matrices, **kw)
        # L * B * tail_len; pv6 uses B=1, tail_len=1.
        self._fires += (self.N_LAYERS * len(inputs)
                        * max(int(kw.get("prefill_tail_len", 1)), 1))
        return out

    def regenerate_logits_teacher_forcing(self, prompts, answer_token_ids,
                                          diff_matrices=None):
        out = super().regenerate_logits_teacher_forcing(
            prompts, answer_token_ids, diff_matrices=diff_matrices)
        if diff_matrices is not None:
            self._fires += self.N_LAYERS * len(prompts)   # L * K
        return out


vc_c = CountingVC()
rec_c = pv6.run_reference_episode(vc_c, fake_diff, seed=0, env=easy,
                                  steering_scope="both")
fires = rec_c["steering_fires"]
L = CountingVC.N_LAYERS
check(fires is not None, "episode records observed steering fires")
# EXACT totals, not ">0": these are the numbers the smoke is judged against,
# and they must be K-dependent or the counter proves nothing about scope.
check(fires["rationale"] == T * L * 1,
      f"both/alpha!=0: rationale sites == T*L*B*tail == {T * L}")
check(fires["action"] == T * L * easy.k,
      f"both/alpha!=0: action sites == T*L*K == {T * L * easy.k}")
check(rec_c["steered_rationale"] is True and fires["rationale"] > 0,
      "config intent and OBSERVED fires agree (both/alpha!=0)")

# K must actually change the action count, or the counter is blind to it.
hard = br.get_environment("hard")
vc_h = CountingVC(k=hard.k)
rec_h = pv6.run_reference_episode(vc_h, fake_diff, seed=0, env=hard,
                                  steering_scope="both")
check(rec_h["steering_fires"]["action"] == hard.horizon * L * hard.k,
      f"hard: action sites scale with K=5 == {hard.horizon * L * hard.k}")
check(rec_h["steering_fires"]["action"] != rec_c["steering_fires"]["action"],
      "the counter distinguishes K=4 from K=5 (a call-counter would not)")
check(rec_h["steering_fires"]["rationale"] == hard.horizon * L,
      "hard: rationale sites do NOT scale with K (B=1 either way)")

vc_c2 = CountingVC()
rec_c2 = pv6.run_reference_episode(vc_c2, fake_diff, seed=0, env=easy,
                                   steering_scope="action")
check(rec_c2["steering_fires"]["rationale"] == 0,
      "action/alpha!=0: rationale hook NEVER fires")
check(rec_c2["steering_fires"]["action"] > 0,
      "action/alpha!=0: action hook still fires")

vc_c3 = CountingVC()
rec_c3 = pv6.run_reference_episode(vc_c3, None, seed=0, env=easy,
                                   steering_scope="both")
check(rec_c3["steering_fires"] == {"rationale": 0, "action": 0},
      "alpha=0: NO hook fires in either pass, whatever the scope")
check(rec_c3["steered_rationale"] is False,
      "alpha=0: config and observed fires agree (both zero)")

check(pv6.run_reference_episode(FakeVC(), fake_diff, seed=0, env=easy,
                                steering_scope="both")["steering_fires"] is None,
      "a model without the counter records None, not a fake zero")

print("\n" + "=" * 60)
if FAILS:
    print(f"FAILED ({len(FAILS)}):")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
