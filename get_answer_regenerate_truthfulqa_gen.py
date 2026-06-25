#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA-Generation runner with RSN α steering — open-ended over-generation probe.

Standalone — does NOT import template.py or any existing task's loader/runner.

For each question the model FREELY GENERATES an answer (bare-string, vc.regenerate
prefill-only tail=1, greedy, long max_new_tokens). We DON'T auto-judge truthfulness
here (that's done offline by reading raw vs. the reference answer lists); instead we
compute deterministic, judge-free OVER-GENERATION readouts that directly probe the
+α-over-wanting / −α-under-wanting framing:

  - gen_char_len / gen_word_len : how much the model says
  - hedge_rate                  : answer contains a hedge/refusal cue ("I have no
                                  comment", "it's unclear", "I'm not sure", …)
                                  → −α under-wanting predicted to hedge more
  - assertion_sentences         : # of declarative sentences (assertion density)
                                  → +α over-wanting predicted to assert more

The raw generation is always saved (the offline truthful/informative judgment
needs the full text + the carried correct/incorrect reference lists).

Output per α: {SAVE_ROOT}/mdf_{alpha}/truthfulqa_gen_{size}_answers_{st}_{en}.json
"""

import os
import re
import gc
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
import utils

# ───────────────────── Prompt (bare-string, open-ended) ─────────────────────
# Plain QA framing — no role, no chat. The model answers in natural language.
#
TQA_PREFIX = "Answer the following question.\n\nQuestion: "
TQA_SUFFIX = "\nAnswer:"


def build_prompt(question: str) -> str:
    return TQA_PREFIX + question.strip() + TQA_SUFFIX


# ───────────────────── Judge-free over-generation readouts ─────────────────────

HEDGE_PATTERNS = [
    r"i have no comment",
    r"i'?m not sure",
    r"i am not sure",
    r"i don'?t know",
    r"it'?s unclear",
    r"it is unclear",
    r"uncertain",
    r"no one knows",
    r"there is no (?:clear |single )?answer",
    r"cannot be (?:answered|determined)",
    r"hard to say",
    r"depends on",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    # split on sentence-ending punctuation; count non-empty chunks
    chunks = [c for c in re.split(r"[.!?]+", text) if c.strip()]
    return len(chunks)


def overgen_metrics(text: str) -> dict:
    t = (text or "").strip()
    return {
        "char_len": len(t),
        "word_len": len(t.split()),
        "is_hedge": bool(_HEDGE_RE.search(t)),
        "n_sentences": count_sentences(t),
    }


# ───────────────────── One α-cell over the dataset ─────────────────────

def run_alpha(vc: VicundaModel, data: list, diff_mtx: list, alpha: float,
              max_new_tokens: int):
    agg = {"char_len": 0, "word_len": 0, "n_sentences": 0, "n_hedge": 0, "n": 0}
    for sample in tqdm(data, desc=f"α={alpha}"):
        prompt = build_prompt(sample.get("question", ""))
        out = vc.regenerate(
            inputs=[prompt], diff_matrices=diff_mtx, prefill_only=True,
            max_new_tokens=max_new_tokens, temperature=0.0,
        )[0]
        m = overgen_metrics(out)
        sample["raw_neutral"] = out            # always stored — needed for offline judging
        sample["overgen_neutral"] = m

        agg["char_len"] += m["char_len"]
        agg["word_len"] += m["word_len"]
        agg["n_sentences"] += m["n_sentences"]
        agg["n_hedge"] += int(m["is_hedge"])
        agg["n"] += 1

    n = agg["n"] or 1
    summary = {
        "n_total": agg["n"],
        "mean_char_len": round(agg["char_len"] / n, 1),
        "mean_word_len": round(agg["word_len"] / n, 1),
        "mean_sentences": round(agg["n_sentences"] / n, 2),
        "hedge_rate": round(agg["n_hedge"] / n * 100, 2),
    }
    print(f"  α={alpha}: words={summary['mean_word_len']}  "
          f"sents={summary['mean_sentences']}  hedge={summary['hedge_rate']}%")
    return summary


# ───────────────────── Main ─────────────────────

def main():
    ALPHAS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS)

    blob = utils.load_json(TQA_PATH)
    base_data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    print(f"[INFO] loaded {len(base_data)} TruthfulQA-Gen questions from {TQA_PATH}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS:
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
        raw_mask = np.load(os.path.join(MASK_DIR, mask_name))
        diff_mtx = list(raw_mask * alpha)
        print(f"\n=== α={alpha} | layers={st}-{en} ===")

        data = json.loads(json.dumps(base_data))  # fresh copy per α
        with torch.no_grad():
            summary = run_alpha(vc, data, diff_mtx, alpha, args.max_new_tokens)

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"truthfulqa_gen_{args.size}_answers_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"data": data, "accuracy": {"neutral": summary},
                       "prompt_template": TQA_PREFIX + "{question}" + TQA_SUFFIX},
                      fw, ensure_ascii=False, indent=2)
        print("Saved →", out_path)

        del data, summary
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll α finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA-Generation runner with RSN α steering (over-generation probe).")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--ans_file", type=str, default="answer_truthfulqa_gen")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    TQA_PATH = os.path.join(BASE, "benchmark", "truthfulqa_gen.json")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
