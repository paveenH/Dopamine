#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluEval-QA discrimination runner with RSN α steering — SIGNED-BIAS hallucination probe.

Standalone — does NOT import template.py or any existing task's loader/runner, so
the established MMLU / GSM8K / betting pipelines are unaffected.

Reads the balanced discrimination JSON from data_halueval.py (each source row
expanded into a right-answer item [gold=No] and a hallucinated-answer item
[gold=Yes]). For each item the model judges: "Is the answer hallucinated? Yes/No".

bare-string + GENERATION (vc.regenerate prefill-only tail=1) + parse Yes/No,
matching the betting/CGT convention and the NMD-mask bare activation distribution.

The headline is NOT accuracy. It is the SIGNED bias of the Yes/No decision:
  - acceptance_rate = P(judge "No" / not-hallucinated) over ALL items
  - FPR = P(judge "Yes" | gold=right)         (calling a true answer fake)
  - FNR = P(judge "No"  | gold=hallucinated)   (letting a hallucination pass)  ← over-wanting / credulity
Prediction: +α → more credulous → FNR↑ / acceptance↑ ; −α → more skeptical → FNR↓.

Output per α: {SAVE_ROOT}/mdf_{alpha}/halueval_qa_{size}_answers_{st}_{en}.json
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

# ───────────────────── Prompt (bare-string, HaluEval-official style) ─────────────────────

HALU_PREFIX = (
    "You are judging whether an answer to a question contains hallucinated "
    "(factually incorrect or unsupported) information.\n\n"
)
# body = knowledge / question / answer block, then the Yes/No anchor
HALU_SUFFIX = (
    "\nDoes the answer contain hallucinated information? "
    "Answer with a single word, Yes or No.\nAnswer: "
)


def build_prompt(knowledge: str, question: str, answer: str) -> str:
    body = []
    knowledge = (knowledge or "").strip()
    if knowledge:
        body.append(f"Knowledge: {knowledge}")
    body.append(f"Question: {question.strip()}")
    body.append(f"Answer: {answer.strip()}")
    return HALU_PREFIX + "\n".join(body) + HALU_SUFFIX


# ───────────────────── Parse Yes/No → 1 (hallucinated) / 0 (not) ─────────────────────

def parse_yesno(raw: str) -> int | None:
    """1 = 'Yes' (judged hallucinated), 0 = 'No', None = unparseable (invalid)."""
    if not raw:
        return None
    s = raw.strip()
    m = re.match(r"(?i)^\W*\b(yes|no)\b", s)
    if m:
        return 1 if m.group(1).lower() == "yes" else 0
    m = re.search(r"(?i)\b(yes|no)\b", s)
    if m:
        return 1 if m.group(1).lower() == "yes" else 0
    return None


# ───────────────────── One α-cell over the whole dataset ─────────────────────

def run_alpha(vc: VicundaModel, data: list, diff_mtx: list, alpha: float,
              max_new_tokens: int, save_all_raw: bool):
    # confusion accumulators keyed by gold (is_hallucination)
    # pred: 1=judged hallucinated(Yes), 0=judged not(No)
    n_right = n_hall = 0
    fp = 0   # gold=right(0)   judged Yes(1)  -> false positive (over-skeptical)
    fn = 0   # gold=hall(1)    judged No(0)   -> false negative (over-credulous)
    tp = 0   # gold=hall(1)    judged Yes(1)
    tn = 0   # gold=right(0)   judged No(0)
    invalid = 0
    n_yes = 0  # total "Yes" among valid

    for sample in tqdm(data, desc=f"α={alpha}"):
        gold = sample["is_hallucination"]
        prompt = build_prompt(sample.get("knowledge", ""),
                              sample.get("question", ""),
                              sample.get("answer", ""))
        out = vc.regenerate(
            inputs=[prompt], diff_matrices=diff_mtx, prefill_only=True,
            max_new_tokens=max_new_tokens, temperature=0.0,
        )[0]
        pred = parse_yesno(out)

        sample[f"pred_neutral"] = ("INVALID" if pred is None
                                   else ("Yes" if pred == 1 else "No"))
        if save_all_raw:
            sample["raw_neutral"] = out

        if pred is None:
            invalid += 1
            continue
        n_yes += pred
        if gold == 0:
            n_right += 1
            if pred == 1: fp += 1
            else: tn += 1
        else:
            n_hall += 1
            if pred == 1: tp += 1
            else: fn += 1

    n_valid = n_right + n_hall
    acc = (tp + tn) / n_valid * 100 if n_valid else 0.0
    fpr = fp / n_right * 100 if n_right else 0.0   # call true answer fake
    fnr = fn / n_hall * 100 if n_hall else 0.0     # let hallucination pass (credulity)
    acceptance = (tn + fn) / n_valid * 100 if n_valid else 0.0  # judged "not hallucinated"
    yes_rate = n_yes / n_valid * 100 if n_valid else 0.0
    summary = {
        "n_total": len(data), "invalid": invalid,
        "invalid_percentage": round(invalid / len(data) * 100, 2) if data else 0.0,
        "n_right": n_right, "n_hall": n_hall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy_percentage": round(acc, 2),
        "FPR_percentage": round(fpr, 2),
        "FNR_percentage": round(fnr, 2),
        "acceptance_rate": round(acceptance, 2),   # P(judge "not hallucinated")
        "yes_rate": round(yes_rate, 2),            # P(judge "hallucinated")
    }
    print(f"  α={alpha}: acc={acc:.2f}%  FNR(credulity)={fnr:.2f}%  FPR={fpr:.2f}%  "
          f"acceptance={acceptance:.2f}%  invalid={summary['invalid_percentage']:.2f}%")
    return summary


# ───────────────────── Main ─────────────────────

def main():
    ALPHAS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS)

    blob = utils.load_json(HALU_PATH)
    base_data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    print(f"[INFO] loaded {len(base_data)} HaluEval-QA discrimination items from {HALU_PATH}")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS:
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
        raw_mask = np.load(os.path.join(MASK_DIR, mask_name))
        diff_mtx = list(raw_mask * alpha)
        print(f"\n=== α={alpha} | layers={st}-{en} ===")

        data = json.loads(json.dumps(base_data))  # fresh copy per α
        with torch.no_grad():
            summary = run_alpha(vc, data, diff_mtx, alpha,
                                args.max_new_tokens, args.save_all_raw)

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"halueval_qa_{args.size}_answers_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"data": data, "accuracy": {"neutral": summary},
                       "prompt_template": HALU_PREFIX + "{knowledge/question/answer}" + HALU_SUFFIX},
                      fw, ensure_ascii=False, indent=2)
        print("Saved →", out_path)

        del data, summary
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll α finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluEval-QA discrimination runner with RSN α steering (signed-bias probe).")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--ans_file", type=str, default="answer_halueval")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_all_raw", action="store_true")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    HALU_PATH = os.path.join(BASE, "benchmark", "halueval_qa_disc.json")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
