#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:29:48 2025

@author: paveenhuang
"""

import os
import re
import json
from typing import List
from pathlib import Path
import numpy as np

INTRO_FMT = "The following are multiple choice questions (with answers) about {subject}."
LABELS = ["A", "B", "C", "D"]
MMLU_POOL_DIR = Path(
    os.environ.get("MMLU_POOL_DIR", "/data2/paveen/RolePlaying/components/mmlu_fewshot")
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ─────────────────────── GSM8K / MATH answer extraction ───────────────────────
# Canonical extraction + correctness logic shared by every GSM8K/MATH entry-point
# (get_answer_gsm8k, closed_loop_gsm8k, track_dopamine_signal, track_hidden_states).
# Keep all scripts importing from here so accuracy stays comparable across runs.

def extract_gsm8k_answer(text: str) -> str:
    text = text.strip()
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}", text)
    if m:
        return m.group(1).replace(",", "")
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def normalize_gsm8k(ans: str) -> str:
    ans = ans.strip().replace(",", "")
    try:
        val = float(ans)
        if np.isinf(val) or np.isnan(val):
            return str(val)
        return str(int(val)) if val == int(val) else str(val)
    except (ValueError, OverflowError):
        return ans


def is_correct_gsm8k(pred: str, gold: str) -> bool:
    return normalize_gsm8k(pred) == normalize_gsm8k(gold)


def gsm8k_difficulty(question: str) -> str:
    ops = len(re.findall(r"[\+\-\*\/]", question))
    if ops <= 2:
        return "Easy"
    elif ops <= 4:
        return "Medium"
    return "Hard"


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def extract_math_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def normalize_math(s: str) -> str:
    return s.strip().lower().replace(",", "").replace(" ", "")


def is_correct_math(pred: str, gold: str) -> bool:
    return normalize_math(pred) == normalize_math(gold)


# ─────────────────────── RSN mask indexing ───────────────────────
# layer_start/layer_end follow HF hidden_states semantics (index 0 = embedding,
# 1..N = decoder layer outputs). Saved masks drop the embedding row
# (detection/nmd.py: `return mask[1:]`), so saved-index i ↔ decoder_layers[i]
# ↔ hidden_states[i+1]. To pick the rows nmd.py actually filled
# (hs[layer_start..layer_end-1]), slice saved[layer_start-1 : layer_end-1].
# Centralize this so the offset never drifts again across consumers.

def mask_slice_for(mask: np.ndarray, layer_start: int, layer_end: int) -> np.ndarray:
    """Slice a saved RSN mask to the rows matching hs[layer_start..layer_end).

    Returns (layer_end - layer_start, H) sub-mask.
    Equivalent to mask[layer_start-1 : layer_end-1].
    """
    if layer_start < 1:
        raise ValueError(f"layer_start must be >=1 (HF hidden_states index), got {layer_start}")
    if layer_end <= layer_start:
        raise ValueError(f"layer_end ({layer_end}) must be > layer_start ({layer_start})")
    if layer_end - 1 > mask.shape[0]:
        raise ValueError(
            f"layer_end-1 ({layer_end - 1}) exceeds saved mask rows ({mask.shape[0]})"
        )
    return mask[layer_start - 1:layer_end - 1]


def decoder_layer_range(layer_start: int, layer_end: int) -> range:
    """Translate HF hidden_states range [layer_start, layer_end) to the matching
    decoder-layer index range. Use this for hook registration so it stays in
    sync with mask_slice_for."""
    return range(layer_start - 1, layer_end - 1)


def softmax_1d(x: np.ndarray):
    e = np.exp(x - x.max())
    return e / e.sum()


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[(p > 0) & np.isfinite(p)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def cleaning(text: str):
    text = text.replace("<|assistant|>", "").replace("\u200b", "").strip().upper()
    m = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", text)
    return m.group(1) if m else text.strip().upper()


def extract_full_correct_text(question_text: str, label_idx: int, label_map):
    prefix = f"{label_map[label_idx]})"
    for line in question_text.split("\n"):
        s = line.strip()
        if s.upper().startswith(prefix):
            return s[len(prefix) :].strip().lower()
    return None


def make_characters(task_name="", custom_roles=None):
    """
    Generate role list for a given task.

    Args:
        task_name: The MMLU task name (e.g., "college_math")
        custom_roles: List of role strings. Use "{task}" as placeholder
                      for task name (e.g., "{task} expert" → "college math expert")

    Returns:
        List of role strings
    
    Reference role templates (now passed via custom_roles):
        # f"non {task_name_formatted} expert",
        # f"{task_name_formatted} expert",
        # f"not an expert in {task_name_formatted}",
        # f"{task_name_formatted} student",
        # "person",
        # "non expert",
        # "expert",
        # "non medical expert",
        # "medical expert",
        # "confident",
        # "unconfident",
    """
    if custom_roles is None:
        return ["neutral"]
    
    task_name = task_name.lower()
    task_name_formatted = task_name.replace("_", " ")
    return [role.format(task=task_name_formatted) for role in custom_roles]


def remove_honest(templates: dict) -> dict:
    """
    Remove "honest" description in (non-) expert prompt
    """
    new_templates = {}
    for k, v in templates.items():
        if isinstance(v, str):
            new_templates[k] = v.replace(" honest", "")
        else:
            new_templates[k] = v
    return new_templates


def construct_prompt(vc, templates, ctx: str, role: str, use_chat=False) -> str:
    """
    Return the final prompt string that can be fed into the model.
    - If use_chat=True: render with tokenizer.apply_chat_template (messages + add_generation_prompt=True)
    - If use_chat=False: fallback to original string templates (default/neg/neutral)
    """
    user_text = templates["neutral"].format(context=ctx)
    if role == "norole" or role == "neutral":
        messages = [{"role": "user", "content": user_text}]
        plain = templates["neutral"].format(context=ctx)
    elif "not" in role or "confident" in role:
        messages = [
            {"role": "system", "content": f"Now you are {role}."},
            {"role": "user", "content": user_text},
        ]
        plain = templates["neg"].format(character=role, context=ctx)
    else:
        messages = [
            {"role": "system", "content": f"Now you are an honest {role}."},
            {"role": "user", "content": user_text},
        ]
        plain = templates["default"].format(character=role, context=ctx)
    if use_chat:
        return vc.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return plain


def record_template(roles, templates):
    tmp_record = []
    for role in roles:
        if "confident" in role:
            role = "neg"
        if role in templates:
            print(f"{role} prompt")
            print(templates[role])
            tmp_record.append(templates[role])
            print("----------------")
        else:
            print("default prompt")
            print(templates["default"])
            tmp_record.append(templates["default"])
            print("----------------")
    return tmp_record


def update(acc, char, status):
    acc[char][status] += 1


def option_token_ids(vc, LABELS):
    ids = []
    for opt in LABELS:
        tok = vc.tokenizer(opt, add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f"Option {opt} maps to {tok}, expected single token")
        ids.append(tok[0])
    return ids


def parse_configs(configs: list[str]):
    """
    Convert ['4-16-22', '1-1-29', 'neg1-11-20']
    → [[4, (16, 22)], [1, (1, 29)], [-1, (11, 20)]]
    """
    parsed = []
    for cfg in configs:
        try:
            parts = cfg.strip().split("-")
            if parts[0].startswith("neg"):
                alpha = -int(parts[0][3:])
                start, end = map(int, parts[1:])
            else:
                alpha = int(parts[0])
                start, end = map(int, parts[1:])
            parsed.append([alpha, (start, end)])
        except Exception:
            raise ValueError(f"Invalid config format: '{cfg}', should be alpha-start-end (e.g., 4-16-22 or neg1-11-20)")
    return parsed


def _task_to_filename(task: str) -> str:
    return task.strip().lower().replace(" ", "_") + ".json"


def _fewshot_exemplar(sample: dict) -> str:
    """
    Construct a single few-shot exemplar block (question + choices + answer).
    """
    labels = LABELS
    lines = [f"Question: {sample['text']}"]
    choices = sample.get("choices", None)
    if choices is not None:
        assert len(choices) == 4, "choices must be exactly 4 texts (A-D)."
        for i, ch in enumerate(choices):
            lines.append(f"{labels[i]}) {ch}")
    ans_idx = sample.get("label", None)
    if ans_idx is None or not (0 <= ans_idx < len(LABELS)):
        raise ValueError("Few-shot exemplar requires a valid 'label' (0..3).")
    lines.append(f"Answer: {LABELS[ans_idx]}")
    return "\n".join(lines)


def build_fewshot_prefix(task: str, k: int = 5) -> str:
    """
    Load the fixed-order few-shot exemplars for `task` from <fewshot_dir>/<task>.json,
    then construct the few-shot prefix:
      INTRO + k exemplars (each ends with "Answer: X")
    Does NOT include the test question; append build_query_block(sample) later.
    """
    file_path = Path(MMLU_POOL_DIR) / _task_to_filename(task)
    if not file_path.exists():
        raise FileNotFoundError(
            f"[Few-shot] Not found: {file_path}. " f"Please prepare 5-shot file under {MMLU_POOL_DIR}/<task>.json"
        )
    support_pool: List[dict] = load_json(str(file_path))
    if not isinstance(support_pool, list) or len(support_pool) == 0:
        raise ValueError(f"[Few-shot] Bad file format or empty file: {file_path}")
    exemplars = support_pool[:k] if k <= len(support_pool) else support_pool
    parts = [INTRO_FMT.format(subject=task)]
    for s in exemplars:
        parts.append(_fewshot_exemplar(s))
        parts.append("")
    return "\n".join(parts).strip()


if __name__ == "__main__":
    print(build_fewshot_prefix("anatomy"))