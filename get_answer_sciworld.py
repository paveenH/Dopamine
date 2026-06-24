#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScienceWorld agentic benchmark with RSN diff injection.

For each (alpha, task) pair:
  - Run N episodes of the ScienceWorld task under neutral / +alpha / -alpha conditions
  - Each episode: observation → prompt → vc.regenerate() with diff_mtx → parse action → env.step()
  - Record: final score, abandonment rate, steps taken, per-step hedging markers

Usage:
  python get_answer_sciworld.py \
    --model llama3 \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --hs llama3 --size 8B --type non \
    --percentage 0.5 --mask_type nmd \
    --configs 0-11-20 4-11-20 neg4-11-20 \
    --task_nums 0 1 2 3 4 \
    --num_episodes 5 \
    --max_steps 50 \
    --data data1
"""

import os
import gc
import csv
import json
import re
import argparse
import numpy as np
import torch

from llms import VicundaModel
import utils

# ---------------------------------------------------------------------------
# Abandonment detection
# ---------------------------------------------------------------------------
ABANDON_PATTERNS = [
    r"\bi\s+cannot\b",
    r"\bi\s+can'?t\b",
    r"\bi\s+don'?t\s+know\b",
    r"\bunable\s+to\b",
    r"\bnot\s+sure\b",
    r"\bgive\s+up\b",
    r"\bimpossible\b",
]
ABANDON_RE = re.compile("|".join(ABANDON_PATTERNS), re.IGNORECASE)

HEDGE_PATTERNS = [
    r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b", r"\bpossibly\b",
    r"\bi\s+think\b", r"\bi\s+believe\b", r"\bnot\s+sure\b",
    r"\buncertain\b", r"\bcould\s+be\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def is_abandon(text: str) -> bool:
    return bool(ABANDON_RE.search(text))


def is_hedge(text: str) -> bool:
    return bool(HEDGE_RE.search(text))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
# NOTE: the historical prompt opened with "You are a science experiment agent."
# That persona line was dropped (same de-roling rationale as the Bandit No-Role
# port): a role primer shifts the activation distribution the NMD mask was
# extracted on and confounds the steering. Both system prompts below are
# persona-free; legacy is therefore NOT byte-identical to the −4/0/+4 results
# under sciworld/mdf_*, so do not compare ACC across the persona boundary.
LEGACY_SYSTEM_PROMPT = (
    "You will receive an observation and a numbered list of valid actions. "
    "Output ONLY the exact text of one action from the list. "
    "Do not explain, do not add punctuation, do not output anything else."
)

ACTION_LINE_SYSTEM_PROMPT = (
    "You will receive an observation and a numbered list of valid actions. "
    "Choose one valid action.\n\n"
    "When I ask for your action, reply with exactly one line:\n"
    "Action: <copy one action from the list>\n\n"
    "The text after Action: must exactly match one action from the list."
)

def build_prompt(
    obs: str,
    valid_actions: list[str],
    history: list[tuple[str, str]],
    prompt_style: str = "legacy",
) -> str:
    """
    Build a text prompt from current observation + valid actions + short history.
    History: list of (action, outcome) pairs (last 3 steps).
    """
    system_prompt = ACTION_LINE_SYSTEM_PROMPT if prompt_style == "action_line" else LEGACY_SYSTEM_PROMPT
    lines = [system_prompt, ""]
    if history:
        lines.append("Recent steps:")
        for act, outcome in history[-3:]:
            lines.append(f"  > {act}")
            lines.append(f"    {outcome[:120]}")
        lines.append("")
    lines.append(f"Observation:\n{obs.strip()}")
    lines.append("")
    lines.append("Valid actions:")
    for i, a in enumerate(valid_actions):
        lines.append(f"  {i+1}. {a}")
    lines.append("")
    if prompt_style == "action_line":
        lines.append("Your action. Reply as Action: <copy one action from the list>.")
    else:
        lines.append("Your action:")
    return "\n".join(lines)


def render_prompt(vc: VicundaModel, prompt: str, use_chat: bool) -> str:
    """Render ScienceWorld prompt either as raw text or a one-turn chat."""
    if not use_chat:
        return prompt
    return vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_action(response: str, valid_actions: list[str]) -> str:
    """
    Match model output to the closest valid action.
    Priority: exact match → substring match → first valid action (fallback).
    """
    resp_raw = response.strip()
    # Newer ScienceWorld prompts may ask for a terminal "Action: ..." line,
    # analogous to IGT's "Chest: N" anchor. Parse the last such line before
    # falling back to legacy exact/substring/number matching.
    action_matches = re.findall(r"(?im)^\s*Action:\s*(.+?)\s*$", resp_raw)
    if action_matches:
        resp_raw = action_matches[-1].strip()

    resp = resp_raw.lower()

    # Exact match
    for a in valid_actions:
        if resp == a.lower():
            return a

    # Substring match
    for a in valid_actions:
        if a.lower() in resp or resp in a.lower():
            return a

    # Fallback: look for action number
    m = re.search(r"\b(\d+)\b", resp)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]

    return valid_actions[0]


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def run_episode(
    vc: VicundaModel,
    env,
    task_name: str,
    variation: int,
    diff_mtx: list[np.ndarray],
    max_steps: int,
    max_new_tokens: int,
    prompt_style: str,
    use_chat: bool,
    save_trace: bool,
) -> dict:
    """Run one episode and return metrics."""
    env.load(task_name, variation)
    obs, info = env.reset()

    history = []
    scores = []
    step_hedges = []
    trace = []
    abandoned = False
    final_score = 0.0

    for step in range(max_steps):
        raw_actions = env.get_valid_action_object_combinations_with_templates()
        if not raw_actions:
            raw_actions = env.get_possible_actions() or [{"action": "look around"}]
        # each element may be a dict {"action": "...", ...} or a plain string
        valid_actions = [
            a["action"] if isinstance(a, dict) else a for a in raw_actions
        ]


        prompt = build_prompt(obs, valid_actions, history, prompt_style=prompt_style)
        rendered_prompt = render_prompt(vc, prompt, use_chat=use_chat)
        response = vc.regenerate(
            inputs=[rendered_prompt],
            diff_matrices=diff_mtx,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            prefill_only=True,
        )[0]

        step_hedges.append(1 if is_hedge(response) else 0)

        abandon = is_abandon(response)
        if abandon:
            abandoned = True
            if save_trace:
                trace.append({
                    "step": step,
                    "observation": obs,
                    "valid_actions": valid_actions,
                    "raw_response": response,
                    "parsed_action": None,
                    "abandoned": True,
                    "score": final_score,
                })
            break

        action = parse_action(response, valid_actions)
        if step == 0:
            print(f"    [dbg] response={repr(response[:80])}  →  action={repr(action)}")
        obs_before = obs
        obs, reward, done, info = env.step(action)
        score = info.get("score", reward)
        final_score = score
        scores.append(score)
        if save_trace:
            trace.append({
                "step": step,
                "observation": obs_before,
                "valid_actions": valid_actions,
                "raw_response": response,
                "parsed_action": action,
                "next_observation": obs,
                "reward": reward,
                "score": score,
                "done": done,
            })
        history.append((action, obs))

        if done:
            break

    result = {
        "final_score": final_score,
        "steps": len(scores),
        "abandoned": abandoned,
        "hedge_rate": float(np.mean(step_hedges)) if step_hedges else 0.0,
        "score_trajectory": scores,
    }
    if save_trace:
        result["trace"] = trace
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError:
        raise ImportError("scienceworld not installed. Run: pip install scienceworld")

    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    env = ScienceWorldEnv("", args.jar_path, envStepLimit=args.max_steps)
    task_names = env.get_task_names()

    # Select tasks
    selected_tasks = []
    for t in args.task_nums:
        if 0 <= t < len(task_names):
            selected_tasks.append((t, task_names[t]))
        else:
            print(f"[Warning] task_num {t} out of range (max {len(task_names)-1}), skipping")

    print(f"Running {len(selected_tasks)} tasks × {len(ALPHAS_START_END_PAIRS)} configs × {args.num_episodes} episodes")

    all_rows = []

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        raw_mask = np.load(mask_path)
        diff_mtx = list(raw_mask * alpha)
        TOP = max(1, int(args.percentage / 100 * raw_mask.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        for task_num, task_name in selected_tasks:
            episode_results = []
            num_variations = min(args.num_episodes, env.get_max_variations(task_name))

            for var in range(num_variations):
                print(f"  Task {task_num} ({task_name}) | var={var} | α={alpha}", end=" ... ", flush=True)
                with torch.no_grad():
                    result = run_episode(
                        vc=vc,
                        env=env,
                        task_name=task_name,
                        variation=var,
                        diff_mtx=diff_mtx,
                        max_steps=args.max_steps,
                        max_new_tokens=args.max_new_tokens,
                        prompt_style=args.prompt_style,
                        use_chat=args.use_chat,
                        save_trace=args.save_trace,
                    )
                episode_results.append(result)
                print(f"score={result['final_score']:.1f}  steps={result['steps']}  abandon={result['abandoned']}")
                gc.collect()
                torch.cuda.empty_cache()

            # Aggregate
            scores = [r["final_score"] for r in episode_results]
            abandons = [r["abandoned"] for r in episode_results]
            hedges = [r["hedge_rate"] for r in episode_results]
            steps = [r["steps"] for r in episode_results]

            row = {
                "model": args.model,
                "size": args.size,
                "alpha": alpha,
                "start": st,
                "end": en,
                "TOP": TOP,
                "task_num": task_num,
                "task_name": task_name,
                "num_episodes": len(episode_results),
                "mean_score": round(float(np.mean(scores)), 3),
                "std_score": round(float(np.std(scores)), 3),
                "success_rate": round(float(np.mean([s >= 100 for s in scores])), 3),
                "abandon_rate": round(float(np.mean(abandons)), 3),
                "mean_steps": round(float(np.mean(steps)), 1),
                "mean_hedge_rate": round(float(np.mean(hedges)), 4),
            }
            all_rows.append(row)

            # Save per-task episode details
            out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            detail_path = os.path.join(out_dir, f"task{task_num}_{args.size}_{TOP}_{st}_{en}.json")
            with open(detail_path, "w", encoding="utf-8") as fw:
                json.dump({"task_num": task_num, "task_name": task_name,
                           "alpha": alpha, "episodes": episode_results}, fw, indent=2)
            print(f"  → {detail_path}")

            gc.collect()
            torch.cuda.empty_cache()

        # Restart env after each alpha block to flush Java heap
        env.close()
        env = ScienceWorldEnv("", args.jar_path, envStepLimit=args.max_steps)

    env.close()

    # Save summary CSV
    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")
    fieldnames = [
        "model", "size", "alpha", "start", "end", "TOP",
        "task_num", "task_name", "num_episodes",
        "mean_score", "std_score", "success_rate",
        "abandon_rate", "mean_steps", "mean_hedge_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[Saved CSV] {csv_path}")
    print("✅  ScienceWorld run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScienceWorld agentic benchmark with RSN steering.")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--configs", nargs="+", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--task_nums", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="ScienceWorld task indices to run")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Episodes (variations) per task")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max tokens for action generation")
    parser.add_argument("--prompt_style", type=str, default="legacy",
                        choices=["legacy", "action_line"],
                        help="legacy = old exact-action prompt. action_line = IGT/CGT-style Action: anchor.")
    parser.add_argument("--use_chat", action="store_true",
                        help="Render each ScienceWorld step through tokenizer.apply_chat_template.")
    parser.add_argument("--save_trace", action="store_true",
                        help="Save per-step observation, raw response, parsed action, and score.")
    parser.add_argument("--jar_path", type=str, default="",
                        help="Path to ScienceWorld JAR file (leave empty to use bundled JAR)")
    parser.add_argument("--ans_file", type=str, default="answer_sciworld")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
