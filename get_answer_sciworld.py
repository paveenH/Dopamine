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

ACTION_NUMBER_SYSTEM_PROMPT = (
    "You will receive an observation and a numbered list of valid actions. "
    "Choose one valid action.\n\n"
    "When I ask for your action, reply with exactly one integer.\n"
    "The integer must be the number of one valid action in the current list."
)

def build_prompt(
    obs: str,
    valid_actions: list[str],
    history: list[tuple[str, str]],
    prompt_style: str = "legacy",
    obs_char_limit: int = 0,
) -> str:
    """
    Build a text prompt from current observation + valid actions + short history.
    History: list of (action, outcome) pairs (last 3 steps).
    """
    if prompt_style == "action_line":
        system_prompt = ACTION_LINE_SYSTEM_PROMPT
    elif prompt_style == "action_number":
        system_prompt = ACTION_NUMBER_SYSTEM_PROMPT
    else:
        system_prompt = LEGACY_SYSTEM_PROMPT
    lines = [system_prompt, ""]
    if history:
        lines.append("Recent steps:")
        for act, outcome in history[-3:]:
            lines.append(f"  > {act}")
            lines.append(f"    {outcome[:120]}")
        lines.append("")
    obs_text = obs.strip()
    if obs_char_limit and len(obs_text) > obs_char_limit:
        obs_text = obs_text[:obs_char_limit] + " …[truncated]"
    lines.append(f"Observation:\n{obs_text}")
    lines.append("")
    lines.append("Valid actions:")
    for i, a in enumerate(valid_actions):
        lines.append(f"  {i+1}. {a}")
    lines.append("")
    if prompt_style == "action_line":
        lines.append("Your action. Reply as Action: <copy one action from the list>.")
    elif prompt_style == "action_number":
        lines.append("Your action number:")
    else:
        lines.append("Your action:")
    return "\n".join(lines)


def get_system_prompt(prompt_style: str) -> str:
    if prompt_style == "action_line":
        return ACTION_LINE_SYSTEM_PROMPT
    if prompt_style == "action_number":
        return ACTION_NUMBER_SYSTEM_PROMPT
    return LEGACY_SYSTEM_PROMPT


def build_user_turn(
    task_name: str,
    obs: str,
    valid_actions: list[str],
    step: int,
    include_task: bool = False,
    prompt_style: str = "legacy",
    obs_char_limit: int = 0,
) -> str:
    lines = []
    if include_task:
        lines.append(f"Task: {task_name}.")
        lines.append("")
    lines.append(f"Step {step + 1}.")
    obs_text = obs.strip()
    # ScienceWorld observations can be long room dumps; under rolling chat these
    # accumulate across turns and blow up the single-step prefill (the OOM cause,
    # since KV cache is per-step not cumulative). Truncate when a limit is set.
    if obs_char_limit and len(obs_text) > obs_char_limit:
        obs_text = obs_text[:obs_char_limit] + " …[truncated]"
    lines.append(f"Observation:\n{obs_text}")
    lines.append("")
    lines.append("Valid actions:")
    for i, a in enumerate(valid_actions):
        lines.append(f"  {i+1}. {a}")
    lines.append("")
    if prompt_style == "action_number":
        lines.append("Your action number:")
    elif prompt_style == "action_line":
        lines.append("Your action. Reply as Action: <copy one action from the list>.")
    else:
        lines.append("Your action:")
    return "\n".join(lines)


def build_rolling_chat_messages(
    prompt_style: str,
    task_name: str,
    obs: str,
    valid_actions: list[str],
    step: int,
    turn_history: list[dict],
    history_window: int,
    obs_char_limit: int = 0,
) -> list[dict]:
    """True ScienceWorld chat: environment=user, model action=assistant.

    Keep the latest K full turns. Older turns are compressed to a short user
    summary so context does not grow linearly across 50-step episodes. Only the
    last `history_window` turns carry a full `user_turn` string (run_episode
    drops the rest), so `old` here is rendered from the compact summary fields.
    """
    messages = [{"role": "system", "content": get_system_prompt(prompt_style)}]

    old = turn_history[:-history_window] if history_window > 0 else turn_history
    recent = turn_history[-history_window:] if history_window > 0 else []
    prefix = ""
    if old:
        lines = [f"Task: {task_name}.", "Earlier actions summary:"]
        for h in old:
            lines.append(
                f"Step {h['step'] + 1}: chose {h['assistant_reply']} "
                f"({h['action']}); score {h['score']}."
            )
        prefix = "\n".join(lines) + "\n\n"

    for i, h in enumerate(recent):
        content = (prefix + h["user_turn"]) if i == 0 else h["user_turn"]
        prefix = ""
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": h["assistant_reply"]})

    current_user = build_user_turn(
        task_name=task_name,
        obs=obs,
        valid_actions=valid_actions,
        step=step,
        include_task=(not turn_history),
        prompt_style=prompt_style,
        obs_char_limit=obs_char_limit,
    )
    messages.append({"role": "user", "content": prefix + current_user})
    return messages


def render_prompt(vc: VicundaModel, prompt: str, use_chat: bool) -> str:
    """Render ScienceWorld prompt either as raw text or a one-turn chat."""
    if not use_chat:
        return prompt
    return vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_action(response: str, valid_actions: list[str]) -> tuple[str, str, int | None]:
    """
    Match model output to the closest valid action. Returns
    (action, parse_type, action_number).

    Order: take the FIRST anchored decision line ("Action: ..." or "<n>") so a
    repeated/looping generation ("Action: 7  Action: 7  …") resolves to its first
    intent, then resolve it. A reply that is a BARE NUMBER (the model selecting by
    list index, e.g. "Action: 7") must go through the number path BEFORE substring
    matching — substring matching a single digit against the action texts mis-fires
    (any action containing that digit, or the first action, gets returned), which
    is what silently mapped "Action: 7" → "look around".
    """
    resp_raw = response.strip()

    # Highest-priority path for action_number. If the model starts with an
    # integer and then hallucinates environment text ("1\nYou open..."), take
    # the opening integer and ignore the rest.
    m0 = re.match(r"^\s*(\d+)\b", resp_raw)
    if m0:
        idx = int(m0.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx], "number", idx + 1

    # Anchored "Action: <x>" line (the action_line prompt). Take the FIRST match
    # so a looped "Action: 7 Action: 7 …" collapses to one decision.
    action_matches = re.findall(r"(?im)^\s*Action:\s*(.+?)\s*$", resp_raw)
    if not action_matches:
        # action_number / loose: also accept an inline "Action: 7" not on its own line.
        action_matches = re.findall(r"(?i)Action:\s*([^\n]+?)(?:\s{2,}|$)", resp_raw)
    if action_matches:
        resp_raw = action_matches[0].strip().rstrip(".")

    resp = resp_raw.lower()

    # Number path FIRST when the reply is (essentially) just an index. This is the
    # intended parse for --prompt_style action_number and for "Action: <n>".
    if re.fullmatch(r"\d+", resp):
        idx = int(resp) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx], "number", idx + 1

    # Exact match
    for i, a in enumerate(valid_actions):
        if resp == a.lower():
            return a, "exact", i + 1

    # Substring match (only meaningful for multi-char action text, never a digit)
    for i, a in enumerate(valid_actions):
        if a.lower() in resp or resp in a.lower():
            return a, "substring", i + 1

    # Last-resort number anywhere in the text
    m = re.search(r"\b(\d+)\b", resp)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx], "number", idx + 1

    return valid_actions[0], "fallback", 1


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
    chat_mode: str,
    history_window: int,
    obs_char_limit: int,
    save_trace: bool,
) -> dict:
    """Run one episode and return metrics."""
    env.load(task_name, variation)
    obs, info = env.reset()

    history = []
    turn_history = []
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


        if use_chat and chat_mode == "rolling":
            messages = build_rolling_chat_messages(
                prompt_style=prompt_style,
                task_name=task_name,
                obs=obs,
                valid_actions=valid_actions,
                step=step,
                turn_history=turn_history,
                history_window=history_window,
                obs_char_limit=obs_char_limit,
            )
            prompt = messages[-1]["content"]
            rendered_prompt = vc.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = build_prompt(
                obs,
                valid_actions,
                history,
                prompt_style=prompt_style,
                obs_char_limit=obs_char_limit,
            )
            rendered_prompt = render_prompt(vc, prompt, use_chat=use_chat)
        prompt_tokens = (
            len(vc.tokenizer(rendered_prompt, add_special_tokens=False).input_ids)
            if save_trace else None
        )
        response = vc.regenerate(
            inputs=[rendered_prompt],
            diff_matrices=diff_mtx,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            prefill_only=True,
            stop_strings=["\n"] if prompt_style == "action_number" else None,
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
                    "prompt_tokens": prompt_tokens,
                    "score": final_score,
                })
            break

        action, parse_type, action_number = parse_action(response, valid_actions)
        if step == 0:
            print(f"    [dbg] response={repr(response[:80])}  →  action={repr(action)} ({parse_type})")
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
                "parse_type": parse_type,
                "action_number": action_number,
                "prompt_tokens": prompt_tokens,
                "next_observation": obs,
                "reward": reward,
                "score": score,
                "done": done,
            })
        history.append((action, obs))
        turn_history.append({
            "step": step,
            "user_turn": prompt,
            "assistant_reply": str(action_number),
            "action": action,
            "score": score,
        })
        # Free the full user_turn text of turns that have fallen out of the
        # rolling window — they are only ever rendered from the compact summary
        # fields, so retaining the long obs/action strings just leaks RAM across
        # the 50-step episode.
        if history_window > 0 and len(turn_history) > history_window:
            turn_history[-(history_window + 1)]["user_turn"] = None

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
                        chat_mode=args.chat_mode,
                        history_window=args.history_window,
                        obs_char_limit=args.obs_char_limit,
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
                "success_rate": round(float(np.mean([s > 0 for s in scores])), 3),
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
                        choices=["legacy", "action_line", "action_number"],
                        help=("legacy = old exact-action prompt. action_line = Action: text anchor. "
                              "action_number = reply with valid action number, then map to action text."))
    parser.add_argument("--use_chat", action="store_true",
                        help="Render each ScienceWorld step through tokenizer.apply_chat_template.")
    parser.add_argument("--chat_mode", type=str, default="single",
                        choices=["single", "rolling"],
                        help="single = one chat turn per step with text history; rolling = true user/assistant turns with sliding window.")
    parser.add_argument("--history_window", type=int, default=8,
                        help="For --chat_mode rolling, keep this many recent full user/assistant turns. "
                             "Lower this if the rolling prefill OOMs (ScienceWorld obs are long).")
    parser.add_argument("--obs_char_limit", type=int, default=600,
                        help="Truncate each observation to this many chars under --chat_mode rolling "
                             "(0 = no limit). Caps the per-step prefill size that causes OOM.")
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
        BASE = f"/{args.data}/paveen/Dopamine/components"

    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
