#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA prompt template (protocol `finqa-v0`). Explicit CoT, direct numeric
answer, no DSL/program required. Neutral role only (no persona), bare-string
(no chat template) -- matching the convention every steering mask in this
repo is extracted under (see CLAUDE.md "chat-template alignment").

Format mirrors the GSM8K default suite's neutral CoT template: `Answer: ` is
the anchor token prefill-only steering injects into (last prompt token), and
`#### <number>` is the frozen final-answer marker, extended here to allow a
percent sign or a leading `$`/comma-grouped number since FinQA gold is
frequently expressed that way -- the SCORER normalizes both sides through the
same function, so format flexibility here does not change what counts as
correct.
"""

FINQA_TEMPLATE = (
    "You are given a financial report excerpt, a table, and a question.\n"
    "Use the report text and the table to answer with a single number.\n\n"
    "Report text (before table): {pre_text}\n\n"
    "Table:\n{table}\n\n"
    "Report text (after table): {post_text}\n\n"
    "Question: {question}\n"
    "Let's think step by step.\n"
    "Give your final numeric answer after '####'. Report a plain number "
    "(e.g. 94, -12.5, 3.2%, $13.4 million) with no explanation after the "
    "marker.\n"
    "Answer: "
)


def build_finqa_prompt(sample: dict) -> str:
    return FINQA_TEMPLATE.format(
        pre_text=sample.get("pre_text", "") or "(none)",
        table=sample.get("table_linear", "") or "(no table)",
        post_text=sample.get("post_text", "") or "(none)",
        question=sample["question"],
    )
