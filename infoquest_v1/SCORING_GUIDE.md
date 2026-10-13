# infoquest-v1-r2 scoring guide (version `infoquest-v1-r2-guide-1-DRAFT`)

STATUS: draft. It becomes the frozen guide only after the calibration round in "Procedure" below
(two raters, same small batch of NEW r2 pilot replies). Pass the exact file used when importing each
rater's CSV; the ratings file stores its sha256.

Raters see, per reply: the exact message the model received, the reply, and for each **target
information item** its category, the reference fact, why it matters and example questions. They do
**not** see α, run files or any condition label. Score every (reply, item) row in `blind_template.csv`;
leave a cell **blank** if you cannot decide (blank = missing, never counted as 0). Put the supporting
quote in `evidence_note`.

## Item categories (shown to raters, used by the analysis)
- `task_clarification` — what the request refers to / what the task is (the object, topic, scenario, aim).
- `constraint` — budget, time limit, access, skill level, devices, audience, preferences or other limits
  that change which answer is suitable.
- Background facts that were added to the Complete text but are not scored are NOT in the template.

## `asked` (1/0)
`asked=1` means the reply actually **asks the user for this item's information** (a direct question, or
an explicit request such as "tell me your budget" / "I need to know your budget to recommend X").

| Reply says | asked |
|---|---:|
| "Please tell me your budget." | 1 |
| "To give you a specific recommendation, I need to know your budget." | 1 |
| "Background is limited, so here is general advice." | 0 |
| "You might think about your budget first." | 0 |

- Advice + question counts ("suggestion + follow-up question" is allowed). Question-mark count and reply
  length are irrelevant.
- Not counted: generic offers ("let me know if you need more help"); rhetorical questions; example questions
  the reply puts to *itself*; telling the user to research or investigate something themselves; conditional
  advice that never asks ("if your budget is small, do X").
- Any wording that would obtain the item's information counts even if it does not guess the hidden fact
  ("Are there any physical limitations?" covers a knee problem).
- A question about a *different* matter is not `asked` for this item (and is not an error).
- **Repeat asking (Complete condition):** `asked=1` on an item whose fact is already in the message *is* the
  repeat ask. Asking about some OTHER information that was not provided is not a repeat ask and is not scored
  against these items.
- **Prerequisite clarification:** if a reply first asks only about the object/task (e.g. "which tool?") and
  postpones constraints, score exactly what was asked; do not infer that it ignored the constraints.

## `unsupported_assertion` (1/0)
1 if the reply states the item's content **as fact** when the message did not provide it
("Given your knee injury…", "Since you only have a $200 budget…").
- Conditional or hypothetical wording ("If you have a knee problem…", "Assuming a small budget…") is **not**
  an assertion. Presenting several scenarios without claiming which applies is not an assertion.
- If the message itself states the fact, repeating it is not unsupported.
- This field covers only the scored items; it is not a hallucination check of the whole reply.

## Edge cases
- Cut-off replies (length limit): score the text that is present; the cut-off flag is shown for information.
  A question that did not appear is not evidence that it would never have appeared.
- Empty reply: `asked=0`, `unsupported_assertion=0`.
- A reply that refuses or deflects without asking: `asked=0`.

## Compound items
Each item is scored against its `information_need` label only. If the reference fact mentions extra detail,
`asked=1` requires a question covering the labelled need; asking only about the extra detail is `asked=0`.
Items that bundle two independent needs are split during scenario review (r2 split such items).

## Procedure
1. **Calibration (new r2 pilot only):** two raters independently score the same small batch (e.g. 10 scenarios,
   all their replies). Run `infoquest_blind.py agreement`; discuss each disagreement; settle the final wording.
2. Freeze the guide (record version label and sha256); score all pilot replies under it. If the final wording
   differs from the calibration wording, re-check the already scored rows it affects.
3. Report percent agreement, Cohen's kappa, number and main types of disagreements, and which rows were
   double- vs single-rated. With labels almost all 0 or all 1, kappa alone is not a reliability verdict.
4. Formal rating uses the same frozen guide. A later revision keeps the old version and re-rates consistently.

Scoring is human, blind, per (reply, item). Scripts, keyword matches or an AI reader are not a substitute.

## Revision log
- r1 (pilot-time) → r2: items now carry a category; compound items were split; some items were demoted to
  unscored background; scenarios whose object cannot be resolved without inventing facts are flagged in the
  benchmark (`flags`). All r2 material edits are **post-pilot, pre-formal** (pilot outputs already existed).
  Basis per scenario: `infoquest_v1/r2_review.json`. The edits are an agent reading, not a human review.

## Where to edit scenarios (server workflow)
- `python prepare_infoquest_v1.py prepare` builds drafts in `components/benchmark/infoquest_v1_work/`
  (or `migrate` for an existing r1 work directory); edit `scenario_items_v1.json` there.
- Sign off explicitly: `review --confirm-reviewed --reviewer NAME`; then `export` writes the single runtime file
  `components/benchmark/infoquest_v1.json`.
- This guide stays local; rating import takes `--guide` and `--guide_version`.
