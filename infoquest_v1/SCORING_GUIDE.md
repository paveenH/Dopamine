# infoquest-v1-r2 scoring guide (version `infoquest-v1-r2-guide-2`)

STATUS: guide-2, frozen for the formal run. It is the guide-1 text (kept unchanged as `SCORING_GUIDE_guide-1-DRAFT.md`) plus the
"Boundary clarifications" section below. The clarifications were written AFTER the alpha=0 r2 pilot was scored (the pilot scorer
reported three ambiguous boundaries) and BEFORE any alpha=+-4 reply existed. They are a post-pilot clarification, not a rule fixed
before any model ran. Pass the exact file used when importing ratings; the ratings file stores its sha256.

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
- **Vague vs specific:** asking for a detail the message did NOT give (e.g. an exact budget when the message only says
  "small") is new information, not a repeat ask. Where a target needs a concrete value, the Complete text states it.
- **Prerequisite clarification:** if a reply first asks only about the object/task (e.g. "which tool?") and
  postpones constraints, score exactly what was asked; do not infer that it ignored the constraints.

## `unsupported_assertion` (1/0)
1 if the reply states the item's content **as fact** when the message did not provide it
("Given your knee injury…", "Since you only have a $200 budget…").
- Conditional or hypothetical wording ("If you have a knee problem…", "Assuming a small budget…") is **not**
  an assertion. Presenting several scenarios without claiming which applies is not an assertion.
- If the message itself states the fact, repeating it is not unsupported.
- This field covers only the scored items; it is not a hallucination check of the whole reply.

## Boundary clarifications (guide-2; written after the alpha=0 pilot, before any formal reply)
One criterion governs all three boundaries: **does the reply explicitly ask the user to supply the information defined by this item's
`information_need` label?** Judge only the wording of the reply against the label (not the full reference fact, not which label would
make conditions differ). A reader who sees the reply should be able to say "it asked me for X".

1. **Invitations ("feel free to provide ...", "if you have ..., let me know / that would be helpful").**
   - Counts (`asked=1`) when the invitation *names* the need's content and asks the user to give it: "If you have a specific area in mind,
     feel free to tell me which" (need: which area); "if you have a particular time frame or type of data in mind, that information would be
     helpful" (need: data type).
   - Does not count (`asked=0`) when it does not name the need: "if you have more details, I can tailor this", "let me know if you need more
     help", "feel free to share more context" (no content named). A yes/no offer to keep talking ("Would you like to explore this further?")
     or to discuss a different matter than the need ("discuss a scenario where you met such a structure" when the need is *which* structure)
     is `asked=0`.
2. **Questions inside advice lists / frameworks.**
   - Does not count: a question that is a step prompt for the user's own thinking ("Define the objective: what is the purpose?",
     "**What do you want to achieve?**" as a bullet in a how-to list) when the reply never asks the user to answer it back to the assistant.
   - Counts: the same question when the reply addresses it to the user as something to tell the assistant ("Tell me what you want to achieve",
     "Could you share ...", a closing "Which of these fits, or what is your goal?").
3. **Broad or partial questions.**
   - Counts when answering the question would give the information the *label* names, even if the question is loose or broader ("What's
     the destination or goal?" for destination; "what kind of trouble are you trying to avoid?" for event to avoid).
   - Does not count when it covers a different or only a generic matter ("what are you trying to do?" is enough only for needs that ARE the goal/
     task/activity/purpose; it does not cover budget, devices, skill level or deadline). "Any specific ideas or goals in mind?" counts only
     for a need that is the goal/activity.
4. **Independent needs.** Each target item is scored on its own label. A reply need not ask all of a scenario's items, and asking one item
   gives no credit to another. Do not infer a need from a neighbouring item.

Examples (asked for the quoted need):
| Reply says | Need | asked |
|---|---|---:|
| "Could you tell me your monthly budget?" | cost limit | 1 |
| "If you have a particular type of data in mind (grades, sales), that would be helpful." | data type | 1 |
| "If you have more details, I can give more targeted advice!" | any | 0 |
| "Step 2: Define your budget. What can you afford?" (list step, no request to the assistant) | budget | 0 |
| "What is your goal, and what is your budget?" | goal / budget | 1 / 1 |
| "What framework are you using?" | skill level | 0 |

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
1. (guide-1 plan, superseded for this round) two-rater calibration was NOT run; the pilot was scored by ONE automated Claude session.
2. guide-2 is frozen for the formal run. The pilot rows were re-scored under guide-2 (old ratings kept, change log saved).
3. No inter-rater agreement / kappa / consensus is reported while there is a single scorer.
4. Formal rating uses guide-2 unchanged. A later revision keeps the old version and re-rates consistently.

Scoring is blind, per (reply, item), by a reader who actually reads each reply. Scripts and keyword matches are not a substitute.
Current round: ONE automated Claude scoring session (model id recorded, not human-verified); no second scorer, so no inter-rater
agreement/kappa/consensus is reported. Do not describe automated ratings as human ratings.

## Revision log
- r1 (pilot-time) → r2: items now carry a category; compound items were split; some items were demoted to
  unscored background; scenarios whose object cannot be resolved without inventing facts are flagged in the
  benchmark (`flags`). All r2 material edits are **post-pilot, pre-formal** (pilot outputs already existed).
  Basis per scenario: `infoquest_v1/r2_review.json`. The edits are an agent reading, not a human review.

## Where to edit scenarios (server workflow)
- `python prepare_infoquest_v1.py prepare` builds drafts in `components/benchmark/infoquest_v1_work/`
  (or `migrate` for an existing r1 work directory); edit `scenario_items_v1.json` there.
- Record the review explicitly: `review --confirm-reviewed --reviewer <name or model id> --reviewer-kind human|automated`; then `export` writes the single runtime file
  `components/benchmark/infoquest_v1.json`.
- This guide stays local; rating import takes `--guide` and `--guide_version`.
