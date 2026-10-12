# infoquest-v1 scoring guide (version `infoquest-v1-scoring-guide-1`)

Raters see, per reply: the exact message the model received, the reply, and for each
**target information item** its category, the reference fact, why it matters and example
questions. They do **not** see α, the model's run files or any condition label.
Score every (reply, item) row in `blind_template.csv`; leave a cell **blank** if you cannot
decide (blank = missing, it is never counted as 0). Put the supporting quote in `evidence_note`.

## `asked` (1/0)
1 if the reply actually **asks the user** about this item (a direct question, or an
explicit request such as "tell me your budget"). Asking may sit next to an answer —
advice + question still counts.
- Count: any wording that would obtain the item's information, even if it does not guess
  the hidden fact ("Are there any physical limitations?" covers a knee injury).
- Do **not** count: generic offers ("Let me know if you need more help", "Need other
  assistance?"); rhetorical statements; conditional advice that never asks
  ("If your budget is small, do X" is advice, not a question — unless it also asks).
- A question about a *different* matter than the item is not `asked` for this item,
  and is not an error either (the reply is not penalised for extra questions).
- Count/length of question marks and reply length are irrelevant.

## `unsupported_assertion` (1/0)
1 if the reply states the item's content **as fact** when the message did not provide it
("Given your knee injury…", "Since you only have a $200 budget…").
- Conditional or hypothetical wording is **not** an assertion ("If you have a knee problem…",
  "Assuming a small budget…", "For example, if…").
- If the message itself already states the fact, repeating it is **not** unsupported.
- Presenting several scenarios without claiming which applies is not an assertion.

## Edge cases
- Cut-off replies: score what is present; the cut-off flag is shown for information only.
- Empty reply: `asked=0`, `unsupported_assertion=0`.
- Reply refuses or deflects without asking: `asked=0`.

## Procedure
1. PILOT only: two raters independently score the pilot export (or a subset); run
   `infoquest_blind.py agreement`, discuss each disagreement, and revise this guide.
2. Freeze the guide before formal rating: save the exact file used and pass it when importing
   each rater's CSV (`infoquest_blind.py import --guide SCORING_GUIDE.md --guide_version <label>`).
   Ratings files store its sha256; merge/agreement refuse files made under different guides.
3. Formal: rate with the frozen guide. If it is revised later, keep the old version and
   re-rate all affected replies consistently.

## Compound items (rule added before any model output)
Each item is scored against its `information_need` label only. If the reference fact bundles
two things (e.g. "room size" + "what it must hold"), `asked=1` requires a question that covers
the **labelled need**; asking only about the bundled extra counts as `asked=0`. Items that
mix two independent needs should be split during scenario review (before freeze), not here.
Requests the model cannot perform (e.g. "send me the file") or with unresolved names are
reviewed before freeze; any change is made before model output and logged below.

### Review log (pre-model changes)
- (none yet)

### Where to edit scenarios (server workflow)
- `python prepare_infoquest_v1.py prepare` builds drafts in `components/benchmark/infoquest_v1_work/`.
- Edit `scenario_items_v1.json` there (repeat `prepare` never overwrites it; `scenarios_v1.json` is derived).
- Sign off explicitly: `review --confirm-reviewed --reviewer NAME`; then `export` writes the single runtime file
  `components/benchmark/infoquest_v1.json`. Changing the shipped `items_src_part*.txt` after a draft exists is an error
  by design (move the work file aside to re-derive).
- This guide stays local; rating import takes `--guide` and `--guide_version`.
