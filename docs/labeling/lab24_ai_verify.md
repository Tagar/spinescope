# lab24: can an AI reader catch the remaining numbering errors on the current labeller?

Status: done (Stage 2 below). This file was written in two stages. Stage 1 (below) is the pre-registration: the blind answers
were written and their hash was committed before the key was opened. Stage 2 (results) is appended after
unblinding. Single AI reader: one agent (the author of this branch) read the packets from the PNG images and
answered the structured question. This is not an estimate of AI reading accuracy in general.

## Question

With the current default (learned) labeller, can an AI reader looking at the verification packet
(`anatomy/verify.py`, MCP `verify_labels`) catch the wrong level names and avoid "correcting" right ones? lab06 used the
old classical labeller, whose discs were mostly misplaced. Positions are now good, so naming is the question.

## Case set (built before any answer was written)

Labeller: `label_spine(study)` (default, learned U-Net, lumbosacral CNN on) on `main` 5195ff5 plus this branch's
packet changes. The numbering was re-collected for every study with `tools/calibrate_labeling.py collect` on this
code (not reused from lab18), with the shared scorer (`_validate.score_anchors`), 12 mm match, and the wrong/right
definition of the collector: a study is right when it has at least one matched disc and every matched disc carries
the true name.

| set | labelled studies run | wrong numbering | right |
|---|---|---|---|
| SPIDER validation | 87 (all) | 2 | 85 |
| qx5rt (CC BY-NC, 4 annotated levels per session) | 55 (all) | 8 | 47 |
| Sudirman | 318 of 515 labelled patients | 39 | 279 |
| total | 460 | 49 | 411 |

Coverage limit: the Sudirman run covers 318 of the 515 labelled patients (the lab18 run's 240 evenly spaced
patients, plus the first 147 patients of the current run, which was stopped for machine load). The 197 unrun
patients are not in the pool, so the wrong cases among them are not found. No cases were added after answering.

Sample: all 49 wrong cases, plus 147 correct cases drawn at random (`random.Random(24)`) from the 411 right studies,
which is exactly 3 per wrong case. Shuffled, relabelled `c001`..`c196`. Composition: SPIDER 2 wrong, qx5rt 8
wrong, Sudirman 39 wrong (the 49 wrong are all in the case set). Three qx5rt sessions (ID15, ID16, ID30) have a
byte-identical T1 file (same md5), so the same picture appears three times; they are kept (each has its own truth
masks) and their labels are identical (c030, c083, c094).

## Packets

`build_packet` now writes, for each case, the mid-sagittal T2 overlay, the mid-sagittal T1 overlay (when a second
sagittal series exists), the localizer with predicted levels (when present), the sacrum zoom 30 mm below the lowest
labelled disc, and a new `overview.png`: a contact sheet of those panels in two columns. The packet notes and
`packet.json` now also carry the lumbosacral CNN probability P(L5-S1) for the lowest candidates, bottom-up
(`lumbosacral_context`). The question text says the CNN is one cue, not a count, and can score a sacral remnant as
L5-S1. The packet holds the predicted levels and no truth, no file names, no patient attributes.

Reader rules (fixed before answering, as in the packet question): lowest_disc_is is the true name of the lowest
disc shown after removing extras; `L5-S1` if the sacrum is below it; `unsure` if the sacrum is not visible or the
count is doubtful. `extra` lists labels that are not discs; `missing` lists unlabelled discs. The reader answered
from the images (`overview.png`, and the single panel when the overview was not enough) and not from the notes alone.

## Stage 1: blind answers (pre-registration)

- Answer file: `/tmp/spinescope-lab24_ai_verify/answers/answers.json` (196 entries, schema-valid).
- Key file (truth) `/tmp/spinescope-lab24_ai_verify/key/key.json` was built in the same step as the packets. The
  reader did not open the truth fields before the answer file was complete and hashed. One disclosure: during the
  blind phase the reader read the key's set and study identifiers once, to check for duplicate studies (three qx5rt
  sessions with byte-identical T1 files). No truth coordinates, labels, or numbering results were printed.
- sha256 of `answers.json`: `633d934c8ca3d734901e629c214011ec0a1088c6a2d0e55b4db7e4fdcd42acd8`
- Answer counts: `L5-S1` 156, `unsure` 40, `L4-L5` 0, `L5-L6` 0. No `missing` and no `extra` were given.
- Revisions made before the hash, from the packet only (CNN line and images), and listed here so they are visible:
  c066 and c073 were first `L5-S1`, changed to `unsure` after the CNN scored the labelled lowest disc at about
  0.00 for L5-S1 while the image was not clear. c072 was skipped in the first pass; it was answered last, from its
  own overview. c196 was also answered in the last pass.

## Stage 2: results (after unblinding; the hash above matched the committed answer file)

Labeller: the labeller that was used to build the case set was `main` 5195ff5. Before scoring, the branch was rebased
onto the current `main` (dccbf9a, which includes the lab22 numbering changes and the lab23 and lab26 changes). The
default labeller was then re-run on all 196 blind studies with the rebased code (`label_spine`, default backend). Its
predicted levels were identical to the labels the reader saw in all 196 cases, so the answers apply to the current
default labeller. The numbering status is the current one (below). The wrong/right split of the 460-study pool was
computed on 5195ff5 and was not recomputed for the 264 studies outside this case set.

Policy as shipped: `apply_answer` with `source="ai_reader"`. The scorer is `_validate.score_anchors`, before and after
the answer. No answer was changed after unblinding.

### 1. Sensitivity, false corrections, label accuracy

| measure | result |
|---|---|
| wrong numberings in the set | 49 (SPIDER validation 2, qx5rt 8, Sudirman 39) |
| wrong numberings where the reader proposed a correction | 0 of 49 |
| wrong numberings flagged by the reader (`unsure`, so not confirmed) | 16 of 49 (33 %) |
| wrong numberings confirmed by the reader (`L5-S1`) | 33 of 49 (67 %) |
| correct studies where the reader proposed a correction | 0 of 147 |
| correct studies flagged (`unsure`) | 24 of 147 (16 %) |
| correct studies made worse by the applied answer | 0 of 147 |
| label accuracy, all 732 truth discs, before / after | 594 / 732 = 0.811 / 0.811 (net 0 labels) |

No `L4-L5`, `L5-L6`, `missing` or `extra` answer was given, so no label moved. Zero corrections is the reader's choice,
not a finding that all 49 were right: the reader never proposed a shift.

### 2. What the policy did to the numbering status (the important result)

Current default labeller, status before review and after the policy:

| group | before: confirmed / probable / unconfirmed | after: confirmed / probable / unconfirmed |
|---|---|---|
| wrong (49) | 0 / 7 / 42 | 0 / 29 / 20 |
| right (147) | 33 / 59 / 55 | 0 / 112 / 35 |

- Wrong numberings announced (confirmed or probable): 7 of 49 before, 29 of 49 after. That is 22 wrong numberings
  that the labeller had as `unconfirmed` and that an AI `L5-S1` answer made `probable`.
- Right numberings: 26 `confirmed` became `probable` (downgrade, from an `L5-S1` answer), and 33 `unconfirmed` became
  `probable` (upgrade). 7 `confirmed` became `unconfirmed` (from `unsure`).
- Mechanism: `apply_answer` writes the review confidence (0.85 for an AI reader, whatever the labeller had) into
  `numbering.confidence`, and `review_status` then gives `probable` when no applicability reason blocks it. The
  labeller's calibrated status is replaced, not combined with. So a reader that only says "L5-S1" lifts a study the
  labeller had as `unconfirmed` to `probable`. This is a defect in `apply_answer`, not in the reader. It was not
  changed here (it is not additive). It must be fixed before any AI answer is applied in production: a review should
  never raise confidence or status. An AI `L5-S1` should leave the labeller's status as it is.

### 3. Why the reader missed most wrong numberings

- The question asks about the lowest disc relative to the sacrum. For 15 of the 49 wrong studies the lowest predicted
  disc was matched and carried the right name (SPIDER validation 2, Sudirman 13). Their error is above the bottom: an
  extra or missing disc higher in the chain, which the question does not ask about (`extra` and `missing` were never
  used). The wrong studies have more predicted discs than truth discs in 48 of 49 cases (median +4).
- 33 wrong studies cannot be scored on the lowest disc: 8 qx5rt (the L5-S1 disc is not annotated there) and 25 Sudirman
  (the predicted lowest disc is more than 12 mm from any annotated disc). These look like positional errors at the
  bottom of the chain, which naming cannot correct (lab06 found the same).
- Of the 124 `L5-S1` answers whose lowest disc could be scored, 123 were right and 1 was wrong. The bottom call was almost
  always right. The reader did not see the errors that were elsewhere.

### 4. Flags versus the labeller's own status

The labeller's status already marks 97 studies `unconfirmed`, and 42 of them are wrong (43 %). The reader's `unsure`
flags mark 40 studies, 16 of them wrong (40 %). The reader's flag adds nothing the labeller's status does not already
give. Its confirmation is what changes the status, and that is the harmful part.

### 5. Packet and question changes (additive, in this branch)

- `overview.png`: one contact sheet with the T2 and T1 mid-sagittal overlays, the localizer and the sacrum zoom. The
  reader answered from the overview; individual panels were opened for 2 cases (c066, c073).
- Lumbosacral CNN probability for the lowest candidates in `packet.json` and in the notes. P(L5-S1) for the lowest
  candidate was at least 0.9 in 47 of 49 wrong studies and in 143 of 147 right ones, so the CNN did not separate them
  here. This matches lab18's finding that sacral remnants are scored as L5-S1.
- Question: a sentence that the CNN is one cue, not a count.
- Tests: `tests/test_labeling_verify.py` (`lumbosacral_context`, the overview sheet, packets with and without the CNN).

These changes did not change the reader's decisions measurably, because the reader could not separate the wrong from the
right studies on the bottom disc. They are harmless and make the packet self-contained.

## Answer to the question

With the current learned labeller, one AI reader looking at this packet did not catch the remaining numbering errors.
It flagged 16 of 49 wrong studies (33 %) and also flagged 16 % of right ones. It proposed no correction, so label
accuracy was unchanged (0.811). Under the current `apply_answer` policy, its `L5-S1` answers lifted 22 wrong numberings
from `unconfirmed` to `probable`, which is harmful. The labeller's own calibrated status is the better signal: 42 of
its 49 wrong-numbering studies in this set are already `unconfirmed`.

## Recommendations for the spine-read skill

1. Do not run AI verification always. Keep it optional, as a packet for a human reviewer or a tested reader.
2. Before any AI answer is applied, fix `apply_answer` so that a review can lower confidence or status and never raise
   them. An AI `L5-S1` should leave the labeller's status unchanged. Add a test for this.
3. If an AI reader is used, change the question to a per-disc count from the sacrum up over the whole chain, and use
   `extra` and `missing`. The current question addresses only the bottom disc, which is where this reader was right.
4. Positional errors need a localisation check first (the lab06 finding still holds).

## Limitations and disclosures

- Single AI reader: one agent, the author of this branch. It read the packets, and the packets show the labeller's own
  per-disc confidence values and the CNN line. It was not blind to those, and it is not an estimate of AI reading
  accuracy in general.
- Sudirman: 318 of 515 labelled patients were run (see the case-set section). The result covers this pool only.
- Labeller version: the case set was labelled on 5195ff5. The 196 cases were re-labelled on the rebased branch
  (`main` dccbf9a) and had identical predicted levels. The pool of 460 was not re-labelled.
- The set was built from the 460 studies. Three qx5rt sessions share one T1 file.
- Two answers were revised before the hash (c066, c073). c072 was skipped in the first pass and answered last, before
  the hash.
- The 39 Sudirman wrong cases dominate the set. Per-set rates are not comparable to the SPIDER validation numbers.
- A full test run on the first base (5195ff5) had 17 failures in `tests/robustness/test_robustness_realistic.py`
  (`test_no_silent_error`). They were labeller behaviour on the realistic phantom (main's source passes them, so they
  came from the older base, not from `verify.py`). The full run on the rebased branch (main dccbf9a plus this
  branch): 1356 passed, 1 skipped, 43 xfailed, 0 failed. ruff, ruff format and mypy are clean.
- Scratch packets, answers, the key and the collected records are under `/tmp/spinescope-lab24_ai_verify/` and are not
  committed. No image or file name from the datasets is in the repository.
