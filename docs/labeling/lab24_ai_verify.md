# lab24: can an AI reader catch the remaining numbering errors on the current labeller?

Status: in progress. This file is written in two stages. Stage 1 (below) is the pre-registration: the blind answers
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
- Key file (truth) `/tmp/spinescope-lab24_ai_verify/key/key.json` was built in the same step as the packets and was
  not opened by the reader before the answer file was complete and hashed.
- sha256 of `answers.json`: `633d934c8ca3d734901e629c214011ec0a1088c6a2d0e55b4db7e4fdcd42acd8`
- Answer counts: `L5-S1` 156, `unsure` 40, `L4-L5` 0, `L5-L6` 0. No `missing` and no `extra` were given.
- Revisions made before the hash, from the packet only (CNN line and images), and listed here so they are visible:
  c066 and c073 were first `L5-S1`, changed to `unsure` after the CNN scored the labelled lowest disc at about
  0.00 for L5-S1 while the image was not clear. c072 was skipped in the first pass; it was answered last, from its
  own overview. c196 was also answered in the last pass.

## Stage 2: results

(appended after unblinding)
