# Label verification with a reader (lab06_verify)

Status: the loop is built and tested on the phantom. The empirical test ran on 20 SPIDER validation volumes.
It did not improve label accuracy in practice. Nothing was tuned on the validation set, and no labeller
defaults were changed.

## What was built

- `anatomy/anchors.py` (additive): `shift_labels(offset)`, `relabel(mapping)`, `drop_disc(label)`,
  `add_disc(label, center, ...)`, `record_correction(...)`. Every change appends
  `numbering['corrections']` with who, source (`programmatic`, `ai_reader`, `human`), why (200 chars max),
  detail and the labels after the change. Confidence is set by the review (see config below). Shifts are
  checked before any change, so a failing call leaves the anchors as they were.
- `anatomy/verify.py`: `build_packet(study, anchors)` (labelled mid-sagittal T2 and T1 via `qa.overlay`, the
  localizer with predicted levels when present, a sacrum zoom at the mid-sagittal slice),
  `parse_answer` (strict JSON validation, fenced JSON accepted), `apply_answer` (atomic: applied to a copy),
  `verify_labels(study, anchors, ask)` for any callable reader. No network code and no new dependency. The
  module docstring has the Anthropic wiring as a snippet.
- MCP (`ai/mcp_server.py`, additive): `verify_labels(study_id)` returns the packet images with the question;
  `apply_label_verification(study_id, answer_json, who)` (the study id is needed because sessions are keyed by
  it); `open_study(path, anchors_path=None)` opens a session with already-verified anchors, so subagents see
  the corrected levels.
- CLI (`cli.py`, additive): `spinescope verify-labels PATH --outdir DIR` writes the packet;
  `--answer answer.json [--out anchors.json]` applies an answer. `brief`, `tasks` and `viewer` accept
  `--anchors` so prompts built after verification use the corrected levels (without it the CLI would label
  again and silently revert them).
- Skill `plugins/claude-code/spinescope/skills/spine-read/SKILL.md`: new step 2a, labels verified before
  fan-out. It also says that a body below a label does not prove the label is a disc.
- Config (`config.py`, additive): `review_confidence_ai` 0.85, `review_confidence_human` 0.95,
  `review_unsure_confidence` 0.3, `review_unplaced_penalty` 0.1, `placed_disc_confidence` 0.3,
  `verify_sacrum_offset_mm` 30, `verify_sacrum_fov_mm` 100. These are judgement defaults, not calibrated.

Answer convention (the question asks for it): `extra` lists labels as displayed that are not discs.
`lowest_disc_is` is the true name of the lowest disc shown after the extras are removed (`L5-S1`: no change;
`L4-L5`: every level moves up one; `L5-L6`: only the lowest disc is renamed, a lumbarized S1; `unsure`: no
structural change). `missing` lists true names of unlabelled discs, placed between their labelled neighbours
or extrapolated from the spacing, with confidence 0.3 and a note that the position was not measured.

Interface deviations from the brief: `apply_label_verification` takes `study_id` first, and `who` is optional.
`open_study` takes an optional `anchors_path`. Both are needed because the MCP server keys sessions by
`study_id`.

## Experiment: the AI reader on 20 SPIDER validation volumes

Protocol.
- Sample: the 20 validation volumes evenly spaced over the sorted validation list (the lab01 subsample).
  Where a volume had a second sequence for the same patient, it was added to the study, so the packet had
  T2 and T1 overlays. The labeller then used its default series: 18 mid-sagittal T2, 1 T1 (single series),
  1 second T2 series of 120 slices.
- Blind packets were built with case ids (`case_01`..`case_20`) and no truth or file names. The reader (the
  agent that wrote this code) answered from the PNGs only, using the packet question and the rule "name the
  lowest disc only when the slice is clearly mid-sagittal and a body is visible below the label". The
  answers were fixed before any scoring. No answer used `missing` or `extra`.
- Scoring: the shared scorer (`_validate.score_anchors`, `aggregate`, `aggregate_extended`), with truth from
  the SPIDER mask of the series the labeller used.

Results (20 volumes, 142 truth discs, 155 predicted discs). Reader answers: `L5-S1` 9, `unsure` 11.

| metric | labeller | after the reader's answers | ceiling (truth-derived answers) |
|---|---|---|---|
| disc recall | 12.0% | 12.0% | 12.0% |
| disc precision | 11.0% | 11.0% | 11.0% |
| label accuracy, matched discs | 0.0% (0 of 17) | 0.0% | 64.7% (11 of 17) |
| label accuracy, all truth discs | 0.0% | 0.0% | 7.7% (11 of 142) |
| off-by-one level, all truth discs | 7.7% | 7.7% | 0.0% |
| volumes with every truth disc labelled right | 0% | 0% | 5% (1 of 20) |
| centre error, median / p90 (mm) | 6.8 / 9.6 | 6.8 / 9.6 | 6.8 / 9.6 |
| false positives per volume | 6.9 | 6.9 | 6.9 |
| midline error, median / p90 (mm) | 22.0 / 29.7 | same | same |

- The reader's answers changed nothing: no answer shifted or dropped a disc. The loop gave no benefit.
- The reader's "L5-S1" answers were wrong: 0 of 9 agree with the positional truth that the labelled bottom disc
  is the lowest true disc. In case 09 the labelled L5-S1 has no true disc behind it (a false positive), and the
  true L5-S1 carries the labeller's L4-L5. A body below the label looked like proof, and it was not. The skill now says so.
- The true lowest disc is matched by any labelled disc in only 2 of 20 volumes. The nearest labelled disc is
  typically 20 to 70 mm away. Naming cannot correct a disc that is not in the right place.
- Ceiling (not a reader result): answers derived from ground truth through the same API. Matched-disc label
  accuracy rises from 0% to 65%, but only 2 volumes had a non-`unsure` truth-derived answer (both `L5-S1`), and only 3
  discs were placed (median centre error 33 mm). Label accuracy is limited by localisation, not by naming.

Why the reader abstained (11 of 20). Off the spine or parasagittal slice: cases 03, 05, 12, 13, 16, 18.
No sacrum in view, or bowel or muscle where the sacrum should be: cases 06, 08, 17, 19, 20. The lumbar
field of view often ends at L5-S1, so the sacrum is not in the picture at all.

## Findings for the labeller and the loop

1. The midline is the dominant failure on these volumes. The median midline error is 22 mm (16 of 20 over
   10 mm), against 11.7 mm in the lab01 baseline over all validation volumes. The overlay shows the wrong
   slice, which the reader can see. The packet has no way to ask for a different midline yet.
2. The loop cannot repair positions, and its ceiling is set by localisation. Fix localisation before counting.
3. A reader's confirmation of the lowest disc needs a check that the labelled disc is a disc: signal between the
   label and the body below, not a body alone. Without that check, reader confirmations are not evidence.
4. `num_vertebrae` cannot identify transitional anatomy. In `overview.csv` the validation split has 6 to 9
   vertebrae in 86 of 87 volumes, and 3 in one; it never equals 5. The brief's suggested
   `num_vertebrae != 5` split therefore flags everything. The transitional flag needs an anatomical cue.
5. Packet bugs found and fixed during the run: the sacrum zoom was too small (now about 600 px), and the zoom
   could use a different slice from the overlay (now the same mid-sagittal slice). Two packets were re-read.
6. A harness mistake was found and fixed before the numbers above were written. The first run scored 10 cases
   (T1 primary) against the mask of the other series, and it also used the wrong index for the lowest
   truth disc. Paired and single-file labelling were checked per case: identical except case 13, which is a 120-slice series.

## Disclosure

- The build step printed the truth disc count for each case before the blind answers were written. Counts
  only, no names or positions. It could have influenced the reader, but no answer used `missing` or `extra`.
- The reader is the same agent that wrote the code. The numbers are for this reader on these packets, not a
  general estimate of AI reading accuracy.
- Scratch packets, answers and the private key stay under `/tmp/spinescope-lab06_verify/` and are not
  committed. No image or file name from the dataset is in the repository.

## Recommendations (not implemented)

- Localisation first: midline and chain errors dominate. A reader cannot correct a misplaced disc.
- Add a packet answer that rejects the slice (`slice_is_midline: false`) and triggers a midline re-pick.
- Before trusting any reader's lowest-disc answer, measure its agreement with truth on a labelled set.
- Derive the transitional flag from an anatomical cue, not from `num_vertebrae`.

Reproduce: `spinescope verify-labels <study> --outdir <dir>` writes the packet. Score answers with
`anatomy/verify.apply_answer` and `anatomy/_validate.score_anchors`. The 20-volume selection is the
`_subsample(spider_cases(root, "validation"), 20)` of `tools/eval_labeling.py`.
