# AI reader surface: measured findings, reading sheets and reviews

This page describes how an AI reader (Claude through the MCP server, or the Claude Code skills) reads the measured
findings of a study, checks them on images, and answers. It states the trust rule, the validation status a reader must
report, and a small pilot in which the AI read ten SPIDER discs blind. Research and education only. Nothing here is a
diagnosis.

## What is there

| piece | where | what it does |
|---|---|---|
| measured findings | `spinescope.ai.reading.measured_findings` | Every registered findings field per disc level (`analysis/findings.py`), with its confidence after the numbering cap, its validation status and the doc that states it |
| validation status | `spinescope.ai.reading.FIELD_STATUS` | One entry per field: `validated`, `partly_validated` or `experimental`, a note in words, and the `docs/analysis/` page with the numbers. A field not listed is experimental |
| reading sheets | `spinescope.render.reading_sheet.reading_sheet`, `ai.reading.render_sheet` | One PNG per measured field: up to three sequences side by side at the measured slice, mm rulers, a scale bar, the anchored disc line, and a banner with the level, the numbering status, the field, the value (unless blind) and the validation status |
| review tasks | `spinescope.ai.briefs.reading_tasks(..., measured=...)` | One task per disc level: confirm or reject each measured value on its sheet |
| review answers | `ai.reading.review_errors`, `review_to_findings` | A review is validated against the measured fields and the findings schema, then converted to a findings file, so `report.merge` merges it as one reader |
| MCP | `list_findings`, `reading_sheet`, `submit_reader_answer`, `tasks` | The same, for the MCP client |
| CLI | `spinescope findings STUDY --outdir DIR [--blind]`, `spinescope review STUDY answer.json --outdir DIR`, `spinescope tasks` | The same, from the command line |
| skills | `plugins/claude-code/spinescope/skills/spine-findings`, `spine-read` | How to read the status, when to say "not assessable", the wording rules, and the review step |

Adding an analysis or a field needs no change here: a field registered in `analysis/findings.py` (or in an analysis
module) appears in `list_findings`, gets its sheet, and is reviewable. Its status is experimental until someone records
a validation in `FIELD_STATUS`.

## Trust rule

An AI review can only lower trust. This is the same rule as the labelling review (`docs/labeling/lab24_ai_verify.md`):

- A **confirm** keeps the measured value and never raises its confidence. The converted confidence is the lower of the
  review's and the measured one.
- A **reject** lowers trust. With an alternative value (a value of the same field that the reader can read on the sheet)
  it is a disagreement that the merge counts. Without one, the value is indeterminate.
- An **unclear** verdict is indeterminate with low confidence.
- No AI verdict is a human confirmation. Every converted note says "not a human confirmation".
- A review does not change the numbering status. A review is refused while the numbering is `unconfirmed`, because it
  names levels, and the abstain policy allows no level name until the labels are verified.

## Validation status and what a reader writes

- `validated`: checked against radiologist gradings on public data, with the pre-registered rule met. The interval is
  still a constraint on a single value.
- `partly_validated`: some values of the field are supported and others are not (for example bulge, but not protrusion
  or extrusion). The note says which.
- `experimental`: not validated, or the rule was not met. It is a measurement to check on the image, never a finding.

The numbers stay in the analysis pages (`docs/analysis/*.md`), not here, because they change when the analyses change.

## Pilot: blind reads of ten SPIDER validation discs (n = 10)

This is a pilot, not a validation. Ten discs are too few for any interval to mean much. The grader was an AI vision
model (the agent that ran the pilot), reading the sheets with mostly low to medium confidence.

Protocol (fixed before any grade was looked at; the scripts were run outside the repository):

1. Validation cases in a seeded random order (seed 0). Per case, the candidates were the library discs matched to a
   truth disc of the same level name (the harness rule), with a Pfirrmann suggestion and a disc contour. One disc per
   case was drawn with seed 1. The first ten cases with a candidate form the pilot.
2. Blind sheets: the Pfirrmann sheet and the disc-contour sheet for each disc, without the measured value. The sheets
   show the level and its numbering status, as a reader would see them.
3. The AI graded Pfirrmann (1 to 5, on sagittal T2, against the standard five-grade description) and the bulge (yes or
   no: the disc contour extends beyond the vertebral margins). The grades were written to a file before the key was
   opened.
4. The radiologist grades (`radiological_gradings.csv`) and the library values were compared after that.

Library rules, as registered: Pfirrmann is the `discs` suggestion; bulge is the `herniation` predictor (excursion of
at least 1 mm after the baseline), the same rule as the SPIDER bulging predictor.

| disc | level | numbering | radiologist Pfirrmann | AI Pfirrmann | library Pfirrmann | radiologist bulge | AI bulge | library bulge |
|---|---|---|---|---|---|---|---|---|
| D01 | L2-L3 | unconfirmed | 4 | 3 | 4 | yes | no | yes |
| D02 | L5-S1 | confirmed | 3 | 4 | 2 | no | no | yes |
| D03 | L1-L2 | confirmed | 1 | 2 | 2 | no | no | no |
| D04 | L3-L4 | confirmed | 3 | 4 | 2 | yes | no | yes |
| D05 | L1-L2 | probable | 3 | 4 | 2 | yes | no | no |
| D06 | L4-L5 | unconfirmed | 3 | 3 | 3 | yes | no | no |
| D07 | L4-L5 | probable | 3 | 4 | 3 | yes | no | yes |
| D08 | L4-L5 | unconfirmed | 3 | 3 | 4 | yes | no | no |
| D09 | L4-L5 | unconfirmed | 5 | 4 | 4 | yes | no | yes |
| D10 | L2-L3 | confirmed | 1 | 2 | 1 | no | no | no |

Summary (n = 10; the Pfirrmann summary uses the ten discs, the bulge summary the seven radiologist-positive and three
negative discs):

| measure | AI reader (blind) | library |
|---|---|---|
| Pfirrmann exact agreement | 2 / 10 | 4 / 10 |
| Pfirrmann within one grade | 10 / 10 | 10 / 10 |
| Pfirrmann mean absolute error (grades) | 0.80 | 0.60 |
| Pfirrmann bias (AI or library minus radiologist) | +0.40 | -0.20 |
| Pfirrmann quadratic weighted kappa (not meaningful at n = 10) | 0.61 | 0.74 |
| bulge sensitivity (7 positive) | 0 / 7 | 4 / 7 |
| bulge specificity (3 negative) | 3 / 3 | 2 / 3 |

Reading of the pilot. On these sheets the AI reader's Pfirrmann grades were within one grade of the radiologist in
all ten cases, but it graded high on average and matched exactly less often than the library. For the bulge it called
every disc normal, which misses all seven radiologist-positive discs. So at this sheet size, with these crops, a
model's visual bulge call is not usable, and the library's contour is the better of the two. Whether a larger, zoomed
bulge sheet changes that is a question for the next wave, which has to decide on a learned model or a vision model
with a validation set of hundreds of discs. Ten discs cannot decide it.

## Limitations

- The sheets are sagittal crops of 60 mm square at the measured slice. They show no axial plane, so a field that needs
  an axial view (foramina, lateral recess) has no sheet until one is added.
- The disc line is the anchored endplate line, not a measurement of the field. A reader checks the number against the
  sheet; the sheet does not compute anything.
- Review readers see the measured value, so they are not blind. A reject is more informative than a confirm for that
  reason. A blind read of the level is the level task.
- Review tasks are not generated while the numbering is unconfirmed, so a study with unverified labels gets positions
  only (see `spine-read` step 2a).
- The status table (`FIELD_STATUS`) is maintained here by hand. Fields registered by analysis modules are experimental
  until an entry is added. The entries for herniation and alignment belong with those modules and should move there when
  those modules are next changed.
- The pilot is not a validation and must not be cited as an accuracy figure.
