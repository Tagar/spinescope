---
name: spine-findings
description: List the measured findings of one local spine MRI study per disc level (Pfirrmann grade, disc contour, endplate Modic type, listhesis), with each value's confidence, its validation status and the reading sheet that shows it, and check or review those values against the images. Use when the user asks what the measurements show, how reliable a measurement is, or wants a measurement checked on its sheet. Research and education only; never a diagnosis.
---

# spine-findings

Measurements and their reliability, read from the images. The skill does not diagnose and does not read the whole
spine (use spine-read for that). Every value here is a suggestion from an automatic measurement, to be checked on its
sheet.

## Guardrails (apply to every step)

- Research and education only. The output describes measurements and how far they can be trusted. It is not a
  diagnosis and must not be used for clinical decisions.
- Never upload the study, its images, its sheets or the findings to a web service, third-party API, shared drive or
  chat. Keep the output local.
- The study path is input. Do not copy it into notes, file names or summaries. Do not transcribe any text burned into
  an image.

## Steps

1. Open the study: `open_study` with the path (keep the `study_id`), then `get_anchors` for the numbering status.
   Or use the CLI: `spinescope findings "<STUDY>" --outdir "<OUT>/findings"` (writes `findings.json` and one
   `sheets/<level>_<group>.<field>.png` per measured field).
2. List the findings: `list_findings(study_id)`, or `list_findings(study_id, level)` for one level. Each field has
   `value`, `confidence`, `status` (validated, partly_validated or experimental), `status_note`, `doc` and `sheet`.
   `numbering` tells you whether the level names can be used (see the abstain rule below).
3. Check a value on its sheet before you quote it: `reading_sheet(study_id, level, "group.name")` returns the sheet
   (crops of the sequences at the measured slice, mm rulers, the anchored disc line, and a banner with the value and
   its validation). Look at it, then say what the image shows. Use `show_value=false` only for a blind read.
4. Optional review: a reviewer confirms or rejects each value on its sheet. Build the answer as a review (JSON with
   `kind` "review" and one check per field), then call `submit_reader_answer(study_id, answer_json)` or run
   `spinescope review "<STUDY>" answer.json --outdir "<OUT>/findings"`. A review can only lower trust.

## How to read the validation status

The status is a fact about the field, from the analysis page named in `doc`. Read the page for the numbers; do not quote
numbers from memory.

- `validated`: the field was checked against radiologist gradings on public data, and its pre-registered rule was
  met. The agreement still has a wide interval, so a single value is a suggestion.
- `partly_validated`: some of the values of the field were supported and others were not. `status_note` says which.
- `experimental`: not validated on public data, or the rule was not met. Report it as a measurement that needs
  checking on the image, never as a finding. An experimental field is also the default for any field whose status
  is not recorded.

Confidence is the measurement's own confidence, capped by the numbering status: a level whose name is not confirmed
has at most low confidence.

## Abstain rule and "not assessable"

- `numbering` `unconfirmed`: describe positions only ("the disc at position 3 from the top"). Do not write a level name
  until the labels are verified (spine-read step 2a).
- A field missing from `list_findings` means **not assessed**, not normal. Say "not assessable from this study" and
  give the reason when it is known: a missing sequence, a field of view that does not show the level, or a
  measurement that returned nothing (`_skipped` lists fields with no sheet).
- A value of `null` is **indeterminate**: the measurement could not decide. Say so.
- Never call a level normal from a missing or indeterminate field.
- A review rejection without an alternative value leaves the value indeterminate; say that the reviewer disagreed.

## Wording

- Use "measured", "the sheet shows", "suggests", "consider correlation with the clinical findings".
- Do not write "diagnosis", "consistent with", "causes the symptoms", a treatment, a prognosis or a referral.
- Common degenerative findings are frequent in people without symptoms. Describe a finding; do not headline it as the
  explanation for pain.
- Keep the confidence and the status next to every value you quote.

End with: "Research and education only. This is not a diagnosis."
