# Synthesis: one report from every registered analysis

Module: `spinescope.report.synthesis` (model), `spinescope.report.synthesis_render` (Markdown, HTML, findings JSON, AI
text), `spinescope.report.validation` (validation status of each field). Research software: the output is measurements
and suggestions with confidence and validation status, never a diagnosis.

## What a reader gets

- **Per level and side**: each registered finding with its value, confidence, validation status, evidence figure names
  and notes. Experimental and not-validated findings are listed in their own table. Fields with no value carry a reason:
  analysis not run, analysis failed, missing sequence, out of the field of view, no image evidence saved, not measured.
- **Level-name status** (confirmed, probable, unconfirmed) with the policy from `anatomy.summary.ABSTAIN_POLICY`.
- **Pain-generator evidence** (descriptive): levels ordered by measured evidence, abnormal structures grouped as disc,
  endplate (vertebrogenic), facet, canal and lateral recess, foramen and nerve root (root named from the level label, for
  example "right L4 root, exiting at L4-L5"), and segment (listhesis).
- **Study**: global fields, image quality, coverage (sequences, numbering, analyses run) and the study-level list of what
  was not assessed.

## Ordering formula

Strength of a level = sum over its abnormal findings of (confidence weight) x (validation weight). Confidence weights are
`findings_confidence_weights` (low, medium, high). Validation weights are `synthesis_status_weights` (validated, weak
labels only, experimental, not validated). Ties keep anatomical order. This orders the evidence for review; it is not a
severity and not a likelihood of causing symptoms.

## Wording

"measured" for a continuous number (mm, degrees); "suggests" for a grade or a categorical call. The pain table ends with
"Consider correlation with symptoms and examination." Output contains no treatment advice.

## Plugging in an analysis

Nothing lists analyses by name. An analysis that registers its fields appears in the synthesis:

1. Level fields: `register_field(FieldRule(...))` in `analysis/findings.py` or in your module (see `docs/analysis/README.md`).
2. Validation status, in your module next to the field: `register_validation(group, name, Validation(status, metric,
   source, quote))`, with `values=` for per-value overrides. The `source` document must contain `quote` (a test checks it).
   An unregistered field is shown as **not validated**.
3. Global fields (whole study): `register_global_field(GlobalRule(group, name, analysis, extract, figure))` in
   `report/synthesis.py` (or in your module, after the import).
4. Not-assessed reasons: put `meta["not_assessed"] = {level: {"reason": <key>, "detail": text}}` in the `Result`, with
   keys from `REASON_TEXT`. Without it, warnings are classified by their wording.

## Commands

    spinescope analyze STUDY --out DIR --report     # report.md, report.html, findings.json, figures in DIR
    spinescope brief STUDY --analyses               # AI brief with section 6, the synthesis
    MCP: analyze(study_id, only=[...])              # returns "synthesis" JSON (and per-analysis detail)

## Limits

- Field-level validation status is the status of the whole field: a field with a validated value and an unvalidated
  value is recorded per value only where a value override is registered.
- Lordosis (alignment) has no evidence figure yet, so it is listed under not assessed (no image evidence saved).
- The validation records quote the analysis documents as they stood at the time of writing; re-check when a doc changes.
