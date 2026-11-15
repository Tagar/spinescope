# Analyses: how to add one, how to validate it

An analysis measures something on the labelled levels of one study (disc height, endplate signal, canal width, ...).
It never diagnoses. Its output is rows (measurements), flags (notable items with a confidence), warnings (missing
inputs) and figures (evidence images). The validation harness scores those outputs against radiologist gradings
and the report wiring puts them into the findings schema.

## The contract

`spinescope.analysis.base`:

```python
@register("my_analysis")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result: ...
```

- `Result(name, rows, flags, warnings, figures, meta)`; `Result.save(outdir)` writes `<name>.csv`, `<name>.json` and
  one PNG per figure. `meta` is PHI-free and must not contain `_`-prefixed keys (they are never serialised).
- Return warnings, not exceptions, when a sequence or anatomy is missing. `run_all` contains exceptions anyway.
- Every threshold comes from `spinescope/config.py`, in mm or relative units, with a source comment (literature,
  anatomy, or SPIDER training; never a validation set or one image). Prefix new keys with the module name
  (for example `canal_ap_stenosis_mm`).
- Keep the module deterministic (no unseeded randomness; sort anything that is iterated over).

## Steps

1. **Module.** Write `src/spinescope/analysis/<name>.py` with the function above. Add `"<name>"` to
   `_BUILTIN_MODULES` in `analysis/base.py` so `available()` finds it (the list is explicit on purpose).
2. **Tests.** `tests/test_analysis_<name>.py`, phantom-based. Vary the phantom (lordosis, noise, missing STIR,
   cropped field of view) and check the warnings path.
3. **Report fields.** In `analysis/findings.py` (or your module), register each field that should appear in a
   structured read:

   ```python
   from spinescope.analysis.findings import Candidate, FieldRule, register_field


   def _extract(res: Result, level: str) -> Candidate | None:
       for r in res.rows:
           if r.get("scope") == "disc" and r.get("disc") == level:
               return Candidate(value=r["grade"], confidence=r["conf"], notes="...", slices=())
       return None


   register_field(FieldRule("disc", "my_field", "my_analysis", _extract, figure="my_strip"))
   ```

   `group` and `name` must be a level field of `data/findings.schema.json` (checked at registration). A field with
   no saved image is skipped, not written: the schema requires at least one image of evidence per field. A level
   whose name is unconfirmed is capped at medium/low confidence automatically.
4. **Validation.** Register a predictor for each SPIDER truth field the analysis can predict (see below), then run
   the harness. Record the rule in your doc before you look at validation numbers.
5. **Doc.** `docs/analysis/<name>.md`: method, literature for every threshold, validation numbers (from
   `docs/analysis/baseline.md` style tables), limitations.

## Validating an analysis against SPIDER gradings

The harness is `tools/eval_analysis.py` (library: `spinescope.analysis.evaluate`).

```sh
.venv/bin/python tools/eval_analysis.py --dataset spider --split validation \
    --analyses discs,endplates --workers 4 --out /tmp/spinescope-<name>/val --pairs
```

Pipeline per patient: load the study (all sequences by default), `label_spine`, run the requested analyses, match
each predicted disc to the nearest reference disc (within 12 mm in the sagittal (y, z) plane, one to one), and score
a disc **only if its predicted level name equals the reference name** at that position. Name disagreements are
dropped and counted as `label_mismatch`; nothing is dropped silently. The report also counts `no_reference`
(predicted disc with no reference disc nearby), `missing_pred` (reference disc not predicted) and `not_assessed`
(analysis gave no value).

Predictors connect an analysis to a truth field, in `analysis/_evaluate_builtin.py` (or your module, if it is
imported before the harness runs):

```python
register_predictor("my_analysis", "disc_narrowing", fn)  # fn(result, anchors) -> {level: (value, score)}
```

`value` is None when the analysis does not assess that level. `score` (optional, higher = more abnormal) enables
AUC.

### Truth fields available (SPIDER)

| field | kind | codes (count in release) | semantics |
|---|---|---|---|
| `pfirrmann` | ordinal 1-5 | 1 (286), 2 (341), 3 (418), 4 (291), 5 (184) | Pfirrmann grade; scale per SPIDER paper |
| `modic` | categorical 0-3 | 0 (1006), 1 (4), 2 (503), 3 (7) | Modic type per disc; exact coding not verified here |
| `up_endplate`, `low_endplate` | binary | 0/1 | endplate code; semantics not verified here |
| `disc_narrowing` | binary | 0 (976), 1 (544) | narrowing present (0/1) |
| `disc_herniation` | binary | 0 (1448), 1 (72) | herniation present |
| `disc_bulging` | binary | 0 (774), 1 (746) | bulge present |
| `spondylolisthesis` | binary | 0 (1478), 1 (42) | listhesis present |

The SPIDER paper defines the codes. The publisher page could not be read while this was written, and the local
dataset folder has no codebook (its README describes files only), so **every code meaning here is unverified**.
The value ranges above are the observed ones; check them against the paper before relying on a code's meaning. Disc levels
follow the dataset convention: IVD 1 is the lowest disc, L5-S1 (`spinescope.datasets.gradings`).

### Predictors already registered

Rules were fixed before any validation number was produced for them:

- `pfirrmann` from `discs`: the disc's `pfirrmann_suggestion`.
- `disc_narrowing` from `discs`: positive when the disc has the existing `low_disc_height` flag (height ratio < 0.6,
  the threshold already in `discs.py`); score = minus the height ratio (for AUC).
- `modic` from `endplates`: highest Modic type over the two endplates (`none` = 0, `Modic 1` and `Modic 1-like` = 1,
  `Modic 2` = 2, `Modic 3` = 3); 0 only when both endplates are known normal.
- `up_endplate` / `low_endplate` from `endplates`: 1 for any Modic-like pattern, 0 for `none`. A proxy: SPIDER's
  endplate codes are not defined by a Modic type.
- `disc_herniation`, `disc_bulging`, `spondylolisthesis`: no analysis predicts them yet, so they are reported as not
  assessed, with counts.

## Metric definitions

Per field, over **scored** pairs only (a disc matched to a reference disc of the same level name, with both a
prediction and a grade):

- **binary**: sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); PPV = TP/(TP+FP); NPV = TN/(TN+FN); prevalence;
  AUC by the rank-sum statistic (ties count half) when a score exists.
- **ordinal**: quadratic-weighted kappa (weights (i-j)^2/(k-1)^2) and linear-weighted kappa; MAE; accuracy; share
  within one grade; confusion matrix (rows reference, columns prediction); one-vs-rest sensitivity, specificity and
  PPV per grade.
- **categorical** (Modic 0-3): accuracy; unweighted kappa; one-vs-rest per class; "any positive" sensitivity and
  specificity (Modic >= 1 against 0).
- **continuous**: MAE, RMSE, bias.
- **Intervals**: 95% percentile bootstrap, 1000 resamples, seed 0, resampling **patients** (discs of one patient are
  not independent). Only headline metrics have intervals.
- **Breakdowns** (point estimates): vendor, field strength, sequence set, transitional flag (see caveat in
  `gradings.py`), reference contrast.
- **Coverage**: predicted discs, reference discs, matched, scored, label mismatch (dropped), no reference (dropped),
  reference not predicted.

Kappa and similar agreement statistics are reported next to published inter-reader values where a source is cited;
none are quoted in these docs yet.

## Reports

`spinescope.analysis.findings.level_findings(results, level, anchors, figures=...)` returns the schema's per-level
object: `{group: {name: {value, confidence, evidence[], notes}}}`. Confidence is the analysis confidence mapped to
low/medium/high, then capped by the level-name status (confirmed: no cap; probable: medium; unconfirmed: low), with a
note saying the name is not confirmed. Fields without image evidence appear under `_skipped`.

## Free-text reports

`spinescope.datasets.report_labels` parses radiologist free text into per-level labels (rule-based; see its module
docstring for the rules and negation handling). `parse_report`, `aggregate_levels`, `coverage_summary`. Only derived
labels are returned; unparsed clauses are kept as generalised templates (levels and numbers replaced), never raw text.

## Commands

```sh
spinescope analyze <study> [--only discs,endplates] [--out DIR]    # JSON to stdout; CSV/JSON/PNG to DIR
# MCP: analyze(study_id, only=[...]) returns rows, flags, warnings, numbering and saved file paths
.venv/bin/python tools/eval_analysis.py --dataset spider --split validation --analyses all --out DIR
```

## Limitations of the harness

- Scores depend on the labeller: a disc placed at the wrong level is dropped, so a poor labeller shows as
  **fewer** scored discs, not as bad metrics. The coverage table is the first thing to read.
- Ground truth for SPIDER is a single radiologist grading per disc (no inter-reader data here).
- Multiprocessing is used by `evaluate(workers>1)`; the worker processes import the built-in SPIDER loader, so
  custom loaders must be registered at import time of a module the workers can import.
