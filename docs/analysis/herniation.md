# Disc contour: bulge, protrusion, extrusion, and posterior excursion (sagittal) (herniation analysis)

Status (dx05): sagittal measurement implemented, calibrated on SPIDER training, validated on SPIDER validation and a weak-label
Sudirman check. The axial contour, side and the lateral zones are **not** measured in this wave (see Limitations). The
herniation predictor does not meet its acceptance rule on the SPIDER validation (see that section).

This page records the rules in the order they were fixed: method, nomenclature and thresholds, the calibration on SPIDER
training, the pre-registered decision rules, and then the validation numbers. No validation number was looked at before the
rules below were written.

## What is measured

Per disc (label from the labeller, `anchors.discs`), on every sagittal T2 (STIR fallback) slice within the disc's width:

- Disc frame and endplate traces from `analysis/discs.py` (`trace_disc`, the dark-line endplates and the thecal-sac (CSF)
  reference). Only discs with traced endplates or an anchored height are measured.
- Reference line: the posterior corners of the body above (inferior endplate, posterior wall corner) and the body below
  (superior endplate) from the labeller's body records (`Vertebra.body`), joined in the slice plane. A disc whose corners
  are more than 8 mm (AP) from its own posterior margin is not assessed.
- Posterior edge: for each height row, the search starts 3 mm anterior to the line (inside the posterior annulus) and runs
  posteriorly; the edge is where the signal reaches `annulus + f x (CSF - annulus)` (f = `herniation_edge_fraction`),
  interpolated on the 1 mm smoothed profile. A row needs two consecutive samples above the level.
- Excursion (per row) = edge - line, in mm, positive posterior. `e_raw` is its maximum over the rows.
- Baseline correction: `e_adj = e_raw - 2.77 mm` (`herniation_baseline_mm`, see the calibration). All classes use `e_adj`.
- Base: the height (mm) over which `e_adj` stays at or above the tolerance around its maximum.
- Primary slice: the measured slice whose plane is nearest the disc centre plane (the mid-sagittal slice).

## Nomenclature and thresholds (with sources)

Fardon DF, Williams AL, Dohring EJ, et al. Lumbar disc nomenclature version 2.0: recommendations of the combined ASSR/NASS
task force. Spine J 2014;14(11):2525-45 (NASS/ASSR/ASNR Lumbar Disc Nomenclature 2.0).

| term | rule used here | source of the number |
|---|---|---|
| normal | `e_adj` < 1.0 mm | tolerance: endplate and edge positions carry about 1 mm error (`analysis/discs.py` documents about a millimetre of endplate bias); engineering choice |
| bulge | 1.0 mm <= `e_adj` < 3.0 mm | Fardon 2014: a bulge extends less than 3 mm beyond the ring apophyses. A sagittal view cannot show the circumference (>25 %), so this is sagittal-only |
| protrusion | `e_adj` >= 3.0 mm and base >= `e_adj` | Fardon 2014: base wider than the displaced material's extent |
| extrusion | `e_adj` >= 3.0 mm and base < `e_adj` | Fardon 2014: displaced material wider than its base (migration not measured, so this is a lower bound) |
| sequestration | not called | needs a separate fragment |

Ring apophyses are not segmented: the body corners stand in for the apophyseal margin, so the 3 mm limit is approximate.

## Calibration on SPIDER training (before validation)

Data: the 179 SPIDER training cases (`subset == training`), labelled with `label_spine(backend="auto")` (the shipped labeller).
Matched discs with the predicted name equal to the reference name: 1220; assessed 1046 (86 %). Not assessed: 68 discs with
no posterior corners in the body records, 65 with corners not at the posterior margin (within 8 mm), 2 with no thecal sac
boundary, the rest without a measurable slice. Coverage is the same at every edge fraction.

Edge fraction (`herniation_edge_fraction`), choice by training AUC of the raw excursion (AUC is invariant to the baseline):

| f | AUC bulging (95 % CI, patient bootstrap) | AUC herniation (95 % CI) |
|---|---|---|
| 0.50 | 0.761 (0.73-0.79) | 0.749 (0.67-0.82) |
| **0.35** | **0.764 (0.73-0.79)** | **0.754 (0.67-0.83)** |
| 0.25 | 0.759 (0.73-0.79) | 0.723 (0.64-0.79) |
| 0.15 | 0.750 (0.72-0.78) | 0.706 (0.62-0.78) |

f = 0.35 is chosen (highest AUC on both fields). The AUC is flat between 0.25 and 0.5, so the choice does not carry much
weight. Half-way (f = 0.5) places the edge on the CSF side of the posterior ligament and dura.

Baseline: the median raw excursion of training discs graded **neither** bulging **nor** herniation (n = 542, f = 0.35) is
**2.77 mm** (IQR 2.29-3.44; 90th percentile 4.65). The labeller's body line lies anterior to the posterior disc margin, and
the posterior ligament and dura add to the edge, so every disc reads about 3 mm before any real displacement. The baseline
is subtracted (`herniation_baseline_mm`). Without it the 1 mm tolerance flags 98 % of normal discs.

Training operating points with the baseline applied (for reference; the rules below are fixed on these):

| rule | sensitivity | specificity | PPV | NPV | flagged |
|---|---|---|---|---|---|
| bulging: `e_adj` >= 1.0 mm | 0.69 | 0.78 | 0.73 | 0.75 | 43 % |
| herniation: `e_adj` >= 3.0 mm | 0.54 | 0.85 | 0.16 | 0.97 | 17 % |

## Pre-registered decision rules (fixed before the validation run)

1. **Edge and baseline**: `herniation_edge_fraction` = 0.35 and `herniation_baseline_mm` = 2.77, as above. Not changed on
   validation.
2. **SPIDER `disc_bulging`** (predictor `herniation`, `bulging_prediction`): positive (1) iff the primary-slice `e_adj` >=
   1.0 mm; score = `e_adj` (for AUC). Not assessed when no slice is measured.
3. **SPIDER `disc_herniation`** (predictor `herniation`, `herniation_prediction`): positive (1) iff the primary-slice class is
   protrusion or extrusion, which is equivalent to `e_adj` >= 3.0 mm; score = `e_adj`. Not assessed when no slice is measured.
4. **Metrics**: AUC of the score with a 95 % patient-bootstrap interval (1000 resamples, seed 0); sensitivity, specificity,
   PPV and NPV at the rule above, with the harness intervals. Predicted discs are matched to reference discs by the
   harness; a label mismatch is dropped and counted.
5. **Acceptance**: a field is supported if the lower 95 % bound of its AUC is above 0.5. Whatever the result, it is reported
   as it is. No threshold is re-chosen on validation. If a rule fails it is reported as failing.

## Validation on SPIDER validation

Run once, with the rules above unchanged: `tools/eval_analysis.py --dataset spider --split validation --analyses herniation
--workers 6 --boot 1000 --seed 0 --pairs` (39 patients, 264 reference discs, 6 label mismatches dropped, 10 without a
reference, 2 reference discs not predicted). Intervals: 95 % percentile bootstrap over patients.
Not assessed: 42 of 264 reference discs (no body corners or corners not at the posterior margin). Scored: 222.

| field (rule) | scored: positives / negatives | sensitivity | specificity | PPV | NPV | AUC of `e_adj` |
|---|---|---|---|---|---|---|
| `disc_bulging`: `e_adj` >= 1.0 mm | 122 / 100 | 0.70 [0.61, 0.79] | 0.82 [0.73, 0.89] | 0.83 [0.74, 0.90] | 0.69 [0.59, 0.79] | **0.82 [0.76, 0.87]** |
| `disc_herniation`: `e_adj` >= 3.0 mm (protrusion or extrusion) | 8 / 214 | 0.38 [0.00, 0.70] | 0.79 [0.72, 0.85] | 0.06 [0.00, 0.14] | 0.97 [0.95, 0.99] | **0.64 [0.38, 0.86]** |

Breakdowns (point estimates; few cases): bulging sensitivity Philips 0.76 (n 89), Siemens 0.67 (n 133); herniation
sensitivity Philips 0.20, Siemens 0.67 (3 of 8 positives are correct overall, so each breakdown is a handful of discs).

Reading against the pre-registered acceptance rule (lower 95 % bound of AUC above 0.5):
- **Bulging: supported.** AUC 0.82 with a lower bound of 0.76. The validation AUC is higher than the training AUC (0.76);
  the difference is within what 39 patients allow, and it is reported, not used to re-tune. At the rule, sensitivity 0.70
  and specificity 0.82 are close to the training operating point (0.69 and 0.78).
- **Herniation: not supported.** AUC 0.64 with an interval of 0.38 to 0.86 on 8 positives. The lower bound (0.38) is below
  0.5, so the pre-registered acceptance rule is not met. The sensitivity interval starts at 0. The predictor is reported as
  not supported; it should not be used to flag herniation on SPIDER-like data until more positive discs are scored.
- Specificity and NPV are reasonable (0.79 and 0.97). PPV is 0.06: SPIDER herniation is rare (4 % of discs here), and a
  flagged disc (excursion of 3 mm or more after the baseline) is usually graded as a bulge or nothing.

## Sudirman (sagittal only, weak labels)

Scope: the 515 labelled Sudirman studies (`lumbar_spine_mri_sudirman_labels`, with the matching DICOM session chosen by
`select_session`). Reference: the free-text radiologist reports, parsed with `datasets/report_labels.py` and aggregated to
the strongest contour per level. Report numbers are the patient folder numbers. Weak labels: the reports are silent on many
normal levels, so a level without a contour mention is unknown, not normal. The SPIDER rules were applied unchanged.

Coverage: 3,504 disc rows (513 patients with rows); 2,948 assessed (84 %). Not assessed: 415 without posterior body corners
and 140 with corners off the posterior margin. Numbering status of the rows: confirmed 596, probable 1,444, unconfirmed 1,464.
Parsed contour labels exist at 603 levels: bulge 461, protrusion 136, extrusion 2, sequestration 4. Zone labels: paracentral
71, central 63, foraminal 16 (others unspecified). Side labels: left 68, right 49, bilateral 1.

| comparison | result (patient-bootstrap 95 % CI) |
|---|---|
| `bulging_suggestion` recall of a report contour (bulge or more), assessed levels | 375 / 464 = **0.81** [0.77, 0.84] |
| `herniation_suggestion` recall of protrusion, extrusion or sequestration | 49 / 86 = **0.57** [0.45, 0.68] |
| flagged herniation, assessed levels | 419 of 2,948 (14 %); 49 of them carry a protrusion-or-worse report label (12 %, a lower bound on precision because reports are silent on many levels) |
| flagged bulging, assessed levels | 1,281 of 2,948 (43 %); 375 carry any report contour label (29 %) |

Zone and side: the sagittal module labels every non-normal finding on the mid-sagittal slice as `central`. Of the 76 report
levels with an explicit zone, 31 are central (41 %). That is the base rate of central in these reports, so it is not evidence
of a zone measurement. Side is not measured (no axial), and the report side (left 68, right 49) is not compared.

The axial labels (canal and disc masks on the axial T1 slices, `lumbar_spine_mri_sudirman_labels`) were not used to measure
posterior disc extent, because no axial measure exists in this module. That comparison is not done.


## Nerve-root relevance (anatomy, "possible" wording)

A mid-sagittal central finding at a level can involve the root that travels down the canal, which exits one level lower
(L5 for L4-L5). The analysis writes `possible_roots` for central findings only, as anatomy. It does not claim compression:
compression needs a measured contact, which this module does not measure. Foraminal and lateral findings are not assessed.

## Limitations

- **Sagittal only.** The axial contour (circumferential extent, focal or broad, zone and side) is not implemented. A first
  attempt traced the posterior contour on axial T2 with a smooth fit; the lateral rays cross fat and muscle with no CSF
  reference, and on the phantom the traced contour was too irregular to fit reliably. It was removed rather than shipped
  unvalidated. Zones beyond central (subarticular, foraminal, extraforaminal) and side are therefore not called.
- **Posterior line.** The reference is the labeller's body corners, not the ring apophyses. The labeller's body geometry
  is required; classical-labeller anchors give "not assessed". The baseline offset absorbs the systematic part of the
  body-line error, but not the noise.
- **Migration** beyond the disc space is not measured, so extrusions are a lower bound.
- **Phantom.** The synthetic phantom's posterior soft band (ligament and dura) is wider than real MRI, and its canal is
  drawn over displaced disc tissue unless `herniation_displaces_sac` is set. Its sacrum (L5-S1) gives false excursions, so
  L5-S1 is not used in the phantom tests.
- **CSF reference.** The sac level is the brightest smoothed point 3 to 25 mm posterior to the disc. If the thecal sac lies
  outside that window, a bright posterior structure can stand in for it, which shifts the edge. The SPIDER crops were
  inspected by eye on one case (the sac edge is not clearly visible in that crop, because the window ends before it).
- **Labels.** SPIDER gradings are one radiologist per disc. The codes of `disc_bulging` and `disc_herniation` are the
  dataset's (unverified here, see `docs/analysis/README.md`).
