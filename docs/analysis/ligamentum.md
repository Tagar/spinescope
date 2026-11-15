# Ligamentum flavum and dorsal epidural fat (axial)

Module: `src/spinescope/analysis/ligamentum.py` (analysis `ligamentum`). Validation tool: `tools/eval_ligamentum.py`.
Tests: `tests/test_analysis_ligamentum.py`. Status: **experimental**. Every confidence is capped at
`ligamentum_max_confidence` (0.3, low). The module measures and suggests; it does not diagnose.

## What it measures

Per labelled disc, on the axial slice nearest the disc (T2 preferred, T1 accepted), with the dural sac centre and
radius from `Anchors.canal` (AP diameter halved) and the posterior direction from the disc's anterior-posterior axis:

1. **Ligamentum flavum thickness, per side (L, R).** Rays leave the sac posterolaterally (midline plus +/- 20, 30, 40,
   50 degrees). Along a ray the LF is the dark band between the bright dural sac and the posterior elements. The band
   is the run around the smoothed minimum where the signal stays at or below the midpoint between the minimum and the
   brighter tissue behind it. The run must close on both sides and reach 3 noise sigmas of contrast. The level value per
   side is the maximum band over that side's rays. Side is taken from the LPS x direction (L = patient left, +x), so a
   row-flipped acquisition gives the same sides.
2. **Dorsal epidural fat, midline, T1 only.** The bright run that starts at the dural edge on the posterior midline ray,
   above the midpoint between the dural (CSF) level and the brightest tissue behind the edge. The run must close, which
   needs a ligamentum or other dark boundary behind the fat. On T2 the CSF and the fat are both bright, so the fat is not
   assessed.

Not implemented: a sagittal fallback (posterior canal margin, lower confidence), dural sac shape (Y or trefoil), and the
Borre 2003 grade (which needs ventral and dorsal fat; only dorsal fat is measured here). The findings schema has a
graded `ligamentum_flavum.thickening` field, and this module writes only `none` or `mild`. Epidural fat has no schema
field, so it appears in the rows and flags only.

## Thresholds and where they come from (verified)

| Quantity | Value used | Source and what it does and does not support |
|---|---|---|
| LF thickness flag | >= 4.0 mm (`ligamentum_thick_mm`) | Convention. Sakamaki T, Sairyo K, Sakai T, et al. *Arch Orthop Trauma Surg* 2009;129(10):1415-1419 (doi 10.1007/s00402-009-0849-1; PMID 19280205) reports thickness above 3.0 mm at L2-L3 as a level-specific marker associated with multilevel stenosis, and finds >3 mm values at L4-L5 in young patients, so it is not a universal cut-off. Altinkaya N, et al. *Spine* 2011;36(16):E1093-E1097 (doi 10.1097/BRS.0b013e318203e2b5; PMID 21343862) associates LF thickness with stenosis at L2-L5 on axial T1 at the facet level, and its abstract does not state a 4 mm cut-off. The same PubMed record quotes Abbas et al. (2010) as arguing that the normal/abnormal boundary should not be set at 4 mm (full citation not checked). **So the 4 mm value is a labelled convention with weak literature support, not a validated threshold.** |
| Epidural fat flag | > 7.0 mm (`ligamentum_epidural_fat_mm`) | Traditional thoracic rule of thumb for posterior epidural fat. A small comparative study reports a mean of 7.35 mm in lumbar controls at L3-L4 (PMC6718140), so normal lumbar fat can exceed it. It only prompts a look. |
| Epidural fat grading (not implemented) | Borre DuS/EF and EF/SpiC | Borre DG, Borre GE, Aude F, Palmieri GN. *Eur Radiol* 2003;13(7):1709-1721 (doi 10.1007/s00330-002-1716-4; PMID 12835988). Grades are on the S1 endplate plane; it needs ventral and dorsal fat. Not computed here. |
| Ray sweep, search, sampling, CNR | 20-50 deg, 12 mm, 0.25 mm, 3 sigma | Judgement and engineering (`config.py`). |

## Method validation on the phantom (additive phantom options)

`PhantomSpec.lf` (disc -> side, thickness) and `PhantomSpec.epidural_fat_mm` add a ligamentum band and a dorsal fat
layer in the cartoon phantom. Both are off by default; the default phantom is unchanged. Measured on an axial T2 slice
through L4-L5 (`tests/test_analysis_ligamentum.py`):

| true LF (mm) | measured, same side | opposite side |
|---|---|---|
| 2 | 1.75 | not measured |
| 3 | 2.5 | not measured |
| 4 | 3.75 | not measured |
| 6 | 5.5 | not measured |
| 8 | 7.5 | not measured |

The estimate is 0.25 to 0.5 mm low (half-maximum edges with smoothing; the sampling step is 0.25 mm). Fat on T1 with a
midline ligamentum: true 3, 6, 9 mm measured 3.25, 7.0, 9.75 mm (high by 0.25 to 1 mm). A flipped acquisition gives the
same sides. The phantom checks geometry and signal. Real anatomy is not checked here.

## Validation on Sudirman (pre-registered before the full run)

Data: Sudirman axial T1 reference slices (label maps for D3, D4, D5 = L3-L4, L4-L5, L5-S1; 515 labelled patients; the
DICOM slice that carries each label map is measured). Label values: 150 thecal sac, 100 posterior elements, 50 disc, 200
area between the anterior and posterior elements. Reports: `Radiologists Report.xlsx` through the rule-based parser
(`report_labels`).

What label 200 covers, checked on rendered reference slices (scratch only): a thin band lateral and posterolateral to
the sac, anterior to the posterior elements, where the ligamentum flavum sits. It is therefore the closest available
axial reference for the LF, but it is an anatomical area label, not a LF truth; its thickness is a geometric proxy.

Pre-registered rules (fixed before the full run):

- **P1 (primary, the LF operating point).** A level is called "suggests thickening" when the maximum LF band over all
  rays of the T1 reference is at least 4.0 mm. Scored against the report label `ligamentum_flavum_thickening` (True =
  positive) at levels that have a measurement and a non-conflicting report value. Sensitivity, specificity, PPV, NPV, with
  95% percentile intervals from 1000 patient-level bootstrap resamples (seed 0).
- **P2 (geometric agreement with label 200).** Bland-Altman bias and 95% limits of agreement between the measured band and
  the label-200 band on the same rays; fraction within 1 mm; Pearson r. Agreement counts only if |bias| <= 1 mm and at
  least 60% are within 1 mm.
- **P3 (decision).** The LF rule is "supported" only if the sensitivity lower bound is at least 0.5 and the specificity
  lower bound is at least 0.6, with at least 10 report-positive scored levels. Otherwise it is a negative result: the
  cap stays and the rule is not re-tuned on these data.
- **P4 (epidural fat).** Lipomatosis is not validated on these data: the notes contain no mention of lipomatosis (checked
  before the run: 0 of 575 reports). Only counts are reported, and no sensitivity or specificity is claimed.
- **P5 (coverage).** Share of report-labelled levels that get a measurement, with the reasons for the rest.

## Results

Full run: 515 labelled patients, 1545 labelled discs (D3-D5), 1481 measured (95.9%); 64 not assessed (no ray gave a
closed band with enough contrast). Only derived numbers are kept (`tools/eval_ligamentum.py`, scratch output).

| Rule | Result | Verdict |
|---|---|---|
| P1 primary: per-level report LF label at the 4.0 mm operating point | **0 scorable levels.** The parser gives an LF label at a level in none of the 19 reports that name the ligament. Two reports give a level-less LF label (`level None`), which per-level scoring drops. | Not scorable. Not supported. |
| P2 agreement with label 200 (1402 measured levels with a label band on the same rays) | Bias +2.55 mm (95% CI 2.39 to 2.72); 95% limits of agreement -3.23 to +8.34 mm; within 1 mm 22.6% (CI 20.4 to 24.8%); Pearson r 0.09; medians 6.75 mm measured, 4.25 mm label 200. | **Fails.** The T1 band is systematically wider than the label-200 band and not correlated with it. |
| P3 decision | P1 cannot be scored and P2 fails. | **Negative result.** The LF measurement is not supported on T1 references; the cap stays. |
| P4 epidural fat | 0 of 575 reports mention lipomatosis. | Not validated; no sensitivity or specificity claimed. |
| P5 coverage | 1481 / 1545 levels measured. | Adequate coverage; the measurement itself does not agree with the reference. |

**Added after P1 was found not scorable (exploratory, not pre-registered).** Patient-level: 21 labelled patients whose
report names the ligament with hypertrophy or thickening (clause-level rule, negation checked; 5 positive). At the same
4.0 mm point the rule flags every named patient: sensitivity 1.00, specificity 0.00 (16 false positives), PPV 0.24,
AUC 0.56 (95% CI 0.12 to 0.95). It does not discriminate. The 4 mm point on this T1 measurement is far below the
measured median, so almost every level is flagged. Read it as a negative result, with the number of patients too small for
any rate to be estimated well.

Phantom checks (above) pass, but they are synthetic. On real T1 the measured band is much wider than the label-200 proxy and
does not track the report, so the axial LF estimate is not fit to be shown as a finding. The module keeps it as an
experimental row with the cap, and it should not be used to call thickening until a labelled T2 reference is available.

## Limitations

- The Sudirman references are T1 only. On T1 the LF is dark-intermediate and close to the sac, so the band can be weak;
  T2 (where the sac is clearer) is not in the reference set.
- Report labels are a rule-based parse of free text, not a radiologist's per-disc grading. Few reports mention the
  ligament (19 of 575), so the positive counts are small and the intervals are wide.
- Label 200 is an area label, so agreement with it is a geometric check, not validation of the LF boundary.
- Axial rays give the thickness along a ray, which only approximates the perpendicular thickness. The maximum over rays is
  an upper estimate for oblique bands.
- Sagittal fallback, sac shape and the Borre grade are not implemented.
- The 4 mm threshold is a convention, not a validated cut-off (see the table above).
