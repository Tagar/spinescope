# Central canal: sagittal AP and axial dural sac area (dx06)

Module: `spinescope.analysis.canal` (registered as `canal`). Tests: `tests/test_analysis_canal.py`.
Validation tool: `tools/validate_canal.py`. The analysis never diagnoses: it measures, suggests a grade with a
confidence, and says what it could not measure.

## 1. Method

Per level, for every labelled disc (the disc level) and every labelled vertebral body (the mid-body level):

1. **Sagittal canal AP (mm).** `anchors.canal["per_level"][level]["ap_mm"]`, the AP chord of the labeller's canal
   mask on the sagittal slice nearest the canal centreline at that height, minus `canal_ap_bias_mm` (0.5, calibrated
   on SPIDER training, `anatomy/canal_ml.py`). It is a dural sac width, not a bony canal diameter. The analysis reads
   it; it does not re-segment. Without a learned canal (classical labeller) the AP is not assessed and a warning says
   so. There is no CSF-signal fallback in this version.
2. **Axial dural sac area (mm^2), disc levels only.**
   - Slab: the axial slab angled to the disc, from `anatomy.correspondence.level_map` (`LevelEntry.slab`).
   - Slice: the slab slice nearest the disc plane (`StackHit.slice`).
   - Seed: the canal centreline point at the disc height (`targets["canal"]["point"]`), projected onto that slice.
   - Sac: Otsu threshold of a square window of +-`canal_axial_halfwindow_mm` (16 mm) around the seed. CSF-bright
     (T2, STIR, PD) or CSF-dark (T1) side of the threshold. The connected component nearest the seed (within
     `canal_axial_centre_tol_mm`, 3 mm) is the sac; holes are filled. A sac touching the window is a lower bound.
   - Obliquity: `angle(slab normal, disc normal)`. The observed area is multiplied by cos(angle). Above
     `canal_axial_obliquity_max_deg` (35 deg) no area is reported. Above 15 deg the confidence is lowered.
   - Plausibility: 15 to 300 mm^2 after correction, else not measured.
3. **Relative value.** `csa_ratio` = area / reference. Reference = the 75th percentile of the study's measured disc
   areas (proxy for a non-stenotic sac; needs at least 3 measured levels, else no ratio). It is reported, not used in a
   grade.
4. **Grade suggestion** (lumbar range only: discs L2-L3 to L5-S1 and bodies L1 to L5; thoracic levels are measured
   but not graded). Per criterion:
   - area: `absolute` below 75 mm^2, `relative` below 100 mm^2;
   - AP: `absolute` below 10 mm, `relative` below 13 mm.
   - `severe`: both absolute. `moderate`: one absolute. `mild`: relative only. `none`: no criterion met. Not assessed
     when neither criterion is measured. This mapping is a convention of this module.
5. **Confidence.** AP and area: the lower of the canal confidence and the area confidence (0.8; 0.6 when the slab is
   more than 15 deg oblique; 0.4 when the sac is truncated). AP only: half the canal confidence (the sagittal-only
   case). Area only: the area confidence. The level-name status cap of the findings layer applies on top.
6. **Contributors (observed only).** Disc height (anchors) is listed. Ligamentum flavum is owned by dx09. Facet and
   foraminal narrowing are not measured here.

Figures: `canal_axial.png` (one crop per measured disc, sac outline in red, area, obliquity and the ratio when there is
a reference), `canal_profile.png` (sagittal AP by level, with the 10 and 13 mm lines).

Findings wiring: `central_canal.stenosis` (value none/mild/moderate/severe; figure `profile`) and
`central_canal.area_mm2` (figure `axial`) are registered with `register_field`. Neither is written without an image.

## 2. Literature thresholds (in `spinescope/config.py`)

| criterion | value | source |
|---|---|---|
| dural sac area, relative stenosis | < 100 mm^2 | Hamanishi C et al., Spine 1994 (dural sac cross-sectional area) |
| dural sac area, absolute stenosis | < 75 mm^2 | Hamanishi C et al., Spine 1994 |
| sagittal AP, absolute | < 10 mm | Steurer J et al., BMC Musculoskelet Disord 2011 (systematic review of quantitative criteria) |
| sagittal AP, relative | 10 to 13 mm | same |

These are cited as given in the wave brief. The paper texts were **not re-checked in this session**; verify the
citations before publication. Schönström's cadaver/CT CSA work (named in the brief) is not cited here because no
source was available to check. Other module constants (window, tolerance, obliquity limits, plausibility gate,
reference quantile) are judgement defaults with their reasons in `config.py`.

Not implemented: Schizas et al. 2010 morphological grade (A to D, from the cauda equina / rootlet distribution), which
needs root-level morphology; Lee et al. 2011 grading, which needs the same. Neither is claimed.

## 3. Pre-registration (written before any validation number was produced)

Data: SPIDER validation split (87 cases, `overview.csv` subset `validation`); Sudirman labelled patients (515, notes
number = folder number, which matches all 515). Nothing was tuned on either.

**V1. Sagittal AP (SPIDER validation).** Reference AP = the labeller's own AP definition applied to the SPIDER canal
mask (label 100): resampled to the labeller's 1.0 mm grid (`resample_slice`, nearest neighbour), largest component,
`ap_chord_mm` at the same z and band, minus the same bias. Metrics: MAE, bias, Pearson r, Bland-Altman 95% limits of
agreement, accuracy and Cohen's kappa for the classes (< 10 mm, 10 to 13 mm, >= 13 mm). 95% CI by patient bootstrap,
1000 resamples, seed 0. No pass gate: this measures the labeller's canal, not stenosis.

**V2. Axial sac area (Sudirman, T1).** Reference = count of label 150 (thecal sac) on the annotated axial T1 slice x
pixel area of that slice (the label map is the DICOM grid). The pipeline's slab and slice must equal the annotated
ones; otherwise the level is excluded and counted. Metrics: MAE (mm^2), bias, Bland-Altman limits, Pearson r, median
relative error; the comparison is on the observed (uncorrected) area, so the segmentation alone is tested; 95% CI by
patient bootstrap. **Gate for `canal_axial_validated = True`:** MAE <= 15 mm^2 AND |bias| <= 10 mm^2 AND Pearson r >= 0.8
on the included levels, with at least 100 included levels. If the gate fails, axial-derived confidence stays capped at
`canal_axial_unvalidated_cap` (0.4) and the area is reported as unvalidated.

**V3. Stenosis suggestion vs weak report labels (Sudirman notes).** Reference positive: the parsed report has a
`stenosis` label (the canal term) with no negation at L3-L4, L4-L5 or L5-S1 and zone not foraminal or extraforaminal.
Reference negative: only negated (normal-type) labels at that level. Conflicting labels are excluded. Primary
prediction: observed area < 100 mm^2 (relative or absolute criterion). Secondary: AP < 13 mm. Metrics: sensitivity,
specificity, PPV, NPV, prevalence, with 95% CI by patient bootstrap. These labels are free text, not a reading of the
images, so the numbers say how the measure agrees with the report, not with truth.
