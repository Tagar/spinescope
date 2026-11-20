# Facet joints: joint fluid per level and side (`analysis/facets.py`, dx08)

Status: **experimental**. Joint fluid is measured per level and side on fat-suppressed images only. Nothing here is
validated against radiologist grades: no sensitivity or specificity is reported, because the public data have no facet
grade that matches the measurement (see Validation). The findings field `facet_L.effusion` / `facet_R.effusion` is
written as a measurement with its evidence image, never as a diagnosis.

## What is measured

| output | how | status |
|---|---|---|
| joint fluid width (mm), per level and side | maximal caliper of the fluid component seeded at the joint site | measured on fat-suppressed images |
| effusion present / absent | width at or above `facets_effusion_min_mm` (2.0 mm) | called on STIR or fat-sat T2 only |
| possible synovial cyst | round fluid (min/max caliper >= 0.6) at least 5 mm wide | flag, wording "possible" |
| fluid vs CSF and muscle | fluid median in the same units as the CSF and muscle references | reported (`fluid_ratio`) |
| parasagittal fallback | sagittal fat-suppressed slice through the site | low confidence, cap 0.3 |
| joint centre from the image (dark joint line), facet angle per side (Karacan convention), tropism (7 and 10 deg) | dx21, `joint_geometry_rows`, `scope = "joint_geometry"` rows; **unvalidated, capped** (section 6) | see section 6 |
| joint-line length and width (joint space is not measured directly), Weishaupt grade | dx21 proxies; the grade is always "not assessed" | see section 6 |
| osteophytes, periarticular oedema, erosions | not implemented | see Limitations |

Side: `L` is patient left (+x in LPS, image right on the display axial), `R` is patient right. Sides come from the
patient coordinates, so a flipped acquisition (rows or columns reversed, same geometry) gives the same result (tested).

## Method

1. **Site.** In the disc plane: posterior to the disc centre by half the body depth plus half the canal AP diameter
   (level map, measured or flagged default). Lateral by `canal_half + facets_site_fraction * (body_half - canal_half)`,
   with `facets_site_fraction = 0.6` (judgement). The canal point is the same posterior point at zero lateral offset.
   The site is not forced to disc height, because the disc plane follows the segment's tilt.
2. **Stack.** Fat-suppressed stacks first (STIR, fat-sat T2), then plain T2 and PD, then axial before parasagittal,
   then the closest plane. The stack must contain the site (within half a slice) and the two reference points (canal
   and muscle), so a stack whose field of view cuts a reference is not used. T1 is never used for fluid.
3. **Sampling.** Square of half side `facets_roi_half_mm` (12 mm) on a `facets_px_mm` (0.5 mm) grid, trilinear, in the
   stack's own plane (`geometry.sample_plane`).
4. **References.** CSF: 3 mm radius at the canal point, same plane. Muscle: 3 mm radius `facets_muscle_offset_mm`
   (20 mm) posterior to the site. Pixels within the canal half width + 1 mm of the canal point are excluded.
5. **Threshold.** `muscle + facets_fluid_frac * (CSF - muscle)`, `facets_fluid_frac = 0.5` (half maximum). A call is
   only made when the CSF-muscle contrast is at least `facets_min_contrast` (0.2) of the sampled intensity range.
6. **Component.** The connected component that has the most pixels within `facets_seed_mm` (4 mm) of the site. Width =
   maximal caliper + one pixel; roundness = minimal / maximal caliper (5 degree steps).
7. **Calls.**
   - No contrast: not assessed.
   - Plain T2 (fat not suppressed): not called; the extent is recorded as `fluid_extent_mm` for review only. On the
     phantom, fat reaches the fluid threshold; plain T2 extents on real studies were 20 to 25 mm in the pilot (not called).
   - Fluid touching the window edge: not called (a lower bound only; large edge-touching regions were not joints, see
     the pilot).
   - Otherwise: effusion present if width >= 2.0 mm; possible synovial cyst if present, width >= 5 mm and roundness
     >= 0.6.
8. **Confidence.** 0.8 for a fat-suppressed, window-contained call; 0.3 for the parasagittal fallback. Level-name status caps it further in the findings
   (`level_findings`): probable names at medium, unconfirmed at low.

## Thresholds and sources

All values live in `spinescope/config.py` (`facets_*`), with a source comment there.

| key | value | source |
|---|---|---|
| `facets_effusion_min_mm` | 2.0 | Schinnerer and Chaput: an effusion of about 1.5 to 2 mm is associated with instability or degenerative spondylolisthesis (from the wave brief). The upper end is used to stay specific. **Citations not re-checked against the papers here.** |
| `facets_cyst_min_mm`, `facets_cyst_roundness_min` | 5.0 mm, 0.6 | judgement; no validated threshold |
| `facets_fluid_frac` | 0.5 | half maximum between muscle and CSF (convention) |
| `facets_min_contrast` | 0.2 | judgement |
| `facets_site_fraction` | 0.6 | judgement (site between the canal wall and the body edge); checked by eye and on phantom only |
| `facets_seed_mm` | 4.0 | judgement: the site is an approximation, a few mm of error is expected; about a third of the sampling half window. Changed from 2.0 after the phantom site was 4.5 mm from the true joint: the seed must reach the joint from the site. Not a literature value |
| `facets_tropism_deg` | 7.0 | Commonly quoted tropism cut-off (Kalichman and Hunter review; secondary source). Noren et al. (1991, Spine 16:530) relates tropism to disc degeneration; its abstract does not give 7 deg (checked 2026-11-15). Used by `tropism_flags` |
| `facets_tropism_strict_deg` | 10.0 | The other commonly quoted cut-off (secondary source). Reported alongside 7 deg; neither is validated |
| `facets_loc_*`, `facets_morph_*` | see section 6 | judgement; pre-registered in section 6.3 |

## Phantom validation (realistic phantom, ground truth in `truth["facets"]`)

Axial STIR (3 mm slices, 0.9 mm pixels), one injected effusion per run, measured width (mm):

| injected (mm) | measured (mm) | bias | fluid ratio | call |
|---|---|---|---|---|
| 0 (healthy) | 0.0 | 0 | none | absent on every level (no false call) |
| 2 | 3.5 | +1.5 | 0.67 | present |
| 3 | 4.3 | +1.3 | 0.69 | present |
| 4 | 5.2 | +1.2 | 0.79 | present; possible cyst (round) |
| 6 | 6.6 | +0.6 | 0.82 | present |

The width overestimates by about 1 mm: partial volume at the fluid edge on a 3 mm slice and the pixel correction. It is
not corrected, because a correction fitted to the phantom would be tuned on one model. Treat widths as upper
estimates of the fluid extent. The site is 3.8 to 4.5 mm from the true joint centre (phantom truth), which is why
`facets_seed_mm` is 4 mm. Tests: `tests/test_analysis_facets.py` (21 tests, phantom-based, including flipped
acquisitions and the plain T2 and T1-only paths).

On the realistic phantom, a plain T2 study makes no call (fat and fluid have similar brightness there, the fat reading
of 1400 against a 1322 threshold gave a false 12 mm "effusion" before the fat rule was added).

## Real-data coverage pilot (Sudirman, 5 patients, not a validation)

Five Sudirman studies with an axial fat-suppressed or STIR fluid series (T2_TSE_TRA_FS, T2_TIRM_TRA_P2 or
T2_TSE_TRA_448-FS). Anchors from `label_spine` (cached per study).

- 3 of 5 studies: the labeller found no discs, so no joint was analysed (a labelling failure, reported as such).
- 2 labelled studies: 30 joint sites. 12 on plain T2 (not called). 18 on STIR: 3 calls (all absent), 15 not called
  (window edge or no contrast). **No positive call.**
- The one large edge-touching fluid region in the pilot (24 mm) was not a joint. The first version called it
  "present"; that call was removed.
- The figure check on one real STIR level shows that the site cross sits between structures and that the fluid
  region was not joint-shaped. The site is **not validated on real images**.

## Validation against radiologist reports (pre-registered, not run)

Pre-registered before any number was looked at: facet effusion present = fat-suppressed width >= 2.0 mm, per level and
side; reference = a non-negated facet mention with the same side in the parsed Sudirman notes (`datasets/report_labels`,
`facet_arthropathy`); metrics: sensitivity and specificity with 95% patient-level bootstrap intervals (1000 resamples,
seed 0), coverage first.

It was not run, for three reasons:
- Only 2 labelled patients with a fat-suppressed axial series, far too few for intervals.
- The notes are keyed by a report number (1 to 575); the mapping to the Sudirman folder IDs is not verified.
- The parsed label is "facet arthropathy", not effusion. Agreement with it would be weak evidence for effusion.

SPIDER has no facet grade, so no harness predictor is registered (`docs/analysis/README.md`, truth-field table). The
SPIDER sagittal fallback failure rate is not measured. Normative facet orientation on Sudirman is not possible,
because orientation is not measured.

## Limitations

- **dx21 (section 6).** The joint line is a dark ridge near the geometric site, not a located joint. Localisation
  succeeded on 47 % of sides and is not closer to the mask proxy than the site. The mask reference is posterior-element
  bone, not the joint, so every dx21 error is against a proxy. The joint-line width is not joint space. Weishaupt grades
  are not assigned. The morphology weak labels have 8 positive and 0 negative levels: not evaluable. Sudirman provides
  one axial T1 slice per annotated disc, so no sagittal or fat-suppressed joint check is possible on this data.

- **Site.** A geometric approximation. Its accuracy on real images is unknown, and the fluid call depends on it. A
  miss gives "absent", not "not assessed". The coverage check is the first thing to read.
- **Fat.** Fat-suppression is required for a call. Non-suppressed fat is not separated from fluid by intensity.
- **Effusion versus cyst.** A round collection is flagged "possible synovial cyst" with no criterion for its origin.
- **Width bias.** About +1 mm on phantom; real-data bias is unknown.
- **Not measured.** Osteophytes, hypertrophy, erosions and subchondral cysts as graded features, the Weishaupt grade
  (Weishaupt, Skeletal Radiology 1999; see section 6), periarticular oedema,
  facet orientation and tropism angles. `tropism_flag` is a pure helper with no inputs from this module.
- **Findings.** The findings evidence cites one series (`meta.series_key`, the most used stack). A level measured on
  another stack is not reported in the findings. The `arthropathy` field is not written.
- **Citations.** Weishaupt (Skeletal Radiol 1999;28:215-219, doi 10.1007/s002560050503) and Noren (Spine 1991;16:530-532)
  checked against their records on 2026-11-15; Karacan 2004 (Spine 29:1132-1136) is a CT study whose angle convention is
  used here. Schinnerer, Chaput are as cited in the brief or as commonly quoted; they have not been
  re-checked against the papers.

## 6. Joint localisation, facet angle, tropism and morphology proxies (dx21)

Status: **experimental, unvalidated until the gates below are met.** Every output of this section is capped at
`facets_morph_max_confidence` (0.2). The Weishaupt grade is never assigned (`grade_status = "not assessed"`).

### 6.1 Method

Inputs: the level-map joint site (section 1 of Method, unchanged) and the axial stack that covers it. The stack is
the annotated one in validation, and the nearest slice in the analysis. Plane: axial, patient coordinates.

1. **Image window.** A 24 mm square at 0.5 mm, centred on the site (`geometry_at`, `facets_roi_half_mm`,
   `facets_px_mm`).
2. **Dark joint line.** Gaussian second derivatives at `facets_loc_sigma_mm` (0.6 mm). Score per pixel:
   max(l1, 0) - 2 max(l2, 0), with l1 >= l2 the Hessian eigenvalues, scale-normalised and divided by the window's intensity
   range. A dark line (cortex between articular processes) scores high across its width and low along it, so
   blobs score low.
3. **Peak and component.** The peak score within `facets_loc_search_mm` (8 mm) of the site. The joint line is the connected
   set of pixels with score >= `facets_loc_component_frac` (0.5) of the peak, within `facets_loc_max_radius_mm` (6 mm)
   of the peak. Fewer than `facets_loc_min_pixels` (6) pixels: not localised.
4. **Outputs.** Joint centre = centroid of the component (mm offsets from the site, LPS-converted). Joint line
   direction = principal axis. Length = maximal caliper, width = minimal caliper (plus one pixel). Width is the
   joint-line thickness, **not** a joint-space measurement: no joint fluid or gap is resolved on the public T1 data.
5. **Facet angle (Karacan et al., Spine 2004;29:1132-1136, convention).** Angle in the axial plane between the joint
   line and the mid-sagittal (AP) direction, 0 to 90 degrees. Smaller means more sagittal, larger means more coronal.
   Karacan's own measurements were on CT; the convention (midsagittal reference line against the facet line, on an axial
   image) is the one used here.
6. **Tropism.** Left minus right angle at a level, |difference|. Reported against 7 and 10 degrees (both
   `facets_tropism_deg` and `facets_tropism_strict_deg`). Neither threshold is validated here.
7. **Morphology proxies.** Joint-line length and width per side, and the ratios over the study's annotated levels
   (level value / median of the study's annotated levels). No grade is assigned.

Source of the mask reference (validation only): the Sudirman label-100 (posterior elements) region lateral to the canal
wall, within 8 mm of the site, on the annotated T1 slice. The mask has no joint, so this is a proxy. The mask angle is
the same principal-axis method applied to the boundary of that region.

### 6.2 Literature used

- Weishaupt D, Zanetti M, Boos N, Hodler J. MR imaging and CT in osteoarthritis of the lumbar facet joints. Skeletal
  Radiology 1999;28(4):215-219. doi:10.1007/s002560050503. Four-point scale, grades 0 to 3 (table 1: joint-space width
  2 to 4 mm normal, narrowing below 2 mm, osteophytes, articular-process hypertrophy, erosions, subchondral cysts; "and/or"
  criteria). Grades are **not** assigned here: the osteophyte, hypertrophy and erosion criteria are not measured.
- Karacan I, et al. Spine 2004;29(10):1132-1136. doi:10.1097/00007632-200405150-00016. Facet angle convention (CT).
- Noren R, Trafimow J, Andersson GB, Huckman MS. Spine 1991;16(5):530-532. doi:10.1097/00007632-199105000-00008.
  Relates facet tropism to disc degeneration (L3-4 to L5-S1); the abstract gives no 7-degree cut-off.
- Masharawi Y, et al. Facet orientation in the thoracolumbar spine: three-dimensional anatomic and biomechanical analysis.
  Spine 2004;29(16):1755-1763. doi:10.1097/01.BRS.0000134575.04084.EF. Table 2: mean (SD) facet angles relative to the
  vertebral midsagittal plane, 240 vertebral columns (ages 20 to 80), vertebra level:
  L3 superior 29.6 (14.4) left, 30.0 (14.8) right, inferior 42.5 (20.9) left, 44.2 (19.3) right;
  L4 superior 37.0 (14.4), 39.4 (15.4), inferior 50.2 (17.1), 54.2 (14.5);
  L5 superior 40.4 (16.1), 49.1 (18.6), inferior 50.6 (13.4), 56.3 (19.8).
  Angles increase from L1 to L5 (more coronal caudally). **The brief's wording "coronal L3-L4 to sagittal L5-S1" is
  the reverse of this table**; the normative check uses the table as published. These are vertebral (bone) angles,
  not joint-level MRI angles, so the comparison is qualitative.
- Kalichman L, Hunter DJ, and the tropism literature: 7 and 10 degrees are the commonly quoted cut-offs (secondary
  sources; primary values not checked here).

### 6.3 Pre-registration (written before any validation number was produced)

Data: Sudirman labelled patients (515, sorted by folder id; `datasets.sudirman.labelled_patients`). Calibration half: the
first 257; validation half: the last 258. Development on the calibration half used the first 20 patients only, to check
that the pipeline runs; no threshold was changed from the defaults in section 6.1 on the basis of any number. Nothing is
tuned on the validation half.

**V1, localisation.** Error = distance (mm) between the image joint centre and the mask reference centre, per annotated
level and side. Comparator: the geometric site (distance to the mask reference). **Gate for `facet_localisation_validated`:**
median image error <= 3.0 mm AND median image error <= median geometric-site error, over all validation levels, with
the 95% patient-bootstrap interval of the median reported (1000 resamples, seed 0). If the gate fails, the image joint is
reported as a negative result and is not used for the angles.

**V2, facet angle.** Image angle vs mask angle (same method), per level and side: MAE, bias, Bland-Altman 95% limits,
Pearson r, patient-bootstrap intervals. **Gate for `facet_angle_validated`:** MAE <= 10 degrees AND |bias| <= 5 degrees
at every annotated level. Normative check: image mean per level against Masharawi 2004 Table 2 (L3 to L5 means), no gate.

**V3, tropism.** Agreement of the |L-R| tropism call (7 and 10 degrees) between image and mask, by level, descriptive only.

**V4, morphology weak labels.** Reference: the parsed report field `facet` (the `facet_arthropathy` label) at the level,
for the same patient. Positive = a non-negated mention; negative = only negated mentions; conflicts and absence dropped.
Primary prediction: mean joint-line length ratio (mean over sides; ratio = level value / study median over annotated
levels) >= `facets_morph_length_ratio` (1.25). Secondary (reported only): mean width ratio <= 0.8. Metrics: sensitivity
(recall) and specificity with patient-bootstrap intervals. **Evaluable only with at least 20 reference positives and
20 reference negatives** in the validation half; otherwise reported as "not evaluable". The notes are free text, so the
numbers say how the measure agrees with the report, not with the images.

**Decision rules.** Localisation is "validated" only if V1 passes; angles are "validated" only if V2 passes. Every
other output stays at `facets_morph_max_confidence` regardless. A failed gate is reported, not re-tuned.

### 6.4 Validation results (Sudirman validation half, 258 patients, 1545 annotated levels)

Run: `SPINESCOPE_THREADS=2 .venv/bin/python tools/validate_facet_morph.py --split validate --workers 2`. Intervals are 95%
patient-bootstrap percentiles (1000 resamples, seed 0). Pipeline: 1542 of 1545 levels had a labelled site (3 without a
level-map site); 1542 levels x 2 sides = 3084 joint sides. The image joint was localised on **1454 of 3084 sides (47 %)**;
1388 sides had both an image joint and a mask reference.

**V1, localisation (gate failed).** Joint-centre error against the mask reference (distance, mm):

| measure | left | right | both sides |
|---|---|---|---|
| image joint, median (95% CI) | 4.24 (4.01 to 4.41) | 4.46 (4.22 to 4.68) | 4.33 (4.14 to 4.48) |
| image joint, MAE | 4.50 | 4.83 | 4.66 |
| geometric site, median (95% CI) | 4.12 (3.98 to 4.23) | 4.20 (4.01 to 4.36) | 4.14 (4.02 to 4.26) |

Gate: median image error <= 3.0 mm **and** <= geometric site. **Failed** on both conditions: the image joint is not better
than the geometric site and is not within 3 mm. `facet_localisation_validated = False`. Interpretation: the image method
finds a dark line near the site, but against a posterior-element mask it is no closer than the site. The mask is a proxy
(no joint in it), so this is not a proof that the joint is wrong; it means the image joint is not validated.

**V2, facet angle (gate failed).** Image angle against the mask angle (same method), n = 1388:

| level | n | MAE, deg (95% CI) | bias, deg | Bland-Altman 95% limits | Pearson r | mean image / mask |
|---|---|---|---|---|---|---|
| L3-L4 | 433 | 21.5 (20.1 to 22.9) | -0.9 | -53.4 to +51.7 | 0.25 | 53.9 / 54.8 |
| L4-L5 | 474 | 20.0 (18.6 to 21.3) | -2.3 | -51.9 to +47.4 | 0.32 | 58.2 / 60.4 |
| L5-S1 | 481 | 16.5 (15.0 to 18.1) | -4.0 | -48.4 to +40.3 | 0.45 | 63.2 / 67.2 |
| all | 1388 | 19.3 (18.3 to 20.2) | -2.4 | -51.3 to +46.4 | 0.37 | 58.6 / 61.0 |

Gate: MAE <= 10 deg at every level. **Failed** (16 to 22 deg at every level). `facet_angle_validated = False`.

**Normative check (no gate).** Image mean angle by level (both sides) rises from 53.9 (L3-L4) to 58.2 (L4-L5) to 63.2
(L5-S1). Masharawi et al. (2004, Table 2, bone angles) inferior-facet means are about 43 (L3), 52 (L4) and 53 (L5), and
superior-facet means about 30, 38 and 45 (left and right averaged). The image values are about 6 to 10 deg above the
inferior-facet means. The **direction** of the trend matches the published one (angles increase caudally). The brief's
phrase "coronal at L3-L4 to sagittal at L5-S1" is the reverse of Masharawi's table and is not used.

**V3, tropism (descriptive).** Per patient, |left - right| at each level (median, deg): image 21.4 (L3-L4), 21.3 (L4-L5),
13.9 (L5-S1); mask 8.8, 12.0, 10.3. Agreement of the 7-deg call with the mask call: 0.58, 0.60, 0.54 (fraction of
patients); at 10 deg: 0.52, 0.56, 0.51. The image method calls tropism in 71 to 80 % of patients at 7 deg, so the call is
dominated by the image-angle noise. Neither threshold is validated.

**V4, morphology weak labels (not evaluable).** The notes give 8 reference-positive levels (facet arthropathy, at a level
on both sides measured) and **no** reference negatives in the validation half. The pre-registered minimum is 20 positives
and 20 negatives, so the rule is **not evaluable**. Descriptively, the primary rule (length ratio >= 1.25) flagged 0 of
the 8 reference-positive levels with the final code (first run: 2 of 8). No sensitivity or specificity is reported.

### 6.5 Amendments and bugs (disclosed)

The validation pipeline ran three times while the code was being fixed. The thresholds and rules in section 6.3 were
never changed. Bugs found along the way, each before the number it affected was reported as final:

1. **Slice choice.** The first dev pass compared on the nearest slice to the site: 38 of 120 levels did not match the
   annotated slice on which the label map lives. Validation evaluates on the annotated slice (`geometry_at(slice_k=...)`).
2. **Notes key.** The morphology lookup first used the wrong key and found no records; the key is `facet_arthropathy`
   (the mapped field name in `aggregate_levels`). Rule unchanged.
3. **Joint-line component (localisation).** When the peak lay outside the joint-line disc, its label was 0 and the
   "component" became the background inside the disc. Found by a unit test (a dark line gave a centre 6.9 mm off-centre).
   Fixed: the peak is searched inside the disc too, and the disc is centred on the site. The first validation run used
   the buggy component: image median 5.76 mm (5.53 to 5.95), angle MAE 23.1 deg (22.2 to 24.1). These superseded numbers
   are kept here for transparency; the gate outcome is the same.
4. **Tropism summary.** The first summary kept one patient per level, so its values were wrong. Now per patient.

Development: the calibration half was used only for a 20-patient plumbing pass (no thresholds set from it).

### 6.6 Interpretation for readers

- The joint line and angle are not validated. They are reported as **experimental** (`experimental: true`, capped at
  `facets_morph_max_confidence`, 0.2) in the `joint_geometry` rows. They are not written as findings.
- The Weishaupt grade is **not assessed**. No grade is supported by the features here: joint-space width is not measured
  (the joint-line thickness is a proxy), osteophytes and erosions are not measured, and the one weak-label rule cannot
  be evaluated.
- Visual check (one level, `L4-L5` left, calibration-half patient 0260, looked at once and not used for any number): the detected joint line lies on the edge of the bright
  articular structure, above the label-100 region. It does not show a joint; the mask is a bone region.
- Joint-space narrowing and the Weishaupt grade remain open. The next step is a labelled joint-level reference (joint-space
  width or a grade read from the images), not more tuning here.
