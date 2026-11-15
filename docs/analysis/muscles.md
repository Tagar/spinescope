# Paraspinal and psoas muscles (`analysis/muscles.py`)

Status: **experimental**. Outputs are capped at confidence 0.5 (`muscles_max_confidence`). Measurements, not diagnoses.

## What it measures

On the axial slice through each of L3-L4, L4-L5 and L5-S1 (slab or axial stack from `anatomy.level_map`), for each
side (patient left `L`, patient right `R`) and each muscle (multifidus `MF`, erector spinae `ES`, psoas `PS`):

- **CSA** (mm²): outline area, corrected to the perpendicular plane (`area * cos(tilt)`, tilt = slab normal vs disc normal).
- **Functional CSA** (mm²): outline minus fat-bright pixels (lean area), same correction.
- **Fat fraction** (%): fat-bright share of the outline (see method).
- **Normalised CSA**: outline area / body cross-section area, both on the same slab (ratio, no tilt correction needed).
- **Kjaer grade** (MF only, see below).
- **Asymmetry** (% of the mean of both sides) of CSA, functional CSA and fat fraction, per level and series.

Rows: `scope = muscle` and `scope = asymmetry`. Flags: `multifidus_fat` (grade >= 1), `paraspinal_asymmetry`
(|CSA asymmetry| > 20 %). Figures: one overlay per level (`ax_<level>`).

## Method

1. **Slice.** `level_map` gives the axial stack/slab that covers the disc (preferred slab first, then the same slab in
   the other contrast). Slices are taken at the nearest slice to the disc.
2. **Coordinates.** Every pixel is mapped to LPS mm (`pix_to_lps_many`). `lat` = x - midline x (patient left positive);
   `ap` = y - disc-centre y (posterior positive, LPS). Left/right and AP therefore do not depend on the image's row or
   column order (tested with row and column flips).
3. **Body size.** Half width and half AP depth from the level's body geometry (`LevelEntry.geometry["body"]`, measured
   by the labeller when available, else `CorrespondenceConfig` defaults; flagged `geometry`).
4. **Windows** (anatomy, in units of half width `hw` and half depth `hd`; `config.py` `muscles_*_frac`):
   - MF: lateral 0.2-1.0 hw, AP 2.0-3.5 hd (behind the canal).
   - ES: lateral 1.0-2.4 hw, AP 1.5-3.5 hd.
   - PS: lateral 1.0-2.3 hw, AP -0.8 to 0.8 hd (middle of the body).
   Outline = window ∩ body outline ∖ subcutaneous ring (12 mm inside the skin), largest component, holes filled.
5. **Body outline.** Smoothed slice, threshold = mean + 3 SD of the two darker corner boxes (taken as air); a fat-relative
   threshold is not used because T2 muscle sits only a few times above the air noise. Fallback if the corners carry
   signal: 25 % of the way from the 5th to the 90th percentile.
6. **Fat and lean references (per slice).** F = median of the bright pixels in the skin ring (subcutaneous fat).
   M = median of the psoas core that is not fat-bright (pixels < 0.6 F); erector spinae if the psoas core is > 50 %
   fat-bright. Fat-bright = pixels above the midpoint (F + M)/2.
7. **Kjaer grade** (BMC Med 2007;5:2, doi 10.1186/1741-7015-5-2, PMID 17254322): grade 0 = 0-10 % fat, 1 = 10-50 %,
   2 = >50 %. The paper's ranges overlap at 10 %; here 10 % exactly is grade 1 and 50 % exactly is grade 1. The grade
   is read from the outline's fat fraction, which is a visual scale re-estimated by a threshold (not a visual read).
8. **Normalisation** follows the common ratio of paraspinal CSA to vertebral body CSA, computed per level and side
   (Front Neurol 2021, doi 10.3389/fneur.2021.656487). The body cross-section is taken as an ellipse of the measured
   half width and depth; the body centre is the disc centre (approximation).
9. **Obliquity.** When the slab tilt to the disc normal is above 12 degrees (`muscles_oblique_tilt_max_deg`, the
   correspondence tolerance), CSA is corrected by cos(tilt) and the confidence is multiplied by 0.6.

## Limitations (read these first)

- **Outlines are windows, not muscle boundaries.** No fascia or muscle edge is segmented, so windows include
  intermuscular fat, vessels, lamina/facet edges and some non-muscle soft tissue. CSA is therefore an upper bound
  on muscle area; the **normalised CSA is dominated by the window size** (ES about 0.9 of the body ellipse, MF about
  0.4), not by muscle. Do not compare normalised CSA with published values until outlines are fascia-bounded.
- **Fat fraction is a fat-bright share of the window.** It includes intermuscular fat and vessels, so it is biased
  upward relative to muscle fat (Dixon fat fraction counts only fat in the muscle). The 10 % Kjaer boundary is applied
  to this share, not to a validated muscle fat fraction.
- **No demographics.** No age or sex normalisation: none is available, by design (PHI-free). Atrophy is only
  "relative to this study": the asymmetry flag, with no published CSA range used.
- **Single slice per level**, the slice nearest the disc. No volume.
- **Oblique slabs**: corrected by cos(tilt) for a planar cut; the window geometry is not rotated with the disc (in-plane
  rotation of the spine is not modelled).
- **Phantom detail.** The phantom's default paraspinal muscle is one ellipse; the multifidus is added only with
  `PhantomSpec(paraspinal_detail=True)` (cartoon only, `realism='cartoon'`).

## Validation

### Public data available

- **SPIDER** (`~/data/public`): sagittal only. No axial slice, so no muscle validation is possible there.
- **Sudirman axial** (CC BY 4.0): axial slabs at L3-S1 for 515 sessions. Used below.
- **OSF qx5rt Dixon** (CC BY-NC 4.0): **not usable for this module.** The NIfTI volumes are sagittal (slice axis =
  right-left, 12 to 15 slices; checked via the vertebral masks, whose extent is 40 mm across slices), so there is no
  axial paraspinal slice to compare with the Dixon fat fraction. The planned T1/T2-versus-Dixon Bland-Altman is
  **not performed**.

### Sudirman axial: measured distributions

Run: all 515 sessions labelled once (`label_spine`, cached) and analysed with `analysis.muscles` (scratch, not
committed). 483 sessions had at least one measured level. Of 17,064 muscle rows, 16,580 were measured; 484 not measured
(outline too small or outside the window, or no fat/lean reference on the slice).

Median fat fraction (%) and CSA (mm²), patient-level 95 % bootstrap (1000 resamples, seed 0):

| level | series | muscle | n patients | fat fraction % (95 % CI) | CSA mm² (95 % CI) |
|---|---|---|---|---|---|
| L3-L4 | T2 | MF | 462 | 7.1 (6.4-7.7) | 572 (567-580) |
| L3-L4 | T2 | ES | 461 | 6.1 (5.5-6.8) | 1335 (1314-1353) |
| L4-L5 | T2 | MF | 470 | 15.4 (14.4-16.4) | 623 (611-633) |
| L4-L5 | T1 | MF | 464 | 21.6 (19.9-23.5) | 627 |
| L5-S1 | T2 | MF | 465 | 18.2 (16.7-19.7) | 653 (629-668) |

Only the headline rows are shown; the scratch run holds the rest. Headline results:

- **T1 versus T2 agreement** (paired, same patient, level and side, MF only, n = 2753): mean difference 4.6 percentage
  points (T1 higher), limits of agreement -16.6 to +25.9. The two contrasts do not agree closely; the fat-bright share
  depends on the contrast.
- **Kjaer grade of MF**: on T2 1094 grade 0, 1598 grade 1, 99 grade 2; on T1 874, 1739, 173. Most outlines are grade 1
  by this measure. The fat-bright share is likely over-counted (intermuscular fat, see limitations), so grade 1
  should not be read as multifidus infiltration in these data.
- **Asymmetry**: median absolute CSA asymmetry 1.0 %; 4.2 % of paired muscle rows exceed the 20 % flag.

### Comparison with published values

- Crawford et al. 2016 (AJNR 37(4):742-748, doi 10.3174/ajnr.A4596): healthy volunteers, **3-T two-point Dixon,
  combined multifidus + erector spinae, fat signal fraction and volume** (not single-slice CSA). Lower lumbar means
  (women/men): L4 21.2 / 15.3 %, L5 31.9 / 25.7 %, where "L4" means the region between superior endplates, not an axial
  slice. Our values use a different method and a different quantity. They are **not directly comparable**; the
  L4-L5 MF fat-bright share (15-21 %) lies in the same range numerically, which is not evidence of agreement.
- Kjaer 2007 grade boundaries: verified above (BMC Med 5:2).
- Single-slice CSA normative values for these outlines were **not** found in the sources checked; none is quoted.

### Radiologist-report weak labels (pre-registered rule)

Rule written before the outcome was computed: a level is "fatty atrophy/infiltration positive" if the report contains
the phrase "fatty" or "infiltration" (or "fatty atrophy") in a sentence about paraspinal muscles or psoas, and the
prediction is positive if MF Kjaer grade >= 1 on either side at that level. Recall = predicted positive / report
positive. Pre-registration disclosure: the free-text search for these terms was run before this rule was written, to
check whether the labels exist.

**Outcome: not evaluable.** Of 575 notes in `Radiologists Report.xlsx`, none contains "fatty" or "infiltration".
The muscle-related mentions are "muscle spasm" (about 200 notes), "intact visualised muscles" (1) and "bulky Lt
iliopsoas muscle with abnormal signal alteration" (1, not fat). Recall is therefore undefined (zero positive reports).
This is a **negative result for the planned validation**, not a validated muscle label.

### Overlay review

Ten random Sudirman L4-L5 slices were reviewed (scratch only, nothing committed or shared). Windows land on the
multifidus and erector spinae in most cases; they also include lateral fat, facet/lamina edges and sometimes miss part
of the psoas (the psoas window follows the body, not the psoas outline). One case had a psoas outline cut by the
body edge. Conclusion: adequate for a coarse, per-slice comparison; not for outline-level measurement.

## Side conventions and tests

- LPS-based left/right; `L` is patient left (+x).
- Tests (`tests/test_analysis_muscles.py`, 31 tests): the Kjaer boundaries; cos correction; asymmetry; windows on the
  patient's left; a synthetic fat/lean slice; the phantom with a left-only multifidus fat band (fat fraction tracks the
  band, grade on the left only); oblique-slab correction and confidence; normal phantom symmetry; flipped rows and
  columns (same answers, same sides); determinism; missing-axial and no-disc warnings.

## Files

- `src/spinescope/analysis/muscles.py`: analysis (registered as `muscles`).
- `src/spinescope/config.py`: `muscles_*` defaults (with sources).
- `src/spinescope/testing/phantom.py`: additive `paraspinal_detail`, `multifidus_fat` (default off).
- `tests/test_analysis_muscles.py`.
