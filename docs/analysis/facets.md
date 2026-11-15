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
| joint-space narrowing, osteophytes, Weishaupt grade, periarticular oedema, facet orientation and tropism angles | not implemented | see Limitations |

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
| `facets_tropism_deg` | 7.0 | Noren et al. (1991, Spine) as commonly cited; 7 to 10 degrees in the literature. Used by `tropism_flag` only, because no angles are measured |

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

- **Site.** A geometric approximation. Its accuracy on real images is unknown, and the fluid call depends on it. A
  miss gives "absent", not "not assessed". The coverage check is the first thing to read.
- **Fat.** Fat-suppression is required for a call. Non-suppressed fat is not separated from fluid by intensity.
- **Effusion versus cyst.** A round collection is flagged "possible synovial cyst" with no criterion for its origin.
- **Width bias.** About +1 mm on phantom; real-data bias is unknown.
- **Not measured.** Joint-space narrowing, osteophytes or hypertrophy, the Weishaupt grade (the facet grading paper I
  can identify is Weishaupt et al., Eur Radiol 1999; the wave brief gave 2001, to check), periarticular oedema,
  facet orientation and tropism angles. `tropism_flag` is a pure helper with no inputs from this module.
- **Findings.** The findings evidence cites one series (`meta.series_key`, the most used stack). A level measured on
  another stack is not reported in the findings. The `arthropathy` field is not written.
- **Citations.** Schinnerer, Chaput and Noren are as cited in the brief or as commonly quoted; they have not been
  re-checked against the papers.
