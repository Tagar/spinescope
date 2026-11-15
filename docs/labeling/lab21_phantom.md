# lab21_phantom: a realistic synthetic lumbar phantom

Branch `lab/lab21_phantom` (worktree `~/work/spinescope-wt/lab21_phantom`). Owner:
`src/spinescope/testing/realistic.py` (new), `src/spinescope/testing/phantom.py` (`realism` option, per-level
geometry, label memoisation), `src/spinescope/testing/robustness.py` (`realism` argument, `cached_case`),
`tools/robustness_report.py` (`--realism`), `tests/test_phantom_realistic.py` (new),
`tests/robustness/test_robustness_realistic.py` (new), `tests/test_anatomy_ood.py` (two realistic variants).
No real image is used anywhere in this work: the phantom is analytic and its parameters come from the literature
listed below. Rendered examples are in `docs/assets/phantom/`.

## Result

- **The realistic phantom exists** (`PhantomSpec(realism="realistic")`). The default stays `"cartoon"`: the
  cartoon results are bit-identical to main on all 69 scenario and backend rows (checked row by row), so no existing
  test changed meaning.
- **The learned labeller does not work on the realistic phantom yet.** It passes 1 of 23 scenarios
  (`hardware_cage_screws`) against 0 of 23 on the cartoon. It is also more often *confidently wrong* on the realistic
  phantom: 17 silent errors (numbering confidence at or above 0.5) against 12 on the cartoon. Those are the failures
  that matter most, so the realistic suite does not hide them.
- **CI now has a learned-labeller gate that needs no real data.** `tests/robustness/test_robustness_realistic.py`
  runs all 23 scenarios with the learned labeller. 22 labelled-correctly and 17 no-silent-error checks are
  `xfail(strict=True)` with the measured numbers. A strict xfail that starts to pass fails the suite, so it must be
  flipped to a real test. Two OOD checks: one realistic variant passes and is a real test, one is a strict xfail.
- **Runtime:** see "CI runtime" below. The realistic robustness file costs about 90 s; it is the price of the
  learned-labeller gate on the phantom.

## Why the phantom had to change

The cartoon has ellipses for vertebral bodies, a single intensity per tissue and one fixed level size. The learned
labeller (trained on SPIDER sagittal T1/T2) saw a body that looked nothing like a vertebra, so its failures on the
cartoon say little about real spines. The realistic phantom aims for the same *kinds* of structure and contrast as
a clinical lumbar sagittal and axial study, built from published numbers.

What the realistic phantom contains (`testing/realistic.py`):

- **Vertebral bodies**: kidney-bean outline (posterior wall concave at the midline), endplates dished 1.2 mm, a
  cortical rim, marrow with smooth random-field texture (+-12 %). Per-level heights and sizes (table below).
- **Discs**: biconvex outline, annulus and nucleus, per-level height, herniations, Modic 1 bands, Schmorl nodes,
  collapse, degeneration (nucleus signal lerp), transitional anatomy (lumbarised S1, sacralised L5).
- **Posterior elements**: pedicles, laminae, facet joints (bone and joint space), spinous processes, ligamentum
  flavum in the interlaminar space. The spinous processes sit at the level of the disc below their body
  (Bogduk 2005).
- **Canal**: thecal sac (dura rim, CSF), cord to the conus at L1, cauda equina strands below it, epidural fat.
  The dural sac ends at the S1 bottom, the sacral canal is narrow below.
- **Sacrum**: S1 plus S2-S5 as tilted, tapering segments (sacral curvature), S1-S2 rudimentary disc.
- **Pelvis and abdomen**: iliac wings (axial slices at the sacral level), psoas, paraspinal muscles with fat streaks
  (fatty infiltration), subcutaneous fat (thicker anteriorly), liver, kidneys, bowel with gas, aorta and IVC
  (flow voids).
- **Alignment**: segmental lordosis from the morphometry below; lumbar lordosis from L1 superior endplate to S1 is
  46 degrees; sacral slope 40 degrees (normal range roughly 40-60 and 35-45 degrees).

## Physics

- Tissue signal = proton density x relaxation, from published T1 and T2 at 1.5 T and 3 T (table below).
- Spin echo: `S = rho (1 - exp(-TR/T1)) exp(-TE/T2)`. Inversion recovery (STIR):
  `S = rho |1 - 2 exp(-TI/T1) + exp(-TR/T1)| exp(-TE/T2)`. Protocol TR/TE/TI are the existing `ACQ` values, so the
  DICOM metadata and the physics agree (TI = 150 ms; at 1.5 T that is close to but not exactly the fat null).
- Partial volume: each pixel averages sub-samples across the slice thickness (2) and in plane (2 x 2), so a voxel
  straddling an endplate gets an intermediate value.
- Noise: the existing Rician model (`noise` relative to the brightest tissue), 3 T with the same SNR factor as the
  cartoon. Surface-coil bias: the existing `bias_field`. Gibbs ringing: optional (`gibbs=True`), k-space truncation
  to 70 % of the band per axis.
- Render cost: label volumes are memoised per phantom on the exact sample points, so the three contrasts of a
  geometry share one labelling pass. Structures are evaluated only near themselves (k-d tree), and the intensity
  is a lookup-table index.

### Morphometry (per level)

Values are approximate central values in the range reported by the cited sources (see Sources). Body sizes grow
from T9 to L5; disc heights grow from T9-T10 to L5-S1.

| level | body height (mm) | body width (mm) | body depth (mm) | disc below the body, height (mm) |
|---|---|---|---|---|
| T9 | 22 | 34 | 28 | 4.5 (T9-T10) |
| T10 | 22.5 | 36 | 29 | 5 (T10-T11) |
| T11 | 23 | 38 | 30 | 5.5 (T11-T12) |
| T12 | 24 | 40 | 31 | 7 (T12-L1) |
| L1 | 25 | 42 | 32 | 8 (L1-L2) |
| L2 | 26 | 44 | 33 | 9 (L2-L3) |
| L3 | 27 | 47 | 34 | 10 (L3-L4) |
| L4 | 28 | 50 | 35 | 11 (L4-L5) |
| L5 | 28 | 53 | 36 | 11 (L5-S1) |

Segmental lordosis (relative turn between adjacent disc planes, T11-T12 to L4-L5, degrees): 5, 6, 8, 9, 11, 12
(the cartoon uses 3, 4, 5, 6, 8, 14). A first attempt at 3 to 10 degrees gave an anteriorly bowed column with no
return above L3; it was corrected before any learned-labeller measurement. Sacral slope 40 degrees; each sacral
segment tilts 12 degrees further towards vertical.

### Relaxation table (approximate literature values, rounded)

rho is relative to CSF = 1. Values are approximate literature ranges, not measurements of one data set. The least
certain rows are fat, marrow, annulus, dura and ligament.

| tissue | rho (rel.) | T1 1.5 T / 3 T (ms) | T2 1.5 T / 3 T (ms) |
|---|---|---|---|
| air | 0 | 1000 / 1000 | 1 / 1 |
| soft | 0.75 | 650 / 900 | 60 / 50 |
| fat | 0.9 | 260 / 370 | 110 / 100 |
| muscle | 0.75 | 870 / 1410 | 47 / 30 |
| marrow | 0.8 | 380 / 450 | 80 / 70 |
| cortex | 0.15 | 300 / 400 | 0.5 / 0.4 |
| annulus | 0.7 | 700 / 1000 | 60 / 40 |
| nucleus | 0.85 | 1000 / 1300 | 110 / 90 |
| csf | 1 | 2500 / 3500 | 2200 / 2000 |
| cord | 0.7 | 800 / 1000 | 80 / 70 |
| vessel | 0.05 | 1200 / 1500 | 100 / 80 |
| modic1 | 0.85 | 700 / 900 | 120 / 100 |
| psoas | 0.75 | 870 / 1410 | 47 / 30 |
| oedema | 0.9 | 800 / 1000 | 150 / 120 |
| hardware | 0 | 1 / 1 | 1 / 1 |
| schmorl | 0.85 | 1000 / 1300 | 110 / 90 |
| bloom | 0.05 | 300 / 400 | 1 / 1 |
| dura | 0.5 | 800 / 1000 | 8 / 6 |
| ligament | 0.6 | 500 / 600 | 25 / 20 |
| facet | 0.9 | 900 / 1200 | 80 / 60 |
| bowel | 0.8 | 600 / 800 | 150 / 120 |
| liver | 0.8 | 550 / 800 | 40 / 30 |
| kidney | 0.8 | 700 / 1000 | 80 / 60 |
| nerve | 0.7 | 800 / 1000 | 80 / 70 |
| remnant | 0.6 | 800 / 1000 | 60 / 50 |

## Sources

- Panjabi MM, Goel V, Oxland T, et al. Human lumbar vertebrae: quantitative three-dimensional anatomy. Spine 1992.
  Lumbar body dimensions (heights, widths and depths grow from L1 to L5).
- Bogduk N. Clinical Anatomy of the Lumbar Spine and Sacrum. 3rd ed. 2005. Body and canal dimensions, the spinous
  process level, the dural sac ending at S2, normal segmental lordosis.
- Frobin W, et al. Measurement of disc height and vertebral height from lateral radiographs (Clinical Biomechanics,
  late 1990s). Disc heights.
- Roussouly P, Gollogly S, Berthonnaud E, Dimnet J. Classification of the normal variation in the sagittal alignment
  of the human lumbar spine and pelvis. Spine 2005. Lumbar lordosis and sacral slope.
- Stanisz GJ, et al. T1, T2 relaxation and magnetization transfer in tissue at 3 T. Magn Reson Med 2005. Muscle and
  other tissue relaxation at 3 T.
- Bottomley PA, et al. A review of 1H nuclear magnetic resonance relaxation in pathology: are T1 and T2 diagnostic?
  Med Phys 1987. Relaxation at 1.5 T.
- Pfirrmann CW, et al. Magnetic resonance classification of lumbar intervertebral disc degeneration. Spine 2001.
  Disc T2 grading.

**Caveat on the citations:** the numbers were entered from the summary values of these works as recalled while
writing, and were not re-checked against the papers in this environment. Check them against the sources before
quoting absolute values from the phantom. Reference details (years and pages) are given to the level I am confident
of.

## Protocol and change log (read this before the numbers)

The learned labeller was first measured on the realistic phantom at version v1 (morphometry, physics, sacrum as the
first version drew it). After that measurement I changed three things. They are listed here, not hidden:

1. **Bug fix (not a choice)**: the sacral segments S2-S5 were chained towards the head. Fixed; they now run inferiorly.
2. **Anatomy change**: the spinous processes were centred on their own body. Clinically they sit at the level of the
   disc below (Bogduk 2005), so they now sit at the inferior endplate level of their body. This was motivated by the
   anatomy and by the probability maps (the posterior elements were read as discs), *not* by the scores.
3. **Speed setting**: supersampling was set to 2 x 2 in plane and 2 through plane. It was chosen for CI time (see
   CI runtime); not tuned on the scores.

Attribution of the scenario passes for the learned labeller (realistic phantom, scenario passes out of 23):
v1 (before 1 and 2): 3 (`t1_only`, `fov_L2_S1`, `scoliosis`). Bug fix 1 only: 4 (`t1_only`, `fov_L2_S1`,
`scoliosis`, `hardware_cage_screws`). Final (bug fix 1 and change 2): 1 (`hardware_cage_screws`). So the
spinous-process move cost 3 passes, so the pass count is fragile: a small geometry change flips several scenarios.
I kept the anatomically motivated geometry. Choosing the geometry that scores best would be tuning the phantom to
the test. With change 2 undone the baseline precision is 0.67 and with it 0.75; neither is right.

Also: a bug found in the first look at the images, not a tuning choice: the psoas had no relaxation entry, so it
rendered black. Fixed before any measurement.

## Results: before (cartoon, main) and after (realistic)

Scenario status, learned labeller (`auto` = `ml`) and classical, for the cartoon (default, before) and the realistic
phantom (after). PASS = all discs found, named and centred within 3 mm (p90), no spurious disc. `silent` = wrong with
numbering confidence >= 0.5. `unsure` = wrong with low confidence (acceptable). Realistic v1 is the first measurement.

| scenario | cartoon: learned | cartoon: classical | realistic v1: learned | realistic (final): learned | realistic (final): classical |
|---|---|---|---|---|---|
| baseline | silent | silent | silent | silent | silent |
| t2_only | silent | silent | silent | silent | silent |
| t1_only | silent | silent | PASS | silent | silent |
| stir_only | unsure | silent | unsure | unsure | silent |
| pd_only | unsure | unsure | unsure | unsure | unsure |
| iso_1mm_3d | unsure | silent | silent | silent | silent |
| thick_6mm | silent | silent | silent | silent | silent |
| few_slices_7 | silent | PASS | silent | silent | silent |
| field_3T | silent | silent | silent | silent | silent |
| fov_T10_sacrum | unsure | silent | silent | silent | silent |
| fov_L2_S1 | silent | silent | PASS | silent | silent |
| fov_L1_coccyx | silent | PASS | silent | silent | silent |
| lumbarised_S1 | silent | silent | silent | unsure | silent |
| sacralised_L5 | silent | silent | silent | silent | silent |
| dark_discs | unsure | silent | silent | silent | silent |
| collapsed_discs | unsure | silent | silent | silent | silent |
| compression_fracture_L2 | silent | PASS | silent | silent | silent |
| schmorl_nodes | silent | silent | silent | silent | silent |
| scoliosis | unsure | silent | PASS | silent | silent |
| hardware_cage_screws | unsure | silent | silent | PASS | silent |
| bias_field | unsure | silent | silent | silent | silent |
| motion_ghosting | unsure | silent | silent | unsure | silent |
| aliasing_wrap | unsure | silent | silent | unsure | silent |

Counts over the 23 scenarios in each column:

| column | PASS | unsure | silent |
|---|---|---|---|
| cartoon, learned | 0 | 11 | 12 |
| cartoon, classical | 3 | 1 | 19 |
| realistic v1, learned | 3 | 2 | 18 |
| realistic (final), learned | 1 | 5 | 17 |
| realistic (final), classical | 0 | 1 | 22 |

Full metrics per row (recall, precision, label accuracy, centre p90, confidence) are in
`python tools/robustness_report.py --realism realistic --json out.json`.

## What the learned labeller does on the realistic phantom

Checked on the midline sagittal slice (probability maps, scratch, not committed):

- Vertebral bodies: clean body class. Real discs: found (recall 1.00 in the baseline).
- Posterior elements (spinous processes, interspinous space): part of their area is predicted as disc. These extra
  detections sit just behind the canal and are numbered as levels, so the labels above them shift by one.
  Precision drops to 0.75 and label accuracy to 0.67 in the baseline, with confidence 0.93.
- Discs above T11 are not present in the phantom, but an extra thoracic disc is often predicted there when the top
  body is in the field of view.

Whether the same confusion happens on real SPIDER spines was **not** tested. That needs a SPIDER check outside this
task; I did not use real images to tune the phantom.

## CI runtime

Measured on this machine, `pytest -q` full suite (the same command as CI here):

- Before (main, cartoon only): 1159 passed, 1 skipped, 43 xfailed, 1 xpassed; 529 s (8 min 49 s).
- After (this branch): 1187 passed, 1 skipped, 83 xfailed, 1 xpassed, 0 failed; 525 s (8 min 45 s).
  The new items account for +28 passes and +40 strict xfails (19 + 8 + 1 new passing tests; 39 + 1 strict xfails).
- The baseline timing is not a clean comparison: I ran phantom probes on this machine while it was running. Use
  the per-file numbers below, which were measured separately without contention.

Parts added by this work (measured without contention):

- `tests/test_phantom_realistic.py` (19 tests): about 5 s.
- `tests/robustness/test_robustness_realistic.py` (46 items, 23 scenarios): about 90 s. Most of it is the learned
  labeller's three-series pass on each scenario and the realistic render (about 2.4 s per scenario).
- `tests/test_anatomy_ood.py` realistic variants: about 9 s.

If 90 s is too much for CI, mark `test_robustness_realistic.py` `slow` and run it in a nightly job; the strict
xfail lists are the gate either way.

## Reproduce

- Cartoon, unchanged: `.venv/bin/python tools/robustness_report.py`
- Realistic: `.venv/bin/python tools/robustness_report.py --realism realistic --json out.json`
- One scenario: `.venv/bin/python -c "from spinescope.testing.robustness import *; print(evaluate(build_case(scenario('baseline'), 'realistic'), 'ml').metrics)"`
- Images: `docs/assets/phantom/` (rendered from `SpinePhantom(PhantomSpec(realism=...))`).

## Limitations

- Not validated against real scans. The phantom is an approximation built from published summary values; it does
  not reproduce any scanner, and it has no motion, breathing, vascular flow, bowel peristalsis or slice-profile
  effects beyond the partial-volume average.
- Relaxation values are rounded and approximate (see table). Fat, marrow, disc, dura and ligament are least certain.
- Iliac wings, kidneys, liver and bowel are simple ellipsoids; the sacrum is a chain of tapered blocks; the
  posterior elements are simple shapes. These are enough for the axial and parasagittal checks, not for shape
  analysis.
- The learned labeller fails on most scenarios on this phantom. This is the reported finding, not a bug to hide. The
  posterior-element confusion was observed on the phantom's probability maps; it is not confirmed for real spines.
- `docs/validation.md` and the SPIDER numbers are unchanged; this work does not change any real-data claim.
