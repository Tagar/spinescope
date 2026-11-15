# lab30_scoliosis: lateral curves, and disc planes that follow them

Branch `lab/lab30_scoliosis` (from main 9f160fc). Owner: `src/spinescope/anatomy/scoliosis.py`,
`tests/test_anatomy_scoliosis.py`, this file. Shared-file edits (additive): `anatomy/ml/postprocess.py` (the
curve hook in `anchors_from_segmentation`), `anatomy/pipeline.py` (the classical backend records that the curve was
not measured), `anatomy/anchors.py` and `data/anchors.schema.json` (the `alignment` field, schema 1.2), `config.py`
(`scoliosis_*`). Schema-version assertions in three existing tests moved from 1.1 to 1.2.

## Why

The learned labeller works on the mid-sagittal slice and its neighbours. With a lateral curve the canal and the
discs at the apex lie off that plane: a disc can be absent from it, and a disc seen on it has a normal that tilts
laterally, which the plane does not show. The fix has two parts: measure the curve, and measure each disc on the
slice through its own lateral centre.

## 1. Lateral curve from the canal, all slices

`measure_curve(seg, cfg) -> LateralCurve` (`anatomy/scoliosis.py`). Input is the canal class of the network output
on every sagittal slice, not only the mid-plane.

- Per slice and height row: the number of canal pixels (the row's cross-section, which peaks on the slice nearest the
  canal's centre). Each row gives a point (z, x, weight).
- Knots every `scoliosis_knot_mm` (5 mm): the weighted centroid of x over the rows in the bin. A bin needs the canal
  on `scoliosis_min_slices` (2) distinct slices, because one slice gives no lateral position.
- Cobb-like angle: the knots are gap-filled and smoothed (Gaussian, `scoliosis_smooth_mm` = 15 mm); the tangent
  angle theta(z) against the vertical; Cobb-like = max(theta) - min(theta) over the interior (end margins trimmed).
- Apex: the point farthest from the chord between the ends; amplitude is that distance; direction is "left" for
  positive LPS x.
- Confidence = coverage x (1 - residual / slice spacing). Active (the curve changes the labelling) when Cobb-like
  >= 10 degrees (`scoliosis_min_cobb_deg`, the clinical scoliosis threshold), amplitude >= 4 mm
  (`scoliosis_min_amplitude_mm`, about one SPIDER slice), and confidence >= 0.3.

The record goes to `Anchors.alignment['coronal']` with `valid`, `active`, `cobb_deg`, `apex_z_mm`, `apex_x_mm`,
`amplitude_mm`, `direction`, `confidence`, `coverage`, the knots and the evidence. Axial and coronal localizers are
not segmented by the labeller, so they give no lateral cue in this version.

## 2. Each disc on its own sagittal plane

When the curve is active (and `scoliosis_relocate` is on), `discs_on_curve` replaces the mid-plane disc set:

1. Candidates: the discs of every sagittal slice whose x is within `scoliosis_disc_reach_mm` (20 mm, half a lumbar
   disc's transverse width) of the canal at that height. A slice further away cannot cut the disc.
2. One candidate per disc: the in-plane (y, z) test of the mid-plane path, closest to the canal first.
3. Pass 2: the 3-D lateral centre of the disc, pooled over the slices within `ml_disc_3d_half_width_mm` (27 mm), picks
   the slice nearest it among the candidates of the same disc.
4. Centre: the 3-D pooled centroid, as the mid-plane path does. Normal: a disc the mid-sagittal slice also shows keeps
   the mid-plane normal (as on main); a disc the mid plane does not show takes the 2-D normal of its own plane. The 3-D
   pooled normal is available with `scoliosis_normal_3d` = True and is not the default (section 4). Height: the largest
   height the disc shows on any slice (a lateral slice cuts only part of it).
5. A disc seen on fewer than three slices is dropped (the 3-D convention).
6. Physical check: a disc that the mid-sagittal slice does not show, and that lies within one level spacing (25 mm)
   in height of another disc, is dropped. Discs the mid-sagittal slice shows are never dropped by this step.

If no disc survives, the mid-plane discs are used and a warning says so. When the curve is negligible, the path is
exactly the one on main.

## 3. Results

All numbers from `/tmp` scripts on this machine (not committed; they use SPIDER and Sudirman, which are not in the
repository). "off" = `scoliosis_relocate=False`, which is main's behaviour (the golden numbers reproduce exactly:
validation recall 0.9952, centre median 0.604 mm). "on" = the default.

### 3.1 How often SPIDER has a lateral curve

The canal's x(z) from the ground-truth masks (voxel centroid per 5 mm bin, the same Cobb-like algorithm, 447
volumes):

- Cobb-like >= 10 degrees: 146 of 447 volumes (training 111 of 360, validation 35 of 87). Median 7 degrees, 90th
  percentile 21.5 degrees, maximum 45 degrees.
- The lateral spread of the canal (max - min of x(z)): median 8.7 mm, 90th percentile 18.2 mm.
- The mask Cobb-like is a tangent-based number and is noisy; the network's curve is compared with it below.

The labeller's curve (active on 31 of 87 validation volumes) agrees with the mask curve on the 10-degree call for
71 of 87 volumes. Amplitude agrees better than the angle: correlation 0.59 with the mask amplitude, median absolute
difference 0.8 mm; the Cobb-like angle correlation is 0.30.

### 3.2 SPIDER: discs (the scorer of `anatomy/_validate.py`)

Curved = the network's curve is active on that volume.

| split (volumes) | set | recall | label acc | centre median / p90 (mm) | normal median / p90 (deg) |
|---|---|---|---|---|---|
| validation (87) | all, off (main) | 0.9952 | 0.9762 | 0.604 / 1.98 | 2.34 / 7.61 |
| validation (87) | all, on | 0.9952 | 0.9952 | 0.615 / 2.22 | 2.34 / 7.61 |
| validation curved (31) | off | 0.9957 | 0.9447 | 0.739 / 2.48 | 3.76 / 9.14 |
| validation curved (31) | on | 0.9957 | 0.9957 | 0.773 / 2.76 | 3.76 / 9.14 |
| training (360) | all, off (main) | 0.9956 | 0.9853 | 0.407 / 1.33 | 2.54 / 6.51 |
| training (360) | all, on | 0.9996 | 0.9913 | 0.394 / 1.19 | 2.55 / 6.53 |
| training curved (150) | off | 0.9905 | 0.9707 | 0.484 / 1.54 | 3.11 / 8.08 |
| training curved (150) | on | 1.0000 | 0.9849 | 0.426 / 1.25 | 3.13 / 8.14 |

Label accuracy is the scorer's exact-name rate over all truth discs. Predicted discs: validation 631 to 629, training
2518 to 2527 (+9; precision 0.9956 to 0.9960).

Training was the calibration split (no threshold was fitted to it; the defaults are judgement values in Config).
Validation was looked at during development and is the held-out number. The validation centre p90 gets worse by 0.24
mm and the training centre p90 gets better by 0.15 mm: the centre change is within split noise. The gain is in
label accuracy on both splits (curved volumes mainly) and in recall on training.

Label accuracy is the scorer's exact-name rate over all truth discs. Predicted discs: validation 631 to 630, training
2518 to 2528 (+10, precision unchanged).

Training was the calibration split (no threshold was fitted to it; the defaults are judgement values in Config).
Validation was looked at during development and is the held-out number. The validation centre p90 gets worse by 0.24
mm and the training centre p90 gets better by 0.15 mm: the centre change is within split noise. The gain is in
recall (training) and labels (both splits, mainly curved volumes).

Training was the calibration split (no threshold was fitted to it; the defaults are judgement values in Config).
Validation was looked at while developing the method and is reported as the held-out number.

The first runs used the 3-D normal for every relocated disc; the default now keeps the mid-plane normal for discs the
mid plane shows (section 4). The rows above are the default.

### 3.3 Sudirman (axial-slab truth, independent of SPIDER)

515 patients (1545 annotated levels); 151 patients have an active curve (453 levels). Per level, with the same scorer as
the benchmark (`_validate_sudirman.score_levels`).

| set | levels | matched | name accuracy | centre median / p90 (mm) | normal median / p90 (deg) |
|---|---|---|---|---|---|
| all 515 patients, main | 1545 | 1532 | 0.919 | 2.13 / 3.83 | 2.47 / 8.57 |
| all 515 patients, on | 1545 | 1532 | 0.922 | 2.12 / 3.83 | 2.47 / 8.57 |
| curved (151 patients), main | 453 | 453 | 0.910 | 2.15 / 3.89 | 2.50 / 8.12 |
| curved (151 patients), on | 453 | 453 | 0.921 | 2.14 / 3.96 | 2.50 / 8.12 |

Recall (matched levels) is unchanged. The normals are unchanged by construction (the mid-plane normal is kept where the
disc is also on the mid plane, and every matched curved disc is). The name accuracy gain is small (+1.1 points on the
curved subset); the centre p90 moves by 0.07 mm on the curved subset.

### 3.4 Phantom (`testing/robustness.py` scoliosis scenario)

- Oracle canal (the phantom's own tissue labels, no network): the curve follows the canal truth. The phantom's canal
  moves about 8 mm at the apex (the discs move 14 mm); canal truth Cobb-like 11.5 to 12 degrees, amplitude 5.2 to
  5.3 mm, apex at z 40 mm. The oracle estimate: cartoon 12.5 degrees and 5.2 mm, realistic 11.3 degrees and 4.6 mm,
  apex at z 37.5 mm. Discs on their own planes match the truth in name, with in-plane centre error under 0.3 mm and
  normal error under 1 degree for the lower discs (the lowest disc of the cartoon, 0.3 to 2.4 degrees, depends on
  the mode).
- Network on the realistic scoliosis phantom: curve active, Cobb-like 36 degrees, amplitude 8 mm, apex at z 52 mm.
  The network over-reads the curve on this out-of-distribution phantom (the canal is 4 to 6 mm off at the lower
  lumbar levels).
- Network on the cartoon scoliosis phantom: curve inactive (Cobb-like 8 degrees, confidence 0.19); the cartoon canal
  is seen on one or two slices per height, so the curve is not measurable there. The single-plane path is kept.

**Caveat, important.** On the realistic phantom the network reads an active lateral curve in 16 of 23 scenarios,
including the straight baseline (Cobb-like about 19 degrees, amplitude 4 to 6 mm). Those are false activations on
out-of-distribution input. They are not a measure of scoliosis handling, and the phantom's realistic-scenario results
must not be read as scoliosis validation (see 3.5).

### 3.5 What the phantom gains are really from

Scoring the 23 realistic scenarios (`testing/robustness.py`, learned labeller):

- main behaviour (relocation off): 1 of 23 correct (the labelled-correctly test)
- relocation on, final version: 2 of 23
- experiment, not shipped: the mid-sagittal path with only the height-clash rule (section 2, step 6) applied to all its
  discs: 17 of 23

So the large phantom gain of the earlier version came from the height-clash rule removing a ghost disc, not from the
lateral measurement. Applied to every mid-plane disc, that rule removes true discs on SPIDER (38_t1 loses L4-L5 and
T11-T12), so it was not kept there, and the final version protects discs the mid-sagittal slice shows. In the final
version relocation changes one scenario: `fov_L1_coccyx` now labels correctly (relocated; its curve is weak, amplitude
4.0 mm, confidence 0.45) and its xfail entry is removed. `hardware_cage_screws` passes with and without relocation.
The other realistic scenarios are unchanged.

A follow-up for someone with the time: a height-spacing check on the mid-sagittal discs themselves would fix the
ghost-disc cases on the phantom, but it changes main's behaviour for every study, so it needs its own benchmark run.

### 3.6 Benchmark

See section 5.

## 4. Normal rule

Two independent truths disagree on the normal of a laterally tilted disc:

- SPIDER's truth is the principal axis of the 3-D mask. The 3-D pooled normal matches that definition, so a 3-D normal
  looks much better on SPIDER (curved subset, training: median 0.74 against 3.1 deg; validation: 0.97 against 3.8).
  Part of that gain is definitional.
- Sudirman's truth comes from axial slabs, independent of the segmentation. There the 3-D normal is worse on the curved
  subset (median 3.55 against 2.50 deg, 453 levels) and overall (2.76 against 2.47 deg, 1545 levels). This is the same
  disagreement the labeller already documents for `ml_disc_normal_3d` (lab09).

Since the no-regression rule covers Sudirman, the default keeps the labeller's normal convention: the mid-plane normal
for a disc the mid-sagittal slice shows (identical to main for those discs, by construction), and the 2-D normal of
its own plane for a disc it does not show. The own-plane section of a laterally displaced disc is a lateral cut, so its
normal is the less reliable one: on SPIDER a 2-D own-plane normal for every relocated disc was worse than the mid-plane
one (curved subset median 4.2 against 3.8 deg), which is why the mid-plane normal is kept where it exists. Centres still
come from the 3-D pooled disc. `scoliosis_normal_3d = True` gives the 3-D normal for every relocated disc.

## 5. Benchmarks (`tools/benchmark.py --check benchmarks/golden.json`)

Final code, default configuration (`scoliosis_relocate` = True, `scoliosis_normal_3d` = False). The check does NOT pass,
and `benchmarks/golden.json` is NOT updated.

Improvements (all judged "improved" by the check):

- spider: label accuracy 98.08% to 100.00%; fully correct 94.25% to 96.55%; disc AUROC 0.797 to 0.863.
- sudirman: fully correct 89.13% to 89.71%.
- phantom_realistic: label accuracy 62.96% to 73.88%; precision 73.37% to 81.71%; disc AUROC 0.650 to 0.821; scenario
  `fov_L1_coccyx` unsure to pass. The phantom gains are on out-of-distribution input (section 3.4 caveat).

Regressions (the check fails on these; each is a measured change, not a rounding effect):

1. `phantom_realistic.recall` 97.83% to 97.10% (-0.72 pp, tolerance 0.50 pp): one true disc lost on the phantom suite.
2. `phantom_realistic.numbering_ece` 0.073 to 0.165 (tolerance 0.020): numbering confidence is less calibrated on the
   phantom, where the network's curve is spuriously active on 16 of 23 scenarios.
3. `spider.disc_ece` 0.061 to 0.078 (tolerance 0.010): disc confidence is less calibrated on validation. The relocated
   discs carry the candidate's own confidence, which is not calibrated for lateral slices. Not diagnosed further.
4. `spider.numbering_auroc` is no longer reported (golden 1.000). With relocation on, every one of the 87 validation
   studies is fully correct on its matched discs, so the AUROC has no negatives and is undefined
   (`anatomy/_validate.py::reliability` returns None). This is a metric artefact of a perfect result, not a loss of
   discrimination; the golden file needs a decision on how to record an undefined AUROC, not a number change.
5. `sudirman.wrong_announced` 11 to 12 (tolerance none): one more probable-status study with a wrong name. The study
   could not be identified from the aggregate output in the time available.

What was tried for the regressions and what the numbers say:

- Height-clash rule applied to all discs: fixed the phantom ghost discs but removed true SPIDER discs (38_t1 loses
  L4-L5 and T11-T12); removed.
- Clash rule protecting discs the mid plane shows, many-to-one: kept 198_t1 wrong (two discs lost); replaced by the
  one-to-one protection now in the code, which fixed 198_t1 and 38_t1/38_t2 and kept SPIDER validation at 0.9952 recall
  and 0.9952 label accuracy on the protected set.

Not yet decided: whether to (a) accept the phantom and calibration regressions with a golden update (this would need
the golden owner's sign-off, as the rule is "update only if the numbers improve"); (b) tighten the activation so that
the phantom's spurious curves are not used (the amplitude and Cobb thresholds do not separate them; a real fix needs
an out-of-distribution check on the canal); or (c) ship with `scoliosis_relocate = False` (measured, no labelling
change), keeping the code for later. The SPIDER benefit is real on both splits; the regressions above are on the
phantom and on calibration, which is why the check fails.

## 6. Limits

- The curve is a canal-centroid estimate from 5 mm slices. Its amplitude is reliable on SPIDER (in-distribution);
  the Cobb-like angle is not (correlation 0.3). Treat the angle as a screening number.
- Out-of-distribution inputs (the realistic phantom) produce false activations. Nothing in the labeller checks for
  this yet; the labelled-correctly outcome on those inputs is not evidence of scoliosis handling.
- Vertebral bodies and the numbering stay on the mid-sagittal plane. A curve can therefore lower the numbering
  confidence (the body below a disc is not seen on that plane); the curve does not change any numbering rule.
- Axial and coronal localizers give no lateral cue (the labeller does not segment them).
- Thresholds (Cobb 10 degrees, amplitude 4 mm, confidence 0.3, reach 20 mm, clash gap one level spacing) are judgement
  values in `Config`, documented there. They were not fitted to any split.

## 7. Reproduce

The runners used for these numbers are in `/tmp/spinescope-lab30_scoliosis/` (not committed: they write per-case
records with case identifiers, and this repository keeps no per-case outputs). The benchmark is the committed
`tools/benchmark.py`.
