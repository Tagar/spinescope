# Sagittal alignment: listhesis, lordosis, segmental angles (dx04)

Module: `spinescope.analysis.alignment` (registry name `alignment`). Measures; does not diagnose. Every number is
traceable to the labeller's body endplates on one mid-sagittal slice per body, and carries the level-name status of
its level.

## Scope and what is not reported

- **Listhesis** (anterolisthesis / retrolisthesis) per disc, as translation in mm and as a percentage of the lower
  body's AP depth. Meyerding grade I to IV for anterolisthesis; V for a slip above 100 %.
- **Lumbar lordosis**: L1 superior endplate to L5 inferior endplate (**L1-L5**). The L1-S1 route is **not reported**
  (see "Sacral endplate" below). Lordosis is measured supine.
- **Segmental (intervertebral) angles** per disc: tilt of the lower body's superior endplate minus tilt of the upper
  body's inferior endplate, positive when lordotic. This is the disc wedge angle.
- **Lateral curvature**: passed through unchanged from the lab30 record (`Anchors.alignment['coronal']`). It is a
  screening record (amplitude first, Cobb-like angle uncalibrated, unvalidated). It is not upgraded here.
- **Not reported**: L5-S1 listhesis and L1-S1 lordosis (sacral endplate, below); any claim of instability.

## Conventions

LPS mm (+x patient left, +y posterior, +z superior).

- Tilt of an endplate: `atan2(dz, dy)` of its anterior-to-posterior corner vector; positive = posterior end higher.
- Translation (the findings schema's `Listhesis.translation_mm`): **positive = anterior**. Measured as the offset of the
  upper body's posterior-inferior corner from the lower body's posterior-superior corner, projected on the lower
  superior endplate. This is a posterior-wall displacement in the Meyerding tradition; the published Taillard procedure
  was not re-read for this implementation.
- Percent slip: |translation| / AP depth of the lower body x 100 (Meyerding).
- Level names: a level whose name is `probable` or `unconfirmed` is reported as `L4-L5 (probable)` and carries a note.
  The findings wiring caps its confidence (`spinescope.analysis.findings`).

## Thresholds and sources

Every threshold is in `spinescope/config.py` with its source. None was set on the validation split.

| key | value | source and status |
|---|---|---|
| `alignment_grade_edges_pct` | 25, 50, 75, 100 % | Meyerding HW. Spondylolisthesis. Surg Gynecol Obstet 1932;54:371-377. Grades I to IV are percentages of the lower body's AP depth; V is above 100 % (spondyloptosis). |
| `alignment_listhesis_min_mm` | 3.0 mm, either direction | Convention. A translation of 3 mm or more is the cut-off most commonly used to call a retrolisthesis or a clinically relevant slip. It is not validated against outcomes. The source of the 3 mm figure needs checking against the primary papers before it is cited in a clinical claim. |
| wall-corner method | `body_plane_geometry` | Corners on the posterior and anterior body walls, with the endplate line met to the wall line. Validated only on the phantom (below). |

Supporting references (background, not thresholds): Kirkaldy-Willis WH, Farfan HF. Instability of the lumbar spine.
Clin Orthop 1982;165:110-123 (the instability concept). Roussouly P et al. Spine 2005;30(3):346-353 (lordosis
classification, standing radiographs). Been E, Kalichman L. Lumbar lordosis. Spine J 2014 (review). The supine versus
standing difference in lordosis is a known effect; the magnitude quoted in those reviews should be checked against
the source before it is cited. **The citations above were written from memory and have not been re-checked against the
papers (volume, pages). Check them before any external use.**

## Limitations of the measurement (phantom evidence)

Phantom tests use the phantom's own body shapes, measured with the labeller's `body_plane_geometry` on the labeller's
plane grid (`tests/_alignment_truth.py`). Results:

- **Straight spine** (no lordosis): translation is exact. Zero slip gives 0.0 mm at every level; a 6 mm posterior
  slip gives -6.0 mm; a 5 mm anterior slip gives +5.0 mm (tests in `tests/test_analysis_alignment.py`).
- **Tilted bodies, default phantom**: the zero-slip baseline is about +3.5 mm at L4-L5 (truth 0 by construction; the
  phantom's posterior walls are not collinear across the lordotic chain). Differences between slipped and unslipped
  spines are exact; the absolute level of the baseline is biased for steeper bodies.
- **Endplate fit, steep bodies.** `body_plane_geometry` fits each endplate over the middle columns of the body's global
  AP extent. A body tilted more than about 20 degrees has its sloped posterior wall inside that range, so the endplate
  comes out too flat. Phantom: L1 at -26 degrees measured -19; the S1 superior endplate at 28 degrees measured about
  13. Ground-truth masks scored with the same fit cancel this bias, so the SPIDER geometric reference below cannot
  reveal it. Fixing it means changing the endplate fit in `anatomy/vertebrae.py`, which is not done in this wave.
- **Sacral endplate.** Because of the bias above, the S1 superior endplate is not used (`SACRAL_ENDPLATE_USABLE =
  False`). Lordosis is L1-L5, and L5-S1 is not assessed. The lowest disc is the most common site of spondylolisthesis,
  so this gap matters clinically and needs the endplate fix.
- **Single slice.** Each body is measured on one mid-sagittal slice (the labeller's). `Anchors` keeps no neighbour
  slices, so the production path has no neighbour check. The neighbour check is done on ground-truth masks in
  `tools/eval_alignment.py`, which reports how much the measurement moves across adjacent slices.
- **Posture.** A supine study underestimates standing lordosis, and a slip on one supine study shows the slip in that
  posture only. Instability is not judged from one study, so the output says "measured", never "unstable".

## Validation (SPIDER, pre-registered)

**Rule fixed before any validation number was produced (2026-11-15).** Scored on the `validation` split only.

- Predictor (`alignment` -> truth field `spondylolisthesis`): 1 when the level's |translation| is at least
  `alignment_listhesis_min_mm` (3.0 mm), in either direction; 0 when measured and below 3.0 mm; not assessed when there
  is no translation (every L5-S1 disc, and any disc without endplates on both sides).
- Score for AUC: |translation| in mm.
- Headline metrics (`tools/eval_analysis.py`): sensitivity, specificity, PPV and NPV, with 95 % percentile bootstrap
  intervals over patients (1000 resamples, seed 0), over the scored pairs. AUC is secondary.
- Truth: the SPIDER `spondylolisthesis` column, with 1 read as "listhesis present". The codebook is not in the local
  dataset, so the code meaning is unverified (see `docs/analysis/README.md`). The prevalence (42 of 1520 discs, 2.8 %)
  suggests 1 is the rare finding.
- Geometric reference (no radiologist involved): the same translation measured on the SPIDER ground-truth vertebra
  masks, with the same functions, on the same slice. Reported as MAE in mm, and as agreement at the 3 mm cut-off. The
  tool is `tools/eval_alignment.py`.
- Any change to the measurement after this point is reported as a new version, with the old numbers kept.

### Results: pre-registered rule (3 mm), SPIDER validation, 39 patients

Coverage: 280 predicted discs, 272 reference discs, 270 matched, 264 scored by the harness (all fields). For
`spondylolisthesis`: 220 scored discs, 44 not assessed (every L5-S1, and discs without endplates on both sides). Only **4
positives** among the 220 (cases 31, 107, 152 and 234; L4-L5 in three, L3-L4 in one).

| metric | value [95 % case-bootstrap CI] |
|---|---|
| sensitivity | 1.00 [1.00, 1.00] (4 of 4; the interval is degenerate with 4 positives and is not informative) |
| specificity | 0.79 [0.74, 0.84] (171 of 216 negatives correct; 45 false positives) |
| PPV | 0.08 [0.02, 0.16] |
| NPV | 1.00 [1.00, 1.00] |
| AUC, score = abs(translation) | 0.94 [0.85, 1.00] |

Reading. The four SPIDER positives have large slips (median 13 mm measured; 7 to 49 mm on the reference masks), so the
measurement finds them. Specificity is poor: at 3 mm the false positives sit close to the noise of the method. On normal
discs the measured translation has a median of 1.1 mm and a 90th percentile of 5.3 mm.

**The false positives are mostly not labeller error.** On the same discs the SPIDER ground-truth masks give a translation
of at least 3 mm for 35 of the 45 false positives. The method itself (posterior-wall displacement on a lordotic, supine,
single-slice measurement) produces 3 mm or more on normal discs. A 3 mm rule therefore has a PPV that cannot be made
useful on this kind of data, whatever the labeller.

### Geometric reference (labeller vs ground-truth masks, same functions, same slices)

`tools/eval_alignment.py`, validation, 39 patients, 220 matched discs. The reference is the same listhesis function on the
SPIDER vertebra masks, on the labeller's slice:

| measure | value |
|---|---|
| MAE, predicted minus reference | 1.63 mm (median absolute 0.92 mm; RMSE 2.82 mm) |
| signed bias | -0.32 mm |
| agreement at the 3 mm cut-off | 0.87 |
| by numbering status: confirmed / probable / unconfirmed (n) | MAE 1.33 (76) / 1.87 (43) / 1.76 (101) mm |
| neighbouring slices, reference: median change (439 pairs) | 0.63 mm |

The labeller's bodies are close to the masks (MAE 1.6 mm). The endplate-fit bias on steep bodies does not show here,
because the same fit is applied to both sides (see Limitations). The neighbour check says one slice moves the reference by
about 0.6 mm (median), so the single-slice measurement is stable to about that much. It does not show the method's noise
floor, which is what the 3 mm false positives above reflect.

Name mismatches: the reference tool drops 7 discs whose level name does not match the reference position (the harness
drops 6; the two use slightly different one-to-one matching). Not measured: 43 (every L5-S1 and discs without endplates on
both sides).

Level names are carried through: of the 220 pairs, 76 are confirmed, 43 probable and 101 unconfirmed. The reported level is
written with its status (for example `L4-L5 (probable)`), and the findings wiring caps its confidence. Unconfirmed
numbering is the most common state on SPIDER, so the figures above mix statuses; these are the numbers the analysis
reports, not a cleaned subset.

### Second threshold, chosen on training (pre-registered before the training run; selected 3 mm)

The 3 mm rule has poor specificity (see Results). A second threshold is chosen on the **training** split only, by a
fixed rule written here before the training run was made:

- Candidate thresholds on |translation|: 2, 3, 4, 5, 6, 8 and 10 mm (`SWEEP_MM` in `tools/eval_alignment.py`).
- Choose the candidate with the largest Youden's J (sensitivity + specificity - 1) on the training pairs; ties go to the
  lower threshold.
- Apply the chosen threshold once to the validation pairs and report sensitivity, specificity and PPV with
  case-bootstrap 95 % intervals (`threshold_sweep`, 1000 resamples, seed 0).
- **Disclosure:** the validation sweep table was printed once while the sweep tool was being written, before the training
  run. So the validation numbers at the chosen threshold are not an independent holdout. The pre-registered 3 mm rule
  stays the primary result.

### Training sweep and outcome of the second threshold

Training split (`eval_analysis.py --split training`, 1022 scored discs, 15 positives in 15 patients). Sensitivity,
specificity, PPV and Youden's J (sensitivity + specificity - 1) for |translation| >= threshold:

| threshold (mm) | TP | FP | FN | TN | sensitivity | specificity | PPV | Youden J |
|---|---|---|---|---|---|---|---|---|
| 2 | 12 | 300 | 3 | 707 | 0.80 | 0.70 | 0.038 | 0.50 |
| **3** | 12 | 163 | 3 | 844 | 0.80 | 0.84 | 0.069 | **0.64** |
| 4 | 8 | 115 | 7 | 892 | 0.53 | 0.89 | 0.065 | 0.42 |
| 5 | 6 | 85 | 9 | 922 | 0.40 | 0.92 | 0.066 | 0.32 |
| 6 | 5 | 64 | 10 | 943 | 0.33 | 0.94 | 0.072 | 0.27 |
| 8 | 5 | 47 | 10 | 960 | 0.33 | 0.95 | 0.096 | 0.29 |
| 10 | 4 | 33 | 11 | 974 | 0.27 | 0.97 | 0.108 | 0.23 |

The rule selects **3 mm**, the same threshold as the pre-registered rule. So there is no second threshold: the validation
result above is the result for the chosen rule, and no new validation pass was run. Even on training the PPV stays
below 0.11 at every threshold. The measure separates positives from negatives on magnitude, but at this prevalence
(about 1.5 % of discs) it cannot give a useful positive call per disc. Any listhesis output should be a measurement to
review, never a flag.

## How to reproduce

```sh
.venv/bin/python tools/eval_analysis.py --dataset spider --split validation --analyses alignment \
    --workers 4 --boot 1000 --seed 0 --pairs --out /tmp/spinescope-dx04_listhesis/val
.venv/bin/python tools/eval_alignment.py --split validation --jobs 3 --out /tmp/spinescope-dx04_listhesis/geo
.venv/bin/python tools/eval_analysis.py --dataset spider --split training --analyses alignment \
    --workers 3 --boot 1000 --seed 0 --pairs --out /tmp/spinescope-dx04_listhesis/train
.venv/bin/python tools/eval_alignment.py --sweep-pairs /tmp/spinescope-dx04_listhesis/train/pairs.csv --out /tmp/spinescope-dx04_listhesis/train_sweep
.venv/bin/python tools/eval_alignment.py --sweep-pairs /tmp/spinescope-dx04_listhesis/val/pairs.csv --out /tmp/spinescope-dx04_listhesis/val_sweep
```

Outputs stay outside the repository (they are derived from SPIDER).
