# Validation of spinescope.anatomy.label_spine

Status: **not yet accurate enough to use**. The targets (disc recall and precision >= 0.9, median
centre error <= 3 mm, label accuracy >= 0.8 on held-out SPIDER) are **not reached**. Numbers are for
the classical labeller on branch `p1/anatomy_classical`.

## Data and protocol

- Dataset: SPIDER lumbar spine MRI (van der Graaf et al., Scientific Data 2024; CC BY 4.0). Sagittal
  T1 and T2 volumes with vertebra, disc and canal masks. All 447 volumes (360 `training`, 87
  `validation`) are run; the labeller has not been calibrated on either subset yet.
- Ground truth from the masks (`spinescope.anatomy._validate.truth_from_mask`): each disc label
  201..2xx is one disc; its voxel centroid is the centre and the smallest-variance axis is the
  normal. Discs are ranked from the bottom and named L5-S1, L4-L5, ... by rank.
- Mask convention, checked on the data: vertebra k lies directly above disc 20k; no vertebra label
  lies below disc 201 in any of the 447 volumes; vertebra and disc counts agree in 425 of 447
  volumes. The sacrum is therefore not labelled. Disc 201 being L5-S1 is an **assumption** that the
  mask alone cannot confirm (the SPIDER paper and dataset page did not state it in a form we could
  read here). Every level name in the tables depends on it.
- Canal truth: the canal median x is taken from the mask through the full image direction matrix.
  Before commit `anatomy: validation canal truth ...` only the x term was used, which gave wrong
  midline truths for the 38 SPIDER volumes whose direction matrix is rotated.
- Scoring: a predicted disc matches a truth disc if their sagittal-plane (y, z) distance is within
  12 mm. Centre error is that in-plane distance. Label accuracy is the exact level name.

## Held-out results (validation, 87 volumes)

| metric | value |
|---|---|
| disc recall | 0.20 |
| disc precision | 0.19 |
| centre error, median | 5.9 mm |
| centre error, 90th percentile | ~9.9 mm (previous run) |
| share of matched discs within 3 mm | ~0.24 (previous run) |
| label accuracy, of all truth discs | 0.06 |
| midline error, median (vs mask canal x) | 11.7 mm; within 3 mm for 17% of volumes |

Breakdown (validation):

| subset | volumes | recall | precision | label accuracy (truth) | median centre error | median midline error |
|---|---|---|---|---|---|---|
| T1 | 38 | 0.26 | 0.27 | 0.12 | 6.2 mm | 6.9 mm |
| T2 | 49 | 0.15 | 0.14 | 0.02 | 5.6 mm | 22.5 mm |
| 1.5 T (Siemens) | 55 | 0.15 | 0.15 | 0.04 | 5.9 mm | 19.4 mm |
| 3.0 T (Philips) | 32 | 0.29 | 0.27 | 0.10 | 5.9 mm | 9.0 mm |

Training (360 volumes, not used for calibration yet): recall 0.18, precision 0.18, label accuracy
0.04, median centre error 4.9 mm, median midline error 9.6 mm. The training and validation numbers
agree, so the labeller is uniformly weak rather than overfit.

## Diagnostics

- Midline on sagittal-only SPIDER: ridge-based criteria (per-slice canal score, track length,
  score per length, score times length, and a centre prior) all leave the median error at
  18 to 22 mm on both subsets. The true canal is not the slice with the strongest CSF ridge.
- Oracle midline: giving the labeller the mask's canal slice (`label_spine` with the midline
  replaced) gives validation recall 0.11, precision 0.11, label accuracy 0.03, median centre error
  6.7 mm. This is **below** the baseline, so midline is not the only bottleneck: the endplate and
  disc chain also fails on SPIDER.
- Failure pattern on the chain (three volumes inspected): the lowest detected interval is often a
  spurious disc below the true L5-S1 (for example -83 mm against a true -42 mm), and levels are
  then numbered from that spurious disc, shifting every label by one. Discs above are placed
  within about 5 mm.
- Phantom xfails (`tests/test_labeling.py`): not fixed in this pass; they document the same
  chain problem on the synthetic spine.

## Not yet done (planned)

- Joint dynamic-programming choice of all discs under spacing and height priors, replacing the
  lowest-disc-first chain; sacrum and L5-S1 identification from several cues with a numbering
  confidence; a straightened image along the column; a per-position disc-likeness score; midline by
  symmetry or canal-surface fitting; calibration of thresholds on `training` only.

## Reproduce

The evaluation scripts used for these numbers are not committed (they run the public data
under `~/data/public/spider`). They compute, per volume: `truth_from_mask`, `label_spine(load_study(...))`
and `score_anchors`, then aggregate with `aggregate`.
