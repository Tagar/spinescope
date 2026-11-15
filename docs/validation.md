# Validation of spinescope.anatomy.label_spine

Status: **not yet accurate enough to use**. Numbers below are the current state of the labeller
(branch `p1/anatomy`), measured on the held-out split of the SPIDER dataset.

## Data and protocol

- Dataset: SPIDER lumbar spine MRI (van der Graaf et al., Scientific Data 2024; CC BY 4.0). Sagittal
  T1 and T2 volumes with vertebra, disc and canal masks.
- Ground truth from the masks (`spinescope.anatomy._validate.truth_from_mask`): each disc label
  201..2xx is one disc; its voxel centroid is the centre and the smallest-variance axis is the
  normal. Discs are ranked from the bottom and named L5-S1, L4-L5, ... by rank. This assumes the
  lowest labelled disc (201) is L5-S1; this is not verified against anatomy.
- Scoring: a predicted disc matches a truth disc if their sagittal-plane (y, z) distance is within
  12 mm. Centre error is that in-plane distance. Label accuracy is the exact level name.
- Held-out: the `validation` subset in `overview.csv` (87 volumes). Calibration uses `training` only
  (no calibration has been done yet; current defaults come from anatomy priors).

## Held-out results (87 volumes, all sequences, both field strengths)

| metric | value |
|---|---|
| disc recall | 0.20 |
| disc precision | 0.19 |
| centre error, median | 5.9 mm |
| centre error, 90th percentile | 9.9 mm |
| share of matched discs within 3 mm | 0.24 |
| normal error, median | 5.1 deg |
| label accuracy, of matched discs | 0.32 |
| label accuracy, of all truth discs | 0.06 |

Breakdown:

| subset | volumes | recall | precision | label accuracy (truth) | median centre error |
|---|---|---|---|---|---|
| T1 | 38 | 0.26 | 0.27 | 0.12 | 5.95 mm |
| T2 | 49 | 0.15 | 0.14 | 0.02 | 6.07 mm |
| 1.5 T | 55 | 0.15 | 0.15 | 0.04 | 6.84 mm |
| 3.0 T | 32 | 0.29 | 0.27 | 0.10 | 5.90 mm |
| Siemens | 55 | 0.15 | 0.15 | 0.04 | 6.84 mm |
| Philips | 32 | 0.29 | 0.27 | 0.10 | 5.90 mm |

## Known failure modes

- Midline: on SPIDER the sagittal slice with the highest canal-ridge score is often lateral
  (x about -20 mm, with the true canal near -3 mm). Foraminal fat and posterior elements give ridges
  as strong as the thecal sac; the cord lowers the sac's contrast at the midline. A canal-band prior
  (45-90% of the AP field) removes the anterior fat edge but not this lateral bias.
- Endplate chain: discs are often missed or mislocated, so numbering is shifted in many volumes.
- The phantom's L2-L3 disc is also missed (see `tests/test_labeling.py`, xfail).

These numbers are the baseline before the planned joint-DP redesign and calibration on training.
