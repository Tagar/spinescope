# Model card: spinescope learned spine labeller (`labeller_v1`)

Entry point: `spinescope.anatomy.label_spine_ml(study) -> Anchors` (optional extra `spinescope[ml]`,
ONNX Runtime on CPU; no PyTorch at runtime). Weights: `src/spinescope/anatomy/ml/assets/labeller_v1.onnx`
(5.5 MB, fp32, 1.38 M parameters). Training and evaluation: `tools/train_labeller.py`.

## Intended use

- Research and education only. Not a medical device; not for diagnosis or treatment decisions.
- Locates the mid-sagittal plane, disc centres, normals, anterior/posterior margins and heights, and
  vertebral-body centres in lumbar sagittal MRI, and numbers levels upward from the lowest disc.
- Output is a suggestion with a confidence. A reader confirms every level before it is used.

## Data

- **SPIDER** lumbar spine MRI dataset: van der Graaf JW, et al. "Lumbar spine segmentation in MR images:
  a dataset and a public benchmark." Scientific Data (2024), doi:10.1038/s41597-024-03090-w; dataset
  doi:10.5281/zenodo.10159290. Licence CC BY 4.0.
- Sagittal T1 and T2 volumes with vertebra (labels 1-25), disc (201-225) and spinal canal (100) masks.
  Label convention checked on the data: disc 20k lies directly below vertebra k, and the lowest disc (201)
  is taken as L5-S1 (checked visually on one volume, where vertebra 1 is L5). SPIDER has **no sacrum label**.
- Split from `overview.csv`: `training` (360 volumes: 172 T2, 158 T1, 30 T2 SPACE) for fitting;
  `validation` (87 volumes: 38 T2, 38 T1, 11 T2 SPACE) held out and used once, for the metrics below.
  Training composition: Siemens 181 (1.5 T 168, 3.0 T 13), Philips Healthcare 148 (3.0 T),
  Philips Medical Systems 31 (1.5 T 30, 3.0 T 1).
  Validation composition: Siemens 55 (all 1.5 T), Philips Healthcare 32 (all 3.0 T).
  **Vendor and field strength are confounded in the validation set**, so the vendor and field rows
  below are the same numbers.
- No GE scanners, no 1.0 T or open-bore systems, no paediatric or post-operative spines, and no
  non-lumbar fields of view are in the training data.
- No SPIDER image or mask is redistributed in this repository or in the weights file.

## Training recipe (`tools/train_labeller.py train`)

- Slices: every sagittal slice whose canal centre lies within 15 mm of the canal centroid (midline plus
  parasagittal). 3,692 slices from 360 volumes.
- Resampling: each slice is resampled to a fixed 1.0 mm grid (rows superior to inferior, columns anterior
  to posterior) by the same code used at inference (`spinescope.anatomy.ml.grid`). Canvas 320 x 320,
  centred on the spine.
- Intensity: per-volume window (0.5 to 99.5 percentiles of foreground voxels), mapped to [0, 1].
- Classes: background, vertebral body (1-99), disc (200-299), spinal canal (100).
- Augmentation: rotation +/-10 deg, scale +/-15 %, translation +/-12 % of the canvas, gamma, contrast,
  brightness, smooth multiplicative bias field, Gaussian noise. **No flips.** Training crops are 256 x 256.
- Network: 2-D U-Net, widths 16/32/64/128/128, 4 down-sampling stages, batch norm, transposed-convolution
  up-sampling, 4-class softmax output. 1,377,972 parameters.
- Loss: cross-entropy with class weights (0.5, 1, 2, 2) plus soft Dice over the three foreground classes.
- Optimiser: AdamW, lr 2e-3, weight decay 1e-4, linear warm-up over 200 steps, cosine decay; batch 12;
  50 epochs; no early stopping and no model selection on any held-out data.
- Seed 20260 (Python, NumPy, PyTorch). Device Apple MPS. About 35 min wall time for training.
- Software: Python 3.12.12, torch 2.15.0, onnx 1.23.2, onnxruntime 1.31.1 (export and evaluation),
  numpy 2.5.4, scipy 1.18.1, SimpleITK 2.5.7. Commit of the training code: `55760a2`.
- Reproduce: `pip install -e '.[dev,train]'`, then
  `python tools/train_labeller.py --spider <root> train --epochs 50 --batch 12 --seed 20260`, then
  `python tools/train_labeller.py --spider <root> evaluate --subset validation`.
  Exact floating-point results depend on the device and may differ across machines.

## Runtime (`label_spine_ml`)

1. Sagittal stack: `study.best("sag")` (T2 preferred, then T1, STIR, PD), as in the classical labeller.
2. Segmentation of every slice on the 1 mm grid, one ONNX session, softmax probabilities.
3. Mid-sagittal plane: slice with the largest canal area, refined by a parabola through the three
   nearest canal areas. Confidence falls when the canal area is flat across slices.
4. Discs: connected components of the disc class on the mid-sagittal slice, plus discs seen only on
   slices one either side (for discs absent from the exact plane). Components below 30 mm2 or with AP/height
   aspect below 1.2 are rejected. Centre, normal (principal axis, superior), AP margins at mid-height and
   height come from the component geometry.
5. Bodies: components of the body class on the mid-sagittal slice, named by the number of discs below them.
6. Numbering: lowest disc is L5-S1; confidence is 0.9 when a body is visible below it and 0.4 when not
   (reported as `numbering.confidence`, with the reason in `numbering.notes`). Because SPIDER has no sacrum
   label, the network does not learn to label the sacrum, so the "body below" check is often unconfirmed.
   The conus is not estimated.

## Held-out results: SPIDER `validation` (87 volumes, run once)

Scoring with `spinescope.anatomy._validate` (truth from the masks; a predicted disc matches a truth disc
within 12 mm in the sagittal plane). Truth centres are 3-D voxel centroids and truth normals are 3-D
principal axes, so centre and normal errors include a small systematic offset from comparing a mid-plane
measurement with a 3-D summary.

| metric | value |
|---|---|
| disc recall | 0.994 |
| disc precision | 0.986 |
| label accuracy, of matched discs | 0.962 |
| label accuracy, of all truth discs | 0.955 |
| centre error, median | 1.5 mm |
| centre error, 90th percentile | 3.1 mm |
| share of matched discs within 3 mm | 0.885 |
| normal error, median | 2.3 deg |

Slice segmentation (all slices of the chosen stack, 1 mm grid, Dice):

| class | Dice |
|---|---|
| vertebral body | 0.889 |
| disc | 0.846 |
| spinal canal | 0.888 |

By group (recall / precision / label accuracy of all truth discs / median centre error):

| group | volumes | recall | precision | label acc. | centre median |
|---|---|---|---|---|---|
| T1 | 38 | 0.993 | 0.982 | 0.952 | 1.61 mm |
| T2 (TSE) | 38 | 0.996 | 0.985 | 0.948 | 1.55 mm |
| T2 SPACE | 11 | 0.989 | 1.000 | 0.989 | 1.42 mm |
| 1.5 T (Siemens, validation) | 55 | 0.990 | 0.998 | 0.976 | 1.53 mm |
| 3.0 T (Philips, validation) | 32 | 1.000 | 0.963 | 0.914 | 1.58 mm |

The 3.0 T / Philips row is the weakest for label accuracy. Vendor and field strength cannot be
separated in this validation set, so the data cannot say which one drives the difference.

For comparison, the classical labeller on the same validation subset (`docs/validation.md`) reaches
disc recall 0.20 and label accuracy 0.06.

Machine-readable output of this evaluation: `eval_validation.json` (per-volume records and groups) is
written to the scratch directory by `tools/train_labeller.py evaluate` and is not committed.

## Synthetic phantom (out of distribution)

`spinescope.testing.phantom` with degeneration, Modic change and a central herniation: five of seven
discs are found and named correctly by count, but the mid-sagittal plane is chosen at x = -15 mm instead
of the canal centre, so centres are about 38 mm off. The phantom is an analytic cartoon with a different
intensity model, so this is a known failure of the model outside SPIDER and not a validation result.
The oracle tests (ground-truth class maps through the same post-processing) pass for all phantom
variants, which isolates the failure to the network's inputs.

## Known limitations

- Trained only on Siemens and Philips 1.5 T and 3.0 T lumbar scans with T1, T2 and T2 SPACE sequences.
  Not evaluated on GE, Canon, Hitachi or other vendors, on other field strengths, on STIR or post-contrast,
  on cervical or thoracic fields of view, or on scans with pathology excluded from SPIDER.
- The numbering rests on the lowest labelled disc being L5-S1. The network cannot see the sacrum as a class,
  so a scan whose lowest disc is L4-L5 will be numbered as if it were L5-S1 with reduced confidence.
- The mid-sagittal plane is taken from the canal area across the stack. A stack with a tilted or partial
  canal, or a missing canal, gives a warning and no anatomy.
- Measured on SPIDER's 3-D truth; not validated against clinical reads. Normal and centre errors are
  against 3-D centroids, not against a radiologist's landmarks.
- Phantom and out-of-distribution inputs fail as described above. Check the midline confidence and the
  warnings before using any output.

## Candidate `labeller_v2` (lab07): trained, evaluated, not shipped

`labeller_v2` adds a fifth class, the sacrum, and was trained with stronger augmentation. It is **not** packaged:
it scored below `labeller_v1` on the same SPIDER validation protocol, so `v1` stays the default. Weights were
not committed; the recipe and tooling are in `tools/train_labeller.py` (`--out` directory, `train` then
`evaluate --model <file>`), and `Config.ml_model` selects a packaged file by name.

### Recipe (differences from v1)

- Classes: background, vertebral body (1-99), disc (200-299), spinal canal (100), sacrum (new). Class weights
  (0.5, 1, 2, 2, 2). 1,377,989 parameters, same U-Net widths 16/32/64/128/128.
- Sacrum pseudo-labels (`sacrum-box-v1`): SPIDER has no sacrum label. On each slice, the box directly below the
  lowest disc (label 201) is labelled sacrum, in that disc's columns, starting 2 mm below its inferior edge and
  ending at 40 mm or at the acquired field of view, whichever is first. Canal pixels are excluded. A box shorter
  than 8 mm (field of view ends above the sacrum) or a lowest disc under 30 mm2 gives no label, so the network
  also sees "no sacrum in view". **The pseudo-label is a position rule, not bone segmentation.** It teaches
  "bone directly below the lowest visible disc", which is not the same as "this bone is S1".
- Augmentation (in addition to v1's rotation, scale, translation, gamma, bias field, noise): partial field of
  view cuts (inferior p=0.3, superior p=0.1, labels cut with the image), in-plane resolution reduction (p=0.3),
  Gaussian blur (p=0.3), stronger bias field, contrast and brightness, a random monotone intensity curve
  (p=0.4, a proxy for other contrasts), and Rician noise (p=0.3). No flips.
- Same optimiser, schedule, batch, seed (20260), epochs (50) and split as v1. Trained on the 360 SPIDER training
  volumes only, on MPS, 2,969 s. Metadata in `train_meta.json`.
- Post-processing: sacrum area below the lowest disc (at least 150 mm2) confirms the lowest disc as L5-S1,
  with the same confidence formula as a body below it (0.9 times the disc term); a field of view that ends less than 8 mm below the lowest disc vetoes that
  confirmation, with a warning. With the v1 model (no sacrum channel) the code path is inert and output is
  unchanged except for two extra `numbering` keys (`sacrum_confirmed`, `fov_below_lowest_disc_mm`).

### Results, SPIDER `validation` (87 volumes, same scorer and protocol, 4 intra-op threads)

| metric | v1 (shipped) | v2 (candidate) |
|---|---|---|
| disc recall | 0.994 | 0.997 |
| disc precision | 0.986 | 0.959 |
| label accuracy, of all truth discs | 0.955 | 0.893 |
| label accuracy, of matched discs | 0.962 | 0.896 |
| off-by-one level, of all truth discs | 1.7 % | 7.3 % |
| volumes with every truth disc named right | 90.8 % | 81.6 % |
| false positives per volume | 0.10 | 0.31 |
| centre error, median / 90th percentile | 1.5 / 3.1 mm | 1.5 / 3.1 mm |
| normal error, median | 2.3 deg | 2.3 deg |
| volumes with a predicted disc below the lowest true disc | 4 | 9 |
| volumes with over-detection (more predicted than true discs) | 5 | 17 |

Sacrum (v2 only; truth = pseudo-label box on the mid-sagittal slice, 85 volumes with a lowest disc on that slice):
sensitivity 0.988 (84 of 85), median Dice 0.905. These numbers are **circular**: the truth is the same position
rule the network was trained on, so they show that the rule was learned, not that the bone is S1.

Numbering confidence (reliability):

| | v1 | v2 |
|---|---|---|
| numbering confidence, all 87 volumes | 0.37-0.39 (all unconfirmed) | 0.82-0.88 (86 of 87 confirmed) |
| volumes with at least one wrong level name (16 for v2, 8 for v1) | confidence 0.39 median, none at 0.5 or above | 94 % at 0.5 or above |

v1's low confidence flags every volume, including the 79 that are fully correct, so it carries no information
about which numbering is wrong. v2's high confidence is wrong in most of its errors, so it is worse for
reliability. Neither model's numbering confidence separates right from wrong numbering; the reader must check the
level names.

### Results, Sudirman lumbar DICOM (100 labelled patients, evenly spaced; 300 levels; in-process)

| metric | v1 | v2 |
|---|---|---|
| level recall | 0.990 | 0.913 |
| label accuracy, of levels | 0.886 | 0.799 |
| label accuracy by level, L3-L4 / L4-L5 / L5-S1 | 0.859 / 0.879 / 0.919 | 0.755 / 0.791 / 0.854 |
| centre error, median / 90th percentile | 2.85 / 4.54 mm | 2.82 / 4.57 mm |
| normal error, median | 2.5 deg | 2.4 deg |

### Robustness, phantom (`tools/robustness_report.py` scenarios, 23 scenarios)

| | v1 | v2 |
|---|---|---|
| PASS | 0 of 23 | 2 of 23 (`fov_T10_sacrum`, `scoliosis`) |
| silent error / unsure | 14 / 9 | 14 / 7 |
| median label accuracy | 0.00 | 0.67 |

The phantom is out of distribution for both models (see the phantom section above), so these are not
validation results; they show that both models fail silently in many scenarios.

### Decision

Ship `labeller_v1` as default. `labeller_v2` is worse than v1 on the primary SPIDER metric (label accuracy
0.893 vs 0.955, off-by-one 7.3 % vs 1.7 %), worse on Sudirman (0.799 vs 0.886) and on numbering reliability. Its
higher disc recall (0.997 vs 0.994) and better phantom label accuracy do not offset that. The likely cause is
the sacrum class competing with the disc class: v2 adds disc detections below the true lowest disc (9 volumes
against 4). That hypothesis is not yet tested. A next attempt would need the sacrum labelled from real sacral
anatomy, or a sacrum-free numbering rule that does not depend on the pseudo-label.

## Lumbosacral check (lab13): `lumbosacral_v1.onnx` (shipped, applied before numbering)

A second network, separate from the labeller, decides whether the lowest disc candidates are the L5-S1 disc or a
sacral disc (an S1-S2 remnant or sacral segment), and removes sacral discs before the names are counted. Entry point:
`spinescope.anatomy.lumbosacral` (`candidate_patches`, `decide`, `drop_lowest`). Pipeline: `_lumbosacral_step` in
`pipeline._finalise`, before fusion. Weights: `src/spinescope/anatomy/lumbosacral_v1.onnx` (448 KB, fp32, one input
`patch` (n, 1, 96, 96), one output `logit`), thresholds and training record in `lumbosacral_v1.json`. Recipe and report:
`tools/lumbosacral.py`, `docs/labeling/lab13_lumbosacral.md`.

### Intended use and data

- Same research-only scope as the labeller. Output is a suggestion; the record in `anchors.numbering['lumbosacral']`
  says what was done.
- Trained on SPIDER **training** volumes only (308 of 360 for the fit, 52 held out for the threshold). Positives: the L5-S1
  disc (mask 201, centre jittered by 2 mm, the mid-sagittal slice and its neighbours). Negatives: L4-L5 and L3-L4 discs,
  the region 33 to 38 mm below L5-S1 (a remnant would be there) and 55 to 65 mm below (sacrum). Patches with less than
  70 % acquired field of view are skipped. 2,323 patches in total (1,080 positive).
- Architecture: 2-D CNN, four conv stages (16/16, 32/32, 64/64, 64), batch norm, ReLU, max-pool, flatten (64 x 6 x 6),
  dropout 0.2, one linear output. AdamW, one-cycle learning rate (max 2e-3), batch 64, 40 epochs, seed 20260, BCE.

### Decision

For the three lowest candidates of the mid-sagittal slice: `keep` when the lowest is the best candidate with
P(L5-S1) >= 0.5; `drop k` when the best candidate is the k-th from the bottom (k >= 1) with P >= 0.8 (judgement default,
not a measured optimum; see the report); `unsure` otherwise, which caps the numbering status at `unconfirmed`. A removed
disc caps the status at `probable`. Without the packaged model the check reports `not_assessed` and nothing changes.

### Results (calibrated default pipeline; before = the same run with the check off; details in docs/labeling/lab13)

| set | before | after | check decisions |
|---|---|---|---|
| SPIDER validation (87) | label acc 0.976, fully correct 81 | identical | keep 87 |
| Sudirman (120 of 515) | label acc 0.906, fully correct 105, wrong 15 (11 probable) | 0.925, 107, wrong 13 (12 probable) | keep 117, drop 3 (all correct), unsure 0 |
| qx5rt (55 sessions, CC BY-NC) | label acc 0.825 (Siemens 0.60, Philips 0.91) | 0.862 (0.67, 0.94) | keep 50, drop 2 (both correct), unsure 3 (names correct) |
| robustness, phantom (23) | 0 pass, 17 silent | 0 pass, 17 silent, status capped | unsure 23 (out of distribution) |

### Limitations

- The check removes at most the lowest discs and never adds one. It does not address mid-chain or top-chain errors; those
  were the remaining errors on SPIDER validation (not sacral) and on Sudirman and qx5rt.
- The main residual failure is a missed sacral disc: the CNN scores it P(L5-S1) = 1.00 (5 of 120 Sudirman patients, 0 of
  87 SPIDER validation). The CNN has no reliable notion of "this is a sacral disc" beyond the training negatives.
- Reliability target not met: wrong numbering stays `probable` on the external sets (12 of 13 Sudirman, 8 of 8 qx5rt).
  The `unsure` branch rarely fires because the CNN is confident on most inputs.
- Out of distribution (the phantom, and the qx5rt Philips sessions with P near 0 everywhere) the check abstains. That is
  the intended direction, not a validation result.
- Not refitted: the volume calibration (`calibration_v1.json`) predates this check and the lab09 post-processing. The
  check only caps the status; it does not change the calibrated confidence.
- No axial cues (iliac crest, transverse processes, sacral ala) are used.

## Attribution

Model weights derived from the SPIDER dataset (van der Graaf et al., Scientific Data 2024,
doi:10.1038/s41597-024-03090-w; dataset doi:10.5281/zenodo.10159290), CC BY 4.0. See `NOTICE`.
