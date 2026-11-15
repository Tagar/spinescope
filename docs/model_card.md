# Model card: spinescope spine labeller

Research and education only. Not a medical device. Not for diagnosis or treatment decisions.

Numbers in this card were checked on `main` at `1f7612a` (2026-11-15). Where a value is historical, the text says so.
The full validation protocol and the rejected candidates are in `docs/validation.md`.

## 1. Model details

The labeller is a set of packaged files and rules. Entry point: `spinescope.anatomy.label_spine(study)` (default
`backend="auto"`), which runs the learned labeller when `onnxruntime` is installed. The learned parts are:

| file | what it is | size | default |
|---|---|---|---|
| `src/spinescope/anatomy/ml/assets/labeller_v1.onnx` | 2-D U-Net, four classes (background, vertebral body, disc, spinal canal), fp32, about 1.38 M parameters | 5.5 MB | yes |
| `src/spinescope/anatomy/lumbosacral_v1.onnx` (+ `lumbosacral_v1.json`) | 2-D CNN on 96 x 96 mm sagittal patches; P(the disc is L5-S1) | 448 KB | yes |
| `src/spinescope/anatomy/calibration_v2.json` | logistic calibration of the numbering confidence, and the status thresholds (`confirmed` at 0.96, `probable` at 0.945) | 1.6 KB | yes |
| `src/spinescope/anatomy/calibration_v1.json` | the previous calibration | 1.4 KB | no (kept for `load_calibration(version=1)`) |
| `src/spinescope/anatomy/axial_v1.json` | calibration of the axial slab cue | 1.6 KB | no (cue off) |

The numbering itself is a rule: the lowest disc is L5-S1 and the others are counted up from it, after the lowest-disc
check. The status is computed from the calibrated confidence and then capped by the evidence (section 5).

Version history of the labeller: `labeller_v1` is the packaged model. `labeller_v2` and `labeller_v3` were trained and
evaluated but not packaged (section 9).

Training code: `tools/train_labeller.py` (commit `55760a2` for the packaged weights). The lowest-disc check is trained by
`tools/lumbosacral.py`. The calibration is fitted by `tools/calibrate_labeling.py`. Each has `--help`.

Licence: code Apache-2.0. The weights were trained on SPIDER (CC BY 4.0) and carry that attribution (section 10).

## 2. Intended use

- Locate the mid-sagittal plane, the lumbar disc centres and normals, and the vertebral bodies in a lumbar sagittal MRI.
- Name the disc levels from the lowest disc upward, and report how sure the labeller is.
- Output is a suggestion with a status. A reader confirms the names before they are used for anything else. A brief or
  viewer shows names only when the status is `confirmed` or `probable`.

## 3. Out-of-scope use

- Diagnosis, grading, or any treatment decision.
- Anatomy outside the lumbar spine: cervical, thoracic, whole-spine or non-spine inputs. The applicability check reports
  these, but it was not tested on real non-lumbar studies.
- Vendors, field strengths and sequences not in the training data (section 4): GE, Canon, Hitachi, 1.0 T, open bore, STIR,
  post-contrast, and paediatric or post-operative spines.
- Naming levels in a study without the lowest lumbar discs in view, or without a usable sagittal stack.
- Use of an AI reader's answer as a confirmation. An AI answer can only lower trust (section 5).

## 4. Training and evaluation data

### 4.1 Labeller (`labeller_v1`)

SPIDER lumbar spine MRI (van der Graaf et al., Scientific Data 2024, doi:10.1038/s41597-024-03090-w; dataset
doi:10.5281/zenodo.10159290), CC BY 4.0. Sagittal T1 and T2 volumes with vertebra, disc and canal masks. No sacrum label.

| subset (from `overview.csv`) | volumes | sequences | vendor and field |
|---|---|---|---|
| `training` (used to fit) | 360 | 172 T2 (TSE), 158 T1, 30 T2 SPACE | Siemens 181 (1.5 T 168, 3.0 T 13); Philips Healthcare 148 (3.0 T); Philips Medical Systems 31 (1.5 T 30, 3.0 T 1) |
| `validation` (held out) | 87 | 38 T1, 38 T2, 11 T2 SPACE | Siemens 55 (all 1.5 T); Philips Healthcare 32 (all 3.0 T) |

The validation set reads each vendor at one field strength only, so vendor and field cannot be separated there.

Training recipe (as recorded by the original training run; not re-run for this card):

- Slices: sagittal slices within 15 mm of the canal centre, resampled to a fixed 1.0 mm grid by `spinescope.anatomy.ml.grid`.
  Canvas 320 x 320 mm, centred on the spine. About 3,700 slices from the 360 volumes.
- Intensity: per-volume window at the 0.5 and 99.5 percentiles of foreground voxels, mapped to [0, 1].
- Augmentation: rotation +/-10 degrees, scale +/-15 %, translation +/-12 %, gamma, contrast, brightness, a smooth bias field and
  Gaussian noise. No flips (anterior-posterior orientation matters). Training crops 256 x 256.
- Loss: cross-entropy with class weights (0.5, 1, 2, 2) plus soft Dice over the foreground classes.
- Optimiser: AdamW, learning rate 2e-3, weight decay 1e-4, warm-up and cosine decay; batch 12; 50 epochs; seed 20260.
  No early stopping and no model selection on held-out data.
- Software at training: Python 3.12, PyTorch 2.15, ONNX Runtime 1.31 for export and evaluation.
- Device: Apple MPS. Results can differ across machines at the floating-point level.

### 4.2 Lowest-disc check (`lumbosacral_v1`)

Trained on SPIDER `training` only (308 volumes for the fit, 52 held out for the threshold). Positives are the L5-S1 disc
(mask 201, centre jittered by 2 mm). Negatives are the L4-L5 and L3-L4 discs, the region 33 to 38 mm below L5-S1 (a sacral remnant) and the
region 55 to 65 mm below (sacrum). 2,323 patches, 1,080 positive. Architecture: four convolution stages, batch norm, dropout 0.2, one output logit.
Thresholds (`lumbosacral_v1.json`): keep when P(L5-S1) at least 0.5 for the lowest candidate; drop the lowest candidate
when the best one above it has P at least 0.8 (a judgement default, not an optimum); otherwise unsure.

### 4.3 Calibration (`calibration_v2`)

Fitted on SPIDER `training` (180 evenly spaced volumes) and Sudirman half A (120 patients). Evaluated on SPIDER validation,
Sudirman half B and qx5rt. The disc model is the v1 disc model, unchanged. The calibration features are the labeller's own
outputs and the lowest-disc check's margin. Feature choice used the external wrong cases (see `docs/validation.md`, 2.2).

### 4.4 Evaluation sets

- SPIDER validation (87 volumes): the held-out set for the labeller and the numbers below.
- Sudirman lumbar DICOM (515 patients, CC BY 4.0): level-wise, three annotated discs per patient. 120 of the patients were
  used to fit the calibration (section 4.3), so the 515-patient figures are not held out.
- OSF qx5rt multi-scanner set (55 sessions, CC BY-NC 4.0, research use): never used for fitting. Numbers only.
- Synthetic phantom (`spinescope.testing.phantom`, 23 scenarios, two looks): mechanics and robustness, not accuracy.

No SPIDER, Sudirman or qx5rt image or mask is in this repository or in the weights.

## 5. Metrics and results (current main)

Definitions are in `docs/validation.md`, section 2.3. Short form: recall and precision are per disc; label accuracy is the
exact level name; "all truth" counts a missed disc as wrong; "announced" is `confirmed` or `probable`; "wrong announced" is an
announced study with a wrong name.

### 5.1 SPIDER validation (87 volumes, held out)

| metric | value |
|---|---|
| disc recall / precision | 99.5 % / 99.2 % |
| label accuracy, matched discs / all truth discs | 98.1 % / 97.6 % |
| fully correct volumes | 94.3 % |
| centre error, median / 90th percentile | 0.60 mm / 1.98 mm |
| centre within 3 mm | 95.4 % |
| normal error, median | 2.3 degrees |
| segmentation Dice: vertebral body / disc / canal | 0.889 / 0.846 / 0.888 |
| announced studies / wrong announced | 59 of 87 / 0 |

Breakdown:

| group | volumes | recall | label acc., all truth | fully correct |
|---|---|---|---|---|
| 1.5 T, Siemens | 55 | 99.3 % | 99.3 % | 94.5 % |
| 3.0 T, Philips | 32 | 100.0 % | 94.3 % | 93.8 % |
| T1 | 38 | 99.3 % | 97.1 % | 92.1 % |
| T2 (TSE) | 38 | 99.6 % | 97.4 % | 94.7 % |
| T2 SPACE | 11 | 100.0 % | 100.0 % | 100.0 % |

The 3.0 T Philips group is the weakest for label accuracy. Vendor and field are confounded here.

### 5.2 External sets

| set | cases | recall | label acc., all truth | fully correct | announced (coverage) | wrong announced |
|---|---|---|---|---|---|---|
| Sudirman, all 515 (includes calibration fit patients) | 515 | 99.2 % | 91.9 % | 89.1 % | 60.2 % | 11 (all probable) |
| qx5rt, all 55 sessions | 55 | 100.0 % | 86.2 % | 85.5 % | 40.0 % | 0 |

qx5rt by vendor (label accuracy, all truth): Philips 93.5 % (39 sessions), Siemens 66.7 % (15), GE 100 % (one session).
The Siemens sessions are the weakest group in any set we have. None of their wrong names was announced.

### 5.3 Numbering status reliability

| set | confirmed | probable | unconfirmed |
|---|---|---|---|
| SPIDER validation | 38 studies, 100 % correct | 21, 100 % | 28, 93 % (2 wrong) |
| Sudirman (515) | 94, 100 % | 216, 95 % (11 wrong) | 205, 78 % (45 wrong) |
| qx5rt (55) | 11, 100 % | 11, 100 % | 33, 76 % (8 wrong) |

Numbering confidence ranks right names above wrong ones: AUROC 1.00 (SPIDER), 0.81 (Sudirman), 0.91 (qx5rt). Calibration
error (ECE): 0.06, 0.02 and 0.05. These are benchmark values; the confidence is not a probability of a clinically correct
level, only of a correct name under the dataset's convention.

### 5.4 Phantom (robustness)

Cartoon look (classical backend): 3 pass, 1 unsure, 19 silent errors of 23. Realistic look (learned backend): 1 pass,
22 unsure, 0 silent errors. The learned labeller is out of distribution on the cartoon look, and the card does not claim
accuracy from the phantom.

## 6. Status and caps (how the outputs are decided)

- `confirmed`: calibrated numbering confidence at or above 0.96, bone below the lowest disc in view, and no cap. Or a human
  has confirmed a reviewed file after the last change (`spinescope.anatomy.review`).
- `probable`: calibrated confidence at or above 0.945, or a capped `confirmed`: a second sagittal series contradicts the
  names, a suspected transitional vertebra is present, or the lowest-disc check dropped a sacral disc. Or a human reviewed the
  names without confirming them, or an AI reader answered (an AI answer is at most `probable`).
- `unconfirmed`: below the probable threshold, or the lowest-disc check is unsure, or an AI reader changed the names and no
  human has looked since, or the applicability check says the bone is not in view. Positions are still given.

Other caps: the image-quality flags can cap confidence and status per level. The transitional-vertebra suggestion is gated: it
is `not_assessed` when the numbering is `unconfirmed`. The conus is experimental and its confidence is capped at 0.5.

## 7. Ethical considerations and privacy

- Research use. The outputs can be wrong, and the card gives no measure of how often a clinician would be misled. Always
  rely on a qualified radiologist's report.
- Training and test data are public research datasets. No patient identifiers, dates, institutions or physicians are in
  the weights or in any output. The tools keep an allowlist of DICOM attributes (`docs/privacy.md`).
- Under-representation: no GE, Canon, Hitachi or 1.0 T scans in the training data. Results for those groups are unknown, and
  the one GE session in qx5rt is not enough to tell.
- Accuracy differs between vendors in the external set: the Siemens qx5rt sessions are less accurate than the Philips ones
  (section 5.2). The cause is not known.
- Transitional anatomy and pathology are not in the labelled data. The labeller may name levels wrongly in those cases with
  the same confidence it gives normal anatomy, so the status is the only safety signal.

## 8. Caveats and recommendations

- Check the status before using the names. `unconfirmed` means positions only.
- Check the warnings. Field of view, two-series conflicts, hardware and artefact flags are reported there.
- Do not treat an AI reader's answer as a confirmation. A human has to confirm the levels (`docs/guide/correcting-labels.md`).
- Measured on SPIDER's 3-D centroids, not against a radiologist's landmarks. Centre errors are therefore optimistic for
  the SPIDER set (`docs/validation.md`, 7).
- The sacrum is not labelled in SPIDER. L5-S1 is the dataset's convention, and the lowest-disc check is a CNN trained on it.
- Re-measure after any change. `tools/benchmark.py --check benchmarks/golden.json` (see `docs/benchmarks.md`).

## 9. Candidates that were not shipped

Each one is in `docs/validation.md`, section 6, with its report. Summary of the two labeller candidates:

- **`labeller_v2`** (lab07): adds a sacrum class, trained on pseudo-labels from a position rule. It was worse than v1 on the
  SPIDER validation set (label accuracy 0.893 against 0.955, off-by-one 7.3 % against 1.7 %, on the v1 of that time) and on
  Sudirman. The likely cause is competition between the sacrum and disc classes. This is a hypothesis and was not tested
  further. Weights were not committed; the recipe is in `tools/train_labeller.py`.
- **`labeller_v3`** (lab25): 2.5-D input (three slices), a posterior hard-negative weight and field-of-view cuts. Trained on
  a 90 % training fold and scored on a 10 % training hold-out (35 volumes), with SPIDER validation not read. No candidate beat
  the single-slice control on the rule written before the scores were read. Not promoted. Weights not committed.

The numbers of these candidates were measured on earlier pipelines and cannot be reproduced on `main` without their weights.
They are not repeated as current results.

## 10. Attribution

Weights derived from the SPIDER dataset (van der Graaf et al., Scientific Data 2024, doi:10.1038/s41597-024-03090-w;
dataset doi:10.5281/zenodo.10159290), CC BY 4.0. Sudirman lumbar DICOM (Mendeley, CC BY 4.0) and OSF qx5rt (CC BY-NC 4.0)
are used for numbers only; no images or weights are derived from them. See `NOTICE` and `docs/datasets.md`.

## 11. Reproduce

```
tools/benchmark.py --check benchmarks/golden.json                       # every benchmark, with the regression check
tools/train_labeller.py --out <dir> evaluate --subset validation        # SPIDER validation metrics, Dice, sacrum record
tools/train_labeller.py --out <dir> train --epochs 50 --batch 12 --seed 20260   # needs the train extra (PyTorch, ONNX)
tools/lumbosacral.py --help                                              # lowest-disc check
```

Training needs the `train` extra (`pip install -e '.[dev,train]'`) and a dataset root at `$SPINESCOPE_DATA/spider`. Exact
floating-point results depend on the device.
