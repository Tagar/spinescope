# Validation of the spine labeller

This file describes how well `spinescope.anatomy.label_spine(study)` finds the lumbar discs and names the levels on
the public data we have, as it ships on `main` (commit `1f7612a`, 2026-11-15). Research use only. Not a medical device.

Every number here was measured on that commit with `tools/benchmark.py --refresh` (full run, 11.8 minutes) or with
`tools/train_labeller.py evaluate`. The benchmark numbers match `benchmarks/golden.json`. Where a number comes from an
earlier commit, the text says so.

## 1. What is validated

`label_spine(study)` uses `backend="auto"`, which is the learned labeller when `onnxruntime` is installed. The
components on the default path:

| step | what it does | file | on by default |
|---|---|---|---|
| labeller | U-Net on the mid-sagittal stack: discs, vertebral bodies, canal | `anatomy/ml/assets/labeller_v1.onnx` (5.5 MB) | yes |
| lowest-disc check | decides whether the lowest disc is L5-S1 or a sacral disc | `anatomy/lumbosacral_v1.onnx` (448 KB) | yes |
| numbering status | calibrated confidence, then `confirmed` / `probable` / `unconfirmed` | `anatomy/calibration_v2.json` | yes |
| applicability | says when not to label (wrong region, short field of view, unseen input) | `anatomy/ood.py` | yes |
| image-quality flags | hardware and artefact flags per level | `anatomy/artifacts.py` | yes; advisory flags hidden |
| transitional vertebra | suggestion only; gated on a confirmed or probable numbering | `anatomy/lstv.py` | yes |
| conus | experimental; confidence capped at 0.5 | `anatomy/conus.py` | yes, marked experimental |
| classical second opinion | classical labeller as an extra fusion vote | `config.classical_second_opinion` | no |
| axial slab cue | disc support from oblique T1 slabs | `fusion.FusionConfig.use_axial` | no |
| top-chain count | count from wide sagittal coverage | `label_spine(..., counting=True)` | no |
| test-time augmentation | agreement of input variants as an uncertainty signal | `config.ml_tta` | no |

The classical labeller (`backend="classical"`) is kept for comparison. It is much less accurate (section 6).

## 2. Protocol

### 2.1 Datasets

| dataset | content | licence | use here |
|---|---|---|---|
| SPIDER lumbar spine MRI (van der Graaf et al., Sci. Data 2024) | 447 sagittal T1 and T2 volumes with vertebra, disc and canal masks; 360 `training`, 87 `validation` | CC BY 4.0 | fit and evaluation |
| Sudirman lumbar DICOM (Mendeley, label companion set) | 515 patients, axial and sagittal DICOM; three annotated discs per patient (L3-L4, L4-L5, L5-S1) | CC BY 4.0 | fit (120 patients of the calibration set) and evaluation |
| OSF qx5rt multi-scanner lumbar set (Al Khalil et al., Sci. Data 2022) | 55 sessions: Philips 39, Siemens 15, GE 1; annotated disc masks L1-L2 to L4-L5 | **CC BY-NC 4.0** | evaluation only; numbers only, no images, no redistribution |
| synthetic phantom (`spinescope.testing.phantom`) | 23 scenarios, two looks (cartoon and realistic) | Apache-2.0 | robustness tests; not a clinical accuracy test |

### 2.2 Splits and how they were used

- **SPIDER training (360 volumes)** fits the labeller (`tools/train_labeller.py`) and the lowest-disc check
  (`tools/lumbosacral.py`). The numbering calibration used 180 evenly spaced training volumes.
- **SPIDER validation (87 volumes)** is the held-out set for every SPIDER number in section 3. It was read during
  development. In lab09 the validation bias table was printed once while exploring; the values that were chosen came
  from the training split. The final configuration was then run once for the report. The disclosure is in
  `docs/labeling/lab09_errors.md`, "Calibration and data use".
- **Sudirman** is split by patient id. The calibration fit used 240 patients (`--limit 240`): half of them (120) to fit
  and the other 120 as the held-out half. The Sudirman numbers in section 3 cover all 515 patients, so they include the
  fit patients. They are not a clean held-out figure. Re-running the calibration protocol gives the held-out number.
- **qx5rt** was never used for fitting. It is the cleanest external check. With 55 sessions, one session moves a rate by
  about 2 points.
- **Pre-registration and blind tests.** The AI-reader experiment (lab24) committed its answer key by hash before the key
  was opened. The calibration protocol is written in `tools/calibrate_labeling.py` (`fit-v2`, `fit-v3`). Candidate models
  were judged by a selection rule written before their scores were read (lab25).
- **Calibration features.** The features were chosen after looking at wrong cases on the external sets (Sudirman and
  qx5rt). That is disclosed in `docs/labeling/lab18_reliability.md`, section 4.3. The external reliability numbers are
  therefore slightly optimistic.

### 2.3 Scoring

Shared scorer: `spinescope.anatomy._validate` (volumes), with the level-wise scorers `_validate_sudirman.py` and
`_validate_qx5rt.py`. The benchmark runner is `spinescope/benchmark.py`.

- **Truth on SPIDER.** Each disc label 201 to 2xx is one disc. Its centre is the 3-D voxel centroid of the mask, and its
  normal is the principal axis. Level names are counted from the lowest disc, which is taken as L5-S1.
- **Matching.** A predicted disc matches the nearest truth disc in the sagittal plane (y, z) within 12 mm.
- **Centre error.** The in-plane (y, z) distance between the predicted and the true centre, in mm. The SPIDER truth is a
  3-D centroid, so this measure favours the learned 3-D centre. The Sudirman and qx5rt truths use other definitions
  (operator-planned slabs and annotated masks), so their centre numbers are the stronger evidence.
- **Recall** is matched truth discs over all truth discs. **Precision** is matched predictions over all predictions.
- **Label accuracy, matched** is the share of matched discs with the exact level name. **Label accuracy, all truth**
  counts an unmatched truth disc as wrong. This is the headline label number.
- **Fully correct** is the share of volumes in which every truth disc is named right.
- **Numbering status** is counted per study: `confirmed`, `probable` (together "announced") and `unconfirmed`.
  A study is correct when at least one disc matches and every matched disc has the right name.
  **Wrong announced** counts announced studies with a wrong name. This is the failure that matters most.

### 2.4 What "label accuracy" means

Label accuracy measures agreement with the dataset's own vertebra convention. SPIDER labels the lowest lumbar vertebra
as 1, and the sacrum is not labelled. Level names come from counting up from the lowest disc, which is taken as L5-S1.
That is an assumption of the dataset, not a checked fact. The SPIDER authors warn that transitional anatomy (a sacralised
L5 or a lumbarised S1) changes that count. SPIDER has no transitional labels, so this validation cannot measure
transitional anatomy. The `num_vertebrae` field in `overview.csv` counts all labelled vertebrae (5 to 9, mostly 6 to 8),
not lumbar ones, so it cannot flag transitional cases. As a weak proxy, the 16 of 87 validation volumes where the vertebra
and disc counts disagree have label accuracy 99.3 % (all truth). That is a small group and a weak check.

The Sudirman and qx5rt annotations name the levels directly, so label accuracy there has no such assumption.

## 3. Headline numbers

All numbers: main `1f7612a`, labeller fingerprint `fc320eaac6e0eb2f`, benchmark run with `--refresh`, no errors.
Source: `benchmarks/golden.json`, which holds the same values (see section 9 for the one that moved by 0.002).

### 3.1 Volumes and level sets

| set | cases | disc recall | disc precision | label acc., matched | label acc., all truth | fully correct | centre median / p90 (mm) | centre within 3 mm |
|---|---|---|---|---|---|---|---|---|
| SPIDER validation (held-out) | 87 | 99.5 % | 99.2 % | 98.1 % | 97.6 % | 94.3 % | 0.60 / 1.98 | 95.4 % |
| Sudirman (all 515 patients; includes fit patients) | 515 | 99.2 % | n/a | 92.4 % | 91.9 % | 89.1 % | 2.13 / 3.83 | 77.6 % |
| qx5rt (all 55 sessions, CC BY-NC) | 55 | 100.0 % | n/a | 86.2 % | 86.2 % | 85.5 % | 0.78 / 2.60 | 92.2 % |

Precision is not defined for Sudirman and qx5rt, which annotate only a few discs per study.

SPIDER validation, other measures: normal error median 2.3 degrees; off by one level 1.9 % of truth discs; 0.06 false
positives per volume; disc ECE 0.06 and AUROC 0.80; numbering ECE 0.06 and AUROC 1.00.

SPIDER segmentation Dice on the mid-sagittal stack (`tools/train_labeller.py evaluate`, validation): vertebral body 0.889,
disc 0.846, spinal canal 0.888.

### 3.2 SPIDER breakdown (validation)

| group | volumes | recall | label acc., all truth | fully correct |
|---|---|---|---|---|
| 1.5 T (Siemens) | 55 | 99.3 % | 99.3 % | 94.5 % |
| 3.0 T (Philips) | 32 | 100.0 % | 94.3 % | 93.8 % |
| T1 | 38 | 99.3 % | 97.1 % | 92.1 % |
| T2 (TSE) | 38 | 99.6 % | 97.4 % | 94.7 % |
| T2 SPACE | 11 | 100.0 % | 100.0 % | 100.0 % |

Vendor and field strength are confounded in this set (Siemens is all 1.5 T, Philips all 3.0 T). The 3.0 T row is weaker for
label accuracy. The data cannot say whether vendor, field strength or contrast causes that.

### 3.3 qx5rt breakdown (CC BY-NC, numbers only)

| vendor | sessions | label acc., all truth | fully correct |
|---|---|---|---|
| Philips | 39 | 93.5 % | 92.3 % |
| Siemens | 15 | 66.7 % | 66.7 % |
| GE | 1 | 100.0 % | 100.0 % |

The Siemens sessions are the weakest group in these sets. Their wrong names are not announced: the benchmark counts zero
wrong announced names for every vendor (section 4). The GE group is one session and says little.

### 3.4 Phantom suites (robustness, not accuracy)

| suite | labeller | pass | unsure | silent error |
|---|---|---|---|---|
| cartoon look | classical (`phantom_cartoon`) | 3 of 23 | 1 | 19 |
| realistic look | learned (`phantom_realistic`) | 1 of 23 | 22 | 0 |

A scenario passes when every visible disc is found, named and centred within 3 mm at the 90th percentile, with nothing
spurious. `unsure` means not a pass and numbering confidence below 0.5. `silent error` means not a pass with confidence at
or above 0.5. The learned labeller is out of distribution on the cartoon look. There it puts an extra disc-like response
below the true L5-S1, so every name moves down one. See `docs/benchmarks.md`, section 3.

## 4. Reliability of announced labels

The status is the safety signal. A brief or viewer shows level names when the status is `confirmed` or `probable`.
`unconfirmed` means positions only.

| set | status | studies | share of set | precision (correct / studies) |
|---|---|---|---|---|
| SPIDER validation | confirmed | 38 | 44 % | 100 % |
| | probable | 21 | 24 % | 100 % |
| | unconfirmed | 28 | 32 % | 93 % (2 wrong) |
| Sudirman (515) | confirmed | 94 | 18 % | 100 % |
| | probable | 216 | 42 % | 95 % (11 wrong) |
| | unconfirmed | 205 | 40 % | 78 % (45 wrong) |
| qx5rt (55) | confirmed | 11 | 20 % | 100 % |
| | probable | 11 | 20 % | 100 % |
| | unconfirmed | 33 | 60 % | 76 % (8 wrong) |

Announced names were wrong in 0 of 59 SPIDER validation volumes (95 % Wilson interval 0 to 6.1 %), in 11 of 310 Sudirman
patients (2.0 to 6.2 %), and in 0 of 22 qx5rt sessions (0 to 14.9 %). Sudirman's 11 wrong announced names are all
`probable`. Most wrong names of every set sit in the `unconfirmed` group.

Confidence: the numbering confidence ranks right above wrong names well on SPIDER (AUROC 1.00) and moderately on Sudirman
(AUROC 0.81) and qx5rt (0.91). The disc confidence is weaker (SPIDER disc AUROC 0.80). These are benchmark values.

Earlier calibration results (lab18, lab22) were measured on earlier pipelines. Their coverage figures are not the current
ones and are not repeated here.

## 5. Failure modes

Observed on the public data and the phantom. Each has a status or a warning on the output.

- **Extra disc below L5-S1.** A sacral remnant or a sacral segment is labelled as a lumbar disc. Every name then moves up
  one level. The lowest-disc check removes most of them. It caps the status when it is unsure.
- **No sacrum class.** SPIDER does not label the sacrum, so the network cannot see it. "Bone below the lowest disc" is often
  not in view, and `confirmed` then needs other evidence. Many correct studies stay `probable`.
- **Errors higher in the chain.** An extra or missing disc at the top of the field is not caught by any cue. The top-chain
  count cannot find a landmark in the public sets, so it stays off.
- **Short field of view.** A stack that ends at or above the sacrum gives `unconfirmed` with a warning.
- **Other vendors and fields.** The model was trained on Siemens and Philips 1.5 T and 3.0 T lumbar scans. The qx5rt
  Siemens sessions are the weakest group (section 3.3).
- **Hardware and artefacts.** Metal, motion ghosting, phase wrap and bias fields are flagged per level, and the flags cap
  confidence. The metal detector is weak: a screw core is near the noise floor on the realistic look. Advisory flags are
  hidden by default (`quality_show_advisory = False`).
- **Out of distribution.** Cervical or thoracic fields and non-spine inputs are reported by the applicability check. Its
  thresholds were calibrated on SPIDER training; real non-lumbar studies were not available to test them.
- **Two sagittal series that disagree.** The status drops to `probable` at most.

## 6. What was tried and rejected

One line each. The linked report has the full results. "Model card" means `docs/model_card.md`, which has the candidate
tables.

| candidate | result | status | where |
|---|---|---|---|
| classical labeller (earlier chain) | SPIDER validation on the protocol of section 2: recall 0.30, label acc. (all truth) 0.087, centre median 5.2 mm, 4.5 false positives per volume (re-measured here) | superseded; kept as `backend="classical"` | `labeling/lab01_eval.md`, `labeling/lab05_midline.md` |
| symmetry midline (classical path) | SPIDER validation midline median error 11.7 mm to 1.4 mm (lab05) | shipped in the classical path only | `labeling/lab05_midline.md` |
| `labeller_v2` (sacrum class) | SPIDER validation label acc. 0.893, against 0.955 for v1 at that time | rejected | model card, "Candidate labeller_v2" |
| `labeller_v3` (2.5-D input, posterior weight, field-of-view cuts) | no candidate beat the control on a held-out training fold; no validation run | not promoted | model card, "Candidate labeller_v3" |
| disc calibrator refit (lab18) | disc AUROC on SPIDER validation fell from 0.82 to 0.60 (ECE about the same) | not shipped; v1 disc model kept | `labeling/lab18_reliability.md` |
| calibration v3 (lab22) | confirmed precision 1.00 on the held-out sets, but a silent error on the realistic T1-only phantom | not shipped; v2 kept | `labeling/lab22_confirmed.md` |
| axial T1 slab cue (lab20) | labels identical with the cue on Sudirman half; 3 status downgrades on correct cases; the calibration was not fitted with it | off by default | `labeling/lab20_axial.md` |
| classical second opinion (lab14) | agreed geometrically on 5 of 267 SPIDER studies | off by default | `labeling/lab14_perf.md` |
| feature logistic instead of the CNN (lab13) | 84 correct and 3 wrong on SPIDER validation (before lab09) | not kept; CNN used | `labeling/lab13_lumbosacral.md`, section 5 |
| test-time augmentation (lab35) | variant agreement did not improve held-out reliability over calibration v2 | off; not shipped | `labeling/lab35_uncertainty.md` |
| top-chain count from wide coverage (lab34) | no usable landmark (no cervical or rib detector in the public sets) | opt-in; no vote | `labeling/lab34_counting.md` |
| AI reader to check names (lab24) | 0 of 49 wrong names corrected; 16 of 49 flagged unsure | flags only; cannot confirm | `labeling/lab24_ai_verify.md` |
| conus locator v2 (lab26) | 0 of 15 tips within 13 mm of an AI-marked reference, which is itself doubtful | stays experimental, capped at 0.5 | `labeling/lab26_conus_v2.md` |
| transitional vertebra by cues (lab28) | no calibrated likelihood ratios; no labelled set | suggestion only, gated | `labeling/lab28_lstv.md` |

## 7. Known limitations

- **Small sets.** SPIDER validation has 87 volumes and qx5rt has 55. One study moves a rate by about 1 to 2 points.
- **Limited training population.** No GE, no 1.0 T, no open-bore, no paediatric or post-operative spine in SPIDER.
- **Centre error against a 3-D centroid.** The SPIDER centre error favours the learned 3-D centre (lab09). Use the Sudirman
  and qx5rt centre numbers as the independent check.
- **The sacrum is not labelled.** L5-S1 is an assumption of the dataset. Transitional anatomy is not measured.
- **No clinical reader study.** No radiologist has read the labels against the images, and no inter-reader agreement was
  measured. Accuracy is against dataset annotations.
- **Sudirman includes fit patients.** The 515-patient figure is not held out (section 2.2).
- **Phantom results are mechanics, not accuracy.**
- **Latency** depends on the machine and on other load. The median SPIDER labelling time in the run of section 3 was 7.8 s,
  with a model evaluation running at the same time. That is not a speed benchmark.

## 8. Reproduce

```
tools/benchmark.py --refresh --out /tmp/spinescope-benchmark         # all suites, about 12 min with the datasets
tools/benchmark.py --check benchmarks/golden.json                    # the same, and fail on a regression
tools/train_labeller.py --out /tmp/spinescope-anatomy_ml evaluate --subset validation   # SPIDER Dice and sacrum record
```

Datasets are read from `$SPINESCOPE_DATA` (default `~/data/public`); see `docs/datasets.md`. Outputs stay outside the
repository.

## 9. Discrepancies found and fixed in this pass

Each old number was checked against a fresh run on `1f7612a`.

- Model card, SPIDER validation table: label accuracy 0.955 (all truth), centre median 1.5 mm and 90th percentile 3.1 mm,
  disc recall 0.994 and precision 0.986. These were the results before the lab09 post-processing fixes. The current run gives
  0.976, 0.60 mm, 1.98 mm, 0.995 and 0.992. The model card now carries the current values only. The old values are recorded
  here, and the change is in the lab09 commit (`6259594`, section 10).
- `docs/benchmarks.md` said the model card and the benchmark disagreed on SPIDER label accuracy and centre error. The
  cause was the stale model card table. The model card's evaluation and the benchmark agree to four decimals on this commit,
  so single-volume and paired-T2 labelling give the same disc and centre numbers here.
- Validation document, earlier version: a status line that said the labeller was "not yet accurate enough to use", with
  numbers from the classical labeller on another branch. Replaced. The classical numbers are re-measured (section 6).
- Phantom realistic disc AUROC 0.650 to 0.648 and ECE 0.199 to 0.200 between the golden run and this run. Within the golden
  tolerance (0.01). The golden fingerprint and these values were updated to the current run.

## 10. Changelog

| date | commit or branch | change to the labeller or its validation |
|---|---|---|
| 2026-11-14 | `1c9f553` | learned labeller is the default, `label_spine(backend="auto")` |
| 2026-11-14 | `ba97583` (lab05) | symmetry midline (classical path): median 11.7 to 1.4 mm |
| 2026-11-14 | `35b525d` (lab08) | numbering status (`confirmed` / `probable` / `unconfirmed`), calibration v1 |
| 2026-11-14 | `6259594` (lab09) | post-processing: SPIDER validation label acc. 95.5 % to 97.6 %; centre median 1.55 to 0.60 mm |
| 2026-11-14 | `48c3094` (lab10) | qx5rt multi-vendor set added as an external check |
| 2026-11-14 | `33fdcc3` (lab12) | applicability check (when not to label) |
| 2026-11-14 | `32c1ced` (lab13) | lowest-disc check (`lumbosacral_v1.onnx`) on by default |
| 2026-11-15 | `2e647fe` (lab18) | calibration v2 (volume features, bottom gap, lumbosacral margin); on Sudirman half B and qx5rt, wrong names announced fell from 11 to 1 and from 8 to 1 |
| 2026-11-15 | `9484831` (lab20) | axial slab cue, off by default |
| 2026-11-15 | `c2b9718` (lab14) | one network pass per study; classical second opinion off |
| 2026-11-15 | `dccbf9a` (lab23) | red team: IO and geometry checks, contrast-mismatch cue |
| 2026-11-15 | `685c6a9` (lab25) | labeller v3 candidates evaluated on a training fold; none promoted |
| 2026-11-15 | `2e7d5ae` (lab24) | blinded AI-reader experiment (no correction power; the policy defect it found is fixed in `2f86036`) |
| 2026-11-15 | `2f86036` | an AI reader's answer can only lower trust; AI relabels stay unconfirmed |
| 2026-11-15 | `a315d39` and `562f479` (lab28) | transitional vertebra suggestion, gated on numbering status; a suspected transitional vertebra caps the numbering at probable |
| 2026-11-15 | `62284ec` (lab32) | human correction end to end; status from provenance |
| 2026-11-15 | `3a1d33d` (lab34) | counting evidence made opt-in |
| 2026-11-15 | `9b543c6` (lab31) | image-quality flags; advisory flags hidden |
| 2026-11-15 | `1f7612a` (lab35) | TTA uncertainty merged, off; not shipped |
| 2026-11-15 | `lab/lab36_docs` | this document rewritten; model card and benchmark re-measured; golden fingerprint updated |
