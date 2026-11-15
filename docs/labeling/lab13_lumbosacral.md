# lab13: lumbosacral junction, which disc is L5-S1

Branch `lab/lab13_lumbosacral`. Owns `anatomy/lumbosacral.py` (decision and drop), `anatomy/lumbosacral_v1.onnx` and
`lumbosacral_v1.json` (packaged weights and thresholds), `tools/lumbosacral.py` (recipe) and the pipeline hook in
`anatomy/pipeline.py` (`_lumbosacral_step`, status caps). The labeller and calibration are not retrained.

Scope: the dominant remaining naming error of labeller v1 is an extra disc below the true L5-S1 (an S1-S2 remnant or a
sacral segment). Every level name then moves up one (lab02, lab08). The segmentation network was not changed (v2 with a
sacrum class made disc detection worse, see docs/model_card.md).

## 1. What the module does

For the lowest three disc candidates (bottom-up) of the labeller's mid-sagittal slice, it cuts a 96 x 96 mm sagittal patch
at 1 mm around each disc centre (the disc at 40 mm from the patch top, so the bone below is in view) and scores it with a
small 2-D CNN: P(this disc is L5-S1). Then:

| decision | rule | effect |
|---|---|---|
| keep | the lowest candidate has the best P and P >= 0.5 | none |
| drop k | the best candidate is k-th from the bottom (k >= 1) with P >= `drop_min_p` (0.8) | the k lowest discs are removed before numbering; names count up from the next disc; status at most `probable` |
| unsure | any other case (a lower disc is plausible but below the drop threshold, or no candidate is clearly L5-S1) | nothing removed; numbering status capped at `unconfirmed` |
| not assessed | no packaged model, no sagittal stack, or no disc | nothing |

The module only removes discs. It never adds one and never moves the others. The transitional offset votes of
`anatomy/fusion.py` still run on the reduced set (a sacralized or lumbarized L5 is still handled there). Placement: the
decision runs in `pipeline._finalise` before fusion and before the sacrum check, so the fused numbering, the calibration
features and the status all see the reduced set. `anchors.numbering['lumbosacral']` holds the record (status, probabilities,
candidates, drop applied, warnings). `LabelDetails.unfused` stays the labeller output before any change.

Not implemented: axial cues (iliac crest, L5 transverse processes against the sacral ala). Sudirman's axial slabs are used
only through the existing slab-description votes of fusion.

## 2. Why a patch classifier, and what the data allow

A feature model over candidate geometry (height, AP depth, tilt, spacing, nucleus signal, bone and canal runs below the
disc, field of view below it) was built first (19 features, standardised logistic, the same `Logistic` class as the
calibration). It was dropped: SPIDER training has one sacral candidate below L5-S1 in the first 200 training volumes
(two in validation), so a feature model has almost no remnant examples to learn from. The baseline is recorded in
section 5.

The CNN is trained on image patches from SPIDER training volumes, with labels from the masks:

- positives: the L5-S1 disc (mask label 201) on the mid-sagittal slice and one slice either side, centre jittered by
  +-2 mm (1,080 patches);
- negatives: the L4-L5 and L3-L4 discs (masks 202, 203), the S1-S2 remnant region 33 to 38 mm below L5-S1 (455 patches,
  where a sacral disc would sit), and the sacral region 55 to 65 mm below (68 patches). A patch with less than 70 % of
  acquired field of view is skipped.

Negatives are placed at anatomical positions of real volumes, not painted discs, so the net learns the appearance of the
lumbosacral layout (disc, sacral body below, lumbar body above) from images.

## 3. Training and threshold selection (no test data used)

- `python tools/lumbosacral.py patches`: 360 SPIDER training volumes to 2,323 patches (52 volumes held out).
- `python tools/lumbosacral.py train`: 2-D CNN, four conv stages (16, 32, 64, 64 channels, batch norm, ReLU), flattened
  6 x 6 x 64 features, dropout 0.2, one logit. AdamW, one-cycle learning rate (max 2e-3), batch 64, 40 epochs, BCE loss,
  seed 20260, torch 4 threads, augmentation: gamma, contrast, brightness, noise. Training on 308 volumes (85 %).
  Exported to ONNX (opset 17): 448 KB, 1 input, 1 output.
- `python tools/lumbosacral.py select --write`: runs the labeller on the 52 held-out training volumes and the CNN on
  its own candidates; sweeps the drop threshold over {0.5, ..., 0.95}; objective = correct - 2 x wrong drops - missed
  drops. **In the held-out volumes after the lab09 post-processing there are no sacral candidates left**, so every
  threshold scores the same and 0.8 is a judgement default (the tie-break is the nearest to 0.8). It is not a measured
  optimum.

Before lab09 the held-out set had one sacral case, which the CNN handled at P = 1.0; that is the only evidence for
the drop path on training data.

## 4. Results

All numbers: default pipeline, calibrated, `backend="ml"`, labeller as merged on main with lab09 (commit 6259594).
"before" = the same run with the sacral-disc check off (`label_spine_details(..., lumbosacral=False)`), "after" = the
default. Nothing was tuned on SPIDER validation, Sudirman or qx5rt.

### 4.1 SPIDER validation (87 volumes)

| | before (labeller v1) | after (with the check) |
|---|---|---|
| disc recall / precision | 0.995 / 0.992 | 0.995 / 0.992 |
| label accuracy, of all truth discs | 0.976 | 0.976 |
| fully correct volumes | 81 of 87 | 81 of 87 |
| centre error, median / p90 | 0.6 / 2.0 mm | 0.6 / 2.0 mm |
| check decisions | | keep 87, drop 0, unsure 0 |

No label changed. The check agrees with the truth on all 87: the lowest candidate is the true L5-S1 in every volume.
The six volumes that are not fully correct are not sacral errors: 53_t1, 53_t2 and 129_t1 miss the top disc (names
correct), 74_t1 has an extra top disc (names correct), and 38_t1 and 38_t2 (one patient) have an extra disc inserted
between L5-S1 and L4-L5, which shifts the names above it. The two naming errors are `unconfirmed`. The four
`probable` volumes are wrong in count, not in names.

### 4.2 Sudirman (120 of 515 patients, evenly spaced; 360 annotated levels)

| | before | after |
|---|---|---|
| label accuracy, of matched levels | 0.906 | 0.925 |
| L5-S1 accuracy | 0.925 | 0.950 |
| fully correct patients | 105 of 120 | 107 of 120 |
| wrong patients, of which `probable` | 15, 11 | 13, 12 |

Check decisions: keep 117, drop 3, unsure 0. The three drops (0146, 0256, 0415) each remove one disc, and the
annotated L5-S1 is the disc that the drop makes lowest in all three. No patient got worse. The truth rank of the
annotated L5-S1 from the bottom is 0 for 111 patients, 1 for 6, 2 for 2, and one patient (0243) has no matched
L5-S1.

Remaining errors, after the check:

- 5 patients (0042, 0073, 0159, 0185, 0233) have one or two sacral discs below L5-S1. The CNN scores the lowest one
  P(L5-S1) = 1.00 and keeps it. This is the main residual failure: the sacral disc looks like L5-S1 to the patch
  classifier.
- 6 patients (0180, 0437, 0456, 0460, 0492, 0506) have L5-S1 at the bottom and wrong names higher up (mid-chain or
  top-chain extra or missed discs). The check does not address these.
- 0243: no annotated L5-S1 matched.

### 4.3 qx5rt (OSF multi-scanner set, 55 sessions, 217 annotated levels; CC BY-NC 4.0, research use only)

| | before | after |
|---|---|---|
| label accuracy, of matched levels (all sessions) | 0.825 | 0.862 |
| Siemens (60 levels) | 0.600 | 0.667 |
| Philips (153 levels) | 0.908 | 0.935 |
| GE (4 levels) | 1.000 | 1.000 |
| sessions with any wrong level | 10 | 8 |

The `before` value is 0.825 in this run, against 0.811 reported for the labeller baseline on this set. The gap is not
reconciled (same labeller and scorer functions; possibly a run difference). Check decisions: keep 50, drop 2 (both
sessions of subject ID33, Philips and Siemens, both corrected), unsure 3. The three unsure sessions (ID17, ID30, ID32)
had correct names before and after, so they are false unconfirmed flags. Every remaining wrong session is `probable`.

### 4.4 Robustness suite (phantom, 23 scenarios, `backend="ml"`)

Before and after: 0 of 23 pass; outcomes unchanged (17 silent errors, 6 unsure). The check returns unsure on all 23
scenarios (the phantom is out of distribution for the patch classifier), which caps every status at unconfirmed (before:
16 probable and 1 confirmed). That is abstention on out-of-distribution input, not a validation result.

### 4.5 Reliability: the target is not met

"Wrong numbering should not be probable" holds on SPIDER validation for the naming errors (both unconfirmed) and on the
phantom (all unconfirmed), but not on the external sets: 12 of 13 wrong Sudirman patients and 8 of 8 wrong qx5rt sessions
are `probable`. The cause is the five missed sacral discs (CNN P(L5-S1) = 1.00) and the mid-chain errors, which the
check does not see. The `unsure` branch fires rarely (3 of 55 qx5rt sessions, 0 of 120 Sudirman patients), because the
CNN is confident on most inputs.

### 4.6 What was not tuned, and what is a judgement

- `drop_min_p = 0.8` is a judgement default. The held-out SPIDER training selection (52 volumes) found no sacral
  candidates after the lab09 post-processing, so every threshold in the sweep scored the same.
- The calibration (`calibration_v1.json`) was not refitted. Its features come from the pipeline before this change and
  the lab09 post-processing, so the calibrated numbering confidence is stale. The check only caps the status, it does
  not change the calibration.
- No axial cue is used.

## 5. Negative result: feature logistic (for the record)

Standardised logistic on the 19 candidate features, fitted on the SPIDER training candidates (1,080, pre-lab09
labeller), decided on SPIDER validation (87 volumes, pre-lab09 labeller; two volumes need a drop): 84 correct and 3
wrong at drop thresholds 0.5, 0.8 and 0.9 (no unsure decisions at any threshold). The three wrong decisions were not
inspected further. It was not kept: the CNN is the candidate that goes forward.

## 6. Reproduce

    .venv/bin/python tools/lumbosacral.py patches --out /tmp/spinescope-lab13_lumbosacral/cnn --jobs 2
    .venv/bin/python tools/lumbosacral.py train --out /tmp/spinescope-lab13_lumbosacral/cnn --epochs 40
    .venv/bin/python tools/lumbosacral.py select --out /tmp/spinescope-lab13_lumbosacral/cnn --jobs 2 --write
    .venv/bin/python tools/lumbosacral.py evaluate --split validation --out /tmp/spinescope-lab13_lumbosacral/eval --jobs 2
    .venv/bin/python tools/lumbosacral.py evaluate --sudirman --limit 120 --out /tmp/spinescope-lab13_lumbosacral/eval --jobs 2
    .venv/bin/python tools/lumbosacral.py evaluate --qx5rt --out /tmp/spinescope-lab13_lumbosacral/eval --jobs 2
    .venv/bin/python tools/lumbosacral.py report --out /tmp/spinescope-lab13_lumbosacral/eval

Training needs torch and onnx (the `train` extra). Inference needs only onnxruntime (the `ml` extra, already a core
dependency). Nothing derived from SPIDER, Sudirman or qx5rt is committed; the scratch outputs stay in `/tmp`.
