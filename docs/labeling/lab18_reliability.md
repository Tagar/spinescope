# lab18: numbering status that is reliable across datasets

Branch `lab/lab18_reliability` (from main 6c5e20f). Owns `anatomy/calibration.py` (new signals, version 2 model),
`anatomy/calibration_v2.json` (packaged calibration), the signal hook in `anatomy/pipeline.py` (`_finalise`), and
`tools/calibrate_labeling.py` (collection of qx5rt, `fit-v2` protocol). `calibration_v1.json` stays loadable
(`load_calibration(version=1)`). The learned labeller, the lumbosacral CNN and fusion votes are unchanged.

Raw outputs are under `/tmp/spinescope-lab18_reliability`; nothing derived from SPIDER, Sudirman or qx5rt is committed.

## 1. Answer

The status was not reliable outside SPIDER: 12 of 13 wrong Sudirman and 8 of 8 wrong qx5rt numberings were `probable`
under v1. With calibration v2 (two new numbering signals, a different fit set, and a probable precision target of
0.97) the same held-out numbers are:

| held-out set | studies | wrong numberings | v1 `probable` and wrong | v2 `probable` and wrong | v1 announced precision | v2 announced precision | v2 coverage (announced / all) |
|---|---|---|---|---|---|---|---|
| SPIDER validation | 87 | 2 | 0 | 0 | 1.000 | 1.000 | 0.86 (v1: 0.98) |
| Sudirman, half B | 120 | 11 | 10 | 1 | 0.908 | 0.986 | 0.59 (v1: 1.00) |
| qx5rt (CC BY-NC) | 55 | 8 | 8 | 1 | 0.852 | 0.972 | 0.66 (v1: 0.98) |

"Announced" means `confirmed` or `probable`. The wrong numberings that are still announced under v2 are 1 Sudirman
and 1 qx5rt study; every other wrong numbering is `unconfirmed`. The price is coverage: on the external sets about a
third to 40 % of studies move from announced to `unconfirmed`, and most of those are correct. Section 5 gives the full
trade-off curves so the operating point can be chosen by the user.

`confirmed` is still unreachable (no sacrum signal before geometry; section 6), so every announced study is `probable`.

## 2. The wrong external cases: what distinguishes them

Wrong numberings fall into three kinds (inspected on overlays and on the records):

1. **Extra disc below L5-S1, the lowest disc is a sacral remnant.** The whole chain is named one level too cranial.
   The lumbosacral CNN scores the remnant P(L5-S1) about 1.0, so the CNN margin is the only cue that fires
   (margin = P(lowest) minus the best P of the discs above it; the correct studies have margin about 1).
2. **Extra disc inserted between L5-S1 and L4-L5 (SPIDER validation patient 38, both series).** The second-lowest
   disc also scores P(L5-S1) about 1.0 (margin 0.01 to 0.03), and the spacing between the two lowest discs is far
   from the study median (gap deviation 1.2 to 1.3).
3. **Top-chain or mid-chain errors (one extra or missed disc at the top of the chain).** No bottom cue sees these; the
   lumbosacral margin is 1.0 and the bottom spacing is regular. Left unconfirmed by nothing in this wave (section 6).

Flag rates (a study is flagged by a cue; numbers from the collected records, all 5 sets):

| cue | SPIDER train (wrong / right) | Sudirman A (wrong / right) | SPIDER val (wrong / right) | Sudirman B (wrong / right) | qx5rt (wrong / right) |
|---|---|---|---|---|---|
| lumbosacral margin < 0.5 | 0 of 3 / 0 of 177 | 3 of 16 / 1 of 104 | 2 of 2 / 3 of 85 | 2 of 11 / 1 of 109 | 5 of 8 / 3 of 47 |
| bottom-pair gap deviation > 0.3 (`gap_bottom_dev`) | 1 of 3 / 2 of 177 | 2 of 16 / 0 of 104 | 2 of 2 / 3 of 85 | 4 of 11 / 1 of 109 | 5 of 8 / 0 of 47 |
| lowest disc height < 0.8 x median (`h_bottom`) | 0 of 3 / 8 of 177 | 3 of 16 / 0 of 104 | 0 of 2 / 7 of 85 | 2 of 11 / 5 of 109 | 6 of 8 / 1 of 47 |

Signals examined and not used:

- **FOV margin below the lowest disc** (`fov_below_mm`): the sign of its association flips between sets (AUROC 0.15 on
  SPIDER training, 0.68 on Sudirman A). Computed and recorded; not in the model.
- **Vote pattern of the fusion** (`vote_shift_frac`, `p_shift`): almost no votes shift the offset (the offset machinery
  is a no-op on real data, lab08), so these are near constant. Not in the model.
- **S1 body confidence and discs below the conus**: the S1 body and the conus are attached *after* numbering
  (`attach_geometry`, `attach_conus`), so they are not available to the numbering step. They were in the first
  collection as constants (always 0) and were removed. Making them available needs the numbering step moved after the
  geometry (section 6).
- **Iliac crest on axials**: not implemented (no axial landmark module on main).

Disc-size and spacing features were already in v1 (`height_dev`, `spacing_dev`); the new bottom-pair features are
the ones that separate the external wrong cases: a bottom disc much shorter than the median and a bottom gap far from
the median spacing.

## 3. Signals and model

`numbering_signals(anchors, lumbo_p, votes, p_shift, coverage_low_z)` (calibration.py) is computed in
`pipeline._finalise` after the sacral-disc check and before relabelling. It is recorded per study
(`LabelDetails.signals`, `volume_feature_dict`; collection records `signals`, `volume_feature_dict`, `lumbosacral`).
Live signals: `lumbo_p0`, `lumbo_margin`, `lumbo_p1`, `lumbo_available`, `h_bottom`, `h_second`, `gap_bottom_dev`,
`fov_below_mm`, `vote_shift_frac`, `p_shift`, `n_discs`.

The version 2 volume model is `VOLUME_FEATURES_V2` = the six v1 volume features + `lumbo_margin` + `gap_bottom_dev`
(standardised logistic, L2 = 3). The disc model is the v1 disc model unchanged (section 4.3).

## 4. Protocol

### 4.1 Sets

- **Fit set**: SPIDER training subset (180 of 360, evenly spaced, as in lab08) + Sudirman half A (120 of the 240
  evenly spaced labelled patients; halves by sorted patient id, even index). 300 studies, 19 with a wrong numbering.
- **Evaluation, never used for any choice**: SPIDER validation (87), Sudirman half B (120), qx5rt (55 sessions, all).

### 4.2 Choices and what was used to make them

1. Feature set and L2: five candidate sets (v1 six; v1 + margin; v1 + gap; v1 + margin + gap; all 11 live signals) x
   L2 in {0.1, 1, 3}, scored by 5-fold out-of-fold log-loss **inside the fit set only**. Selected: v1 + margin + gap,
   L2 = 3 (OOF log-loss 0.179 against 0.203 for v1 alone). The full 11-signal set is clearly worse (0.27 at best,
   overfits 19 wrong cases). Full table: `calibration_report_v2.json` (`fit.candidates`).
2. Thresholds: out-of-fold predictions on the fit set. `confirmed` target precision 0.98 (unchanged; not reached, so
   the threshold falls back to the default and the cap applies); `probable` target precision **0.97** (v1: 0.90).
   Reason: the base rate of correct numbering is 0.94 on the fit set and 0.85 to 0.92 on the external sets, so a 0.90
   target admits almost every study. The threshold chosen on the fit set is 0.945 (probable), 0.96 (confirmed).
3. Disc calibrator: **v1 disc model shipped unchanged**. A refit on the fit set was computed and reported but not
   shipped: it lowered SPIDER validation disc AUROC from 0.82 to 0.60 (ECE about the same). This was decided after the
   held-out disc numbers were seen, so it is a judgement on evaluation data, stated here.

### 4.3 Disclosure: what the evaluation sets saw

The wrong-case analysis (section 2) and the univariate AUROC table were computed on all five sets, including Sudirman
half B and qx5rt, because the brief asks for that analysis. The features were then chosen by fit-set out-of-fold
log-loss only, but the choice of which signals to compute was informed by those external cases. Treat the qx5rt and
Sudirman half B numbers as held-out for fitting and thresholds, not as blind validation of the feature idea. The
cleanest test is a new external set; qx5rt is the closest we have.

## 5. Results

### 5.1 Numbering status, before (v1 calibration and thresholds) and after (v2)

Announced = `confirmed` or `probable`. "Wrong announced" counts wrong numberings with status `probable` or `confirmed`.

| set | version | announced | coverage | precision of announced | wrong announced | wrong `unconfirmed` | ECE | AUROC |
|---|---|---|---|---|---|---|---|---|
| SPIDER validation (87) | v1 | 85 | 0.98 | 1.000 | 0 | 2 | 0.041 | 1.000 |
| | v2 | 75 | 0.86 | 1.000 | 0 | 2 | 0.064 | 1.000 |
| Sudirman half B (120) | v1 | 120 | 1.00 | 0.908 | 11 | 0 | 0.087 | 0.835 |
| | v2 | 71 | 0.59 | 0.986 | 1 | 10 | 0.062 | 0.925 |
| qx5rt (55) | v1 | 54 | 0.98 | 0.852 | 8 | 0 | 0.160 | 0.596 |
| | v2 | 36 | 0.66 | 0.972 | 1 | 7 | 0.047 | 0.915 |

ECE and AUROC are of the numbering confidence (volume level, 10 bins), from the shared scorer
(`anatomy._validate.reliability`). The v1 "before" uses v1 thresholds with the same caps.

Mean numbering confidence, wrong vs right: Sudirman B 0.54 vs 0.92 (v2) against 0.96 vs 0.99 (v1); qx5rt 0.46 vs 0.92
(v2) against 0.87 vs 0.88 (v1). On SPIDER validation the wrong cases fall to 0.001 (v2) from 0.08 (v1).

The SPIDER validation ECE rises (0.041 to 0.064) because the two wrong studies move from near 0.1 to near 0 and the
correct studies from 0.96 to 0.93, so the calibration is slightly under-confident there. That is a cost on the SPIDER
distribution, not a regression on the external sets.

### 5.2 Selective prediction, v1 and v2 curves (announced if numbering confidence >= t)

Sudirman half B (wrong announced in brackets):

| t | v1 coverage | v1 precision | v2 coverage | v2 precision |
|---|---|---|---|---|
| 0.50 | 1.00 | 0.908 (11) | 0.95 | 0.947 (6) |
| 0.80 | 0.99 | 0.908 (11) | 0.85 | 0.971 (3) |
| 0.90 | 0.96 | 0.930 (8) | 0.78 | 0.989 (1) |
| 0.95 | 0.94 | 0.929 (8) | 0.53 | 0.984 (1) |

The shipped v2 probable threshold is 0.945 (71 announced, coverage 0.59, precision 0.986; the grid above uses 0.95).

qx5rt:

| t | v1 coverage | v1 precision | v2 coverage | v2 precision |
|---|---|---|---|---|
| 0.50 | 0.98 | 0.852 (8) | 0.85 | 0.936 (3) |
| 0.80 | 0.80 | 0.864 (6) | 0.84 | 0.935 (3) |
| 0.90 | 0.76 | 0.857 (6) | 0.78 | 0.953 (2) |
| 0.95 | 0.64 | 0.886 (4) | 0.66 | 0.972 (1) |

Shipped v2 threshold 0.945 gives 36 announced on qx5rt (coverage 0.66, precision 0.972, same as the 0.95 row) and
71 announced on Sudirman B (coverage 0.59, precision 0.986; the 0.95 row has 63).

SPIDER validation: at t = 0.945, v2 announces 75 of 87 with precision 1.00 (v1: 85 with precision 1.00). Full grids
(0.50 to 0.95, step 0.05) are in `calibration_report_v2.json` (`sets.*.curve_v1`, `curve_v2`).

v1's confidence is nearly flat (most studies above 0.9), so its curve cannot trade coverage for precision: it stays
at coverage 0.94 to 1.00 with precision 0.91 to 0.93 on Sudirman B. v2 reaches precision 0.97 at coverage 0.85 and
0.99 at coverage 0.80 on the same set.

### 5.3 Labels (unchanged)

The labeller and fusion are unchanged, so disc recall/precision and label accuracy are the same as main (SPIDER
validation label accuracy 0.976 at lab13; Sudirman and qx5rt per lab13 section 4). Only the status and the numbering
confidence changed.

## 6. Limitations

- **`confirmed` is unreachable.** The status cap needs bone below the lowest disc (`sacrum_below`). In v1 the labeller
  reports no S1 body at numbering time (`attach_geometry` adds one later, 0.4 confidence on the volume checked). Making
  the S1 body available means moving the calibrated numbering step after the geometry is attached (one function call in
  `label_spine_details`, plus recomputing the features). Not done here: it changes the order of the pipeline.
- **The conus locator fails in the default pipeline.** `_conus_after_labels` logs `conus locator failed: 'numpy.float64'
  object does not support item assignment` on SPIDER volumes (the call works when made directly). This is a lab16 path,
  outside this wave; it does not affect numbering status.
- **Coverage cost.** On the external sets 34 to 41 % of studies become `unconfirmed`, and most of them are correct
  (Sudirman B: 39 correct and 10 wrong unconfirmed; qx5rt: 12 correct and 7 wrong unconfirmed). The probable target
  (0.97) is a policy choice made on the fit set; section 5.2 shows the alternatives (on Sudirman B a threshold of 0.80
  keeps coverage 0.85 at precision 0.97), and none of them was chosen on the held-out sets.
- **Small counts.** 19 wrong numberings in the fit set; 2 in SPIDER validation; 11 and 8 in the held-out external sets.
  The fitted weights and thresholds are indicative.
- **Sudirman and qx5rt are subsets for the fit only.** Sudirman: 240 of 515 patients (evenly spaced). qx5rt is CC BY-NC
  and research use only; its numbers appear here, its data does not.
- **Wrong top-chain errors are not caught.** Cue class 3 in section 2 needs a top-of-chain signal (for example a
  thoracic or conus-based count), which is not on main.
- **No iliac-crest or axial landmark cue** (not implemented).

## 7. Reproduce

    .venv/bin/python tools/calibrate_labeling.py collect --split validation --out /tmp/spinescope-lab18_reliability --jobs 3
    .venv/bin/python tools/calibrate_labeling.py collect --split training --limit 180 --out /tmp/spinescope-lab18_reliability --jobs 3
    .venv/bin/python tools/calibrate_labeling.py collect --qx5rt --out /tmp/spinescope-lab18_reliability --jobs 3
    .venv/bin/python tools/calibrate_labeling.py collect --sudirman --limit 240 --out /tmp/spinescope-lab18_reliability --jobs 3
    .venv/bin/python tools/calibrate_labeling.py fit-v2 --out /tmp/spinescope-lab18_reliability --write

`fit-v2` writes `calibration_report_v2.json` into the output directory and, with `--write`, `calibration_v2.json`.
The protocol is fixed in the tool (`V2_CANDIDATES`, `V2_L2_GRID`, `V2_FOLDS`, `_v2_sets`). Collection is about 2 h on a
loaded 14-core machine.
