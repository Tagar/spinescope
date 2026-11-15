# lab35: test-time augmentation (TTA) uncertainty for the learned labeller

Branch `lab/lab35_uncertainty` (from main 62284ec). Owns `anatomy/ml/tta.py` (variants and agreement), the TTA hook in
`anatomy/pipeline.py`, the `Config.ml_tta*` fields, `anatomy/ml/postprocess.py` (`plane_shift`, additive), the TTA
features and loader in `anatomy/calibration.py`, and the `--tta` collection and `fit-tta` protocol in
`tools/calibrate_labeling.py`.

Status: **optional, off by default, not shipped.** TTA disagreement did not improve the numbering status over calibration
v2 on the held-out sets, so no `calibration_tta_v1.json` is packaged. `Config.ml_tta` stays `False`. The code path is kept
and tested so the result can be reproduced.

Raw outputs are under `/tmp/spinescope-lab35_uncertainty`. Nothing derived from SPIDER, Sudirman or qx5rt is committed.

## 1. Answer

Variant agreement is informative about disc position (centre spread predicts disc centre errors above 3 mm on SPIDER
validation and qx5rt), but it does not flag wrong numberings better than calibration v2:

| held-out set | studies | wrong numberings | v2 announced / wrong / precision / coverage | TTA announced / wrong / precision / coverage |
|---|---|---|---|---|
| SPIDER validation | 87 | 2 | 75 / 0 / 1.000 / 0.86 | 77 / 0 / 1.000 / 0.89 |
| Sudirman half B | 120 | 11 | 72 / 1 / 0.986 / 0.60 | 100 / 2 / 0.980 / 0.83 |
| qx5rt | 55 | 8 | 36 / 1 / 0.972 / 0.66 | 40 / 1 / 0.975 / 0.73 |

"Announced" means `confirmed` or `probable`, at the thresholds each model chose on the fit set. The v2 baseline is the
shipped `calibration_v2.json` scored on the same records as TTA. TTA raises coverage on every set, and on Sudirman half B it
announces one more wrong numbering. The qx5rt wrong numbering stays announced. That is not an improvement, so nothing ships.

## 2. Variants (anatomy/ml/tta.py)

The order is `TTA_ORDER`; `Config.ml_tta_variants` takes the first N. The default N is 4.

| variant | what changes | network passes | valid for sagittal spine MRI |
|---|---|---|---|
| `rot_pos`, `rot_neg` | in-plane rotation of every slice by +/-4 deg (`ml_tta_rotation_deg`) about the slice centre; the probability maps are rotated back to the base grid before any anatomy step | 1 each | yes (brief: 3-5 deg) |
| `gamma` | contrast: power 0.8 on the normalised slices (`ml_tta_gamma`) | 1 | yes |
| `plane_up`, `plane_down` | the analysed mid-sagittal plane moved one slice (re-centring) | 0 (reuses the output) | yes |
| `resolution` | network on a 1.2 mm grid (`ml_tta_resolution_mm`) | 1 | yes |
| `lr_reverse` | slice order reversed | 1 | valid, but degenerate here (see 6) |

Aggregation (`aggregate`): each base disc is matched to the nearest variant disc within half the smallest level spacing
(12.5 mm, sagittal plane; the duplicate-disc rule). Per base disc: `presence`, `name_agree` (same count-from-bottom label),
`spread_mm` (largest 3-D centre displacement of a matched variant disc), `mean_prob`. Per study: the six features of
`TTA_STUDY_FEATURES` (`tta_count_agree`, `tta_chain_agree`, `tta_presence_min`, `tta_name_min`, `tta_spread_max`,
`tta_bottom_name`). The features are computed on the labeller's chain before any numbering step, and recorded in
`volume_feature_dict`, `numbering.tta`, and per-disc evidence lines.

Checks: the rotation maps were checked numerically (a blob rotated by +/-4 deg lands at the predicted pixel; the
round-trip error is 0.01 of the image range). A test rotates the oracle probability maps and brings them back: the discs come
back within 1.5 mm.

## 3. Protocol (fit-v2 style)

- Fit set: SPIDER training subset (180, evenly spaced, as lab18) + Sudirman half A. 300 studies, 19 wrong numberings.
- Evaluation, not used for any choice: SPIDER validation (87), Sudirman half B (120), qx5rt (55).
- Candidates: `v2_baseline` (the v2 volume set) and five TTA sets, each with L2 in {0.1, 1, 3}, chosen by 5-fold out-of-fold
  log-loss on the fit set only. Thresholds from the out-of-fold predictions (targets 0.98 / 0.97, as v2). The v1 disc model
  is kept.
- Decision rule, written before the held-out TTA numbers were computed: ship only if, on the held-out external sets, TTA has
  fewer or the same wrong announced numberings as v2 with equal or higher announced precision, and no new wrong announced on
  SPIDER validation. Otherwise keep the code path off and document it.
- Records were re-collected with `--tta` on current main (`collect --tta`, same subsets). Their base features are not
  identical to the lab18 caches. Example: 257_t2 now has `sacrum_below` 1, where the lab18 cache has 0. Main has changed
  since lab18, so the published v2 numbers are not reproduced exactly. All v2 comparisons here use the new records. The
  shipped `calibration_v2.json` was fit on the older records, so it is slightly out of distribution on current main.

Result of the selection: `v2+presence+name` (L2 = 3) had the lowest fit-set out-of-fold log-loss (0.158, against 0.176 for
`v2_baseline` refit on the same records). Its thresholds were confirmed 0.96 and probable 0.90. The lower fit-set loss did
not carry over: on Sudirman half B it announces 28 more studies and one more wrong numbering than v2. Rule outcome: **not
shipped**.

## 4. Disc level: does centre spread predict disc errors above 3 mm?

Over all matched predicted discs with a TTA record (`fit-tta`, `disc_spread`):

| set | discs | centre error > 3 mm | AUROC spread vs error > 3 mm | AUROC spread vs wrong label | flagged at spread > 1 / 2 / 3 mm (precision of error > 3 mm) |
|---|---|---|---|---|---|
| SPIDER validation | 626 | 29 | 0.88 | 0.64 | 204 (0.12) / 55 (0.27) / 19 (0.32) |
| Sudirman half B | 360 | 74 | 0.66 | 0.67 | 135 (0.30) / 20 (0.55) / 5 (0.60) |
| qx5rt | 217 | 17 | 0.88 | 0.35 | 68 (0.22) / 20 (0.35) / 12 (0.42) |

Presence is nearly always 1 (AUROC 0.5 for both targets), so it carries no information. Name disagreement gives AUROC 0.47 /
0.81 / 0.61 for a wrong label, so it is not consistent. Spread is a usable ranking signal for large centre errors, but
the flags are weak: most flagged discs are correct, and the precision at 3 mm is 0.3 to 0.4 on the external sets.

## 5. Cost

Single-study timing on one SPIDER validation study (one run each, not a median, machine under no other load in this run):
base `label_spine_details` 2.06 s, with 4 variants 3.63 s. The overhead is 3 extra network passes plus the warps. The
brief's 0.5 s figure is the labeller alone. TTA with N=4 costs about 1.6 s per study, roughly 0.8 times the base run. The
plane variants are free. Keep N small: N=2 (rotations) is the cheapest option that still tests the in-plane geometry.

## 6. Notes on specific variants

- L-R slice-order reversal: the labeller is a 2-D network with one input channel per slice (`[batch, 1, h, w]`), so the
  network output of each slice does not depend on its neighbours. Reversing the order and reversing the output back returns
  the base result by construction. The variant can only detect a non-determinism in the plane choice. It is kept for
  completeness and excluded from the default count. It was not measured separately.
- The plane variants re-use the base network output, so their disagreement comes only from the mid-sagittal plane
  choice and the 3-D refinement.

## 7. Decision and what ships

- `Config.ml_tta = False` (default). `ml_tta_variants = 4`. The default path is unchanged: the golden benchmark and the
  default tests are the gate (see the run log under `/tmp/spinescope-lab35_uncertainty`).
- `calibration_tta_v1.json` is not packaged. With `ml_tta=True`, the numbering confidence is v2, and a warning says the TTA
  calibration is not shipped. The TTA evidence is still recorded in `numbering.tta` and per-disc evidence.
- The `fit-tta` protocol and the `--tta` collection stay in the tool, so a future candidate can be fitted and scored the same way.

## 8. Limitations

- Held-out sets are 87, 120 and 55 studies with 2, 11 and 8 wrong numberings. A difference of one study is within noise.
- The fit set has 19 wrong numberings. The selected candidate was chosen by log-loss on that set.
- The variant set is small (N = 4). A larger N, or variants that change the input more (larger rotations, intensity
  jitter), was not tested.
- Main has drifted since lab18. The v2 baseline on the new records is the fair comparison, but the shipped v2 file was fit
  on the older records.
- qx5rt is CC BY-NC and research use only; its numbers appear here, its data does not.

## 9. Reproduce

    .venv/bin/python tools/calibrate_labeling.py collect --split training --limit 180 --tta --out /tmp/spinescope-lab35_uncertainty/tta --jobs 3
    .venv/bin/python tools/calibrate_labeling.py collect --split validation --tta --out /tmp/spinescope-lab35_uncertainty/tta --jobs 3
    .venv/bin/python tools/calibrate_labeling.py collect --sudirman --limit 240 --tta --out /tmp/spinescope-lab35_uncertainty/tta --jobs 3
    .venv/bin/python tools/calibrate_labeling.py collect --qx5rt --tta --out /tmp/spinescope-lab35_uncertainty/tta --jobs 3
    .venv/bin/python tools/calibrate_labeling.py fit-tta --out /tmp/spinescope-lab35_uncertainty/tta

`fit-tta` writes `calibration_tta_report.json` into the output directory. `--write` would ship `calibration_tta_v1.json`; it was
not run.
