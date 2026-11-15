# lab22: numbering status `confirmed` is reachable

Branch `lab/lab22_confirmed` (merged with main c2b9718, which includes lab14_perf). Raw outputs are under
`/tmp/spinescope-lab22_confirmed/final`; nothing derived from SPIDER, Sudirman or qx5rt is committed.

## 1. Answer

- The numbering evidence now exists before the calibrated numbering decision. The S1 body below the lowest disc is
  measured from the labeller's own segmentation before numbering, so `sacrum_below` is no longer a constant 0.
- With the shipped calibration v2 (default, unchanged) and the new order, `confirmed` is reachable. On the held-out sets
  the confirmed studies have precision 1.00 (81 studies pooled: SPIDER validation 42, Sudirman half B 25, qx5rt 14).
- Calibration v3 was fitted on the final pipeline with the lab18 protocol. Its held-out wrong-announced counts are not worse
  than v2, and it reaches confirmed precision 1.00. **It is not shipped**: it moves the realistic T1-only phantom to a
  silent error (section 5), so v2 stays the default. The candidate file is outside the repo.

## 2. Stage order (`pipeline.label_spine_details`, ML backend)

1. Segmentation: the primary sagittal stack once (`label_spine_ml_segmented`); the second sagittal series once more if present.
2. Anchors: discs and vertebrae from the network output.
3. Lumbosacral check: sacral candidates removed (`_lumbosacral_step`).
4. Vertebral geometry (`_geometry_step`, `vertebrae.measure_geometry(seg)`): bodies, canal, posterior elements, and the bone
   below the lowest disc (`below_lowest_disc_mm`, 2nd percentile of the body points below the disc's inferior margin).
5. Conus (experimental, weak): located on the primary segmentation (`attach_conus(seg=)`); its tip votes in the fusion.
6. Second series and fusion votes (`fuse_numbering`), then relabelling. Geometry names follow the relabel (`Anchors._sync_geometry`).
7. Calibrated numbering: `sacrum_below` from geometry (`s1_below_mm >= sacrum_min_visible_mm`), the status caps, the status.
8. Applicability (`apply_applicability`), then `_recap_status` and `refresh_conus`.

Network passes: one per stack (checked on SPIDER: 1 for T1-only, 2 for a T1+T2 pair; the conus reuses the primary).
A test records the order and asserts no stack is segmented twice (`tests/test_labeling_lab22.py`).

## 3. What changed in code

- `vertebrae.py`: S1 measured on the slice holding most of the bone below the disc, without the body-area floor;
  `extent_mm` and `below_lowest_disc_mm` on each body record (optional, added to the schema).
- `calibration.py`: `sacral_signals` (`s1_present`, `s1_below_mm`, `s1_ap_ratio`, `t1t2_bottom`), kept separate from
  `numbering_signals` so the version 2 key set is unchanged; `choose_thresholds(targets=, fallback=)`; `TARGET_PRECISION_V3`
  (confirmed 0.99, probable 0.97); `calibration_path(3)`; `load_calibration` default is `DEFAULT_CALIBRATION_VERSION` (2).
- `conus.py`: `refresh_conus` (level, low-lying flag, status factor and experimental cap in the final names); `confidence_raw`.
  The empty-cluster failure of the tip median is fixed by main's consensus rule (no consensus, not located).
- `pipeline.py`: `_finalise` stages as above; `_recap_status` re-applies the caps after applicability (applicability
  recomputed the status from the confidence alone, which could undo the caps, so a study with no bone below could be
  `confirmed`). This is a behaviour fix on main as well.
- `fusion.relabel_anchors`: carries geometry through relabelling.
- `tools/calibrate_labeling.py fit-v3`: the lab22 protocol (section 4); `--write` writes `calibration_v3.json` only if the
  acceptance rules pass.

## 4. Protocol

- Fit: SPIDER training subset (180) + Sudirman half A, 300 studies, 19 wrong (same as lab18).
- Evaluation, never used for any choice: SPIDER validation (87), Sudirman half B (120), qx5rt (55).
- Candidates (5 sets x L2 in {0.1, 1, 3}, fit-set out-of-fold log-loss): `v2_selected`, `+s1_below`, `+morph` (AP ratio),
  `+t1t2_bottom`, `+all_v3`. Selected: `v2_selected`, L2 = 3 (OOF log-loss 0.175; `+s1_below` 0.176, `+t1t2_bottom` 0.179,
  `+morph` 0.183, `+all_v3` 0.187). None of the new geometry signals improved the fit-set log-loss, so the selected model is the
  v2 feature set with `sacrum_below` now live (weight 0.27) and `classical_agree` constant 0 (classical opinion is opt-in).
- Thresholds (out-of-fold, fit set): confirmed 0.975 (target 0.99), probable 0.905 (target 0.97).
- Acceptance (fixed before the evaluation): wrong-announced not worse than v2 (SPIDER 0, Sudirman B 1, qx5rt 1 is the
  reference); confirmed precision >= 0.99 pooled over held-out sets with >= 20 studies.

## 5. Results (held-out, final pipeline, `calibration_report_v3.json`)

Coverage = share of the set's studies at that status; precision = share of those studies with the right numbering.

| set | model | confirmed (n, cov, prec) | probable (n, cov, prec) | unconfirmed (n, cov, prec) | announced wrong |
|---|---|---|---|---|---|
| SPIDER validation (87) | v2 (shipped) | 42, 0.48, 1.00 | 33, 0.38, 1.00 | 12, 0.14, 0.83 (2 wrong) | 0 |
| | v3 candidate | 32, 0.37, 1.00 | 45, 0.52, 1.00 | 10, 0.12, 0.80 (2 wrong) | 0 |
| Sudirman half B (120) | v2 | 25, 0.21, 1.00 | 47, 0.39, 0.98 | 48, 0.40, 0.79 (10 wrong) | 1 |
| | v3 candidate | 6, 0.05, 1.00 | 85, 0.71, 0.99 | 29, 0.24, 0.66 (10 wrong) | 1 |
| qx5rt (55) | v2 | 14, 0.25, 1.00 | 22, 0.40, 0.95 | 19, 0.35, 0.63 (7 wrong) | 1 |
| | v3 candidate | 13, 0.24, 1.00 | 25, 0.45, 0.96 | 17, 0.31, 0.59 (7 wrong) | 1 |

Shipped v2 thresholds are confirmed 0.96 and probable 0.945. Wrong-announced counts under v2 on the new pipeline match lab18
(0 / 1 / 1), so the reordering does not add wrong announcements. Confirmed is new: lab18 had none.

The v3 candidate meets the acceptance rule (wrong-announced 0 / 1 / 1; confirmed pooled 51 studies at precision 1.00).
Its probable threshold dropped from 0.945 to 0.905, and it announces more Sudirman studies (91 vs 72) with the same single
wrong case.

## 6. Why v3 is not shipped

On the realistic phantom (`tests/robustness/test_robustness_realistic.py`), the T1-only case is `unsure` under v2 and a
`silent_error` under v3 (same labels, label accuracy 0.67; v3 announces the numbering). The repo treats silent errors on
that phantom as a gate, so v3 would regress main's test suite. The held-out rule did not include this gate and the threshold
was not chosen with it in view. Revisiting v3 needs a decision on the probable threshold (0.905 vs 0.945) and a rerun of the
robustness gate; the candidate is kept at `/tmp/spinescope-lab22_confirmed/final/calibration_v3_candidate.json`.

## 7. Limitations

- The S1 geometry signals did not add information in the out-of-fold fit (section 4). The bone below the lowest disc
  matters through `sacrum_below` only. SPIDER fields often end at L5-S1, so the S1 body is absent there.
- `confirmed` precision 1.00 is on 81 pooled studies: a small sample, so the true error rate could still be near 1%.
- Coverage is low on Sudirman B and qx5rt (confirmed 21-25%), mainly because of unconfirmed studies that are often right.
- `classical_agree` is 0 by default since lab14 (classical opinion opt-in); v2 was fitted with it on, so its weight is inert now.
- Label accuracy on SPIDER validation from this run is 98.1% (87 volumes, 97.7% fully correct). The coordinator quoted
  97.6% / 94.3% for main; not reconciled here (same scorer, different run).

## 8. Reproduce

    SPINESCOPE_THREADS=4 .venv/bin/python tools/calibrate_labeling.py collect --split validation --out OUT --jobs 2
    SPINESCOPE_THREADS=4 .venv/bin/python tools/calibrate_labeling.py collect --split training --limit 180 --out OUT --jobs 2
    SPINESCOPE_THREADS=4 .venv/bin/python tools/calibrate_labeling.py collect --qx5rt --out OUT --jobs 2
    SPINESCOPE_THREADS=4 .venv/bin/python tools/calibrate_labeling.py collect --sudirman --limit 240 --out OUT --jobs 2
    .venv/bin/python tools/calibrate_labeling.py fit-v3 --out OUT            # add --write to ship (rules must pass)
