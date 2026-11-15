# lab20: axial slab evidence (disc-likeness, slab vote, missed-disc candidates)

Branch `lab/lab20_axial`. Owns `anatomy/axial.py`, `anatomy/_calibrate_axial.py`, `anatomy/axial_v1.json`,
the additive `axial_slab_vote` in `anatomy/numbering_checks.py`, the `FusionConfig.use_axial` switch in
`anatomy/fusion.py`, the `fusion=` argument of `label_spine` / `label_spine_details`, and the phantom option
`write_standard_study(slab_contrasts=...)`. Tests: `tests/test_labeling_axial.py` (17 tests, phantom-based).

## 1. What the cue does

Oblique axial slabs (usually T1 or T2, one per disc) are angled to a disc. The labeller's disc position is an
independent measurement, so a slab gives three things:

* **Disc support** (`DiscAxial`): a predicted disc covered by a T1 slab (normal within 12 degrees, centre inside
  the slab block) gets a disc-likeness log likelihood ratio (LLR) at its centre. Below the calibrated
  `support_llr` it is `unsupported` (possible false disc). Uncovered discs are not scored.
* **Missed-disc candidates** (`MissedCandidate`): disc-like peaks in a slab slice with no predicted disc within
  12 mm in that plane. Warnings only.
* **Slab vote** (`numbering_checks.axial_slab_vote`, cue name `axial_slab`): the lowest T1 slab's disc, if it is
  disc-like, gives the same likelihood as the unchecked `slab` vote. It replaces the `slab` vote when on.

Disc likeness is a two-class Gaussian naive Bayes on three ellipse features of the normalised slice:
`FEATURES = ("mean", "cv", "ring")` (inner-ellipse mean, coefficient of variation, ring contrast). Order is stored
in `axial_v1.json` and checked at load time.

The cue abstains (`AxialReport.available=False`, one warning, no vote) when the study has no T1 axial slab, or when
`axial_v1.json` is missing. SPIDER has no axial slabs, so it always abstains there.

## 2. Calibration (`axial_v1.json`, version v1)

Fitted by `python -m spinescope.anatomy._calibrate_axial --out ...` on the Sudirman labelled axial T1 slices
(annotated IVD centroids as positives; pixels at least 25 mm from every labelled region of the same slice,
8 per slice, as negatives).

* Split: patients sorted by id; even positions are FIT, odd positions are TEST. The naive Bayes means, variances,
  `support_llr` (5th percentile of FIT positive LLRs, so 95 % of FIT discs are supported) and `missed_llr`
  (maximises F1 of peaks against annotated discs on FIT) use FIT only. TEST is reported, never used to fit.
* Fit set: 774 positive and 6192 negative slices. TEST: 771 positives, 6168 negatives.
* Support on TEST: TPR 0.949, FPR 0.012 (negatives contain no vertebral bodies; see section 4).
* Missed-peak precision and recall on TEST: 0.45 and 0.77, with about 715 false peaks per 258 patients on the labelled
  slices. Too many to be a useful warning on their own (see section 4).

The calibration was fitted on the Sudirman labels, which follow the lowest-slab naming convention. The positions
and disc-likeness are independent of that convention, but the level names are not used by the calibration.

## 3. Held-out Sudirman evaluation (TEST half, 257 patients, 771 annotated levels)

Both runs use the same code and the same patients (`labelled_patients()`, odd positions). Scoring is the shared
level-wise scorer `_validate_sudirman.score_levels` (name of the nearest predicted disc, matched within 12 mm).

| Measure | axial off (default) | axial on |
|---|---|---|
| Label accuracy (exact level name, 771 levels) | 0.914 | 0.914 |
| Discs matched within 12 mm | 0.984 | 0.984 |
| Median centre error (mm) | 2.1 | 2.1 |
| Patients with all three names correct | 227 / 257 (0.883) | 227 / 257 (0.883) |
| Patients with a numbering error (matched disc, wrong name) | 26 | 26 |
| Status `confirmed` (n, fraction all-correct, mean confidence) | 230, 0.922, 0.998 | 228, 0.921, 0.997 |
| Status `probable` | 12, 0.50, 0.77 | 13, 0.54, 0.78 |
| Status `unconfirmed` | 15, 0.60, 0.66 | 16, 0.63, 0.59 |
| Patients whose status changed | | 3 (all `confirmed` or `probable` to a lower status) |
| Patients whose labels changed | | 0 |

What the status changes are: in 2 of the 3, labels were correct and the lowest disc was flagged unsupported
(confirmed to probable, confidence about 0.99 to 0.67 and 0.82). In the third, correct labels fell from probable
(0.53) to unconfirmed (0.08) with all three discs unsupported. So on this half the cue only lowers confidence on
correct cases. It never changes a label.

**Detection of wrong numbering** (26 patients with a matched disc carrying a wrong name, cue on):

* Flagged by the axial evidence (vote pointing away from the provisional lowest disc, or any unsupported disc):
  12 / 26 (46 %). Vote alone: 4 / 26. Unsupported disc alone: 8 / 26.

**False alarms on correct cases** (231 patients with all names right, cue on):

* Vote pointing away: 0 / 231.
* Any unsupported disc: 56 / 231 (24 %). Of the 771 annotated levels, 47 are unsupported while correctly named
  and within 12 mm, which is about 7 % of correctly positioned discs. Calibration expected about 5 %, so the
  support threshold is 5 % by construction on FIT positives, so 7 % on TEST is close, but precision as a numbering flag is poor.

**Position / support cross-tab** (cue on, annotated levels): supported and within 12 mm: 628 (name right) plus 51
(name wrong). Unsupported and within 12 mm: 47 (name right) plus 6 (name wrong). Not covered by a T1 slab: 33.
This is the independent part of the check: disc-likeness agrees with the annotated position for most levels.

**Missed-disc candidates** (cue on): 1963 candidates over 237 of 257 patients (about 7.6 per patient). Sudirman
annotates only three discs and the calibration negatives contain no bodies, so most of these are unannotated
structure rather than missed discs. Not usable as a warning without a per-slice rule that removes them. They are
not a basis for a default-on recommendation.

## 4. Caveats (read before using any number)

* **Circular naming check.** Sudirman level names follow the lowest-slab convention: the lowest oblique slab is
  named L5-S1 by the dataset. So "the lowest slab agrees with the name" is partly true by construction and is
  not independent evidence of correct numbering. The independent parts are the disc position (matched within
  12 mm, median 2.1 mm) and the disc-likeness of the predicted disc in its slab (cross-tab above).
* Only T1 slabs are calibrated. Sudirman labels cover only the three annotated discs (L3-L4, L4-L5, L5-S1).
* Transitional-anatomy cues (iliac crest level, Castellvi, iliolumbar ligament, sacral promontory) are not
  implemented here: there is no data to calibrate or test them on. `numbering_checks` still says "not evaluated".
* The calibration negatives contain no vertebral bodies, so a body-like region can still score as disc-like.
* SPIDER: 12 validation cases (both sequences) gave identical anchors, offsets, confidences and statuses with the
  cue on or off, and no axial report. The cue does not change SPIDER results. Those 12 are a subset, not all 87.
* The phantom is a cartoon: the tests check mechanics (coverage, matching, abstention), not LLR values.

## 5. Recommendation: keep `use_axial` default OFF

* On the held-out Sudirman half the labels are identical with the cue on or off. The only effect is 3 status
  downgrades on correct cases.
* Turning it on changes the calibrated confidence inputs (`p_applied` and the raw confidences feed
  `volume_feature_vector`). The shipped calibration was not fitted with the cue, so the confidences would be
  miscalibrated. A refit is needed before the default changes.
* The disc-support flag is too noisy as a warning (56 false alarms vs 8 detections). The missed-disc candidates
  are not usable (section 3).
* The slab vote is the one part with no false alarms on correct cases (0 / 231), but it fires on only 4 of 26
  wrong-numbering cases, and its evidence is partly circular (section 4).

Proposed next step (not done here, needs the refit owner): after lab18_reliability refits the calibration, consider
ON with the `axial_slab` vote only, and keep the support flag and missed-disc candidates as diagnostics (not
warnings).

## 6. Stable names (for the calibration refit, lab18_reliability)

* Config switch: `anatomy.fusion.FusionConfig.use_axial` (default `False`); `label_spine(..., fusion=FusionConfig(...))`
  and `label_spine_details(..., fusion=...)`.
* Vote: cue name `anatomy.numbering_checks.AXIAL_SLAB_CUE == "axial_slab"`, from `axial_slab_vote(report, anchors)`.
  It replaces the `slab` vote (cue `"slab"`) only when `use_axial` is on and the axial report is available.
* Record: `FusionResult.axial` and `fusion_record(...)["axial"]` (present only when the cue is on and available),
  keys from `AxialReport.to_dict()`: `available`, `contrast`, `discs`, `slabs`, `missed`, `warnings`.
* Module and data: `anatomy.axial` (`assess_axial`, `AxialReport`, `AxialConfig`, `FEATURES = ("mean", "cv", "ring")`),
  `anatomy/axial_v1.json` (keys `version`, `features`, `disc`, `non`, `support_llr`, `missed_llr`, `provenance`).
* Volume features affected when on: `p_applied` and the raw per-disc confidences, which feed
  `calibration.volume_feature_vector`. The cue changes the lowest-disc posterior, so the calibrated confidence changes.

## 7. Reproduce

    SPINESCOPE_THREADS=4 .venv/bin/python -m spinescope.anatomy._calibrate_axial --out /tmp/spinescope-lab20_axial/axial_fit.json --workers 2

The held-out runs used a scratch driver (not committed) that calls `label_spine_details(study, fusion=FusionConfig(use_axial=...))`
per labelled patient and scores with `score_levels`. Raw outputs stay in `/tmp/spinescope-lab20_axial/`.
