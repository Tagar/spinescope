# lab08: numbering fusion, calibrated confidence and the abstain policy

Branch `lab/lab08_fusion`. Owns `anatomy/pipeline.py`, `anatomy/numbering.py`, and the new modules
`anatomy/fusion.py` (votes and decision) and `anatomy/calibration.py` (features, calibrator, packaged
weights). Learned labeller (`anatomy/ml/`) is called through its public API and not edited.

Numbers below come from `tools/calibrate_labeling.py` (`collect`, then `fit`). Raw outputs stay in
`/tmp/spinescope-lab08_fusion`; nothing derived from SPIDER or Sudirman is committed.

## 1. What the default pipeline now does

`label_spine(study)` (backend `auto`, i.e. the learned labeller when available):

1. Discs, bodies and midline from the learned labeller (`label_spine_ml`), unchanged.
2. Second sagittal series of the same study (T1 when T2 is primary) is run through the same labeller.
3. The classical labeller runs on the primary stack as an independent second opinion.
4. `fuse_numbering` (`anatomy/fusion.py`) collects votes on the lowest-disc offset (-1, 0, +1; see
   `numbering_checks`) and decides the final offset:

| Vote | Source | Weight |
|---|---|---|
| Oblique slab geometry | `numbering_checks.slab_vote` | as documented in lab04 |
| Sacral tilt (lowest disc most tilted) | `numbering_checks.tilt_vote` | as documented in lab04 |
| Conus tip prior | `numbering_checks.conus_vote` | as documented in lab04 |
| Level name written in an oblique slab description that passes through the lowest disc (`L4-5 OBL AX`) | `fusion.description_votes` | one vote per named slab; `name_p_match` 0.95, `name_p_other` 0.02 |
| Classical tilt, only when the classical discs agree geometrically | `fusion.fuse_numbering` | weak: likelihood raised to 0.5 |
| Second sagittal series | recorded as a neutral vote; the label agreement feeds the calibrated confidence and warnings | 0 (consistency, not an offset vote) |

Every cue likelihood is floored at 0.05 before combining (`numbering_checks.combine`), so one cue
cannot veto. Consequence: one named slab alone gives P(shift) about 0.6 (candidate, not applied); two
independent slabs that name the same level give about 0.9 (applied).

5. Relabelling. A non-zero offset is applied (disc names, vertebra names and the conus level shift
   together, `fusion.relabel_anchors`) only when its posterior is at least **0.9** (`FusionConfig.relabel_min_p`).
   Below that the provisional names stay and the record carries `transitional_candidate: true` with a warning.
   `transitional` and `transitional_kind` are set only when applied.
6. Every vote is recorded in `anchors.numbering['votes']` (method, best offset, probabilities, weight,
   evidence), with `posterior`, `offset`, `applied_offset`, `transitional`, `transitional_kind`,
   `transitional_candidate`, `second_series` and `classical` records.

## 2. Calibration

Per-disc `confidence` and the study-level `numbering.confidence` are now calibrated probabilities, not
heuristics. Model: standardised logistic regression with an L2 penalty (`calibration.Logistic`, Platt
style), fitted on SPIDER training only, shipped as `src/spinescope/anatomy/calibration_v1.json`.

Disc features (target: the predicted disc matches a true disc within 12 mm and has the right level name;
false positives are wrong):

- `logit_raw`: logit of the learned labeller's disc probability.
- `height_dev`: |log(height / study median height)|, disc-size plausibility.
- `spacing_dev`: mean relative deviation of the spacing to the neighbouring discs from the study median, spacing regularity.
- `second_present`, `t1t2_agree`: a second sagittal series exists, and the same disc with the same label is found in it.
- `sacrum_below`: vertebral bone visible below the lowest disc.

Volume features (target: every matched disc has the right name, and at least one is matched):

- `logit_offset`: logit of the posterior of the offset the final names carry.
- `sacrum_below`, `second_present`, `t1t2_frac` (share of discs confirmed by the second series),
  `classical_agree`, `min_logit_raw` (the weakest disc).

Status thresholds (`anatomy/numbering.py`): confirmed and probable thresholds are the lowest calibrated
values on the training studies whose training precision reaches the target. Targets are policy, not
fitted: **0.98** for confirmed and **0.90** for probable. The training base rate of correct numbering is
about 0.96, so targets above it are needed to separate the groups.

Hard caps applied after the thresholds (`pipeline._apply_status_caps`):

- `confirmed` needs vertebral bone visible below the lowest disc. Otherwise the status is at most
  `probable`. This is a policy decision, not a calibration result: SPIDER labels are relative to the
  lowest lumbar vertebra, so they cannot tell L5-S1 from L4-L5 (see section 5).
- A relabelled study, and a study whose second series shows a disc below the primary's lowest disc, is at most `probable`.

## 3. Abstain policy (for downstream modules and briefs)

`anchors.numbering['status']` is one of:

| Status | Meaning | What a brief or viewer should do |
|---|---|---|
| `confirmed` | calibrated P(numbering right) >= confirmed threshold, bone below the lowest disc, no contradiction | Show level names as the labelled levels. |
| `probable` | calibrated P >= probable threshold, or capped from confirmed | Show names with a "to confirm" marker. Do not use them as settled facts. |
| `unconfirmed` | below the probable threshold | Show disc positions, but not level names as facts. Say that the numbering is unconfirmed. |

`anchors.numbering['confidence']` is the calibrated probability (0 to 1). `confidence_raw` keeps
the labeller's own heuristic from main, for comparison. `calibrated` says whether the calibrated values are
in use (False for the classical backend and when the packaged calibration is missing; the fallback then
uses the fused posterior, multiplied by 0.9 when no bone is visible below the lowest disc). Warnings
are added for every status other than confirmed.

Caution: the calibrated numbering confidence is high for the sacral-disc failures on Sudirman (section 4.3, 5.2).
Downstream code should show the status, and treat the number as a secondary value.

Dependency on the v2 labeller: with v1 the network never reports bone below the lowest disc (section 5),
so `confirmed` is not reachable on SPIDER. Confirmation needs a sacrum class in the labeller (v2).

## 4. Results

Protocol. One case per SPIDER volume (as in `tools/eval_labeling.py`): a T2 volume is labelled with its T1
sibling as second sagittal series; a T1 volume alone. Scored with the shared scorer
(`spinescope.anatomy._validate`: `score_anchors`, `aggregate`, `reliability`). Sudirman: level-wise, on the 3
annotated discs per patient (`score_levels`), unannotated predicted discs are not counted. Calibration is fitted
on SPIDER training only; validation and Sudirman are reported, not used. Training uses an evenly spaced subset of
180 of the 360 training volumes, and Sudirman an evenly spaced 60 of 515 patients (the machine was shared and
heavily loaded; the subsets are recorded). Validation uses all 87 volumes.

### 4.1 Labels (learned labeller, before and after fusion)

| Set | Cases | Recall | Precision | Label accuracy | Centre median / p90 mm | Labels changed by fusion |
|---|---|---|---|---|---|---|
| SPIDER training (subset) | 180 | 0.998 | 0.993 | 0.976 (1254 matched) | 1.5 / 2.7 | 0 |
| SPIDER validation | 87 | 0.994 | 0.986 | 0.962 (625 matched) | 1.6 / 3.1 | 0 |
| Sudirman (annotated levels) | 60 | n/a | n/a | 0.944 (177 matched) | n/a | 0 |

Fused and unfused label accuracy are identical in all three sets. Across the 267 SPIDER and 60 Sudirman studies
fusion never shifted a name and never raised a transitional candidate: the highest posterior of a non-zero offset
in any of them is 0.15 (the apply threshold is 0.9 and the candidate threshold 0.5).
Per vendor on validation: Siemens label accuracy 0.986 (numbering correct in 54 of 55), Philips 0.914 (29 of 32).

### 4.2 Disc confidence: raw model probability vs calibrated

| Set | Raw ECE | Calibrated ECE | Raw AUROC | Calibrated AUROC | Wrong discs with confidence < 0.5 (raw -> calibrated) |
|---|---|---|---|---|---|
| SPIDER training | 0.014 | 0.009 | 0.687 | 0.716 | 0% -> 13% |
| SPIDER validation | 0.008 | 0.013 | 0.752 | 0.860 | 9% -> 21% |
| Sudirman (annotated) | 0.028 | 0.048 | 0.375 | 0.658 | 0% -> 10% |

Mean confidence of wrong vs right discs, validation: 0.88 vs 0.95 (raw), 0.74 vs 0.97 (calibrated).

Reading. On SPIDER the raw probability is already near-calibrated in ECE terms (almost every disc is above 0.9), so
calibration does not improve ECE on validation (it rises from 0.008 to 0.013). What it does add is ranking: wrong
discs are now separated from right ones (AUROC 0.75 -> 0.86 on validation) and a fifth of wrong discs carry a
confidence below 0.5 instead of almost none. On Sudirman the raw confidence has no ranking power at all (AUROC 0.38,
every disc is about 0.97), and the SPIDER calibration recovers only part of that (AUROC 0.66). The calibration
transfers poorly to a different domain (clinical DICOM, Siemens 1.5 T); that is a limit of a 6-feature model fitted
on one dataset, not something to tune on Sudirman.

Fitted disc weights (standardised logistic; training): `logit_raw` 0.30, `height_dev` 0.32, `spacing_dev` -0.67,
`second_present` -0.91, `t1t2_agree` 1.24. `sacrum_below` has no variance (0 in every study) and gets weight 0.

### 4.3 Study numbering confidence

| Set | Studies | Numbering correct | Heuristic on main (ECE) | Uncalibrated fallback (ECE / AUROC) | Calibrated (ECE / AUROC) | Mean confidence wrong vs right (calibrated) |
|---|---|---|---|---|---|---|
| SPIDER training | 180 | 0.961 | 0.570 | 0.157 / 0.632 | 0.026 / 0.953 | 0.79 vs 0.97 |
| SPIDER validation | 87 | 0.954 | 0.563 | 0.160 / 0.720 | 0.039 / 0.991 | 0.44 vs 0.95 |
| Sudirman | 60 | 0.900 | 0.508 | 0.043 / 0.611 | 0.090 / 0.648 | 0.85 vs 0.98 |

The heuristic on main gives about 0.39 to every study whatever its numbering (its AUROC of 0.9 on SPIDER comes from
tiny differences between values in 0.37 to 0.40, so it is not informative). Calibration fixes its scale: the
right studies move from about 0.39 to about 0.95.

Note the number of wrong studies: 7 in the SPIDER training subset, 4 in validation, 6 in Sudirman. The volume-level
calibration rests on very few failures and its ECE is noisy.

**The calibrated numbering confidence is confidently wrong on Sudirman.** The calibrated values of the six wrong
Sudirman studies are 0.999 (0069), 0.929 (0088), 1.000 (0107), 0.181 (0395), 0.991 (0414) and 1.000 (0459). Five of six
are above 0.9. The volume calibrator leans mainly on the fused offset posterior, and that posterior is high for these
studies because the cues all count from the same extra disc (section 5.2). So on Sudirman `confidence` must not be read
alone: the status (which is capped at `probable`, see 4.4) is the safer signal, and the calibration does not yet meet the
brief's target for this failure mode.

### 4.4 Status (abstain policy) and what it catches

| Set | confirmed | probable (precision) | unconfirmed (precision) |
|---|---|---|---|
| SPIDER training | 0 | 178 (0.972) | 2 (0.000) |
| SPIDER validation | 0 | 85 (0.977) | 2 (0.000) |
| Sudirman | 0 | 58 (0.914) | 2 (0.500) |

Confirmed is empty on every set: v1 never sees bone below the lowest disc (section 5), so the cap holds the status at
probable. Of the 4 wrong validation studies, 2 are `unconfirmed` and 2 `probable`. Of the 6 wrong Sudirman studies,
1 is `unconfirmed` and 5 are `probable`. So on Sudirman the policy catches 1 in 6 wrong numberings, and `probable`
is not a safe signal for the lowest level name in that set (5 of 58 probable studies are wrong).

### 4.5 Second series, classical second opinion, transitional

- Second sagittal series present on 112 of 267 SPIDER studies (T2 volumes with their T1 sibling). The T1 and T2 give
  the same label and place for 93% of the primary's discs (739 of 795). A disc below the primary's lowest disc, seen only
  in the second series, was found in 1 case.
- Classical second opinion: agrees geometrically (within 6 mm in the sagittal plane, most discs matched) in 5 of 267 SPIDER
  studies. The median matched fraction is 0 (maximum 0.75). The classical discs are at about the right heights (for one study:
  84 vs 82 and 117 vs 118 mm) but 15 to 20 mm off in AP, which the (y, z) gate counts as a miss. A z-only gate is the
  obvious next step; it was not run because the collection is expensive on the shared machine. Until then the
  classical tilt vote is effectively never cast.
- Transitional: no candidate and no applied shift on SPIDER or Sudirman. `num_vertebrae` in `overview.csv` is 7 for
  most volumes (446 of 447 are not 5), so it does not identify transitional anatomy and the split the wave brief asked for
  is not informative on SPIDER.

## 5. Limitations and findings

1. **No sacrum in v1.** The learned labeller returned no bone below the lowest disc in any of the 327 studies (267 SPIDER,
   60 Sudirman), and in the three SPIDER volumes checked directly the grid extended 40 to 85 mm past the lowest disc (`anatomy/ml` has no sacrum class). This is why
   `confirmed` is unreachable. Confirmation of L5-S1 needs the sacrum class (the v2 training).
2. **The main failure is a lowest-disc error that every cue agrees with.** Of the six wrong Sudirman studies, three
   (0069, 0107, 0459) have an extra predicted disc below the annotated L5-S1, and the whole chain is named one level too
   cranial. For 0069 I checked the geometry: the predicted disc at z -85 mm sits below the annotated L5-S1 at -51 mm, is
   6 mm high and strongly tilted, which fits an S1-S2 (sacral) disc. Study 0088 has one annotated disc named one level up
   beside a missed disc mid-chain; 0395 finds none of its three annotated discs (the chain is displaced). On SPIDER
   validation, case 38 shows the same sacral pattern as a two-level shift (two extra discs below). The T1 sees the same
   discs, so the second-series check agrees, and the tilt and slab cues are fooled by the same disc. Nothing in fusion
   detects it, so the status stays `probable` and the calibrated confidence stays high (section 4.3). An S1-S2
   discriminator (disc size and shape, sacral body below) is the missing cue.
3. **SPIDER labels follow a rank convention, not anatomy.** Disc names are counted from the lowest lumbar vertebra. The
   calibration therefore measures agreement with that convention (count consistency, disc detection, second-series
   agreement). It does not measure whether the lowest disc is L5-S1, and it cannot tell us how often the fusion's shifts
   are right on SPIDER. The shifts are exercised by synthetic tests only (`tests/test_labeling_fusion.py`).
4. **Fusion is a no-op on real data so far.** No label changed in 327 real studies. The offset machinery is correct on
   synthetic evidence (tests) but untested on real transitional anatomy, which the clinical data here did not show.
5. **Small failure counts.** 4 wrong numberings in validation and 6 in Sudirman. The volume calibration and the status
   thresholds rest on about 10 failures; treat the numbers as indicative.
6. **Domain shift.** The SPIDER-fitted calibration over-states numbering confidence on Sudirman (ECE 0.04 with the
   uncalibrated fallback, 0.09 calibrated).
7. **Subsets.** Training 180 of 360 (evenly spaced), Sudirman 60 of 515 (evenly spaced), validation all 87.
8. **Side finding, not in the package.** `tests/_spider.load_t2_t1` gives a study on which `label_spine` finds no discs
   for SPIDER validation case 35, while the package loader (`load_study`) finds all 3. The package path is what the
   calibration uses; whoever owns `tests/_spider.py` should check its orientation handling.

Recommended next steps: (a) the v2 sacrum class, then re-fit with `sacrum_below` informative; (b) a z-only gate for the
classical opinion; (c) an S1-S2 discriminator as a fusion cue; (d) more SPIDER-independent wrong cases before trusting the
volume-level calibration.


## Reproduce

```
python tools/calibrate_labeling.py collect --split training --limit 180 --out /tmp/spinescope-lab08_fusion --jobs 4
python tools/calibrate_labeling.py collect --split validation --out /tmp/spinescope-lab08_fusion --jobs 4
python tools/calibrate_labeling.py collect --sudirman --limit 60 --out /tmp/spinescope-lab08_fusion --jobs 4
python tools/calibrate_labeling.py fit --out /tmp/spinescope-lab08_fusion --write   # writes calibration_v1.json
```

The fit uses only the recorded training subset (`training_subset.json`) and never reads validation or
Sudirman records.
