# lab09_errors: error analysis of the learned labeller and post-processing fixes

Scope: the default learned labeller (`label_spine(backend="ml")`, `labeller_v1.onnx`, see `docs/model_card.md`).
Only `src/spinescope/anatomy/ml/postprocess.py` (existing function signatures kept) and two additive
`Config` fields changed in `config.py` were touched. The network, weights and scorer are unchanged.

Before = main at `1c9f553`. After = branch `lab/lab09_errors`. Same scorer (`spinescope.anatomy._validate`),
same protocol. Nothing was tuned on `validation`; see "Calibration and data use" for the exact provenance.

## Headline

| benchmark | metric | before | after |
|---|---|---|---|
| SPIDER validation (87) | label accuracy, of all truth discs | 95.5% | 97.6% |
| | centre error, median / p90 (mm) | 1.55 / 3.09 | 0.60 / 1.98 |
| | share of matched discs within 3 mm | 88.5% | 95.4% |
| | level errors among matched discs | 3.8% | 1.9% |
| | volumes with every truth disc labelled right | 90.8% | 94.3% |
| SPIDER training (360, not for the network) | label accuracy, of all truth discs | 97.1% | 98.8% |
| | centre error, median / p90 (mm) | 1.47 / 2.70 | 0.41 / 1.34 |
| | level errors among matched discs | 2.7% | 1.0% |
| | volumes with every truth disc labelled right | 94.2% | 96.7% |
| Sudirman (515 patients, 1,545 annotated discs) | label accuracy (level-wise) | 90.5% | 91.2% |
| | centre error, median / p90 (mm) | 2.63 / 4.44 | 2.13 / 3.83 |
| Robustness suite, ML rows (23 scenarios) | passing scenarios | 0 | 0 |
| | silent errors (confident and wrong) | 14 | 4 |

Caveats that matter when reading the table:

- SPIDER's truth centre and normal are 3-D voxel statistics of the mask. The 3-D centroid fix is therefore a
  measurement-definition match on SPIDER. The Sudirman centre improves too (median 2.63 to 2.13 mm), and its
  independent truth is a different definition (operator-planned slab), so that gain is the stronger evidence.
- The midline change moves the median to 0 mm on SPIDER but widens the tail on validation (p90 3.2 to 4.4 mm).
  See "Midline" below.
- The robustness phantom is out of distribution (see `docs/model_card.md`). It has no passing ML scenario before
  or after. Its "silent error" count fell because the numbering confidence dropped (body-below check lost),
  not because labels got right.

## What was run

    # ML labeller via the module spec (the `ml` alias in tools/eval_labeling.py is broken on main, see below)
    python tools/eval_labeling.py --backend spinescope.anatomy.ml.labeller:label_spine_ml \
        --dataset spider --split validation|training --out /tmp/spinescope-lab09_errors/<run> --jobs 8
    python tools/robustness_report.py --json <path>      # prints every backend; the ml rows were used
    spinescope.anatomy._validate_sudirman.evaluate(workers=6).summary()   # all 515 labelled patients

Outputs (summary.md, cases.csv, gallery PNGs, per-volume tags) stayed under `/tmp/spinescope-lab09_errors/`.
Nothing derived from SPIDER or Sudirman is committed. Runs took about 2.5 min (validation) and 12 min (training)
on 8 processes, with the prediction cache keyed by the labeller source.

## Failure taxonomy (automatic, all 447 SPIDER volumes, before the fixes)

Classification is by the cached predictions against the SPIDER truth column (`scripts` are not committed;
the logic is: unmatched predictions are classed by z relative to the truth column: above the top truth disc
(coverage), below the lowest (caudal), or between truth discs (interior); missed truth discs by level; midline
more than 5 mm from the truth canal median).

| failure | validation volumes before / after | training volumes before / after | what it is |
|---|---|---|---|
| interior false disc (split fragment or duplicate) causing a +1 level shift above it | 3 / 2 | 15 / 8 | one disc split into two components, or a neighbour-slice disc duplicating a neighbouring level |
| caudal (sacral) false disc | 2 / 0 | 2 / 0 | strip below L5-S1 with a horizontal normal, or ~50 mm below the column |
| missed T9-T10 / T10-T11 at the top of the field | 4 / 3 | 3 / 3 | top disc at the edge of the field, not found |
| missed L2-L3 / L4-L5 (middle) | 0 | 3 / 1 | a disc absent from the mid-plane and its neighbours |
| midline error above 5 mm | 8 / 8 | 6 / 3 | the plane is off the canal median (152, 177, 234 in validation) |
| discs above the lowest annotated disc (thoracic) | 3 / 3 | 3 / 3 | not wrong in fact: SPIDER annotates only some thoracic discs (see below) |

Volumes with any issue: validation 16 before, 14 after (of 87); training 30 before, 18 after (of 360).

Visual review. I looked at the overlay pages for 38_t2, 73_t2, 242_t1 and 38_t1 (after the fixes), and read the
disc coordinates of 209_t1, 29_t2, 251_t1, 73_t2 and 242_t1 directly. Those cases are the ones described below.
The other ~40 worst cases were classified by the automatic rules above, not by eye. That is a gap in the
review as asked, and I say so rather than claim more.

Specific cases that drove the fixes:

- 38_t2 (before): the L5-S1 disc was two components separated by a one-pixel diagonal seam (4-connected labelling),
  plus a 31 px vertical strip at z = -51 mm (sacral). Both were false, and together they shifted levels by 1-2.
- 73_t2 and 145_t1 (before): a second component on the lowest disc, adjacent to the first one.
- 242_t1 (before): an L3-L4 disc present on the mid-slice and again on a neighbour slice, 11.8 mm apart in the
  sagittal plane. The mid-plane dedupe used pre-refinement centres and missed it.

## Fixes (postprocess.py, with config defaults in config.py)

1. **8-connected instances** (`_components`). A split along a one-pixel diagonal seam is one disc. Fixes the
   split-disc case (38_t2, 73_t2) and removes one source of numbering shifts. Test: diagonal seam is one disc;
   a one-pixel gap keeps two.
2. **Normal tilt plausibility** (`extract_discs`, `Config.ml_disc_max_tilt_deg = 70`). A component whose normal tilts
   more than 70 degrees from the superior-inferior axis is a sagittal strip, not a disc. Source: SPIDER training truth
   normals tilt at most 59.6 deg (99.9th percentile 56.8 deg), so 70 is a margin, not a fit. Tests: vertical strip
   rejected; flat disc kept.
3. **Caudal outliers** (`drop_caudal_outliers`, new). The lowest disc is dropped while it lies more than the largest
   level spacing (`level_spacing_mm[1]` = 50 mm, existing config) below the next disc, with a warning. Source: an
   anatomical prior, not a fit. Tests: isolated caudal disc dropped; normal spacing kept; a lone disc never dropped.
4. **3-D centroid** (`refine_disc_3d`, `_disc_3d`, `Config.ml_disc_3d_half_width_mm = 27`,
   `ml_disc_3d_ap_half_mm = 22`). The mid-plane centre sits about 1.3 mm anterior of the SPIDER 3-D centroid at every
   lumbar level (training median dy -1.34 mm; dz +0.26 mm). The offset is a measurement definition, not a network
   error: a disc's sagittal centroid moves across its width. Pooling the disc pixels over slices out to +/-27 mm and
   taking their centroid removes it (median dy -0.1 mm). The window was chosen on the training split by sweeping the
   slice half-width (table below). 27 mm = eight 3.34 mm SPIDER slices; the error plateaus at 8 and beyond.
5. **Normal from the 3-D disc** (`Config.ml_disc_normal_3d`, default **off**). Implemented and tested. On SPIDER it cuts the
   normal error from 2.5 to 0.7 deg, but SPIDER's truth normal is the same 3-D principal axis, so that is partly
   definitional. On Sudirman, whose truth is the independent axial-slab normal, it is worse (median 2.47 to 3.28 deg).
   The default keeps the mid-plane normal. Flip the flag to test the 3-D normal on another dataset.
6. **Duplicates after refinement** (`drop_duplicate_discs`, new). Discs whose refined (y, z) centres are within
   half the smallest level spacing (12.5 mm, existing config) are merged, keeping the more confident one. Fixes
   duplicates that only become close after the 3-D step (242_t1).
7. **Midline = canal-area median** (`mid_sagittal`). The plane is the slice where cumulative canal area reaches half
   its total. The truth midline is the median of canal voxels, and voxel count per slice is proportional to canal
   area, so this is the same definition. Training median error 1.18 mm to 0.00 mm; more than 5 mm on 1.7% to 0.8%
   of volumes. It also moves the phantom plane from x = -15 mm (the known failure in the model card) to x = 0.
   The validation p90 rose from 3.2 to 4.4 mm, which is the one regression.

Tests: `tests/test_labeling_postprocess.py` (15 tests: instance connectivity, tilt, caudal outliers, 3-D centroid,
3-D normal (opt-in) with the default kept, duplicates). The existing `tests/test_anatomy_ml.py` oracle tests pass.

## Calibration and data use (please read)

- Training split: the 3-D window (sweep of 0 to 12 slices, `scripts/exp_centre.py`), the midline estimator (compared
  on 360 training volumes), the tilt bound (training truth), the caudal bound (config) and the dedupe threshold
  (config) were chosen on training only.
- I printed the validation bias table once while exploring, before the training-split bias analysis. The chosen
  values are the training ones. The validation numbers above are one run of the final configuration.
- The normal decision (`ml_disc_normal_3d` off) was taken on Sudirman, which is independent of SPIDER validation.
  That is a decision on a second dataset, disclosed here.

Training-split sweep for the 3-D window (median 3-D centre error of matched discs, mm; training split):

| slices either side | half-width (mm) | median 3-D centre error | p90 | normal median (deg) |
|---|---|---|---|---|
| 0 (mid-plane centroid) | 0 | 1.53 | 2.98 | not meaningful (flat set) |
| 1 | 3.3 | 1.40 | 2.73 | 3.6 |
| 3 | 10 | 0.92 | 2.22 | 1.3 |
| 5 | 17 | 0.56 | 1.75 | 0.9 |
| 8 | 27 | 0.45 | 1.47 | 0.8 |
| 12 | 40 | 0.44 | 1.37 | 0.8 |

## Vertebral centres, heights and margins

Not changed, and not validated. The scorer has no vertebra-centre truth, and disc height and AP margins have no
truth in SPIDER masks as used here. The disc height is still from the mid-plane component. The margins are still
at mid-height on the mid-sagittal plane. Both are candidates for a truth-based check (for example, mask-derived
heights and AP extents from the 3-D masks).

## Midline

The canal-area median fixes the systematic error, but not the tail. Remaining wrong-midline volumes on validation
(152, 177, 234: 9 to 16 mm before the fix) are still off by more than 5 mm. Those are canal-segmentation or
scoliosis cases, and the model team should look at them with the canal class.

## Remaining failure modes, prioritised for the model-training agent

1. **Adjacent false discs at the lowest level (largest single cause).** A second component on or next to L5-S1
   (38_t1, 38_t2 still shifted +1 after the fixes), and a neighbour-slice disc that duplicates a level (209_t1,
   and the interior cases 29_t2, 242_t2, 251_t1, 233_t2, 241_t1, 36_t1, 34_t1 in training). The post-processing
   only merges within 12.5 mm and cannot tell a fragment from a real disc at 13-25 mm. A model fix would be an
   instance-aware head or per-level training with slice-consistency. A candidate post-processing rule: discs closer
   than the minimum level spacing (25 mm) are the same disc. I did not test it. On 251_t1 the more confident of the
   two is the wrong one, so it would need a tie-breaker.
2. **Discs at the top of the field.** T9-T10 and T10-T11 are missed when they sit at the edge of the field of view
   (validation 129_t1, 53_t1, 53_t2; training 18_t1, 18_t2, 18_t2_SPACE). Training on more top-of-field examples, or
   a field-of-view-aware head, should address it.
3. **Midline tail.** Canal-segmentation failures above 5 mm (validation 152, 177, 234; training 110, 166).
4. **Thoracic discs above the annotated range are scored as false positives.** These are not model errors, and
   SPIDER's truth covers only part of the thoracic spine (`overview.csv` num_vertebrae counts thoracic too, but the
   masks do not always label every thoracic disc). The "over-detected" tag therefore overstates precision errors. Use
   the coverage-aware count for precision.
5. **Numbering without a sacrum class.** Numbering is unconfirmed on 100% of SPIDER volumes ("no vertebral body below
   the lowest disc"), so the L5-S1 name rests on the lowest disc. Transitional anatomy cannot be tested on SPIDER
   beyond counts: by `num_vertebrae`, the 5-level group (1 volume in training, 0 in validation) is too small to
   say anything, and the other groups show offsets spread across 6, 7, 8 and 9 counts. The phantom's lumbarised_S1 and
   sacralised_L5 scenarios are wrong too (label accuracy 0.71 and 0.00).
6. **Out of distribution: phantom and non-SPIDER scans.** Phantom label accuracy is 0 in most scenarios, both before
   and after. I did not isolate the cause; the labels are shifted by about one level against the phantom truth names. Robustness is unchanged except that
   confidence moved; see the table below.

## Robustness (ML rows, `tools/robustness_report.py`)

Before to after, per scenario (status, label accuracy, p90 centre error, numbering confidence):
23 scenarios, 0 pass before and after. Silent errors 14 to 4; unsure 9 to 19. Notable: fov_T10_sacrum and fov_L1_coccyx
improve recall (0.89 to 1.00, 0.80 to 1.00) and the first one gets label accuracy 1.00. Scoliosis label
accuracy 0.00 to 0.17, p90 9.97 to 3.55 mm. The numbering confidence drops from 0.87 to 0.38 in the phantom
scenarios where the body-below check no longer finds a body on the canal-median slice. That is the cause of the
silent-to-unsure shift, not a labelling gain.

## Known tool issues (not fixed here)

- `tools/eval_labeling.py --backend ml` fails on main: it resolves `spinescope.anatomy.ml.label_spine`, which the
  package does not export (it exports `label_spine_ml`). Workaround used: `--backend spinescope.anatomy.ml.labeller:label_spine_ml`.
- `tools/robustness_report.py` has no `--backend` flag. It runs every backend in `BACKENDS`; the ML rows were used.

## Tests and checks

`ruff check .`, `ruff format --check .`, `mypy` clean. New tests in `tests/test_labeling_postprocess.py`.
