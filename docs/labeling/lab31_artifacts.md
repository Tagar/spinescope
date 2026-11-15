# lab31: post-operative hardware and image artefacts

Owner: lab31_artifacts. Code: `src/spinescope/anatomy/artifacts.py` (detectors, `assess_quality`, `apply_quality`),
wired last in `label_spine` (`pipeline._quality_stage`), surfaced by `anatomy/summary.py` (`quality_summary`), with the
additive `Anchors.quality` field (schema `data/anchors.schema.json`, `$defs/Quality`, and `numbering.quality_cap`).
Tests: `tests/test_artifacts.py`.

## Why

Metal (screws, rods, cages, cement) and image artefacts (susceptibility blooming, motion ghosting, phase wrap, bias
field, failed fat suppression, low SNR) degrade the level names and every measurement downstream. The labeller does
not know about them, so the study must say so: a flag per level, a study score, and a lower confidence where the flag
covers the level. Calibrated values are never overwritten silently.

## What the record contains

`anchors.quality` (JSON-safe, no file paths, no DICOM attributes):

- `levels[label]`: `flags` (kind -> confidence in [0, 1]), `flagged` (kinds at or above
  `Config.quality_flag_min_confidence`), `confidence` (the largest).
- `study`: `score` in [0, 1] (1 minus the largest level confidence; 1 = no evidence), `flags[kind]` with `present`,
  `confidence` and the detector `detail`. Advisory kinds carry `advisory: true` and `would_flag` but never `present`.
- `caps`: one record per capped level: `level`, `raw` (the calibrated value), `capped`, `reason`.
- `warnings`: plain text, also appended to `anchors.warnings`.

Every detector works on relative intensity (fractions of the study's own tissue reference, or of the background
noise) or on sizes in mm. Nothing is in absolute signal units or pixel coordinates.

## Detectors and policy

Each kind is computed for every study. Only the kinds in `Config.quality_capping_kinds` (currently `motion_ghost`) can
cap a level or the numbering. The others are ADVISORY: reported with their confidence and evidence, `would_flag` in the
study record, listed in the summary as "advisory (not capped)", and never in `flagged`, the level confidence or the
score. The policy follows the real-data review below, not the phantom alone.

| Kind | Rule (sagittal stack the labels came from) | Level mapping | Status |
|---|---|---|---|
| `motion_ghost` | Lag autocorrelation of the high-pass residual over the spinal window, along the phase (row) axis. Evidence = peak prominence at the best lag, outside the vertebral period (taken from the labelled disc spacing), median over slices. Threshold `quality_ghost_prominence` = 0.15. | study-wide (every level) | CAPPING |
| `metal` | Compact dark region inside the spinal window: core below 0.2 of the tissue reference, area 4 to 400 mm^2, at least 2.5 mm thick, more than half of its 6 mm ring is tissue. Needs 3 slices (`quality_void_min_slices`). Rim ratio recorded. | the adjacent discs of the body it sits in | advisory |
| `fusion_or_cage` | The same voids, when their centre is within half a disc height of a disc centre. | that disc | advisory |
| `wrap` | Outer 8 % of rows: mean brightness at least 1.5 times the middle third. | the disc nearest that edge | advisory |
| `low_snr` | Tissue reference over robust background sigma, below `quality_min_snr` = 8. | study-wide | advisory |
| `fat_sat_failure` | STIR subcutaneous rim over the interior, at or above `quality_stir_fat_ratio` = 0.6. | none | advisory |
| `bias_field` | 90th over 10th percentile of tissue along the anterior-posterior profile. | none | detail only (never a hit) |

Caps (`apply_quality`, only for capping kinds):

- a level flagged by a capping kind (confidence at or above `quality_flag_min_confidence` = 0.5) gets its disc
  confidence capped at `quality_disc_cap` = 0.5 when the raw value is higher; the raw value stays in the disc's
  evidence and in `quality.caps` (`raw`, `capped`, `reason`);
- a `confirmed` numbering becomes `probable` when a level is flagged; `numbering.quality_cap` records from/to and the
  reason; the numbering confidence number is unchanged;
- a study score below 0.3 caps the numbering to `unconfirmed`.

Failures are isolated: a detector that raises adds a warning ("image quality not assessed") and the labels stay as
they were (`tests/test_artifacts.py::test_quality_failure_never_sinks_the_labels`).

## Development history (what was rejected, and why)

Recorded so the next reader does not repeat it.

- **Air-based ghosting.** The first ghost rule used the background air. The phantom and real soft tissue leave almost
  no air, so it saw nothing. Replaced by the lag-autocorrelation peak.
- **Lag-2 peak.** The first autocorrelation search picked the boundary lag (2) on every scan. The search is now over
  interior lags only.
- **Vertebral period read as ghosting.** Anatomy repeats at the vertebral period (about 30 rows on the phantom).
  Lags near multiples of the labelled disc spacing are excluded; this is data-driven, not a set lag.
- **Edge "tissue above the outermost disc".** Thoracic anatomy above the labelled levels is normal, so the rule
  flagged the baseline. Replaced by the edge-band brightness ratio, which separates the wrap scenario from the
  baseline and six other scenarios in both phantom looks (wrap about 1.9 to 2.0; the rest about 0.9 to 1.05).
- **Bias profile.** The anterior-posterior profile is confounded by anatomy: an isotropic 1 mm phantom with no bias
  reads 3.2, and the bias scenario reads 1.48 against 1.16 for the baseline. Not separable, so advisory only.
- **STIR fat rim.** On the phantom the subcutaneous rim is soft tissue, not fat (ratio 1.02 on a normal study), so the
  rule is advisory only.
- **Void core intensity.** On the realistic look the void cores sit at the noise floor and the baseline has voids
  with the same rim signature as the hardware; intensity alone does not separate them. Requiring three slices
  (not two) removed the baseline false positives, at the cost of the thinnest hardware.

## Validation

Robustness suite, 23 scenarios, both phantom looks (`spinescope.testing.robustness`). Labeller: classical for the
cartoon, learned for the realistic. "Capping" is what changes labels' status; "advisory" is what the detector would
flag but does not cap. Score is the study score (1 = none).

Cartoon phantom:

| scenario | capping | advisory (would flag) | score |
|---|---|---|---|
| baseline | - | fat_sat_failure | 1.00 |
| t2_only | - | - | 1.00 |
| t1_only | - | - | 1.00 |
| stir_only | - | fat_sat_failure | 1.00 |
| pd_only | - | - | 1.00 |
| iso_1mm_3d | - | bias_field | 1.00 |
| thick_6mm | - | - | 1.00 |
| few_slices_7 | - | - | 1.00 |
| field_3T | - | - | 1.00 |
| fov_T10_sacrum | - | - | 1.00 |
| fov_L2_S1 | - | - | 1.00 |
| fov_L1_coccyx | - | - | 1.00 |
| lumbarised_S1 | - | - | 1.00 |
| sacralised_L5 | - | - | 1.00 |
| dark_discs | - | fat_sat_failure | 1.00 |
| collapsed_discs | - | fat_sat_failure | 1.00 |
| compression_fracture_L2 | - | fat_sat_failure | 1.00 |
| schmorl_nodes | - | fat_sat_failure | 1.00 |
| scoliosis | - | fat_sat_failure | 1.00 |
| hardware_cage_screws | - | fat_sat_failure, metal | 1.00 |
| bias_field | - | - | 1.00 |
| motion_ghosting | motion_ghost | - | 0.21 |
| aliasing_wrap | - | wrap | 1.00 |

Realistic phantom:

| scenario | capping | advisory (would flag) | score |
|---|---|---|---|
| baseline | - | bias_field | 1.00 |
| t2_only | - | bias_field | 1.00 |
| t1_only | - | bias_field | 1.00 |
| stir_only | - | bias_field | 1.00 |
| pd_only | - | - | 1.00 |
| iso_1mm_3d | - | - | 1.00 |
| thick_6mm | - | bias_field | 1.00 |
| few_slices_7 | - | bias_field | 1.00 |
| field_3T | - | bias_field, fusion_or_cage, metal | 1.00 |
| fov_T10_sacrum | - | bias_field, fusion_or_cage, metal | 1.00 |
| fov_L2_S1 | - | bias_field | 1.00 |
| fov_L1_coccyx | - | bias_field | 1.00 |
| lumbarised_S1 | - | bias_field | 1.00 |
| sacralised_L5 | - | bias_field | 1.00 |
| dark_discs | - | bias_field | 1.00 |
| collapsed_discs | - | bias_field, fusion_or_cage, metal | 1.00 |
| compression_fracture_L2 | - | bias_field | 1.00 |
| schmorl_nodes | - | bias_field | 1.00 |
| scoliosis | - | bias_field, fusion_or_cage, metal | 1.00 |
| hardware_cage_screws | - | bias_field, fusion_or_cage, metal | 1.00 |
| bias_field | - | bias_field | 1.00 |
| motion_ghosting | motion_ghost | bias_field | 0.33 |
| aliasing_wrap | - | bias_field, metal, wrap | 1.00 |

Reading the tables:

- `motion_ghosting`: flagged by the capping rule on both looks (cartoon score 0.21, realistic 0.33). No false
  positives on the other 22 scenarios of either look.
- `aliasing_wrap`: flagged as wrap (advisory) on both looks, at the outermost superior level.
- `hardware_cage_screws`: metal and fusion are advisory on both looks. On the realistic look the metal rule also fires
  on `field_3T`, `fov_T10_sacrum`, `collapsed_discs`, `scoliosis` and `aliasing_wrap`, so hardware precision on the
  realistic phantom is 1 of 6 flagged scenarios.
- `bias_field`: NOT detected (the bias profile does not separate it from anatomy). This is a miss.
- The STIR fat rim (advisory) fires on the cartoon's soft-tissue skin rim (phantom limitation, not a real fat failure).


## Real data

Benchmark cache of this run (SPIDER validation 87, Sudirman 513 labelled patients with quality records, qx5rt 55). No
studies have labelled hardware or artefacts in these datasets as far as we know, so precision is measured by visual
review of a sample of hits (mid-sagittal slices rendered to /tmp, not committed).

| suite | capping flags (motion ghost) | metal advisory | fusion/cage advisory | wrap advisory | low SNR advisory | STIR fat advisory |
|---|---|---|---|---|---|---|
| SPIDER (87) | 0 | 83 | 82 | 13 | 13 | not run |
| Sudirman (513) | 0 | 476 | 468 | 178 | not run | 3 |
| qx5rt (55) | 0 | 25 | 18 | 1 | not run | not run |

No study on any real suite is capped, so the caps add no regression risk to the real-data numbers (see the golden
check below).

Visual review of hits (precision on the reviewed sample):

- metal: 4 SPIDER studies sampled (2_t1, 19_t2, 38_t1, 145_t2). All four show normal lumbar anatomy, no hardware:
  0 of 4 true hits.
- wrap: 3 SPIDER studies flagged (93_t1, 101_t2, 2_t1) and one control not flagged (107_t2_SPACE). The flagged edges
  show normal anatomy and the black field-of-view padding, not folded anatomy: 0 of 3 true hits.
- Sudirman and qx5rt hits were not reviewed; given the SPIDER rates they are assumed to be the same kind of false
  positive, which is why none of these can cap.
- motion ghosting: 0 hits on every real suite, so precision is not measurable. The prominence on SPIDER is about 0
  (max 0.018), far below the 0.15 threshold.

Golden check (`tools/benchmark.py --check benchmarks/golden.json`, final code): PASS, no regressions, 127 metrics within
tolerance; the capping policy leaves real-data labels and statuses unchanged. An earlier version that let metal, wrap and low SNR
cap failed the golden check (SPIDER announced coverage 67.8 % -> 2.3 %, Sudirman 60.2 % -> 1.6 %): that
is the evidence for the advisory policy. Do not re-enable a hardware or wrap cap without a real-data review that
clears the false-positive rate.

## Limitations

- Detectors are classical and not calibrated: `quality_*` values are judgement defaults. They have been tuned on the
  synthetic phantom and checked on public SPIDER data; a hit rate on real data says how often they fire, not how
  often they are right.
- Metal and cage attribution is per level by proximity (a void is assigned to the nearest disc or the discs touching
  its body). It does not separate a screw from a rod or cement, and it does not give the instrumented span.
- Realistic-look false positives remain (see the phantom table). The metal rule is the weakest: on the realistic look a
  screw core is at the noise floor.
- Ghost and wrap flags are study-wide or edge-wise. Ghosting applies to every level at the same confidence; a
  localised ghost would be missed or over-counted.
- Wrap only detects brightness at a field edge. A fold that does not brighten the edge band is not seen.
- The quality flags cap confidence and status. They do not correct a label; the verification path (`verify_labels`)
  remains the way to confirm levels.
- Research use only. Not a diagnosis.
