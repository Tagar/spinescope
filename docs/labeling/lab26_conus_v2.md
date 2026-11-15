# lab26_conus_v2: conus locator v2, reference set and the decision on the experimental flag

Branch `lab/lab26_conus_v2` (from main c2b9718). Owner: `src/spinescope/anatomy/conus.py`,
`tools/conus_reference.py`, `docs/labeling/lab26_*`, `tests/test_anatomy_conus.py`. Shared-file edits:
`config.py` (conus v2 block). `anatomy/pipeline.py` is untouched; the conus hook and the experimental flag stay as they were.

## Decision

**The conus stays experimental.** Its confidence stays capped at `Config.conus_max_confidence = 0.5` and the
record stays marked `experimental`. The reason is the reference: on 60 public-data cases marked by one AI reader
before the v2 locator ran, the v2 tip was within half a vertebra (13 mm) on 0 of 15 located tips (v1: 1 of 35).
The reference cannot prove the locator wrong either, because its own marks are implausibly low (see "Reference
reliability"). Neither the accuracy bar (85% within half a vertebra) nor the calibration bar can be shown with these data.

## What v2 does (conus/2)

1. **Sac slices, all sagittal slices.** The canal area of every slice of the segmented stack is measured. The sac's
   mid-portion lies on the slices whose area is at least `conus_sac_area_frac` (0.6) of the largest; the nearest
   `conus_max_slices` (5) to the mid-sagittal plane are analysed.
2. **Interpolation.** The cord evidence of the two sac slices that bracket the mid-sagittal plane is blended linearly
   by distance to that plane (`_bracket_weights`). The tracking runs on this one profile. Each sac slice is also
   tracked on its own, for the consistency vote.
3. **Cord evidence per row** (as v1): canal centreline and half-width, core (cord, within 2 mm) against CSF flank
   (from 0.6 of the half-width outwards), window-relative. Cord-like rows need the CSF in the bright (T2/STIR) or dark
   (T1) half of the window and a contrast of at least 0.25.
4. **Tracking from the top.** The first cord run of at least 6 mm is followed caudally. Non-cord gaps shorter than
   `conus_stop_mm` (8 mm) are bridged; a gap of 8 mm or more ends the cord (width collapsed to roots and filum).
   Reaching the lowest canal row means the tip is a lower bound (`fov_end`, confidence capped at 0.5).
5. **Consistency.** With two or more sac-slice tips, at most half may agree (within 6 mm) with the interpolated tip,
   otherwise the conus is not located. With a second sagittal contrast in the study (T1 next to T2) its tip is located
   too (one more network run, about 1 s on CPU): agreement keeps the confidence, disagreement halves it, absence
   gives a factor of 0.9.
6. **Level and low-lying flag**: unchanged from v1 (`level_at`, `low_lying`).

New config (`config.py`, conus v2 block): `conus_sac_area_frac`, `conus_max_slices`, `conus_min_agree_share`,
`conus_bridge_mm` (not used by the tracker; the stop length does the work, see below), `conus_stop_mm`,
`conus_cross_contrast`, `conus_xc_agree_factor`, `conus_xc_disagree_factor`, `conus_xc_absent_factor`.
`conus_slice_halfwidth_mm` is kept but no longer used. Public signatures are unchanged; `conus_from_segmentation`
gained keyword-only `cross` and `cross_contrast`.

Note: `conus_bridge_mm` is declared in the config but the tracker bridges gaps shorter than `conus_stop_mm`. Remove it
or wire it up in a follow-up; it has no effect today.

## Validation

### (a) Phantom (ground truth `conus_z`), oracle canal from the phantom labels

`.venv/bin/python tools/conus_reference.py phantom --out /tmp/...`. Errors in mm.

| Phantom | Truth z | Located z | Error | Level | Confidence |
|---|---|---|---|---|---|
| cartoon, default (T2) | 78.16 | 78.84 | 0.68 | L1 | 0.72 |
| cartoon, default (T1 polarity) | 78.16 | 78.84 | 0.68 | L1 | 0.72 |
| cartoon, noise 0.06 | 78.16 | 78.84 | 0.68 | L1 | 0.72 |
| cartoon, conus at T12 | 113.72 | 113.84 | 0.12 | T12 | 0.47 |
| cartoon, conus at L2 | 42.31 | 42.84 | 0.53 | L2 | 0.72 |
| cartoon, conus at L3 | 6.34 | 6.84 | 0.50 | L3 | 0.72 |
| realistic, default (T2) | 73.48 | 73.46 | 0.01 | L1 | 0.75 |
| realistic, default (T1) | 73.48 | not located | n/a | n/a | n/a |
| realistic, conus at L3 (T2) | 4.96 | not located | n/a | n/a | n/a |

The two realistic misses are the consistency rule at work: the sac-slice tips disagree in z, so the locator abstains
rather than give a tip. The cartoon phantom is easy: its cord is a drawn 8 mm tube that ends sharply.
The realistic T2 tip is right to 0.01 mm. Unit tests (21 in `tests/test_anatomy_conus.py`) cover tracking, bracketing,
the consistency rule and the cross-contrast factor.

### (b) Reference set: 40 SPIDER validation + 20 Sudirman, marked by eye before the v2 locator ran

Cases: the seed-16 samples from lab16 (`tools/conus_audit.py`), so the same 60 studies. Blinded crops
(`tools/conus_reference.py render`): the midline sagittal slice (learned canal, the crop centred on the canal), disc lines
from the learned labeller (no conus), a height ruler in LPS mm. No locator output was drawn. Marks were typed into
`docs/labeling/lab26_reference_marks.csv` and committed (commit 9ec9956) before `run` was executed on v2.

Caveats that matter for reading the numbers:

- **One AI reader, low confidence.** Every mark is "uncertain" except the 16 "not visible" (no separable cord, or
  the slice is off the canal). The ruler reading is about +-5 mm by eye. No clear marks were recorded.
- **The reader had seen the lab16 audit table** (v1 grades and tips) before marking. The marks are not independent of
  v1 in that sense; they were read from the crops, not from the v1 numbers, but this should be stated.
- **The marks are implausibly low.** Of the 44 tip marks, 31 are at L3 or below (L3 15, L4 8, L3-L4 7, L4-L5 1),
  median z 35 mm. A normal conus sits at T12 to L2 (Saifuddin et al. 1998, mean in the lower third of L1). So the
  reader more likely saw where the grey cord fades into roots than where the conus ends. That makes the marks a poor
  ground truth. This is the main limitation of this section.

Results (scores in `docs/labeling/lab26_reference_scores.csv`; summary from `tools/conus_reference.py score`):

| | v1 (lab16, main) | v2 (this branch) |
|---|---|---|
| Cases | 60 (44 with a tip mark, 16 not visible) | 60 |
| Located, total | 44 | 21 |
| Located on tip marks | 35 | 15 |
| Located on not-visible marks (false tip) | 9 | 6 |
| Abstained on tip marks | 9 | 29 |
| Within half a vertebra (13 mm), of located tip marks | 1 (2.9%) | 0 (0%) |
| Median error, located tip marks | 74 mm | 87 mm |
| Level agreement, located tip marks | 0 | 0 |

v2 errors are systematically positive: its tips sit 50 to 120 mm above the marks. The v2 located tips have a median
of z 107 mm (T12-L1 to L1, the anatomically usual region), which is where the marks disagree with anatomy. The two
methods put the tips in the same region, so the disagreement is with the marks, not between the two methods.

Confidence: all 15 located tip marks have confidence 0.17 to 0.50 (the cap). None is within half a vertebra, so the
confidence does not separate right from wrong. Calibration is impossible on this reference.

Consistency (c): 8 SPIDER studies have a T2 tip and a T1 tip. Their T1-T2 z differences are 0.1, 3.9, 5.2, 5.9, 7.9,
9.0, 20.0 and 49.7 mm; 4 of 8 agree within 6 mm. The T2/T2-SPACE pairs of the same patient: 129 (82.2 vs 81.2 mm)
and 136 (110.9 vs 106.9 mm) agree within 4 mm; the other pairs were not located on both. Small samples, and not
independent of the reference.

### Calibration

Not possible: no located tip fell within half a vertebra, so there is no positive class to calibrate against.
The reference is also biased (above), so any mapping fitted to it would be fitted to the reader.

### Threshold exploration (disclosed)

One exploratory run changed the tracker to a 20 mm stop, a 6 mm bridge and a 0.15 contrast minimum, on the 44 tip-marked
reference cases only. It located 16 tips and gave 0 within half a vertebra, median error 82 mm. The defaults were
not changed. Setting thresholds on the reference set would be tuning on validation, which the wave rules forbid; this
run is recorded only to show that loosening the thresholds does not fix the agreement.

## What is still wrong (why the flag stays)

1. The cord-likeness signal (window-relative core against CSF flank, on the network canal mask) is noisy on 3-5 mm sagittal
   T2. The tracker follows it and stops at the first spot where it fails, typically T12-L1 to L1, while a grey cord is
   still visible to the reader. Tracking the cord on the canal mask alone is not enough.
2. The reference marks are biased low (above), so the comparison does not measure accuracy against anatomy.
3. Two thirds of the real cases give no tip at all (cord not separable, slice off the canal). A conus locator that
   abstains this often cannot be the sole source of a low-lying flag.

## Next steps (not done here)

- Get a reference that is not one reader: at least two radiologists marking the cord end on the same crops, with
  inter-reader agreement reported. Without it the accuracy bar cannot be measured.
- Use the sagittal intensity profile of the cord (the mid-canal line on T2, not the canal mask) as the tracking signal,
  and use the cauda equina (roots) as the stop criterion rather than a contrast drop.
- Wire `conus_bridge_mm` into the tracker or remove it.
- Feed the conus into numbering fusion only after it passes the same bar (currently it does not).

## Reproduce

    .venv/bin/python tools/conus_reference.py render --out /tmp/spinescope-lab26_conus_v2/ref   # blinded crops (done before the v2 run)
    .venv/bin/python tools/conus_reference.py run --out /tmp/spinescope-lab26_conus_v2/v2
    .venv/bin/python tools/conus_reference.py score --out /tmp/spinescope-lab26_conus_v2/v2 --marks docs/labeling/lab26_reference_marks.csv
    .venv/bin/python tools/conus_reference.py phantom --out /tmp/spinescope-lab26_conus_v2/phantom
    .venv/bin/python -m pytest -q tests/test_anatomy_conus.py

The v1 baseline is `run` on main (c2b9718).
