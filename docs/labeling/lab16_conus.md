# lab16_conus: conus medullaris locator

Branch `lab/lab16_conus`. Owner: `src/spinescope/anatomy/conus.py`, `tools/conus_audit.py`,
`tests/test_anatomy_conus.py`. Shared-file edits: `config.py` (conus block), `anatomy/pipeline.py`
(one hook in `label_spine_details`).

## Why

The learned labeller (default since main 1c9f553) returns discs and numbering but no conus, while the
classical labeller reported a rough tip. The conus matters clinically (low-lying conus, tethered cord) and
it is a weak numbering cue.

## What it does

`anchors.conus` is now written by `attach_conus(study, anchors, cfg)`, called once in
`label_spine_details` after the labels are final (so the numbering fusion is unchanged).

1. Sagittal series: T2 first, then STIR, then T1 (`pick_conus_stack`). CSF is bright on T2/STIR and dark on T1.
2. Canal: the learned canal class (`spinescope.anatomy.ml.segment_stack`, public API) on the 1 mm plane grid.
   The midline slice is `mid_sagittal`. Slices within `conus_slice_halfwidth_mm` (4 mm) of it are analysed.
3. Per row, the canal centreline (middle of the largest canal component, median-smoothed) and the canal half-width.
4. Profiles across the canal, window-relative (the volumes carry offsets, so raw ratios are meaningless):
   core = samples within 2 mm of the centreline; CSF = samples from 0.6 x half-width outwards.
   Cord-likeness = 1 - core/CSF on T2/STIR, 1 - CSF/core on T1. A plausibility gate requires CSF in the bright
   half of the window on T2 (dark half on T1).
5. Cord-like rows are median-filtered along z; the longest run (>= 6 mm) gives the cord. Its inferior end is the tip.
   A run that reaches the lowest canal row in the field of view is flagged `fov_end` (tip is a lower bound).
6. Slices whose tips agree within 6 mm of the median give z and xyz. Confidence = per-slice contrast and run
   length, times agreement among slices that show a tip, times the numbering-status factor
   (confirmed 1.0, probable 0.9, unconfirmed 0.7), capped at 0.5 when the tip is at the field-of-view end.
7. Level: the disc whose margin contains the tip (`"L1-L2"`, position `at disc`), else the vertebra between two
   adjacent discs with its third (`upper third` / `middle third` / `lower third`, from the disc margins).
   Gaps in disc labels are not named. Above or below all discs the end vertebra is named without a third.
8. Low-lying: tip below the L2-L3 disc margin (`low_lying: true`). Without the L2-L3 disc, the level name decides.
   Warning wording depends on confidence: at or above 0.6 "low-lying conus ... confirm on the sagittal T2";
   below 0.6 "possible low-lying conus ... too low to call". L3 is flagged as an indeterminate zone.

`anchors.conus` keys: `z`, `xyz` (LPS mm), `center` (= xyz, kept for the classical key), `level`, `position`,
`confidence`, `low_lying` (true / false / null), `fov_end`, `slices`, `agreeing_slices`, `contrast`, `evidence`.
If the locator finds nothing, a classical tip already present is kept with a warning; otherwise `conus` is `None`.
The hook catches exceptions, so the labels are never lost.

## Thresholds and sources

No conus ground truth exists in SPIDER, Sudirman or qx5rt, so no threshold is calibrated. The values are
judgement, documented in `config.py`:

| Parameter | Value | Basis |
|---|---|---|
| `conus_core_half_mm` | 2.0 | Cord is about 8 mm AP in the lumbar region (anatomy). |
| `conus_flank_inner_frac` | 0.6 | Keeps the CSF samples clear of the cord rim (judgement). |
| `conus_cord_contrast_min` | 0.25 | Cord about half as bright as CSF on T2 (judgement). |
| `conus_csf_min_rel` | 0.5 | CSF is the brightest (T2) or darkest (T1) fluid (physics). |
| `conus_min_run_mm` / `conus_full_run_mm` | 6 / 20 | A cord run is at least a few mm; 20 mm gives full credit (judgement). |
| `conus_slice_halfwidth_mm` | 4 | About one slice either side on 3-4 mm sagittal spacing. |
| `conus_agree_mm` | 6 | About half a vertebral body (anatomy). |
| `conus_status_factor` | 1.0 / 0.9 / 0.7 | Judgement: unconfirmed level names lower the confidence. |
| `conus_low_lying_disc` | L2-L3 | Wilson DA, Prince JR. AJNR 1989;10(2):259: termination at L2-L3 or above normal, L3 indeterminate, L3-L4 or below abnormal. |

Distribution reference (not used as a threshold): Saifuddin A, Burnett SJ, White J. Spine 1998;23(13):1452-6,
504 adults without deformity; mean tip in the lower third of L1, range mid-T12 to upper L3.
Citations come from a web search summary of the papers, not from reading the papers. Check them before publication.

## Phantom (ground truth `conus_z`)

Oracle canal segmentation from the phantom labels (the learned labeller is out of distribution on the phantom).
Tests are in `tests/test_anatomy_conus.py` (16 tests, all passing).

| Phantom | Truth | Located | Error |
|---|---|---|---|
| default (T2) | L1 mid, z 78.2 | L1, middle third, z 78.8 | 0.6 mm |
| default (T1 polarity) | L1 mid, z 78.2 | L1, middle third, z 78.8 | 0.6 mm |
| strong lordosis | L1 mid, z 65.1 | L1, z 65.1 | 0.0 mm |
| noise 0.06 | L1 mid, z 78.2 | L1, z 78.8 | 0.6 mm |
| conus at T12 | T12 mid, z 113.7 | T12, z 113.8 | 0.1 mm |
| conus at L3 | L3 mid, z 6.3 | L3, z 6.8, low-lying flagged with the L3 note | 0.5 mm |
| cord removed from the image | not present | not located (warning: no cord signal) | n/a |

## Visual audit (AI reader, not ground truth)

Set-up: `tools/conus_audit.py`, seed 16, outputs under `/tmp/spinescope-lab16_conus/` (renders not committed).
SPIDER: 40 of the 49 validation sagittal T2-type volumes (33 plain T2, 7 SPACE; the sample is seeded).
Sudirman: 20 of the patients in `patient_sessions` (seeded). Every case ran without error.
Each render is the sagittal slice nearest the tip, with the disc lines and the tip line.
Grades (written by one AI reader, from the render alone):

- **correct**: the cord end is visible and within half a vertebra of the tip;
- **wrong**: the cord end is visible and more than half a vertebra away;
- **can't tell**: no separable cord, or the slice does not show the thecal sac;
- **not located**: no tip.

Totals: SPIDER validation (40): located 30 = 6 correct, 1 wrong, 23 can't tell; not located 10.
Sudirman (20): located 14 = 0 correct, 0 wrong, 14 can't tell; not located 6.

The audit does not show the tip is reliable. Only 6 of 30 SPIDER tips were verifiable. About 8 of the
30 SPIDER tips and 7 of the 14 Sudirman tips sat on a slice that does not show the thecal sac in the
render, so the slice choice (the learned midline plus a 4 mm window) is itself a failure point.
Several confident tips were not supported by the image (172_t2 at 0.90, 118_t2 at 0.90, 93_t2 at 0.72).
So the confidence is overstated and is not calibrated.
The Sudirman studies gave one analysed slice each (about 5 mm sagittal spacing), so agreement across
slices never applied there.

| Dataset | Case | Grade | Tip z (mm) | Level (third) | Conf | Agree/slices | Low-lying | Note |
|---|---|---|---|---|---|---|---|---|
| SPIDER val | 2_t2 | correct | 60.6 | L1 (middle third) | 0.72 | 1/1 | no | grey cord tapers near L1 |
| SPIDER val | 11_t2 | can't tell | 90.1 | L3-L4 (at disc) | 0.20 | 1/3 | yes | crop off the canal |
| SPIDER val | 31_t2 | can't tell | 87.8 | T12-L1 (at disc) | 0.64 | 1/1 | no |  |
| SPIDER val | 35_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 35_t2_SPACE | can't tell | 184.4 | T12-L1 (at disc) | 0.46 | 3/5 | no |  |
| SPIDER val | 38_t2 | can't tell | 162.7 | T11 (middle third) | 0.29 | 1/3 | no |  |
| SPIDER val | 53_t2 | can't tell | 138.8 | L1 (upper third) | 0.72 | 1/1 | no |  |
| SPIDER val | 59_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 75_t2 | can't tell | 217.9 | L1 (middle third) | 0.24 | 1/3 | no |  |
| SPIDER val | 93_t2 | can't tell | 168.9 | L1 (middle third) | 0.72 | 1/1 | no | slice off the canal; confidence 0.72 |
| SPIDER val | 101_t2 | can't tell | 101.6 | L1-L2 (at disc) | 0.58 | 1/1 | no | crop empty above L3 (slice off the canal) |
| SPIDER val | 107_t2 | correct | 119.2 | T12-L1 (at disc) | 0.72 | 1/3 | no | grey cord tapers at T12-L1 |
| SPIDER val | 107_t2_SPACE | correct | 119.4 | T12-L1 (at disc) | 0.82 | 5/5 | no | cord tapers at T12-L1 |
| SPIDER val | 112_t2 | can't tell | 129.0 | L1 (lower third) | 0.24 | 1/3 | no |  |
| SPIDER val | 115_t2 | can't tell | 88.9 | L1-L2 (at disc) | 0.60 | 2/3 | no |  |
| SPIDER val | 118_t2 | can't tell | 187.7 | T12-L1 (at disc) | 0.90 | 2/3 | no | confidence 0.90, cord seems to run lower |
| SPIDER val | 129_t2 | can't tell | 82.2 | T12-L1 (at disc) | 0.14 | 1/3 | no |  |
| SPIDER val | 129_t2_SPACE | not located | - | - | - | - | - |  |
| SPIDER val | 136_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 136_t2_SPACE | wrong | 106.9 | L1 (upper third) | 0.67 | 3/5 | no | grey cord core visible down to about L1-L2; tip looks high (low certainty) |
| SPIDER val | 140_t2 | can't tell | 209.4 | T12 (upper third) | 0.40 | 2/3 | no |  |
| SPIDER val | 145_t2 | can't tell | 80.3 | T12 (middle third) | 0.53 | 1/3 | no |  |
| SPIDER val | 152_t2 | can't tell | 176.8 | T12 (upper third) | 0.60 | 2/3 | no |  |
| SPIDER val | 152_t2_SPACE | can't tell | 182.5 | T11-T12 (at disc) | 0.83 | 4/5 | no |  |
| SPIDER val | 161_t2 | correct | 137.4 | L1 (lower third) | 0.60 | 2/3 | no | root rings begin just below L1-L2 |
| SPIDER val | 161_t2_SPACE | correct | 133.9 | L1 (lower third) | 0.90 | 5/5 | no | root rings begin just below L1-L2 |
| SPIDER val | 165_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 168_t2 | can't tell | 143.2 | L1 (middle third) | 0.72 | 1/3 | no |  |
| SPIDER val | 171_t2 | correct | 56.9 | L1 (upper third) | 0.72 | 1/1 | no | sac narrows to roots at T12-L1/L1 |
| SPIDER val | 172_t2 | can't tell | 99.9 | L1 (middle third) | 0.90 | 2/3 | no | confidence 0.90, no thecal sac in the crop |
| SPIDER val | 177_t2 | can't tell | 216.0 | T11-T12 (at disc) | 0.78 | 2/3 | no |  |
| SPIDER val | 179_t2 | can't tell | 100.3 | T12-L1 (at disc) | 0.61 | 1/1 | no |  |
| SPIDER val | 192_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 198_t2 | can't tell | 129.9 | L1 (middle third) | 0.38 | 2/3 | no |  |
| SPIDER val | 208_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 210_t2 | can't tell | 70.8 | T12-L1 (at disc) | 0.47 | 2/3 | no |  |
| SPIDER val | 234_t2 | not located | - | - | - | - | - |  |
| SPIDER val | 234_t2_SPACE | not located | - | - | - | - | - |  |
| SPIDER val | 236_t2 | can't tell | 110.6 | T12 (middle third) | 0.68 | 1/3 | no |  |
| SPIDER val | 257_t2 | not located | - | - | - | - | - |  |
| Sudirman | 0006 | can't tell | 20.8 | L1 (upper third) | 0.71 | 1/1 | no |  |
| Sudirman | 0011 | can't tell | 33.3 | T12 (upper third) | 0.72 | 1/1 | no |  |
| Sudirman | 0024 | can't tell | 54.6 | L1 (lower third) | 0.72 | 1/1 | no |  |
| Sudirman | 0161 | can't tell | 97.6 | T12 (lower third) | 0.72 | 1/1 | no |  |
| Sudirman | 0248 | can't tell | 105.7 | T12-L1 (at disc) | 0.72 | 1/1 | no |  |
| Sudirman | 0250 | can't tell | 117.5 | L1 (lower third) | 0.53 | 1/1 | no |  |
| Sudirman | 0255 | can't tell | 123.5 | T11 (no third) | 0.43 | 1/1 | no |  |
| Sudirman | 0267 | can't tell | 68.9 | T12-L1 (at disc) | 0.60 | 1/1 | no |  |
| Sudirman | 0291 | not located | - | - | - | - | - |  |
| Sudirman | 0320 | can't tell | 135.3 | L1 (upper third) | 0.72 | 1/1 | no |  |
| Sudirman | 0336 | can't tell | 113.5 | L3 (lower third) | 0.33 | 1/1 | yes | low-lying flag (L3) on a slice with no visible sac; confidence 0.33 so worded as unconfirmed |
| Sudirman | 0342 | can't tell | 117.0 | T12 (upper third) | 0.55 | 1/1 | no | sac continues well below the tip; cord end not separable |
| Sudirman | 0350 | not located | - | - | - | - | - |  |
| Sudirman | 0381 | can't tell | 122.8 | T12 (upper third) | 0.46 | 1/1 | no |  |
| Sudirman | 0409 | can't tell | 114.9 | T12-L1 (at disc) | 0.39 | 1/1 | no |  |
| Sudirman | 0460 | can't tell | 113.9 | T10 (no third) | 0.33 | 1/1 | no | tip above the disc range (T10); slice off the canal |
| Sudirman | 0469 | not located | - | - | - | - | - |  |
| Sudirman | 0509 | not located | - | - | - | - | - |  |
| Sudirman | 0536 | not located | - | - | - | - | - |  |
| Sudirman | 0549 | not located | - | - | - | - | - |  |

## Limitations and next steps

- No ground truth, so accuracy is unknown. The audit is a visual AI-reader grade, not an expert read.
- Slice selection and the crop are the weakest links (see above). The next step is to locate the cord on the
  slices that contain the thecal sac, not the midline slice only, and to require a visible sac.
- Confidence is not calibrated; treat it as an ordering, not a probability.
- The conus level is not fed to numbering fusion. `fusion.py` runs before this hook, so the conus cue still uses
  only the classical tip there. Feeding this result into fusion is a separate change.
- Thresholds are judgement; they were set by looking at three SPIDER training cases (21, 47, 222) to fix the
  window-relative contrast (raw ratios were meaningless because the volumes carry offsets). They were not tuned
  on validation.
- Not compared with the classical tip on the audit cases.
- Sudirman studies are one slice each at this spacing; a wider window or spacing-aware sampling would help.

## Reproduce

    .venv/bin/python tools/conus_audit.py spider --split validation --n 40 --seed 16 --out /tmp/spinescope-lab16_conus/spider_val
    .venv/bin/python tools/conus_audit.py sudirman --n 20 --seed 16 --out /tmp/spinescope-lab16_conus/sudirman
    .venv/bin/python -m pytest -q tests/test_anatomy_conus.py
