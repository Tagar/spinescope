# Mid-sagittal plane by mirror symmetry (lab05_midline)

Status: the midline error is cut from a median of 11.7 mm to 1.4 mm on the SPIDER held-out split. The
wave target (median <= 2 mm and >= 90 % within 3 mm) is **met for the median and not for the 3 mm share**
(71 % within 3 mm on validation). Disc recall and label accuracy move, but the disc chain is still the
main limit (see the downstream section).

## What changed

- `src/spinescope/anatomy/midline_robust.py` (new):
  - `estimate_stack_midline(stack, config) -> MidlineEstimate` for one sagittal stack, no axial data.
  - `estimate_midsagittal(study, config) -> MidlineEstimate` for a study: the best sagittal stack, refined
    by axial CSF blobs through `midline.refine_midline_axial` when axial slices cross the canal.
  - `MidlineEstimate` holds `x_mm` (scalar midline, LPS x), a curve `knots_z`/`knots_x` with `x_at(z)`,
    `slice_weights` (one per slice, Gaussian in x around the midline), `bands` (the knots, with their
    vote prominence), `confidence` (0..1), `evidence` and `warnings`.
- `src/spinescope/anatomy/midline.py`: `select_midline` calls the symmetry estimate by default
  (`midline_method="symmetry"`). The earlier canal-ridge method is kept as `midline_method="ridge"`, and it
  is also the fallback when the symmetry estimate is unavailable. Public signatures unchanged.
- `src/spinescope/config.py`: new `midline_*` defaults, each with a comment (mm, relative or method values).
- `tests/test_midline_robust.py`: 19 phantom tests (shifted spines, three contrasts, cropped and flipped
  fields, scoliotic and rotated variants, axial refinement, degenerate inputs, determinism, label_spine
  integration, ridge method still runs).
- `sagittal.py` is unchanged: its track and ridge functions are still used by the labeller and the
  axial refinement.

## Method

1. Each sagittal slice is resampled to about 1 mm in-plane (`midline_grid_mm`), and the stack is sorted by x.
2. For each candidate plane `x0` (1 mm grid) the mirror pairs `x0 - d` and `x0 + d` are compared by
   normalised cross-correlation, averaged over `d` = 3 to 20 mm (`midline_sym_d_mm`). Slices are linearly
   interpolated in x, so the candidate is continuous. Two similarities are averaged: raw intensities and a
   high-pass (image minus a 6 mm Gaussian, `midline_hp_sigma_mm`), which keeps structures and drops the
   slowly varying coil and fat signal.
3. The peak is refined to sub-grid precision by a parabola. Its prominence is the height above the best value
   more than 4 mm away, relative to the height above the median.
4. Patch votes: the same curve in overlapping 40 mm square patches (`midline_patch_mm`, step 20 mm). Each patch
   has its own mirror peak. Patches with prominence below 0.5 (`midline_patch_min_prominence`) are ignored.
   Outside the spine (bowel, fat, air) the patches are asymmetric and vote weakly, so the spine dominates.
5. Curve: at each vertical position the weighted median of its votes gives a knot. The curve is the
   piecewise-linear interpolation of the knots. The scalar `x_mm` is the median of the curve over the knot
   range.
6. Slice weights: `exp(-0.5 ((x_slice - x_mm) / 10 mm)^2)`.
7. Confidence = 0.4 prominence + 0.4 vote agreement (votes within 3 mm of the curve, weighted) + 0.2
   residual score. Capped at 0.5 when fewer than two knots remain (global plane only).

Runtime: about 1.5 s per volume in the labeller (`tools/eval_labeling.py` reports 1.51 s median per volume).

## Calibration and held-out protocol

- Calibration on SPIDER `training` only (360 volumes). The choices made there were: the patch vote
  aggregation (weighted median of votes with prominence >= 0.5 beat a single global peak, a straight-line
  fit, and the high-pass alone), the removal of a deviation gate around the global plane (it cost 0.1 mm
  on training), and piecewise-linear knots over a straight line (equal within noise on SPIDER, needed for
  scoliosis on the phantom).
- Held-out: SPIDER `validation` (87 volumes), run once with the final configuration.
- Ground truth: `truth_from_mask` canal median x (canal label 100, full transform). Note that the canal
  voxels spread over about 30 mm in x and the canal x can vary by 30 mm along z (for example volume 1_t1: from
  -7 mm to +26 mm), so the scalar truth is a summary of a curve.
- Error: |x_mm - canal median x|. Shares are over volumes.

## Results

Held-out (SPIDER validation, 87 volumes, 0 errors):

| metric | value |
|---|---|
| midline error, median | 1.43 mm |
| midline error, 90th percentile | 9.85 mm |
| within 2 mm | 59 % |
| within 3 mm | 71 % |
| within 5 mm | 87 % |

Training (calibration, 360 volumes): median 1.43 mm, p90 4.99 mm, within 2 mm 62 %, within 3 mm 73 %,
within 5 mm 90 %. The global plane alone (no patch votes) gives median 1.70 mm and 69 % within 3 mm on training.

By vendor, field and sequence (validation; training in brackets):

| group | volumes | median mm | p90 mm | within 2 mm | within 3 mm |
|---|---|---|---|---|---|
| Philips 3.0 T | 32 (149) | 1.42 (1.64) | 4.65 (5.36) | 62 % (56 %) | 72 % (72 %) |
| Siemens 1.5 T | 55 (168) | 1.66 (1.23) | 11.22 (4.29) | 56 % (67 %) | 71 % (75 %) |
| Philips 1.5 T | - (30) | - (1.37) | - (3.40) | - (70 %) | - (83 %) |
| Siemens 3.0 T | - (13) | - (3.53) | - (5.82) | - (46 %) | - (46 %) |
| T1 | 38 (158) | 1.38 (1.42) | 5.05 (4.93) | 66 % (60 %) | 76 % (73 %) |
| T2 | 38 (172) | 1.86 (1.44) | 6.35 (4.97) | 53 % (63 %) | 66 % (74 %) |
| T2 SPACE | 11 (30) | 1.41 (1.78) | 13.27 (6.15) | 55 % (63 %) | 73 % (70 %) |

Siemens 3.0 T is 13 training volumes and absent from validation; Philips 1.5 T is 30 training volumes and absent
from validation. Those rows are small samples and should not be read as vendor effects.

Confidence: weak. On validation, the area under the ROC curve for confidence against "within 3 mm" is
0.60 (training 0.59). Volumes with confidence >= 0.7 (41 % of validation) are within 3 mm 81 % of the time,
against 65 % below 0.7. Use confidence to flag, not to gate.

Baseline on the same split (ridge midline, `midline_method="ridge"`, same cases file): median 11.72 mm,
p90 32.54 mm, within 2 mm 11 %, within 3 mm 17 %, within 5 mm 24 %.

## Downstream effect (classical labeller, SPIDER validation)

Run with `tools/eval_labeling.py --backend classical --split validation` (symmetry, the default) and with a
scratch backend that sets `midline_method="ridge"` (baseline). Same cases, same scorer.

| metric | ridge midline (baseline) | symmetry midline |
|---|---|---|
| disc recall | 19.6 % | 29.6 % |
| disc precision | 19.1 % | 32.2 % |
| label accuracy (of all truth discs) | 6.2 % | 8.7 % |
| label accuracy (of matched discs) | 31.7 % | 29.6 % |
| centre error, median | 5.9 mm | 5.2 mm |
| share of matched discs within 3 mm | 23.6 % | 29.0 % |
| off-by-one level (of matched) | 39.0 % | 43.0 % |
| midline error, median | 11.7 mm | 1.4 mm |
| false positives per volume | 6.0 | 4.5 |
| volumes with every truth disc labelled right | 0.0 % | 1.1 % |

The midline fix raises recall and precision, but the label accuracy is still 9 %. The disc chain (numbering from
the lowest disc, endplate tracing) is now the dominant failure, as the baseline notes said. The labeller does
not use the curve x(z) yet: discs are projected onto the scalar plane (`pipeline._disc_geometry`), so the
integration of the curve is a follow-up.

## Approaches tried and rejected (training numbers unless stated)

- Canal CSF ridge per slice (`track_canal`, the old method): median 11.7 mm on validation. Its per-slice score
  profile does not peak at the canal (volume 221_t2: peak at 25 mm, canal at 6 mm).
- CSF tube contrast along the canal track in a +/-10 mm window: median 3.8 mm on 40 training volumes, worse
  than the symmetry estimate of that version on the same 40 (2.1 mm). Lateral slices pick up other bright
  structure.
- Full-width vertical bands (40 mm) of symmetry: median 1.8 mm over 500 band estimates on 40 training volumes
  (the 2-D patch votes are the final choice).
- Central crop of the field of view (70 % and 50 % of the in-plane width): global median 1.1-1.5 mm on 40
  training volumes, no clear gain.
- Gradient-magnitude similarity: worse (2.0 mm median on 40). A high-pass at a smaller scale (1.5 mm) was
  worse, and at 3 mm about the same as raw intensities. The final 6 mm scale was chosen in the module-level
  training run and was not swept separately.
- Blur along x to remove the slice-alignment effect: worse on the phantom (an 8 mm shift gives 7.5 mm instead of
  8.1 mm, and a flipped stack 5.3 mm instead of 6.0 mm).
- Linear-only x(z) through the votes: scoliosis not followed (phantom curve flat), same accuracy on SPIDER.
- Deviation gate around the global plane (12 mm): costs about 0.07 mm median and 3 points within 3 mm on training.
- Mirror pairs along the sagittal normal only: no tilt model (see limits).

## Limits

- **Axial rotation.** A sagittal stack cannot represent a mid-sagittal plane tilted about the vertical axis.
  On the phantom at 8 degrees the estimate is 4.3 mm off the canal; the likely cause is that the body centres
  sit on the tilted plane, so the symmetry axis averages over the cross-section. A tilted-plane search (mirror pairs along the plane's
  normal in the axial plane) is the natural next step. Not implemented.
- **Body outline vs spine.** On the phantom the body outline is centred at x=0 while the spine is shifted to
  8 mm. The outline then competes with the spine for the symmetry axis. A flipped raw stack gave a 2 mm error;
  `load_study` normalises display orientation, so this is mostly a phantom effect. The test allows 3 mm.
- **Slice alignment.** Candidates on sampled slices compare unblurred pairs and others compare interpolated
  ones. The effect is visible on the phantom (5 mm spacing) and is not removed by blurring. The patch
  aggregation absorbs most of it.
- **Target not met.** 71 % within 3 mm, not 90 %. The residual error looks systematic per volume: on training
  volume 185_t2_SPACE the estimate is 28.1 mm against a canal median of 21.5 mm, and the votes agree with each
  other, so it is a consistent offset rather than noise. The likely cause is a body-to-canal offset that a
  body-symmetry cue cannot see; a canal-specific cue would be needed for the rest. This is a hypothesis, not
  something measured here. A coronal reformat of 185_t2_SPACE, checked by eye, shows the estimated curve through
  the vertebral-body column and the mask canal offset to one side of it. That is consistent with the hypothesis,
  but it is one volume.
- Single sagittal volumes only. Axial refinement is used when axial slices cross the canal (phantom: 0.0 mm error).
- The scalar midline is the median of the curve over the knot range; the truth is the median of canal voxels over
  the canal extent. They agree when the canal extent matches the spine extent (true on SPIDER and the phantom).

## Reproduce

    python tools/eval_labeling.py --backend classical --dataset spider --split validation \
        --out /tmp/spinescope-lab05_midline/eval --cache /tmp/spinescope-lab05_midline/cache --jobs 8

The midline error per volume is the `midline_err_mm` column of `cases.csv`; the median and p90 are in
`metrics.json` (`midline_err_median_mm`, `midline_err_p90_mm`). The shares within 2 and 3 mm are computed from
`cases.csv`. Phantom tests: `pytest tests/test_midline_robust.py`.
