# Labelling evaluation tooling and baseline failure taxonomy (lab01_eval)

Status: baseline of the current classical labeller (`spinescope.anatomy.label_spine`, main at `dc43dc0`),
measured on the SPIDER `validation` split (87 volumes, all sequences). This is a measurement, not a
calibration: nothing was tuned on `validation`, and no defaults were changed.

## What the tooling does

- `src/spinescope/anatomy/qa.py`: `overlay(study, anchors, truth)` draws the mid-sagittal slice nearest the
  labeller's midline with predicted discs (centre, plane line, label, confidence colour), ground-truth
  discs with connectors to their matches, and the canal centreline. `case_report(...)` writes one PNG page
  per case (overlay, a disc-plane reformat per disc, match table).
- `tools/eval_labeling.py`: runs a labeller over a split with a process pool and a per-case JSON cache
  (keyed by a hash of the labeller source and the backend spec, so unchanged labellers are read from cache).
  It writes `summary.md`, `metrics.json`, `cases.csv`, `gallery.html` and one QA page per worst case, into a
  folder outside the repository. Backends: `classical`, `ml` (if `spinescope.anatomy.ml` exists), or any
  `module:function` that takes a `Study` and returns `Anchors`.
- `spinescope.anatomy._validate` (additive): per-disc match records (`DiscMatch`), missed truth discs,
  level-error helpers (`level_index`, `level_offset`), per-level recall, the numbering error histogram,
  reliability (confidence bins, ECE, AUROC), extended pooled metrics (off-by-one rates, whole-volume
  correctness, false positives per volume), `group_scores`, and `truth_from_phantom`.

Reproduce (outputs stay under `/tmp`):

    python tools/eval_labeling.py --backend classical --dataset spider --split validation \
        --out /tmp/spinescope-lab01_eval/run --cache /tmp/spinescope-lab01_eval/cache --jobs 8

A full validation run takes about 15 s from scratch on an 8-process pool (median 0.28 s per volume in the
labeller), and about 8 s from cache.

## Protocol

- Truth: SPIDER masks (`truth_from_mask`). Disc label 201..2xx is one disc. Discs are ranked from the bottom
  and named L5-S1, L4-L5, ... by rank (the convention of `disc_names`). The lowest label is assumed to be
  L5-S1; this is not verified against anatomy. Checked here: the disc labels are contiguous from 201 in all
  87 validation volumes and their count matches `num_discs` in `overview.csv`, so the rank naming is
  internally consistent.
- Matching: a predicted disc matches the nearest unused truth disc within 12 mm in the sagittal (y, z)
  plane. Correct = matched with the exact level name.
- Midline truth: median LPS x of the canal voxels (fixed in this change, see below).
- Overview columns read: `new_file_name`, `subset`, `Manufacturer`, `MagneticFieldStrength` only.

## Overall (validation, 87 volumes, 0 errors)

| metric | value |
|---|---|
| disc recall | 19.6% |
| disc precision | 19.1% |
| label accuracy, of matched discs | 31.7% |
| label accuracy, of all truth discs | 6.2% |
| off-by-one level, of matched discs | 39.0% |
| off-by-one level, of all truth discs | 7.6% |
| volumes with every truth disc labelled right | 0.0% |
| false positives per volume | 5.98 |
| centre error, median (mm) | 5.9 |
| centre error, 90th percentile (mm) | 9.9 |
| share of matched discs within 3 mm | 23.6% |
| normal error, median (deg) | 5.1 |
| midline error, median (mm) | 11.7 |
| midline error, 90th percentile (mm) | 32.5 |

These agree with the held-out baseline in `docs/validation.md` (recall 0.20, label accuracy 0.06, matched
label accuracy 0.32).

## Breakdown

By sequence:

| sequence | volumes | recall | precision | label acc (truth) | off-by-one (matched) | centre median mm | midline median mm |
|---|---|---|---|---|---|---|---|
| T1 | 38 | 25.6% | 26.5% | 11.7% | 32.9% | 6.2 | 6.9 |
| T2 | 38 | 15.0% | 14.1% | 2.6% | 55.0% | 4.7 | 21.9 |
| T2 SPACE | 11 | 14.6% | 13.7% | 0.0% | 23.1% | 7.9 | 33.2 |

By field strength and vendor. These two splits are identical: every 1.5 T volume is Siemens and every 3.0 T
volume is Philips in this split, so field and vendor cannot be separated here.

| group | volumes | recall | precision | label acc (truth) | off-by-one (matched) | centre median mm | midline median mm |
|---|---|---|---|---|---|---|---|
| 1.5 T / Siemens | 55 | 14.8% | 14.8% | 4.3% | 41.9% | 5.9 | 19.4 |
| 3.0 T / Philips | 32 | 29.0% | 27.2% | 10.0% | 36.1% | 5.9 | 9.0 |

Per level (truth level names; matched and label accuracy among the matches):

| level | truth discs | matched | recall | label acc (matched) |
|---|---|---|---|---|
| L5-S1 | 87 | 16 | 18.4% | 18.8% |
| L4-L5 | 87 | 13 | 14.9% | 23.1% |
| L3-L4 | 87 | 12 | 13.8% | 16.7% |
| L2-L3 | 86 | 21 | 24.4% | 33.3% |
| L1-L2 | 86 | 23 | 26.7% | 39.1% |
| T12-L1 | 86 | 26 | 30.2% | 38.5% |
| T11-T12 | 58 | 9 | 15.5% | 44.4% |
| T10-T11 | 37 | 2 | 5.4% | 50.0% |
| T9-T10 | 15 | 1 | 6.7% | 0.0% |

Recall is lowest at the bottom of the lumbar spine (L3-L5, 14-18%), where the numbering anchor is.

Numbering error histogram (matched discs; predicted level minus truth level, + = labelled too cranial):

| level error | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|
| discs | 5 | 8 | 39 | 40 | 22 | 9 |

Only 39 of 123 matched discs (32%) carry the right level number. Of the 84 that do not, 71 are labelled too cranial.

Confidence calibration (correct = matched with the exact level name; 643 predictions):

- ECE 0.240, AUROC of confidence for correctness 0.69.
- 603 of 643 predictions (94%) fall in the 0.2-0.3 bin, with mean confidence 0.28 and accuracy 5%.
- Predictions above 0.5 confidence number 30 in total and are 23% correct. Confidence is therefore not
  calibrated on this split: most predictions sit at one low value, and the confident ones are often wrong.

## Failure taxonomy

Built from the automatic per-volume tags (share of the 87 volumes) and from reviewing 18 of the worst overlays
by eye. Volume IDs are SPIDER file stems used as examples.

| tag | volumes | share | meaning |
|---|---|---|---|
| wrong_midline | 66 | 76% | midline more than 5 mm from the truth canal |
| midline_at_stack_edge | 10 | 12% | wrong midline in the outer 10% of the sagittal coverage |
| no_match | 47 | 54% | no predicted disc matches any truth disc |
| partial_column | 23 | 26% | fewer than half of the truth discs matched |
| over_detected | 35 | 40% | more predicted discs than truth discs |
| spurious_lowest_disc | 37 | 43% | the most caudal predicted disc matches no truth (with other matches present) |
| numbering_shift | 27 | 31% | most matched discs carry a non-zero level error |
| truth_outside_stack | 0 | 0% | truth midline outside the sagittal coverage (none) |

Labeller warnings present (share of volumes): thin interval rejected as not disc-like 51%; lowest disc less
tilted than the disc above 40%; conus not located 40%; vertebral body too long for one body (disc may be
missed) 32%; only a few mm of bone below the lowest disc 9%; numbering confidence low 6%.

Observed modes, in order of impact:

1. **Lateral mid-sagittal slice (dominant).** The labeller's midline is off by more than 5 mm in 76% of
   volumes, usually lateral. Overlays show the truth discs 10-45 mm off the displayed plane; the predicted
   discs then sit on posterior elements, foramina or paraspinal fat, and the drawn canal runs along the
   posterior elements rather than the thecal sac (examples: 112_t1, 75_t2, 107_t2_SPACE, 152_t2,
   118_t2_SPACE, 61_t2_SPACE). Because every disc is read from this slice, one wrong midline gives
   no-match volumes even when the chain itself is plausible. Only 11% of volumes are within 2 mm of the
   truth midline.
   - Direction (signed = predicted minus truth; + = toward patient-left): T2 errors are two-sided (median
     signed error -0.1 mm, median absolute error 21.9 mm; 47% positive). T2 SPACE leans positive (median
     +33 mm; 8 of 11 volumes positive). T1 is small and leans positive (median +3.9 mm, 68% positive; absolute
     median 6.9 mm).
   - Edge cases: in some volumes the chosen slice is the first or last slice of the stack (e.g. 179_t2,
     35_t1), so the ridge peak is at the boundary of the coverage (10 volumes).
2. **Spurious discs below the true L5-S1; numbering from a false bottom.** Intervals in the sacrum or below
   the lumbosacral disc are read as discs, and every level above is numbered from that false bottom, so the
   column shifts cranially by one to three levels (examples: 101_t2 +1, 210_t2 +2 with two spurious discs,
   74_t2 +3 with three, 136_t2_SPACE +2 to +3). The labeller's warning "lowest disc is less tilted than the
   disc above (unusual for L5-S1)" is raised in 40% of volumes and accompanies these cases.
3. **Short field of view: the bottom level is a default guess.** When only a few mm of bone are visible
   below the lowest disc, the labeller names it L5-S1 by default and says so (9% of volumes). In 118_t1 the
   result is a two-level shift.
4. **Missed discs and rejected thin intervals.** Real discs are dropped as "thin" (rejected in 51% of
   volumes), and long vertebral bodies (a missed disc) are flagged in 32% (examples: 192_t1 with four
   discs predicted for six truth discs; 179_t2 with four predicted for seven). Partial columns follow.
5. **Over-detection.** More predicted discs than truth discs in 40% of volumes. In several overlays the extra
   discs are thoracic (T8-T9 to T10-T11) and lie on a lateral slice (examples: 152_t2, 136_t2_SPACE,
   107_t2_SPACE).
6. **Confidence does not flag the failures.** Most discs, right or wrong, carry the same low value (0.26 or
   0.30: the labeller gives 0.3 when no endplate margins are found, and 0.85 times that when the numbering
   confidence is below 0.8). Some wrong discs are confident: 210_t2 T10-T11 (0.71) and T11-T12 (0.72) are both
   two levels off, and 19_t1 L4-L5 (0.81) is a false positive. Low numbering confidence is reported in only 6%
   of volumes, so it does not mark the failures either.
7. **Conus.** Not located in 40% of volumes (cord signal in the thecal sac not found).

T1 versus T2: T1 has the best recall (25.6%) and the smallest midline error (median 6.9 mm). T2 and T2 SPACE
have recall 15% and midline medians of 22 and 33 mm, so the midline failure of mode 1 is the main reason T2
is worse. In T2 the matched discs are also more often off by one (55% versus 33% on T1). Field strength and
vendor are confounded in this split (see the table above), so the 1.5 T and 3.0 T differences cannot be
attributed to field strength.

## Data caveats and corrections

- **Midline truth fixed in this change.** `truth_from_mask` computed the canal x from the sitk x index alone,
  which is only correct when the sitk x axis runs along LPS x. On some SPIDER volumes it runs along y (e.g.
  136_t2_SPACE: truth +50 mm instead of -1.9 mm). The canal voxels now go through the full image-to-LPS
  transform, as the disc truth already did. Effect on the validation split: midline median 12.4 to 11.7 mm,
  90th percentile 34.3 to 32.5 mm. Disc truth and all disc metrics are unchanged. `docs/validation.md`
  describes the midline failure with the old truth; its numbers should be re-checked before they are quoted.
- Truth naming assumes the lowest labelled disc is L5-S1 (not verified against anatomy). All level-name
  numbers above inherit this assumption.
- The worst-case ranking is by severity: a missed truth disc counts 1, a matched disc with the wrong level
  name counts 0.5, averaged over the truth discs, ties broken by midline error. The first 30 are all
  no-match volumes, mostly T2.
- Images derived from SPIDER are not committed; the gallery lives under `/tmp`.

## Next steps (for the labelling wave)

- Midline: the canal ridge picks lateral CSF-like signal on T2 and on SPACE volumes. A sagittal midline
  check against the axial CSF blobs and a T1-versus-T2 consistency test should come before any chain work.
- Sacrum: a rule that a disc is only accepted below the last vertebral body if its endplates follow the
  sacral slope, plus a check against the posterior elements, would remove the false-bottom shift.
- Confidence: replace the constant floor with evidence-based terms (endplate contrast, margin agreement,
  chain agreement across both sequences) and re-calibrate on `training` only.
