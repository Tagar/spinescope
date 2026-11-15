# lab17_orientation: rotated volumes, slice axis and the paired T1+T2 path

Branch `lab/lab17_orientation`. Owns `io/volumes.py`, `series.py` (role fields), `geometry.py`
(`plane_margin_deg`, additive), `datasets/spider.py` (`SpiderStudy.volume/mask`, `to_study`),
`tests/_spider.py`, and the new tests in `tests/test_io_volumes_orientation.py`. Shared-file edits:
`config.py` (three additive fields in the `Config` block at the end, `volume_*` and `plane_ambiguity_deg`),
and one assertion updated in `tests/test_datasets_spider.py` (the mask test now expects the mask to get the
same display transform as its image). No SPIDER data or images are committed.

## 1. The bug was in the test helper, not in the volume reader

Reproduced on the SPIDER paired T1+T2 studies (190 pairs: 153 training, 37 validation).

- 73 of 447 volume files have a rotated direction matrix (`AnatomicalOrientation = ASL`, directions
  `P+, S-, L-` for index axes i, j, k). Their thickest axis is k (3.3 mm), so they are sliced along the L
  direction. The other 374 files are standard (`RAI`, identity, thickest axis i).
- Truth (checked, not assumed): the mid-slice of a rotated T2 (validation case 107) shows the lumbar column in
  profile, with vertebrae, discs, canal and sacrum. The canal runs along the S-I index axis (spread 91 vs 1.4
  voxels along the slice axis), and the header matrices agree with the `ASL` label. These are genuine sagittal
  acquisitions stored with a permuted index order, and the direction matrices are correct.
- The slice axis chosen by `read_volume` (largest spacing) is correct for all 447 files. The old rule and the new
  rule pick the same axis for every SPIDER file.
- `tests/_spider.py` (the paired helper) hard-coded the slice axis to index 0 and wrote `plane: sag` into the
  role. For rotated files index 0 runs along P, so the stack was coronal, while the hard-coded role hid it.
  `Study.find` filters on the geometric plane, so `best('sag')` returned None and the study became
  `no_sagittal`. Case 35 went through the same path and labelled 0 discs; the single-file path labels 3.
- SimpleITK and the built-in MetaImage reader agree on all 447 SPIDER volumes (data, spacing, origin, direction).

## 2. Fix

- `io/volumes.py`: `choose_slice_axis(spacing, direction)` returns a `SliceAxis` decision. If only the thickest
  axis is clearly thickest (`volume_isotropic_ratio` = 1.15), it is the slice axis even when oblique; the
  deviation from the nearest canonical axis is recorded. If voxels are near-isotropic, spacing does not decide:
  the candidate closest to a canonical axis wins, ties going to L, then P, then S, and the decision is flagged
  ambiguous. Both readers share this rule. `meta['volume']` records the axis, rule, spacing, deviation,
  alignment, ambiguity and obliquity. A non-3-D input raises `ValueError`.
- Why spacing stays decisive for oblique slabs: preferring the canonical in-plane axis when the thick axis is
  tilted would slice a 30-degree sagittal slab along an in-plane direction. Tests pin this behaviour.
- `series.py`: `Role` gains `obliquity_deg` and `plane_ambiguous`. A normal within `plane_ambiguity_deg` (15)
  of being equidistant between two canonical axes (sag/cor at about 38 to 52 degrees) is flagged, and the
  evidence says so. The plane label itself is still the nearest canonical plane (`plane_of`, unchanged).
- `datasets/spider.py`: `SpiderStudy.volume` returns display orientation, as `load_study` does for a file.
  `mask` gets the same display transform. `to_study(sequences)` builds the study with `from_stacks`, the same
  classification path as `load_study`.
- `tests/_spider.py`: `load_t2_t1` is built on `SpiderStudy.to_study()`. The hard-coded stack builder is gone.

## 3. Verification

- Parity over all 190 SPIDER pairs: each paired T2 is sagittal and identical to `load_study` on the same file
  (volume, geometry, key). Mask shapes match.
- Synthetic regression tests (`tests/test_io_volumes_orientation.py`, 330 parametrised cases): sagittal layouts, standard and
  SPIDER-rotated, rotated 0 to 90 degrees about each of x, y, z. Each case checks voxel geometry against
  SimpleITK's index-to-point transform, the slice-axis decision, obliquity, plane label and ambiguity through
  `load_study`, MetaImage reader agreement, and label-grid alignment. Plus near-isotropic tie-breaks,
  oblique-slab behaviour, `plane_margin_deg`, the non-volume error, and SPIDER dataset checks (case 35 and 107
  pairs, and case 35 disc counts on both paths, skipped when SPIDER or onnxruntime is missing).
- Full suite in the worktree: 1109 passed, 1 skipped, 43 xfailed, 1 xpassed (the `pd_only` XPASS also occurs on
  main). ruff, ruff format and mypy are clean.

## 4. Before and after on SPIDER validation

Paired path (`load_t2_t1`, the lab12 path; backend `auto`; 37 pairs; shared scorer):

| Metric | main (before) | branch (after) |
|---|---|---|
| `no_sagittal` verdicts | 7 | 0 |
| Verdicts (lumbar / thoracolumbar) | 21 / 9 | 24 / 13 |
| False-rejection rate (`eval_applicability`) | 0.189 | 0.000 |
| Recall | 0.796 | 0.996 |
| Precision | 0.990 | 0.992 |
| Label accuracy (matched) | 0.971 | 0.977 |
| Label accuracy (all truth discs) | 0.773 | 0.973 |
| Volumes with all truth discs labelled | 0.757 | 0.946 |
| Centre error median / p90 (mm) | 0.52 / 1.88 | 0.61 / 1.90 |
| Numbering unconfirmed (`eval_applicability`) | 15 | 9 |

The seven formerly rejected cases (35, 107, 129, 136, 152, 177, 234) are now labelled. Four of them come out
`thoracolumbar`, which caps thoracic discs above T10 as designed. Training split: 13 of 153 pairs had no
sagittal stack before, and 0 of 153 after.

Single-file path (`tools/eval_labeling.py --backend auto --split validation`, 87 volumes): unchanged. Recall
99.5%, precision 99.2%, label accuracy 98.1% (matched) and 97.6% (all truth), centre median 0.6 mm, p90 2.0 mm.
0 of 87 per-case rows changed. Single-file SPIDER volumes are not affected because their slice axis and
display orientation did not change.

Baseline validation numbers were collected before the change, but no parameter was chosen by looking at them.
The defaults (1.15 ratio, 20 degree alignment, 15 degree ambiguity) are judgement defaults in `config.py`, set
from the header survey and the geometry, not from the labelling metrics.

## 5. Limits and follow-ups

- SPIDER has no obliquely acquired volumes: all 447 are axis-aligned. The oblique and isotropic branches are
  covered only by synthetic tests.
- `plane_ambiguous` is recorded but not yet used by `anatomy/ood.py`. A stack flagged ambiguous could be
  reported as a caveat rather than trusted as sagittal. That is not in this branch.
- `AnatomicalOrientation` in the MetaImage header is ignored; it agreed with the direction matrices on all files.
- Training (`tools/train_labeller.py`) read volumes in raw orientation; inference uses display orientation. The
  ML grid canonicalises slices from geometry (`anatomy/ml/grid.py`), so the difference should not matter, but
  this was not tested separately.
- The paired-path numbers above use the shared scorer on the paired T2 labels. They are not the lab12 numbers,
  which were verdicts only.
