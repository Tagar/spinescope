# lab02: Sudirman lumbar DICOM benchmark for `label_spine`

Branch `lab/lab02_sudirman`, labeller = `main` (classical `label_spine`, no tuning on this dataset).
Dataset: Sudirman et al., Lumbar Spine MRI Dataset (Mendeley Data, doi:10.17632/k57fr854j2.2, CC BY 4.0),
with its label companion set (doi:10.17632/zbf6b4pttk.2) and radiologist notes (doi:10.17632/s6bgczr8s2.2),
both CC BY 4.0. Unlike SPIDER, this is clinical DICOM (axial and sagittal series exported from a
scanner), so it tests the loader and classifier as well as the labeller.

## Headline

- Scored all 515 labelled patients (1,545 annotated discs; no subsampling, 0 failed patients). The run
  takes about 50 s on 6 processes.
- `label_spine` finds an annotated disc within 12 mm (in the sagittal y, z plane) for **36.6 %** of
  levels, and the nearest disc carries the right name for **34.5 %** of those (**26.0 %** if the distance
  is ignored). The found discs are placed well: median centre error 4.7 mm, p90 9.7 mm, median normal
  error 2.6 deg.
- The dominant failure is naming, not position. Among levels whose nearest disc has the wrong name, the
  name is one level too high in 694 cases and two levels too high in 339. Discs are found near the true
  z (median dz 0.6 mm when matched), but they are named as if the lowest detected disc were L5-S1.
  The lumbosacral numbering step is the cause to fix first.
- These numbers are far from the targets in `docs/validation.md`. They are a baseline for the labelling
  wave, not a result to ship.

## Data and ground truth

Folder layout (`$SPINESCOPE_DATA`, default `~/data/public`):

- `lumbar_spine_mri_sudirman/01_MRI_Data/<patient>/<session>/<series>/*.ima`: 515 patients, 558 sessions,
  3,761 series folders, 48,345 DICOM files. 477 patients have one session, 33 have two and 5 have three.
- `lumbar_spine_mri_sudirman_labels/` (new registry entry `lumbar_spine_mri_sudirman_labels`, 1.68 GB in
  two zips):
  - `05_Final_Ground_Truth_Data/Label_Images/L1_<patient>_<D>.png`: 320 x 320 label maps, one per patient
    and disc `D3`, `D4`, `D5`. Values 50 = intervertebral disc, 100 = posterior elements, 150 = thecal sac,
    200 = area between anterior and posterior elements. Checked by overlay on the T1 slice; 50 is the disc.
  - `04_Intermediary_Ground_Truth_Data/T1_Output/T1_<patient>_<D>.png`: the axial T1 slice the map was drawn on.
- `lumbar_spine_mri_sudirman_notes/raw/Radiologists Report.xlsx` (new registry entry
  `lumbar_spine_mri_sudirman_notes`): free text per patient, 575 rows.

How the ground truth is built (`spinescope.datasets.sudirman`):

1. Each label map is paired with its reference slice (`T1_Output`). The reference equals one DICOM axial
   T1 slice pixel for pixel: normalised cross-correlation is at least 0.99997 for all 1,545 levels (515 of
   515 patients, 3 levels each), so no resampling or registration is needed. The search covers every
   unenhanced axial T1 stack (`ax_t1`, `ax_t1_2`, and their slabs). The acceptance threshold is 0.98.
2. The disc centre is the centroid of the IVD label on the matched slice, mapped to LPS with the slice
   geometry (`Stack.pix_to_lps`). The normal is the axial slab normal (the disc axis), pointing superior.
   The z resolution is the slice position, so about 2 mm.
3. Level names. The dataset calls D3, D4 and D5 "the last three IVDs". The matched slabs are the three
   lowest axial slabs in every session (D5 is the lowest one in all 515 patients, for 2 to 12 slabs per
   session). Mapping: D3 = L3-L4, D4 = L4-L5, D5 = L5-S1. This is an assumption, supported by two checks:
   - The radiologist notes name L4-5 in 316 patients, L5-S1 in 204, L3-4 in 100, and L1-2 or L2-3 in
     one each. The notes therefore cover the same three levels.
   - Sagittal check on one patient (red = truth, green = labeller): the lowest truth disc sits directly
     above the sacrum, and the three truth discs sit on the three discs visible in that slice.
4. Session selection. Patients with several sessions have the labelled slices in one session only. The
   session with the most matched discs (then the best NCC) is used, so folder order is never relied on.

Limitations of this ground truth: it is a 2D disc position per level, not a segmentation of the disc
plane. Only three levels are annotated, so the shared scorer's precision (0.16) is structurally low.
The level names rest on the assumption above. There are no sagittal labels.

## Scanner, vendor and field strength

All 558 sessions were acquired on one scanner: **Siemens MAGNETOM Essenza, 1.5 T** (DICOM Manufacturer
SIEMENS, ManufacturerModelName MAGNETOM_ESSENZA, MagneticFieldStrength 1.5). Main series, as classified:

| Series | Sessions with it | Parameters (typical) |
|---|---|---|
| Sagittal T1 TSE (`T1_TSE_SAG_320`) | 523 | TR 648 ms, TE 9.2 ms |
| Sagittal T2 TSE (`T2_TSE_SAG_384`) | 556 | TR 3500 to 4500 ms, TE 94 ms |
| Axial T1 TSE (`T1_TSE_TRA`) | 518 | TR 715 ms, TE 11 ms, 4 mm, 320 x 320, 0.688 mm |
| Axial T2 TSE (`T2_TSE_TRA_384`) | 534 | TR 4000 ms, TE 88 ms |

Because there is one vendor and one field strength, the vendor and field breakdown has one row. The
sequence breakdown is by axial slab count below. It is an acquisition difference, not an anatomical one.
The dataset has no transitional-anatomy annotation, so that split cannot be made.

## Series classification (`load_study`)

Survey of all 558 sessions (8,581 stacks), comparing each stack's role with its DICOM series description.

| Finding | Count | Action |
|---|---|---|
| Siemens "PosDisp" position-display copies (1 to 3 images, with the acquisition's TR/TE) were keyed as `sag_t2`, `cor_t1`, `ax_t2`, or `sag_t2_2` and so on | 1,583 stacks (18 %) | **Fixed** in `series.py`: "POSDISP" marks a localizer. All 1,583 are now `loc_*`. |
| The disc metrics are unchanged by that fix (the labeller uses the main `sag_t1` / `sag_t2` keys, which POSDISP never held) | | Checked: identical recall, label accuracy, and centre error with and without the fix |
| Series named `T2_TSE_SAG` but with coronal geometry (normal along y) | 7 stacks in 4 sessions | Not fixed: classification follows geometry, which is right. Those sessions have no sagittal T2 key. |
| `5002 New Series` single sagittal images with T2 parameters, keyed `sag_t2_2` / `sag_t2_3` | 16 stacks in 13 sessions | Not fixed. Their nature is unclear (likely a secondary capture). Not used by the labeller. |
| Two unenhanced axial T1 series in one session (`ax_t1` and `ax_t1_2`); the labelled slices sit in the second one in 5 sessions | 5 sessions | Not fixed. Downstream code that takes `ax_t1` may pick the wrong series. |
| Axial series split into 2 to 12 slabs by tilt (slab count per session) | all sessions | Expected. The loader keys slabs by z order. |

## Results (`label_spine` on all 515 patients, 1,545 annotated discs)

Level-wise scoring (`spinescope.anatomy._validate_sudirman.score_levels`). "Matched" means the nearest
predicted disc lies within 12 mm in the sagittal (y, z) plane. Centre error is that in-plane distance.

| Metric | Value |
|---|---|
| Level recall (matched within 12 mm) | 0.366 |
| Label accuracy among matched levels | 0.345 |
| Label accuracy, distance ignored | 0.260 |
| Centre error, matched: median / p90 | 4.66 mm / 9.74 mm |
| Centre error, matched: |dz| median / p90 | 1.42 mm / 5.18 mm |
| Centre error, matched: |dy| median | 3.70 mm |
| Normal error, matched: median | 2.58 deg |

By annotated level:

| Level | n | Recall | Label accuracy (matched) | Centre error median |
|---|---|---|---|---|
| L3-L4 | 515 | 0.400 | 0.374 | 3.6 mm |
| L4-L5 | 515 | 0.390 | 0.383 | 5.0 mm |
| L5-S1 | 515 | 0.309 | 0.258 | 6.2 mm |

By number of axial slabs in the session (acquisition layout, not anatomy):

| Axial slabs | Levels | Recall | Label accuracy (matched) |
|---|---|---|---|
| 3 | 834 | 0.400 | 0.353 (118 of 334) |
| 4 | 417 | 0.331 | 0.319 (44 of 138) |
| 5 | 192 | 0.307 | 0.458 (27 of 59) |
| 6 to 12 | 102 | 0.343 | 0.171 (6 of 35) |

The 6-to-12 group mixes several layouts with few sessions each, so its rates are unstable.

Shared scorer (`_validate.score_anchors` and `aggregate`, for comparison with SPIDER; greedy matching,
12 mm, all predicted discs counted):

| Metric | Value |
|---|---|
| Disc recall | 0.361 |
| Disc precision | 0.162 (structurally low: about 8 predicted discs per study, 3 annotated) |
| Label accuracy (matched) | 0.346 |
| Label accuracy over all truth discs | 0.125 |
| Centre error median / p90 | 4.66 mm / 9.81 mm |
| Centre within 3 mm | 0.290 |
| Normal error median | 2.64 deg |

Name errors among the 1,545 levels, by how many levels the predicted name sits above the truth:

| Predicted minus truth (levels) | 0 (correct) | +1 (one below) | +2 or more | -1 (one above) | -2 (two above) | -3 or more |
|---|---|---|---|---|---|---|
| Count | 401 | 66 | 10 | 694 | 339 | 35 |

Unmatched levels (979 of them) have their nearest predicted disc a median 42 mm away (p25 27, p75 55 mm),
which is about one disc spacing. The labeller's disc chain is shifted by a level, not unstable.

## Reading the failure

- Where the labeller finds a disc at the right height, the position is good: matched dz median 0.6 mm
  and matched normal error 2.6 deg.
- The name is one or two levels too high in most failures. Each annotated level has a predicted disc at
  its height, but the labeller names the lowest disc it finds as L5-S1 (or the next one as L4-L5), so the
  whole chain moves up. The numbering rule ("count up from the lowest disc, L5-S1 when a body is visible
  below it") is the likely cause. Its evidence is a visible body below the lowest disc, which the sagittal
  field of view often cuts short here.
- This matches the SPIDER baseline (recall 0.20, label accuracy 0.06 in `docs/validation.md`): the disc
  chain is weaker on this data than the position check suggests, and naming fails most often.

## Reproduce

```
.venv/bin/python -c "
from spinescope.anatomy._validate_sudirman import evaluate
ev = evaluate(workers=6)          # all labelled patients; pass patients=[...] for a subset
print(ev.summary())
"
```

The dataset-marked test `tests/test_labeling_sudirman.py` runs three patients when the data is on disk and is
skipped otherwise. Unit tests for the scorer, the NCC matching, and the notes parser need no data
(`tests/test_labeling_sudirman_units.py`).

## Not done

- No changes to the labeller itself (the task was to measure `main`). The naming failure is the first
  thing to test in the labelling wave.
- Disc-plane segmentation is not in the ground truth; the centre is a slice-level IVD centroid (about
  2 mm z quantisation).
- The level names rest on the D3/D4/D5 convention, checked against the notes and the sacrum on one patient,
  not confirmed against a source that names the discs.
