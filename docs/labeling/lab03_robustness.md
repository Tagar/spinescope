# Labelling robustness benchmark (lab03)

A synthetic benchmark that any vertebral labelling backend must pass. It runs in CI on every push, needs no
real data, and fails when a backend that used to get a case right gets it wrong. The cases come from
`spinescope.testing.robustness` (scenario catalogue, scoring) on top of the phantom in
`spinescope.testing.phantom` (analytic anatomy with exact ground truth).

Run it:

    .venv/bin/python -m pytest tests/robustness -q          # the suite (about 17 s)
    .venv/bin/python tools/robustness_report.py             # per-scenario, per-backend table
    .venv/bin/python tools/robustness_report.py --json out.json

## What is tested

Each scenario renders a sagittal protocol from the phantom, runs `label_spine(study, backend=...)` and scores
the discs with the shared scorer in `spinescope.anatomy._validate` (the same one used on SPIDER). The
field of view is respected: a truth disc counts only if its projection onto the sagittal plane the labeller
uses lies inside the stack.

Two checks per scenario, in `tests/robustness/test_robustness.py`:

* `test_labelled_correctly`: a **pass** needs every in-view disc found (recall 1), no spurious disc
  (precision 1), every name right (label accuracy 1) and a p90 centre error of at most 3 mm.
* `test_no_silent_error`: a wrong result must not come with numbering confidence at or above 0.5. The
  labeller may be wrong, but it must say so.

Each scenario ends in one of three outcomes (`classify` in `robustness.py`):

| outcome | meaning |
|---|---|
| pass | all of the above hold |
| unsure | not a pass, and the labeller reports numbering confidence below 0.5 (the acceptable failure) |
| silent_error | not a pass, and confidence is 0.5 or higher (the failure that matters most) |

Scenarios the current backend fails are `xfail(strict=False)`, with the failure observed on the phantom as the
reason. When a backend fixes one, its test shows XPASS and the suite stays green. A reason only
describes what happens now. Remove the entry when the case is fixed.

## Ground-truth conventions

* Disc names count up from the lowest lumbar vertebra, as in SPIDER: the lowest disc is L5-S1, or L6-S1 when
  S1 is lumbarised. Label accuracy is measured against that convention, so a sacrum-relative answer for a
  six-lumbar spine counts as wrong, which is the failure we want to see. The anatomical name (S1-S2) is kept in
  `phantom.truth["discs"]["L6-S1"]["anatomical"]`.
* Sacralised L5: the L5-S1 disc is marked `partial`. It keeps its name and the bone bridge is on one side.
* Truth is exact (it comes from the phantom), so a centre error is a labeller error, not a disagreement about
  where the disc is.

## Scenarios

Group: `protocol` = which sequences and slices; `fov` = field of view; `transitional` = lumbar count and
sacral anatomy; `pathology` = disc and bone changes; `anatomy` = alignment; `hardware` = implants;
`artefact` = image artefacts.

| scenario | group | what it is | why it matters |
|---|---|---|---|
| baseline | protocol | standard 3-contrast sagittal set (T2, STIR, T1), 13 slices at 5 mm, default anatomy | reference point; should always pass |
| t2_only | protocol | sagittal T2 only | many sites acquire one sequence |
| t1_only | protocol | sagittal T1 only | T1 endplates are weaker |
| stir_only | protocol | sagittal STIR only | the STIR-only study is common in trauma |
| pd_only | protocol | sagittal proton density only | PD is the sequence with the least canal contrast |
| iso_1mm_3d | protocol | 3D isotropic 1 mm sagittal volume, 9 slices, 160 x 112 matrix | 3D sequences give a dense thin-slice volume with few slices in the midline |
| thick_6mm | protocol | 6 mm slices, 9 slices | thick slices blur endplates |
| few_slices_7 | protocol | 7 sagittal slices at 5 mm | a short acquisition; little for the midline search |
| field_3T | protocol | 3 T intensity and noise variant, T2 and T1 | the same anatomy at a different field strength |
| fov_T10_sacrum | fov | field of view from T10 to the sacrum, thoracic ribs, the T9-T10 disc in view | thoracic levels at the edge of the lumbar view |
| fov_L2_S1 | fov | field of view from L2 to S1 only | upper discs out of view; the labeller must not invent discs |
| fov_L1_coccyx | fov | field of view from L1 with 40 mm of air below the sacrum (the phantom has no coccyx) | extra field of view below the sacrum must not confuse numbering |
| lumbarised_S1 | transitional | six lumbar vertebrae, full S1-S2 disc | the count from the sacrum is one level low |
| sacralised_L5 | transitional | partial L5-S1 disc with a bone bridge on one side | the lowest disc is partial and can be missed |
| dark_discs | pathology | degenerated discs with signal loss (dark nucleus) at L2-L3 and L4-L5 | dark discs weaken the endplate lines |
| collapsed_discs | pathology | discs 50% and 60% collapsed at L3-L4 and L4-L5 | thin discs are rejected as not disc-like |
| compression_fracture_L2 | pathology | anterior wedge of L2 with STIR and T2 oedema | a wedged body changes the body-to-disc geometry |
| schmorl_nodes | pathology | Schmorl nodes at the L3 superior and L4 inferior endplates | a bright or dark node in the endplate can read as a disc edge |
| scoliosis | anatomy | 14 mm lateral apex at L2 with 12 degrees of axial rotation | the midline sagittal plane cuts the levels off centre |
| hardware_cage_screws | hardware | L4-L5 cage, L4 and L5 pedicle screws, posterior rod (signal voids with blooming) | implant voids sit in the disc and body the chain follows |
| bias_field | artefact | surface-coil intensity bias, strong gradient from posterior to anterior | endplate contrast changes across the slice |
| motion_ghosting | artefact | ghost copies along the phase axis | ghosts look like extra endplates |
| aliasing_wrap | artefact | 60 rows of anatomy fold back onto the opposite edge | wrapped anatomy can sit where the sacrum should be |

## Phantom options

All added to `PhantomSpec` or `SpinePhantom.render` with defaults that leave the existing output unchanged
(checked byte for byte: the renders and truth of the original phantom are identical before and after).

| option | effect |
|---|---|
| `scoliosis_mm`, `scoliosis_rot_deg` | C-shaped thoracolumbar curve, apex L2, zero at the sacrum |
| `thoracic_top` | top thoracic body (11 default; 10 or 9 adds levels) |
| `transitional` | `lumbarised_s1` (six lumbar, S1-S2 disc) or `sacralised_l5` (partial L5-S1 disc) |
| `collapsed` | fraction of disc height lost, per disc |
| `fracture` | anterior wedge fraction per body; the body gets STIR/T2 oedema |
| `schmorl` | Schmorl node at a body's superior or inferior endplate |
| `hardware` | `rod`, `screws:<body>` (signal void with blooming), `cage:<disc>` |
| `ribs` | rib tubes off the thoracic bodies (visible off the midline) |
| `bias_field` | multiplicative surface-coil bias |
| `field_strength` | 1.5 (default) or 3.0: intensity and noise variant; written to DICOM |
| `render(ghost=)` | motion ghosting, copies along the row (phase) axis |
| `render(wrap_rows=)` | phase wrap-around: extra rows fold onto the opposite edge |
| contrast `pd` | proton density (TR 3000, TE 15); classified as `pd` by the series classifier |

The phantom is a cartoon. Intensities and the artefact models are chosen to exercise the labeller, not to
simulate a scanner. Results on these scenarios say nothing about accuracy on real scans; that is what the
SPIDER and Sudirman validations in `docs/validation.md` are for.

## Current status (classical backend)

Generated by `tools/robustness_report.py` on the phantom. Columns: recall, precision and label accuracy are
over the in-view truth discs; centre p90 is over the matched discs; conf is the labeller's numbering
confidence.

| scenario | group | backend | status | recall | precision | label acc | centre p90 (mm) | conf |
|---|---|---|---|---|---|---|---|---|
| baseline | protocol | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 1.89 | 0.70 |
| t2_only | protocol | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 1.89 | 0.90 |
| t1_only | protocol | classical | FAIL (silent) | 0.83 | 1.00 | 0.83 | 5.24 | 0.90 |
| stir_only | protocol | classical | FAIL (silent) | 1.00 | 1.00 | 1.00 | 3.61 | 0.90 |
| pd_only | protocol | classical | FAIL (silent) | 0.00 | 0.00 | 0.00 | n/a | 0.50 |
| iso_1mm_3d | protocol | classical | FAIL (silent) | 0.60 | 1.00 | 0.20 | 1.24 | 0.90 |
| thick_6mm | protocol | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 1.13 | 0.90 |
| few_slices_7 | protocol | classical | PASS | 1.00 | 1.00 | 1.00 | 2.20 | 0.90 |
| field_3T | protocol | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 0.39 | 0.70 |
| fov_T10_sacrum | fov | classical | FAIL (silent) | 0.67 | 0.67 | 0.67 | 2.76 | 0.90 |
| fov_L2_S1 | fov | classical | FAIL (silent) | 1.00 | 1.00 | 1.00 | 3.09 | 0.90 |
| fov_L1_coccyx | fov | classical | PASS | 1.00 | 1.00 | 1.00 | 1.99 | 0.90 |
| lumbarised_S1 | transitional | classical | FAIL (silent) | 0.71 | 1.00 | 0.14 | 0.59 | 0.90 |
| sacralised_L5 | transitional | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 1.89 | 0.90 |
| dark_discs | pathology | classical | FAIL (silent) | 1.00 | 1.00 | 1.00 | 7.89 | 1.00 |
| collapsed_discs | pathology | classical | FAIL (silent) | 0.50 | 1.00 | 0.00 | 3.37 | 0.70 |
| compression_fracture_L2 | pathology | classical | PASS | 1.00 | 1.00 | 1.00 | 0.38 | 1.00 |
| schmorl_nodes | pathology | classical | FAIL (silent) | 0.67 | 1.00 | 0.17 | 0.41 | 0.70 |
| scoliosis | anatomy | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 1.71 | 0.70 |
| hardware_cage_screws | hardware | classical | FAIL (silent) | 0.83 | 0.83 | 0.50 | 8.00 | 0.70 |
| bias_field | artefact | classical | FAIL (silent) | 0.83 | 1.00 | 0.50 | 2.13 | 0.70 |
| motion_ghosting | artefact | classical | FAIL (silent) | 0.83 | 0.83 | 0.83 | 1.41 | 0.90 |
| aliasing_wrap | artefact | classical | FAIL (silent) | 0.83 | 0.83 | 0.67 | 2.89 | 0.90 |

3 of 23 scenario-backend pairs pass.

Reasons for each failure (the xfail text in `tests/robustness/test_robustness.py`):

* `baseline`: L2-L3 is missed: the chain jumps to L1-L2 and the labels above shift by one level
* `t2_only`: same L2-L3 miss as the baseline; the labels above it shift by one level
* `t1_only`: L2-L3 is found 7 mm from its true centre (weaker T1 endplates); the other discs are fine
* `stir_only`: all six discs are named correctly but the p90 centre error is 3.6 mm, above 3 mm
* `pd_only`: no canal on PD (CSF is not bright): the chain sits 50-70 mm from the true discs, none matched
* `iso_1mm_3d`: 9 slices of 1 mm: 3 of 5 discs are found, L2-L3 is skipped and the labels above shift
* `thick_6mm`: same L2-L3 miss at 6 mm slices; the labels above it shift by one level
* `field_3T`: same L2-L3 miss at 3 T; the labels above it shift by one level
* `fov_T10_sacrum`: the lumbar discs are right, but the T9-T10, T10-T11 and T11-T12 discs are 12-14 mm from their true centres and fall outside the match radius
* `fov_L2_S1`: all four discs are named correctly but L4-L5 is 3.3 mm off, so p90 is 3.1 mm (limit 3 mm)
* `lumbarised_S1`: the S1-S2 disc under the lumbarised S1 is not found, so L5-L6 is taken for L5-S1 and every name is one level off (1 of 7 correct)
* `sacralised_L5`: same L2-L3 miss as the baseline; the partial L5-S1 disc itself is labelled correctly
* `dark_discs`: the two dark discs (L2-L3, L4-L5) are centred 7-9 mm from their true positions
* `collapsed_discs`: collapsed discs (4-5 mm high): 3 of 6 discs are found and the labels above them shift
* `schmorl_nodes`: Schmorl nodes at L3 and L4 break the chain: L4-L5 is not found and the labels above shift
* `scoliosis`: the L2-L3 miss again, at the apex of the 14 mm curve; the labels above it shift
* `hardware_cage_screws`: the L4-L5 cage void breaks the chain: L3-L4 and L4-L5 sit 11-12 mm from their true positions and the labels above shift
* `bias_field`: same L2-L3 miss as the baseline; the bias field does not change the chain
* `motion_ghosting`: ghost copies move the sacral end of the chain: L5-S1 is 26 mm from its true position
* `aliasing_wrap`: wrapped rows move the sacral end: L5-S1 is 34 mm off and L4-L5 is labelled at the L5-S1 position

## Reading the status

* 3 of 23 scenario and backend pairs pass. Every failure is a silent error: the labeller reports numbering
  confidence between 0.5 and 1.0 when it gets the labels wrong. No scenario ends as "unsure". The classical
  confidence is therefore not calibrated on these cases. That is the most useful result of this benchmark.
* One failure dominates the protocol and artefact rows: the disc at L2-L3 is missed by the endplate chain on
  the default anatomy. The labels above it shift by one level, and the numbering confidence stays at 0.7.
  The same miss is behind the existing xfail in `tests/test_labeling.py`.
* `pd_only` is the worst case: no canal is found on PD, and every chain disc is about 50 to 72 mm from its true
  position, so nothing matches.

## Not covered yet

* No learned backend: `label_spine` accepts only `classical` (`BACKENDS` in `anatomy/pipeline.py`). The report
  picks up any backend added to `BACKENDS` automatically; an `ml` backend is not on main yet.
* The phantom has no coccyx. Thoracic levels go no higher than T9 (`thoracic_top=9`, the highest used here).
* The scenarios are single-study. Multi-study and mixed-series cases belong in the dataset tests.
