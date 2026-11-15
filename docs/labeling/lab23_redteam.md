# lab23 red team: `load_study` + `label_spine` (default backend)

Nasty but realistic inputs built from the phantom (`spinescope.testing.phantom`) and perturbed with pydicom or
numpy. Each row is one attack. "Before" is main at 9484831, "after" is this branch. Regression tests are in
`tests/test_labeling_redteam.py` (50 tests, 160-192 px matrices, one whole-spine case marked `slow`).

Expected outcome is one of: an explicit error that names the cause, a warning, an abstention (no level names), or
the same labels as the clean reference (invariance). "Same" means the same disc names with centres within 2 mm.

## Caveat: the phantom as reference

The learned labeller is out of distribution on the cartoon phantom. For the clean phantom it places the discs
correctly (within 1 to 2 mm of truth) but puts an extra disc-like response below the true L5-S1, so every level
name moves down one. The numbering stays `unconfirmed` (P about 0.28), which is the right outcome. Because of this,
the attacks are checked for invariance against the clean phantom run through the same pipeline, not against the
phantom truth. The naming also depends on pixel size (1.2 mm and 0.6 mm give different bottom discs), so
comparisons across pixel sizes use positions, not names.

## Attack table

| # | Attack | Expected | Before (main) | After (this branch) | Fix or test |
|---|---|---|---|---|---|
| A00 | clean phantom, 192 px, 1.2 mm, 13 slices | reference | 6 discs T12-L1..L5-S1, numbering unconfirmed | same | reference fixture |
| A01 | no PixelSpacing on any slice | explicit error naming the cause | generic "no DICOM images with plane geometry" | same type; message names `missing_pixel_spacing 13` | `io/dicom.py` reasons; `test_missing_pixel_spacing_is_an_explicit_error` |
| A02 | no SliceThickness | works | same labels | same labels | `test_missing_slice_thickness_does_not_matter` |
| A03 | ImagePositionPatient missing on 4 of 13 slices | warning, no silent truncation | 4 slices dropped silently; only a "not uniform" warning; 0 discs | warning "4 image(s) ignored for missing or invalid ImagePositionPatient..."; 0 discs (abstain) | `io/dicom.py` per-frame reasons and `_geometry_warnings`; `test_some_slices_without_position_are_reported` |
| A04 | ImagePositionPatient missing on all slices | explicit error | generic message | message names `missing_position 13` | same as A01 |
| A05 | duplicate InstanceNumber on every slice | same labels | same labels | same labels | `test_duplicate_instance_numbers_do_not_change_labels` |
| A06 | one slice with IOP rotated 5 deg in plane | warning naming the slice, not confident | 0 discs; warning "slice spacing is not uniform" (misleading) | warning "1 slice(s) have an orientation 5.0 deg away from the other 12 (inconsistent ImageOrientationPatient?)"; 0 discs | `_geometry_warnings`; `test_one_slice_with_a_rotated_iop_is_warned_and_not_confident` |
| A07 | one slice with IOP negated (reversed) | warning, not a "0 deg" message | 0 discs; same misleading warning | "180.0 deg away" (signed angle) | `test_one_slice_with_a_negated_iop_is_warned` |
| A08 | IOP vectors scaled to length 1.3 | explicit exclusion (corrupt header) | labelled correctly by accident (silent normalisation) | excluded: error "bad_orientation 13" | policy change, see note 1; `test_scaled_iop_is_excluded_not_normalised` |
| A09 | IOP skewed (column axis 14 deg off orthogonal) | exclusion, not a wrong normal | 0 discs, no reason | excluded: error "bad_orientation 13" | `_geometry_problem`; `test_skewed_iop_is_excluded` |
| A10 | IOP all zeros | explicit error | uncaught `ValueError: zero-length vector` | error "bad_orientation 13" | `_geometry_problem`; `test_zero_iop_is_an_explicit_error_not_a_crash` |
| A11 | NaN voxels (3200) on two slices | warning; no NaN in output | same labels, no warning | warning "3200 non-finite pixel value(s)..."; same labels | `load_stack_pixels` replaces NaN/inf; `study._pixel_qa`; `test_nan_pixels_are_replaced_and_warned` |
| A12 | inf voxels on one slice | warning; no inf in output | same labels, no warning | warning "1921 non-finite..."; same labels | same; `test_inf_pixels_are_replaced_and_warned` |
| A13 | 4 inferior slices all zero | warning; labels not invented | 0 discs, no warning | warning "4 of 13 slice(s) are constant"; 0 discs | `study._pixel_qa`; `test_constant_slices_are_warned` |
| A14 | every voxel zero | abstain with reason | 0 discs, applicability reason | plus constant-slice warning | `test_all_zero_volume_abstains` |
| A15 | three slices saturated at 4095 | warning | 5 discs (T12-L1 lost), no warning | 5 discs, warning "3 of 13 slice(s) are constant (... saturated)" | `study._pixel_qa`; `test_saturated_constant_slices_are_warned` |
| A16 | signed int16, background at -2048 | same labels as unsigned | **0 discs, "Not a lumbar spine study"** (silent wrong verdict) | same 6 discs as the clean reference | **bug fixed**: window collapsed; `grid.intensity_offset` and labeller shift; `test_signed_int16_with_negative_background_matches_unsigned`, `test_intensity_offset_is_zero_for_unsigned_and_negative_for_signed` |
| A17 | 8-bit pixels (rescaled to 0..255) | same labels within 3 mm | 6 discs, confidence 0.18 | same | `test_uint8_pixels_match_the_clean_reference` |
| A18 | secondary capture (RGB screenshot, SC SOP class) | ignored as non-MR | generic "no DICOM images" error | error names `secondary_capture 13` | `test_secondary_capture_is_not_anatomy` |
| A18b | MR SOP class with RGB pixels | ignored | generic error | error names `colour_or_screenshot 13` | `test_colour_mr_is_not_anatomy` |
| A19 | 1024 x 1024 matrix (0.3 mm) | labelled, time within budget | 8 discs (2 extra at T10-T11, T11-T12 in the larger FOV), about 4 s | same (about 4 s) | test uses a 256 px proxy: `test_large_matrix_proxy_labels_consistently` |
| A20 | tiny 64 x 64 matrix | no crash; correct or abstain | 0 discs, reason given | same | `test_tiny_matrix_abstains_without_crashing` |
| A21 | single sagittal slice | abstain | 0 discs | same | `test_single_slice_abstains` |
| A22 | two sagittal slices | abstain | 0 discs | same | `test_two_slices_abstain` |
| A23 | 200 slices at 1 mm | no crash; correct or abstain | 4 discs (field of view covers the lower spine), 25 s | same labels, 12 to 18 s | `test_many_slices_do_not_crash` (60-slice proxy) |
| A24 | two studies in one folder (two patients) | warning; no other patient in output | 6 discs from the largest study; "holds 2 studies" warning; equal counts not flagged | plus "several studies tie on image count, so the choice is arbitrary"; no PHI from the other patient in the output | `io/dicom.py` tie warning; `test_mixed_patients_warn_and_do_not_leak_the_other_patient` |
| A25 | two sagittal T2 series (rescan), different FOV | warning; not confident | 6 discs, numbering 0.08; labeller warns | same; asserted not confident | `test_two_sagittal_rescans_with_different_fov_are_not_confident`. No load-level warning yet (limitation) |
| A26 | T1 pixels declared as T2 (TE/TR say T2) | warning; lower confidence | 7 discs, numbering **0.52** (higher than the correct 0.28) | 7 discs, numbering 0.31, warning "intensities contradict the declared T2 contrast (canal is darker than bone)" | `ood.contrast_polarity` cue; `test_t1_labelled_as_t2_is_flagged_by_the_pixels` |
| A27 | STIR description on a T2 image | physics wins; same labels | same labels | same labels | `test_stir_description_on_t2_pixels_does_not_change_labels` |
| A28 | localizer description on a sagittal image | abstain, no sagittal | 0 discs, "no sagittal series" | same | `test_localizer_description_leaves_no_sagittal` |
| A29 | left/right flipped IOP (row direction negated) | abstain, or correct | 0 discs | 0 discs; reason still reads "Not a lumbar spine study" (misleading, see limitations) | `test_lr_flipped_iop_abstains` |
| A29b | pixels mirrored, IOP unchanged | abstain, or correct | 0 discs | same (same misleading reason) | `test_mirrored_pixels_with_unchanged_iop_abstain` |
| A30 | 35 deg oblique sagittal | abstain or unsure | 0 discs | same (same misleading reason) | `test_extreme_obliquity_35_deg_abstains` |
| A30b | 45 deg oblique sagittal | abstain (plane flagged) | 0 discs | 0 discs (abstain) | `test_extreme_obliquity_abstains[45.0]` (parametrised with A30) |
| A31 | anisotropic pixels (0.72 x 2.16 mm) | correct positions in mm | 5 discs | 5 discs; every disc within 4 mm of an isotropic-reference disc (name-agnostic) | `test_anisotropic_pixels_are_labelled_at_their_true_size` |
| A32 | interleaved InstanceNumber order | same labels | same labels | same labels | `test_interleaved_instance_order_does_not_change_labels` |
| A33 | three consecutive slices missing | gap warning, abstain or unsure | 0 discs; "not uniform" and gap warnings | same warnings; 0 discs | `test_gap_in_slices_is_warned` |
| A34 | scout (3-plane localizer) only | abstain with a clear reason | 0 discs, "no sagittal series" | same | `test_scout_only_study_abstains_with_a_reason` |
| A35 | non-spine (noise) | abstain | 0 discs | same | `test_non_spine_noise_abstains` |
| A36 | spine outside the field of view | abstain | 0 discs | same | `test_spine_out_of_field_of_view_abstains` |
| A37 | whole spine, 1150 mm field of view, 384 px at 3 mm, 60 slices | no crash; correct or abstain | **IndexError** in `numbering.disc_names` after 44 s | no crash; 0 discs (abstain); 50 to 66 s (slow) | `numbering.disc_names` uses the `V<n>` fallback; `test_whole_spine_field_of_view_does_not_crash` (marked `slow`) |
| A38 | cervical-only field of view | abstain | 0 discs | same | `test_cervical_only_field_of_view_abstains` |
| A39 | (not an attack: duplicate of A00) | | | | removed |

Other checks made on every case: no patient name, ID, accession or institution string, no other patient's
tokens, and no input path appears in the serialised anchors (`label_spine(...).to_dict()`).

## Fixes and where they are

1. **Signed pixel data (A16).** The loader keeps signed values as stored (the existing signed-reading contract,
   `test_signed_pixels_are_read_as_signed`) and records the signed background in `stack.meta['signed_offset']`
   (`io/dicom.py`, `_signed_offset`). The learned labeller and the polarity cue read their network input through
   `grid.network_input_volume`, which removes the offset only for signed DICOM. The window assumes a background
   near zero, so without the shift it collapses onto the tissue. Unsigned DICOM and `.mha` or NIfTI volumes
   (SPIDER has a background of -1000) take the unchanged path, so the shipped model's inputs are unchanged.
   Limitation: the classical labeller, conus and vertebra-geometry code still read raw `stack.vol` for signed data.
2. **Whole-spine crash (A37).** `numbering.disc_names` indexed the name table without the `V<n>` fallback that
   `vertebra_name` already has.
3. **Geometry drops and reasons (A01, A03, A04, A08 to A10, A18, A18b).** `io/dicom.py` records each excluded
   frame with a reason (`missing_position`, `missing_orientation`, `missing_pixel_spacing`, `bad_position`,
   `bad_orientation`, `bad_pixel_spacing`) instead of skipping silently. The "no images" error names the counts. A
   frame whose IOP is not unit-length or orthogonal beyond `Config.iop_orthonormal_tol` (0.01, about 0.6 deg) is
   excluded.
4. **Orientation inconsistency (A06, A07).** A series with one or two slices whose orientation differs from a run
   of at least four others is warned, with the angle; those slices stay a separate stack.
5. **Pixel QA (A11 to A15).** `study._pixel_qa` warns on constant slices (every voxel equal: empty, zero-filled or
   saturated) and on clipped slices (`Config.saturated_slice_frac` 0.25). Non-finite voxels are replaced with the
   nearest finite extreme and counted.
6. **Contrast contradiction (A26).** `ood.contrast_polarity` = mean window-normalised intensity of the network's
   canal pixels minus that of its body pixels. The declared T2 contrast with polarity below
   `-Config.ood_contrast_polarity_min` (0.05), or T1 above it, adds the same off-distribution factor as the
   entropy cue, and a warning. Calibration (final code): 10 SPIDER training T1 volumes gave -0.16 to -0.40, and 10 training T2
   gave +0.21 to +0.50 (sample seed 0; no tuning on validation). Phantom T2 +0.23, phantom T1 -0.41.
7. **Ambiguous study choice (A24).** Two studies with the same image count now also warn that the choice is
   arbitrary.

## Edits outside the allowed list

- `src/spinescope/anatomy/numbering.py`: `disc_names` now uses `vertebra_name` for both ends (one-line root-cause
  fix for A37). This file was not on the list; it is outside the files being restructured (pipeline, calibration,
  fusion), and the change is only the fallback that the file already had. Revert if the numbering owner prefers.
- `src/spinescope/config.py`: new knobs `iop_orthonormal_tol`, `saturated_slice_frac`,
  `ood_contrast_polarity_min`, each with a source comment.

Not touched: `pipeline.py`, `calibration.py`, `fusion.py`.

## Policy change (note 1)

A08 (IOP vectors of length 1.3) used to be labelled correctly, because the geometry normalised the vectors
silently. It is now excluded with the reason `bad_orientation`. A header whose vectors are 30 % too long is corrupt
in a way the reader cannot judge, so refusing it (with a reason) is safer than guessing. Rounding-level deviations
(up to 0.01) are still accepted.

## Limitations (not fixed here)

- **Misleading abstention reason (A29, A29b, A30).** These abstain with "Not a lumbar spine study", which is not
  the cause. The cause is a spine that is not where the network expects (mirrored or obliquely cut). A better
  message needs an orientation check in `ood.py` (for example, canal found only after an AP flip); not done.
- **Whole-spine runtime (A37).** A 1150 mm field of view at 1 mm grid costs 50 to 66 s and abstains. The grid is
  set by the field of view, not the matrix, so this is a quadratic cost. A field-of-view guard belongs in the
  labeller's input handling, but the right limit needs real whole-spine data; not decided.
- **Phantom naming shift.** On the cartoon phantom the bottom level name moves one level (see the caveat above).
  Real-data checks (SPIDER acceptance) are the authority; the phantom is not used for accuracy claims.
- **No load-level warning for two sagittal T2 series (A25).** The labeller warns and its confidence drops; a
  study-level "two sagittal series" warning would be clearer.
- **NumPy warnings.** `anatomy/column.py` emits "Mean of empty slice" RuntimeWarnings on sparse inputs (dozens
  per test run). Not changed: outside the allowed files, and the results are unaffected.
- The shipped labeller's inputs are unchanged (the labeller and grid code are byte-identical to main; the signed
  offset only applies to signed DICOM). The polarity cue changes a result only when it contradicts the declared
  contrast; the 20 SPIDER training volumes sampled do not contradict. Full SPIDER validation numbers should be
  re-checked by the suite run (see the test run in the branch report).
