# Focal vertebral marrow lesions and diffuse marrow signal (sagittal T1 / T2, Dixon when present)

Status: **experimental**. The focal flags are **off** because the pre-registered rule failed validation (negative
result, below). The focus rows are descriptions only. The diffuse flag stays on, capped at 0.4, with no public
validation. Nothing here is a diagnosis, and no malignancy is implied by any output.

Module: `src/spinescope/analysis/marrow.py` (registered as `marrow`). Thresholds: `config.py`, block `marrow_*`.
Tests: `tests/test_analysis_marrow.py` (23 tests, phantom-based).

## What it measures

Per vertebral body, on every reference sagittal slice (the same traced endplates and disc frames as `endplates`):

1. **Body core.** At least 10 mm (`marrow_focal_endplate_mm`) from both endplates, and clear of a 1.5 mm cortical rim.
   The endplate band is 3 to 8 mm (`endplate_band_mm`), and the Modic-type band in the phantom reaches 9 mm. So a
   10 mm margin keeps endplate changes and Schmorl nodes out of the focal search, as the endplate lesion search does
   (`lesion_endplate_mm`).
2. **Marrow reference.** T1 and T2 are divided by the body's own mid-body marrow median, pooled over the body's
   slices. Pooling matters: a focus covers one or two slices, and on the lesion slice a per-slice median is
   set by the focus (a planted haemangioma was missed that way). A two-pass rule drops pixels that already deviate
   by more than 20% (`marrow_reference_exclude_ratio`).
3. **Foci by contrast pattern** (ratio to marrow, `marrow_focal_*`):
   - bright T1 and bright T2 (ratio at least 1.20 on each): haemangioma-like, or focal fat-like when Dixon fat
     fraction is at least 0.5 or, without Dixon, STIR is at most 0.60 of marrow. Without a fat-sensitive sequence the
     class is "T1 and T2 bright: haemangioma or fat (undecided)".
   - dark T1 and dark T2 (at most 0.80 and 0.85) with STIR not bright (at most 0.85 when present): bone island-like.
   - dark T1 (at most 0.80) that is also darker than the adjacent disc nucleus, and not in the bone island pattern:
     atypical. A reader sees "atypical focal marrow lesion; consider dedicated evaluation" only when the flag switch is on.
4. **Linear exclusion.** A focus at most 4 mm wide and at least 2.5 times as long is a channel (basivertebral vein)
   and is not reported (`linear_excluded` in meta). Minimum focus area is a 4 mm diameter disc.
5. **Clustering and size.** Pixel components on neighbouring slices are one focus when their centres lie within 8 mm.
   Size is the larger bounding-box side of the best slice, in mm.
6. **Diffuse marrow.** Per body: marrow T1 divided by the adjacent discs' nucleus T1 (median over slices). Flag
   `diffuse_t1_low` when the ratio is below 0.90 (`marrow_diffuse_disc_ratio`), with the wording "diffusely low T1
   marrow signal; can reflect marrow reconversion or infiltration; consider correlation".
7. **Dixon fat fraction** (when water and fat stacks with `role.dixon` are present): per body and per focus, F/(F+W).
   Its reference is age-unaware and gets no normative judgement.

**Not measured:** the paraspinal-muscle reference the brief asks for. Muscle is parasagittal, and this analysis uses
one sagittal stack, so the disc is the only reference. The meta field `muscle_reference` says so.

## Literature (verified by search, abstract level; full texts not read)

- Gaudino S, et al. A systematic approach to vertebral hemangioma. *Skeletal Radiol* 2015;44:25-36.
  doi:10.1007/s00256-014-2035-y. Coarsened vertical trabeculae with T1 and T2 hyperintensity (typical haemangioma).
- Ross JS, et al. Vertebral hemangiomas: MR imaging. *Radiology* 1987;165:165-169. doi:10.1148/radiology.165.1.3628764.
  Chemical-shift and histology: fat explains the T1 brightness. (The brief's "Carroll 1997" was not found and is not cited.)
- Baudrez V, Galant C, Vande Berg BC. Benign vertebral hemangioma: MR-histological correlation. *Skeletal Radiol*
  2001;30:442-446. doi:10.1007/s0025610300442. T1 hyperintensity with adipocytes, T2 with vessels and oedema.
- Vande Berg BC, et al. Normal variants and frequent marrow alterations that simulate bone marrow lesions at MR imaging.
  *Radiol Clin North Am* 2005;43:761-770. doi:10.1016/j.rcl.2005.01.007. Lumbar marrow on non-fat-suppressed T1
  normally sits above the disc; thoracic marrow can sit below it.
- Zhang W, et al. *Medicine (Baltimore)* 2026;105(3):e47196 (IDEAL-IQ, 74 healthy volunteers, age-stratified).
- Baum T, et al. *J Magn Reson Imaging* 2015;42:1018-1023 (PDFF L1-L5, 28 young adults).

Thresholds: no threshold has a published numeric cut-off here. The ratio cut-offs are judgement, uncalibrated, and
stated as such in `config.py`. The literature fixes the patterns and the anatomical logic, not the numbers.

## Pre-registration and the order of events (please read)

The config values (`marrow_*`) were set by judgement. During phantom debugging three method details changed, and all
three were fixed on the phantom, not on any public data: the core margin went from 8 to 10 mm (the phantom's Modic band
reaches 9 mm), the marrow reference became pooled over slices with a two-pass exclusion (a per-slice median was set by
the planted focus), and the basivertebral vein was moved inside the traced body. Then the SPIDER validation split was run
**once** with the resulting config (the call burden below). No threshold was changed after that run. The Sudirman check used a rule fixed in its script before the run (below).
The SPIDER **training** split was not used. Calibrating on training would have been the proper route for a new rule, but
the brief says a rule that fails validation is a negative result and not to be re-tuned, so nothing was re-tuned.

## Validation

### Phantom (ground truth)

Planted lesions (`PhantomSpec.marrow_lesions`): haemangioma 10 mm (L3), focal fat 10 mm (L1), bone island 8 mm (L2),
atypical 10 mm (L4); basivertebral vein in every body; diffuse T1-dark marrow in L5; Modic-type band at L4 inferior;
Schmorl node; 8 mm slices of 8 mm thickness (partial volume). Results:

- Each planted focus gets its class. Ratios match the phantom intensities (haemangioma T1 1.31 against 950/720 = 1.32).
- Reported sizes are 8.4 to 12 mm for planted 8 to 10 mm (bounding box plus smoothing; one to two pixels).
- Only the atypical focus is flagged when the switch is on. The basivertebral vein, the Modic band, the Schmorl node and
  partial volume produce no focus. The normal phantom gives no focus, with noise, lordosis and 3 T variants.
- Diffuse: L5 at 0.79 against the disc. Normal bodies 1.6 to 2.1.

### SPIDER validation (39 cases, call burden only; SPIDER has no focal truth)

Run with the committed config (`marrow_focal_flags` off, so rows only; counts from the rows and the flags).

| Quantity | Result |
|---|---|
| Studies with at least one focus row | 33 of 39; median 11 foci per study; 786 foci in total |
| Focus classes | bone island-like 430; T1 and T2 bright (undecided) 340; atypical 16 |
| Atypical call per study (flag switch on) | 7 of 39 studies, 0.18 (95% Wilson 0.09-0.33), 16 calls |
| Diffuse T1 flag per study | 5 of 39, 0.13 (0.06-0.27) |

Atypical calls must be rare, and 0.18 per study is far above that. Many foci sit at a T1 ratio near 0 (background or
edge artefact in the SPIDER volumes), and the dark pattern is common in ordinary marrow. This is the failure the brief
describes.

### Sudirman weak-label check (60 studies: 12 haemangioma notes, 48 controls)

Pre-registered before the run, in `/tmp` script form: positives are notes that name a haemangioma (12); controls are
the first 48 notes by patient number with no marrow-type term (haemangioma, bone island, Modic, marrow, basivertebral,
Schmorl, and similar). The first session of each patient was used, default config. A level is called when it has a
bright T1 and T2 focus (haemangioma-like or undecided). Level names come from the note clause that names the
haemangioma (D12 mapped to T12). The labels are weak: a note is one radiologist's free text, and controls are assumed
negative.

| Quantity | Result (95% Wilson) |
|---|---|
| Positive notes with a level stated | 10 (12 reference levels: 5 called, 7 missed) |
| Level recall | 5/12 = 0.42 (0.19-0.68) |
| Level precision | 5/22 = 0.23 (0.10-0.43) |
| Positive notes with at least one call | 11 of 12 |
| Controls with at least one called level | 39 of 48 = 0.81 (0.68-0.90) |
| Study without a sagittal series | 1 of 60 |

The operating point fails both ways: the calls are not specific (controls are called on the sacrum and the lower
lumbar bodies), and the recall and precision are low. Haemangioma in this set is 2% of notes (12 of 575).

### qx5rt Dixon fat fraction (numbers only; CC BY-NC, research use)

Per body, F/(F+W) over the binary body mask (`Labels/L1`-`L5`), 161 body rows over 34 sessions (2 sessions had no Dixon).
Some files named `.nii.gz` are plain NIfTI, so they were read as bytes.

| Level | n | median % | IQR % | range % |
|---|---|---|---|---|
| L1 | 30 | 69.6 | 55.7-79.4 | 16.3-86.0 |
| L2 | 32 | 70.1 | 54.6-78.7 | 16.4-85.4 |
| L3 | 33 | 72.9 | 61.7-80.9 | 14.7-86.8 |
| L4 | 33 | 68.9 | 61.3-79.7 | 16.8-87.3 |
| L5 | 33 | 75.3 | 61.4-79.6 | 16.8-87.3 |

By vendor (medians): Philips Achieva 68.2 (n 121), Elition 69.7 (20), Ingenia 79.3 (20).

Comparison against published values is **not a valid normative test**. The qx5rt medians (about 70%) are roughly double
the published healthy-adult PDFF means (Zhang 2026: 28-64% by age and sex; Baum 2015: 32-39% in young adults). The
likely reason is the definition: this is an uncorrected two-point F/(F+W) on images whose calibration the release does
not document, not confounder-corrected PDFF. The imaging files carry no age, so the comparison is age-unaware. No
normative judgement is made.

## Limitations

- Negative result: the focal rule is over-called on SPIDER and Sudirman. The focal flags are off by default, and the
  focus rows are for a reader to inspect, not to read as findings. The most likely causes are the dark-pattern
  threshold on real marrow texture, zero-signal foci at the body edge, and a T1 reference that is uncalibrated across
  vendors.
- The disc-only reference (no muscle). Diffuse T1 has no truth to validate against.
- No STIR in SPIDER and Sudirman, so the STIR fat rule and the bone-island STIR check run only on the phantom.
- Dixon is validated only as a measurement on qx5rt, not as a focal discriminator.
- Sizes are bounding boxes of a thresholded focus: about 1 to 2 mm too big at a 1.2 mm pixel and too small at the
  edges of round foci. Foci taller than the core (body height minus 20 mm) are clipped in height.
- The one-sagittal-stack geometry means the lateral extent of a focus is not known.
- Haemangioma numbers come from 12 notes. The confidence intervals are wide, and none of this supports a rate.

## Not done (open for the integrator)

- Calibrate the focal thresholds on SPIDER **training** (not validation), then score validation once, as a new rule.
  Fix the dark pattern and the zero-signal foci first.
- Add the muscle reference from a parasagittal or axial stack.
- Read the Zhang 2026 and Baum 2015 full texts and check the Dixon calibration before any normative comparison.
- No findings-schema field, no viewer overlay, no MCP exposure in this wave.
