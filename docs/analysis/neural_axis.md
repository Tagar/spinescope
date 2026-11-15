# Neural axis findings: conus, perineural cysts, dural sac ratio, filum terminale

Module `spinescope.analysis.neural_axis` (`analyze(study, anchors, config) -> Result`, name `neural_axis`).
Tests `tests/test_analysis_neural_axis.py`. Validation tool `tools/eval_neural_axis.py`. Status: **experimental**.
Every output is a measurement with a confidence capped at `neural_axis_max_confidence` (0.5), and the wording is
"possible", "likely incidental", "consider correlation". Nothing here is a diagnosis.

## What it outputs

| Row scope | Source | Output | Flag |
|---|---|---|---|
| `conus` | `anchors.conus` (lab16/lab26 locator) | level, position, z, confidence, `experimental: true` | "possibly low-lying conus" only if the tip is below the L2-L3 disc and confidence >= 0.6 |
| `perineural_cyst` | sagittal STIR (else T2), sacral canal below L5-S1 | size (major axis, mm), minor axis, slices, large flag | a candidate of at least 5 mm |
| `dural_sac_ratio` | midsagittal sac AP / body AP depth at L5 and S1 | ratio, limit | above the published limit |
| `filum` | axial T2 at L5-S1 | diameter, thickened (true, false or null) | thickened above 2 mm |
| `filum_fatty` | axial T1 at L5-S1 | candidate diameter | candidate fatty filum |

Not implemented, on purpose:

- **Syrinx / central cord T2 bright above the conus.** Rare in a lumbar field of view, and no reference here has
  syrinx labels. Not measured.
- **Clumped nerve roots (arachnoiditis).** Delamarter's classification is a reader's grade, not a threshold. A
  robust measure would need labelled roots, and none exists locally. Not measured.
- **Lumbar root-sleeve cysts.** On a midline-plane sagittal they are not separated from the sac. Only sacral cysts
  are searched.

Missing inputs give warnings, never exceptions: no sagittal STIR or T2, no L5-S1 anchor, no axial series, or a failed
segmentation each name the item that was not assessed.

## Method

### 1. Conus

`conus_rows` reads `anchors.conus`. The locator is not changed. Its record is already capped at
`conus_max_confidence = 0.5` and marked experimental (lab26: 0 of 15 located tips within half a vertebra).
The low-lying flag needs `low_lying` true (tip below the L2-L3 disc margin, Wilson DA, Prince JR. AJNR 1989;10:259)
and confidence at or above `conus_low_lying_min_confidence` (0.6). **Because of the cap, the flag cannot fire
on the current locator.** A test pins this (`test_low_lying_flag_is_dormant_under_the_experimental_cap`). The
flag is not lifted to get output; it stays dormant until the locator passes a reference.

### 2. Perineural (Tarlov) cysts

On the sagittal series (STIR preferred, because fat suppression makes sacral marrow dark, so CSF-like cysts stand out;
T2 otherwise), with the learned canal segmentation:

1. **References.** CSF is the median intensity of the lumbar thecal sac (above the L5-S1 margin). Marrow is the median
   of the sacral pixels in a 15 mm band around the sacral canal, outside the sac. If CSF and marrow differ by less than
   `neural_axis_cyst_min_contrast_frac` (0.15) of the window, the search is **not assessed** (typical of T2 fast spin echo,
   where fat and CSF are both bright).
2. **Candidates.** Per sagittal slice: pixels within `neural_axis_cyst_csf_tol_frac` (0.08) of the window of the CSF
   value, below the L5-S1 margin, within 25 mm AP of the sacral canal. On slices that carry the sac (canal area at least
   0.6 of the peak, `neural_axis_sac_area_frac`) the sac and a 1 mm margin are excluded. On lateral slices nothing is
   excluded, because the largest canal blob there may be the cyst itself.
3. **Shape.** Components not touching the grid edge, with solidity (pixel area over convex hull) at least 0.8 and
   minor/major axis at least 0.3. Size is the major axis from the second moments (a disc of diameter D gives D).
4. **Size and link.** A candidate is kept at 5 mm or more. Candidates on adjacent slices are linked when their centres are
   within 6 mm plus half their sizes. A cyst is the link group; its size is the largest member.
5. **Wording.** A cyst of 20 mm or more: "large perineural cyst: consider correlation with symptoms". Otherwise:
   "possible perineural cyst, likely incidental". Confidence is 0.3 for one slice, plus 0.1 per extra slice up to three,
   capped at 0.5.

### 3. Dural sac ratio

Midsagittal thecal sac AP extent (first to last canal pixel in the row) at the L5 and S1 body centre, divided by the
body AP depth from the labeller (`vertebrae[*].body["ap_depth_mm"]`). Limits from Oosterhof N, et al. Spine 2001
(PMID 11477262): L5 above 0.48, S1 above 0.57. They were derived in Marfan and control adults, and later population
studies show wide overlap with normal subjects, so a ratio above the limit is a low-confidence measurement only.

### 4. Filum terminale, axial

Axial slice nearest the L5-S1 disc (within the slab). Its sac centre is the canal point on the midsagittal plane at the
disc height. A 16 mm patch is sampled on 0.5 mm steps.

- **Thickness (T2).** The CSF reference is the median in a 6 mm disc (the sac, not the filum, so a thick filum does not
  fill it). The filum is the dark component at the half-maximum between CSF and the darkest pixel in a 1.5 mm centre
  disc, with its centroid within 3 mm of the centre. Diameter is the equivalent diameter. Thickened above 2 mm
  (Raghavan N, et al. AJNR 1989;10:27, normal up to 2 mm, cited from its summary; no PMID verified). Within one in-plane
  pixel of 2 mm the call is **null** (indeterminate), not "thickened".
- **Fatty filum (T1).** A bright candidate at the half-maximum of the centre disc. It must exceed the CSF by 0.2 of the
  window and by 0.7 of the slice's fat level (99th percentile above CSF). Cord-like tissue is brighter than CSF on T1
  too, so the relative rule keeps cord out (tested). **Fat saturation is not checked**, so this is a candidate
  (Fatty filum criteria: T1 hyperintense matching subcutaneous fat, signal loss on fat-saturated images; the source
  checked was a teaching review, not a primary paper).

## Thresholds and sources

| Key (`config.py`) | Value | Source |
|---|---|---|
| `conus_low_lying_disc`, `conus_low_lying_min_confidence` | L2-L3, 0.6 | Wilson & Prince 1989 (reused) |
| `neural_axis_cyst_min_mm` | 5 | judgement: about five sagittal pixels; the 4-5 mm slice spacing does not give reliable smaller sizes |
| `neural_axis_cyst_large_mm` | 20 | judgement; Paulsen RD, Call GA, Murtagh FR. AJNR 1994;15:293 (PMID 8192075): treated symptomatic cysts 1.6-3.0 cm |
| `neural_axis_cyst_csf_tol_frac`, `..._min_contrast_frac` | 0.08, 0.15 | judgement (window-relative, offsets carried) |
| `neural_axis_cyst_min_solidity`, `..._min_aspect` | 0.8, 0.3 | judgement (blob shape) |
| `neural_axis_cyst_link_mm` | 6 | judgement |
| `neural_axis_sac_area_frac` | 0.6 | as `conus_sac_area_frac` (lab26) |
| `neural_axis_dsr_l5`, `neural_axis_dsr_s1` | 0.48, 0.57 | Oosterhof N, et al. Spine 2001 (PMID 11477262) |
| `neural_axis_filum_thick_mm` | 2.0 | Raghavan N, et al. AJNR 1989;10:27 |
| `neural_axis_filum_search_mm`, `..._csf_radius_mm`, `..._centre_mm` | 8, 6, 3 | judgement |
| `neural_axis_fatty_contrast_frac`, `neural_axis_fatty_relative_frac` | 0.2, 0.7 | judgement (T1 cord versus fat, see above) |
| `neural_axis_max_confidence` | 0.5 | all outputs unvalidated: capped and marked experimental |

Citations were checked against search summaries of the papers, not the full texts. Re-check the page numbers before
any publication.

## Pre-registered rule for the cyst call

Fixed before the validation numbers were produced (it is in `tools/eval_neural_axis.py` and in the config above):

- A study is **called** when the analysis gives a `perineural_cyst` row of at least 5 mm.
- **Sudirman**: positives are reports naming a Tarlov or perineural cyst (10 reports, found by regex and read by hand).
  Negatives are reports without that wording (weak: the wording is not in the report, which says nothing about whether
  a cyst is absent). A seeded sample of 40 of the 505 is run (seed 0). The reports are free text, so the labels are weak.
- **SPIDER**: no cyst labels, so every call on the validation split is a candidate false positive (burden only).

## Validation (public data, numbers from `tools/eval_neural_axis.py`)

### Sudirman, weak labels (labeller anchors, STIR or T2 sagittal, 50 studies)

| | value | 95% Wilson CI |
|---|---|---|
| Positives run (reports naming a Tarlov cyst) | 10 | |
| Sensitivity (7 of 10 called) | 0.70 | 0.40 to 0.89 |
| Negatives run (40 sampled reports without the wording) | 40 | |
| Specificity (18 of 40 not called) | 0.45 | 0.31 to 0.60 |
| Call rate on negatives | 0.55 | |

The specificity is poor. More than half of the sampled negatives get a call. Some of these may be real cysts that the
reports do not mention. Others are sac-edge or other CSF-bright sacral shapes. The rate is not a reliable estimate of
true false positives until each call is read on the image.

### SPIDER validation split (canal mask label 100, lowest IVD as L5-S1, T2 sagittal, 17 studies assessed)

| | value | 95% Wilson CI |
|---|---|---|
| Studies in the split | 39 | |
| Assessed (sacrum in the field of view, contrast gate passed) | 17 | |
| Not assessed | 21 | (no sacral marrow in the field of view, or CSF and marrow alike) |
| Studies with a call | 7 of 17 (0.41) | 0.22 to 0.64 |
| Largest call per study (mm) | 9 to 62 | calls of 48 and 62 mm are not plausible Tarlov cysts |

Most SPIDER lumbar fields of view end at or near the L5-S1 disc, so the sacral region is not in the image for most
studies. The burden is therefore based on 17 studies, and the sizes show the search picks up CSF-bright shapes that are
not cysts.

**Process note.** While building the SPIDER harness, the disc centre was first computed from image indices rather than
LPS. That bug was found by checking anchor placement on the validation split (six cases), not by reading any count.
The thresholds were not changed after the SPIDER or Sudirman numbers were seen.

### Phantom (ground truth, oracle canal, `tests/test_analysis_neural_axis.py`)

| Case | Truth | Measured | Note |
|---|---|---|---|
| Planted cyst, 10 mm at 12 mm lateral, sacral | z -96.2, y 81.5, 10 mm | z -96.2, y 81.4, 10.2 mm, 3 slices | |
| Default (no cyst) | none | no call | |
| Large cyst, 24 mm | 24 mm | "large", wording for correlation | |
| Thin filum, radius 1.0 (2 mm) on T2 | 2.0 mm | 2.03 mm, thickened null (at the limit) | |
| Thick filum, radius 2.5 (5 mm) on T2 | 5.0 mm | 5.23 mm, thickened | |
| Fatty filum, radius 1.5 on T1 | 3.0 mm | 3.5 mm | a half-pixel high at 1.4 mm axial pixels |
| Cord-like filum on T1 | not fatty | not called | |
| Sac ratio, canal radius 6 / 9 | 0.375 / 0.56 | within 0.08 of 2r/32; flag at 9 | body depth is a fixed 32 mm in the phantom |

The default phantom's sac (canal radius 8 mm) gives a ratio of 0.56 at L5. That is above the L5 limit, because the
phantom's sac is wider than a typical adult sac. The phantom is not calibrated for the ratio, and it is not changed to
pass.

### Visual check

Renders in `/tmp/spinescope-dx14_cord_cysts/vis/`: the cyst outline sits on the planted blob at the sacrum, and the
filum outline sits in the sac centre on the axial. Public-data overlays were not rendered to docs/assets.

## Limitations

- **No reference for the conus, the filum or the syrinx.** Sudirman reports never name the conus, and the only filum
  mention is one lipoma. The filum and fatty-filum measurements are checked on the phantom only.
- **Cyst specificity is poor** on both public sets (above). Calls need a human read before any use.
- **Cysts touching the sac merge with it** and are missed. Lumbar root-sleeve cysts are not searched.
- **Sacral marrow on T2 fast spin echo** often matches CSF, so the search is not assessed there.
- **The labeller's disc height is the anchor's height, not a measured one** on SPIDER (the harness uses the disc's z
  extent across slices, about 8 mm if it is one slice). This shifts the sacral margin by a few millimetres.
- **Axial resolution** (about 1 mm on the phantom, often 0.5 to 1 mm in clinical axials) is close to the filum size.
- **The dural sac ratio uses a body AP depth from the labeller's body records**, which exist only on the learned path.
  Without them it is not assessed.
- **Confidence is not calibrated.** It orders the outputs and is capped at 0.5.

## Reproduce

    SPINESCOPE_THREADS=2 .venv/bin/python tools/eval_neural_axis.py sudirman --n-negatives 40 --seed 0 --out /tmp/spinescope-dx14_cord_cysts/val
    SPINESCOPE_THREADS=2 .venv/bin/python tools/eval_neural_axis.py spider --split validation --out /tmp/spinescope-dx14_cord_cysts/val
    SPINESCOPE_THREADS=2 .venv/bin/python -m pytest -q tests/test_analysis_neural_axis.py
