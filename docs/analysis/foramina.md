# Neural foramina: location, foraminal height, AP width, foraminal fat and a perineural fat proxy

Analysis `foramina` (`src/spinescope/analysis/foramina.py`). One row per (disc, side), `scope = "foramen"`, with the
disc name, the exiting root (upper vertebra), the anchors' numbering status, and the measurements below. It measures
and flags; it does not diagnose and it does not claim compression (DESIGN.md principle 4).

## 1. What is measured, and what is not

| Quantity | Measured? | How |
|---|---|---|
| Foramen plane (parasagittal slice) | yes | fat search in a lateral window (section 2) |
| Foramen centre | yes | fat search behind the disc (section 2) |
| Foraminal height (mm, pedicle border to pedicle border) | yes, when both borders are found | half-level borders on the profile along the disc normal |
| AP width at the disc level (mm) | yes, when both borders are found | half-level borders on the profile along the disc AP axis |
| Foraminal fat area (mm^2) | yes | connected fat-bright region of the window around the centre; NaN when it opens into the window edge |
| Perineural fat around the exiting root | **proxy only** | which of four directions from the centre keep fat-bright tissue in their outer quarter; the root is not located, so this is not a reader's Lee grade (section 4) |
| Lee et al. 2010 grade (0-2) | derived from the proxy | section 4; grade 3 (root morphology change) is **not assigned** |
| Wildermuth-style grade (0, 2, 3) | derived from the proxy | section 4 |
| Root morphology | no | "not assessed" in every row |
| Axial support (disc contour, facet, osteophyte) | no | not used; the brief's optional axial step is not in this version |

Coverage: if no sagittal T1 plane lies in the lateral window on a side, that side is **not assessed** with the reason
"coverage" (narrow coverage, for instance Sudirman's 15-slice, 4.8 mm sagittal T1 whose lateral extent depends on the
case). If the centre is not fat-bright, the foramen is not assessed.

## 2. Method (in order)

1. **Midline and sides.** Midline is `anchors.midline_x`. Patient left is +x in LPS, so the lateral distance of a plane is
   `+(x - midline)` on the left and `-(x - midline)` on the right (`side_sign`, `lateral_mm`). A missing midline is a
   warning and no rows.
2. **Parasagittal candidates.** Planes whose lateral distance lies in `foramen_lateral_mm` = (10, 25) mm. This is a search
   window (section 3). Its lower bound keeps the spinal canal out: epidural fat is fat-bright on T1 too.
3. **Fat index.** `intensity / p99`, where p99 is the 99th percentile of the T1 stack. Index 1.0 means the series' bright
   tissue (subcutaneous fat, marrow). This is a relative scale; it assumes the field of view contains bright fat (true for
   lumbar fields, but a check is absent: a stack with no bright fat would scale wrongly).
4. **Centre search.** For each candidate plane, the centre is searched on a grid: AP offsets from the posterior disc margin in
   `foramen_ap_search_mm` = (-2, 12) mm, and offsets along the disc normal within `foramen_si_search_mm` = 6 mm. The score of a
   candidate is the mean fat index of a box of half size `foramen_box_mm` = 4 mm. The best centre per plane is kept; the
   best plane is the one with the highest score. `score_margin` is its lead over the median plane.
   `edge_plane` is true when the chosen plane lies within one slice of either window edge.
5. **Fat-bright centre.** The centre is fat-bright when its box mean reaches the window half level, `(1.0 + p10 of the
   window) / 2`, and the window has the contrast in step 7. The box mean is used, not the centre sample, because a nerve root
   at the centre is grey on T1.
6. **Pedicle borders.** On the profile along the disc normal (height) and along the disc AP axis (width), through the
   centre, out to `foramen_profile_half_mm` = 14 mm: the border on each side is the first crossing of the half level
   `(1.0 + p10 of the profile) / 2` moving away from the centre. Missing borders leave `rims_found = False`; the height and width
   are then NaN and the centre test still stands.
7. **Contrast.** A profile or window needs `(1.0 - dark) >= foramen_min_rim_contrast` = 0.25 (dark = its 10th percentile).
8. **Area.** Connected component of the window's fat-bright mask containing the centre; mm^2 from the pixel pitch.
9. **Rays (proxy).** Four rays from the centre (anterior, posterior, superior, inferior), of length half the measured width
   (AP rays) or half the measured height (SI rays), or `foramen_ray_default_mm` = 5 mm where unmeasured. A direction is
   **obliterated** when no sample in its outer part, from `foramen_ray_min_frac` = 0.25 of the length to the length, reaches
   the half level. This is a geometric proxy: it asks whether fat-bright tissue reaches the outer part of each direction. It
   does not know where the root is.
10. **Gradings** (section 4) from the number and pattern of obliterated directions.
11. **Normative height.** Height against the Hasegawa 1995 normal group (section 5): `z = (h - mean) / SD`, and a flag for
    height at or below 15 mm.
12. **Study-relative values.** `height_minus_side_median_mm` (height minus the median of the study's assessed heights on the same
    side) and `fat_index_rel_study_median` (centre fat index over the median of the study's assessed centres).
13. **Naming.** The foramen at disc `L4-L5` carries the **L4** root (the upper vertebra). Anatomy, not a compression claim. The
    disc name carries the anchors' numbering status in every row and flag. Side is the patient's (left = +x).
14. **Flags.** Assessed foramina with >= 2 obliterated directions or height <= 15 mm. Confidence is capped at
    `foramen_flag_max_confidence` = 0.3 until the validation below supports more.
15. **Confidence per row.** 0.6, halved for an edge plane and for incomplete borders, at most 0.1 when the centre is not
    fat-bright, and at most the disc's anchor confidence.

## 3. Search windows (conventions, not measurements)

These are chosen, not measured, and they are the first thing to check if a result looks wrong:

| Config field | Value | Why |
|---|---|---|
| `foramen_lateral_mm` | (10, 25) mm | clear of the canal (lower); the lumbar foramina sit lateral to the pedicle (upper). The edge-plane share is reported |
| `foramen_ap_search_mm` | (-2, 12) mm | the root lies between the posterior disc margin and the facet |
| `foramen_si_search_mm` | 6 mm | the foramen spans the disc between the two pedicles |
| `foramen_box_mm` | 4 mm | roughly a foramen-sized box for the fat score |
| `foramen_profile_half_mm` | 14 mm | half-length of the profiles; well beyond the Hasegawa mean heights (20-22 mm, so a full height fits) |
| `foramen_ray_min_frac` | 0.25 | a direction with fat in its outer quarter counts as fat-bright |
| `foramen_ray_default_mm` | 5 mm | ray length when a border is not found |
| `foramen_min_rim_contrast` | 0.25 | relative contrast needed to place a border |

## 4. Gradings, with thresholds and sources

* **Lee et al. 2010**, AJR 194:1095-1098 (doi 10.2214/AJR.09.2772). Grade 0 normal; 1 perineural fat obliterated in two
  opposing directions (superior and inferior, or anterior and posterior) with no root change; 2 obliterated in all four
  directions with no root change; 3 nerve-root collapse or morphologic change. Here: `lee_grade` = 0 for no obliterated
  direction, 1 for an opposing pair, 2 for all four. One direction, or a non-opposing pair, is outside the scheme (`None`).
  Grade 3 is never assigned: it needs the root's shape, which the ray proxy does not see.
* **Wildermuth et al., Radiology 1998;207:391-398** (doi 10.1148/radiology.207.2.9577486), as reproduced in the Lee 2010
  comparison: 0 normal; 1 deformed but completely surrounded by fat; 2 partially surrounded; 3 complete obliteration. The proxy
  cannot tell 0 from 1 (no shape), so `wildermuth_grade` is 0 for no obliterated direction, 2 for one to three, 3 for four.
  Caveat: the criteria were checked through a secondary source, not the 1998 paper's text.
* **Perineural status** (the wording of the field): preserved (0 directions), partially obliterated (1 to 3), fully
  obliterated (4). The proxy, not the root, decides.

The findings schema (`data/findings.schema.json`) gives `foramen_R.lee_grade` and `foramen_L.lee_grade`, and its
description cites "Lee 1980". The grading in this module is Lee et al. 2010 (AJR). The schema text is shared and is not
edited here (flagged to the integrator).

## 5. Normative heights and the critical height

* **Hasegawa T, et al. 1995**, J Bone Joint Surg Am 77(1):32-38 (cryomicrotome study in cadavers). Foraminal height is the
  maximum distance between the inferior margin of the upper pedicle and the superior margin of the lower pedicle. Normal group
  (no nerve-root compression), mean +- SD in mm: L1-L2 20.25 +- 2.34; L2-L3 21.84 +- 2.30; L3-L4 22.20 +- 2.52; L4-L5 20.47 +- 2.73;
  L5-S1 21.72 +- 2.53. No value for T12-L1 (not tabulated). Cadaver values, neutral supine; MRI measurements are coarser (section 7).
* Critical height: 15 mm, the height at or below which the same study saw nerve-root compression in 4 of 5 foramina.
  A cadaver threshold, used as an operating point, not an MRI standard.

## 6. Pre-registration (written before any validation run)

This section fixes the validation rule. The counts below were read from the report text (positives, negatives, sided labels)
and from the case counts; no metric was computed before this section was committed.

**Data facts that set the design.** Sudirman radiologist reports (515 patients with a session; 575 report numbers keyed to the
same patient numbers): **11 reports** carry a foraminal label, **9 with a side**, **13 positive** labels and **0 explicit negative**
labels. So negatives cannot be estimated from these reports, and sensitivity rests on few positives. SPIDER has no foraminal
grading; its T1 sagittal series are used only for geometry (section 6.3).

**6.1 Truth (Sudirman, weak labels).** For each (disc, side):
* positive: an explicit positive foraminal label on that side at that disc (`stenosis`, zone `foraminal`, side left or right);
* negative: an explicit negated foraminal label at that disc on that side, or unsided, with no positive on that side or unsided;
* excluded: everything else. An unsided positive at a disc excludes both sides of that disc (it could mean either side). Absence of
  a mention is **not** a negative.

**6.2 Scored pairs and operating points.**
* Scored pair = (patient, disc, side) with an **assessed** foramen and a truth. Disc names are the anchors' own names; no shift is
  allowed, so a name disagreement is not scored and is counted.
* **P1** (primary): positive if >= 2 directions obliterated (the proxy for Lee grade 1 or 2). AUC score: n obliterated directions.
* **P2** (secondary): positive if foraminal height <= 15 mm. AUC score: minus the height.
* Metrics: sensitivity, specificity, PPV, NPV, AUC; 95% percentile CIs from 1000 bootstrap draws resampling **patients**, seed 0.
* **Guard:** with fewer than 20 positives, or with 0 negatives, the numbers are reported as counts only. Specificity is reported
  as not estimable if there are no negatives. No CI is claimed beyond what the bootstrap gives.
* Coverage: assessed foramina over all (disc, side) rows, with the reason counts for not-assessed rows.

**6.3 SPIDER geometry sanity (training split only; validation split not used).** For each training case with a sagittal T1: the
foraminal height by disc, median and mean, against the Hasegawa normal mean. **Pre-registered criterion:** a level passes if its
median height is within 2 SD (Hasegawa SD) of the normal mean. Failure rates (not assessed by reason) are reported. No threshold or
window is tuned on these numbers; if they fail, the doc says so.

**6.4 Not part of the rule.** Lee grade 3 (root morphology), the axial step, and any comparison with dx05. No other operating point
will be scored after the fact; any further point is labelled exploratory.

## 7. Limitations

* The perineural grade is a proxy. The root is not located on T1; the rays do not see root shape, and the centre is a fat maximum,
  not the root's centre. A "fat-bright in the outer quarter" test cannot be relied on as a Lee reader grade. Use it as a prompt.
* Sagittal T1 in the Sudirman series is 4.8 mm apart, and the foramen plane is chosen from 15 slices. The lateral search is coarse,
  so the chosen plane can be a slice away from the true foramen (edge-plane share reported).
* The fat reference (p99 = 1.0) needs bright fat in the field of view.
* The Hasegawa values are cadaver values, neutral supine. Supine MRI is not standing; the height is a comparison, not a norm.
* Level names carry their numbering status; a probable or unconfirmed name is reported as such and is not confirmed by this analysis.
* Sudirman's reports name foramina rarely, and their wording is free text parsed by rules (see `datasets/report_labels.py`). A
  missing or wrong report label is a truth error, and a true stenosis the report did not write down is a false negative in this table.

## 8. Results

Results are appended after the pre-registered run (section 6). See the commit history for the run date and the outputs under
`/tmp/spinescope-dx07_foramina/` (not committed: they derive from the public datasets).
