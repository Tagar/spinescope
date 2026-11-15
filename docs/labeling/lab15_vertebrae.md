# lab15: vertebral body, canal and posterior geometry

Adds the measurements that alignment, canal, foramina, facet, muscle and endplate modules need, on top of the
disc centres and level names from `label_spine`. Everything comes from the learned labeller's segmentation
(`anatomy/ml`, classes body / disc / canal), so it runs only with the learned backend (`backend="auto"` or
`"ml"`). The classical backend leaves these fields as None.

## What is added

All fields are optional and additive. `SCHEMA_VERSION` is now `"1.1"`; 1.0 documents still load (the new
fields default to None). The brief's `data/anchors.schema.json` does not exist in the repository, so the
schema is documented here and in `anchors.py`.

| Field | Where | Content |
|---|---|---|
| `Vertebra.body` | `anchors.vertebrae[name].body` | `centre` (3-D, LPS, pooled over all slices), `centre_plane`, `superior_endplate` / `inferior_endplate` (anterior and posterior corners in LPS, `angle_deg`), `height_mm` (anterior, middle, posterior thirds, mean), `ap_depth_mm`, `width_mm`, `complete`, `confidence`, `slice`, `evidence` |
| `Anchors.canal` | top level | `points` (centreline `[x, y, z]`, superior first), `per_level` (disc and mid-body labels to `{z_mm, ap_mm, kind, slice}`), `confidence`, `method`, `evidence` |
| `Anchors.posterior_elements` | top level | `lamina_proxy`, `spinous_tip` (per body, LPS point and slice), `confidence`, `limits` (see below), `method` |

Disc anterior and posterior margins were already on `Disc` and are unchanged; they are validated below.

## How it is computed

Module `anatomy/vertebrae.py` (`attach_geometry`, `measure_geometry`, `body_plane_geometry`,
`posterior_elements`) and `anatomy/canal_ml.py` (`canal_geometry`, `centreline_from_points`, `ap_chord_mm`).
The hook is `pipeline.label_spine_details`, after `_finalise`, on the ML path only. Names come from the final
disc labels, so body `j` (discs below it) is the upper vertebra of the disc below it, the same convention as
`numbering.vertebra_name`. Relabelling (`shift_labels`, `relabel`) carries the geometry names along.

- **Body centre.** Every body-class pixel of the body's zone (the discs below each pixel define the zone),
  on all slices, LPS mean. No largest-component cut and no per-slice area floor: those dropped lateral body
  parts and biased the centre by 4 to 6 mm.
- **Endplates and heights.** On the sagittal slice nearest the body's own centre (the spine is displaced
  laterally across a stack, so one mid-plane is not enough). The top and bottom boundary of the largest
  body component per column give the endplates; least-squares lines over the central 60 % of the AP
  depth, corners at the body's anterior and posterior extremes. Height is the vertical gap between the
  endplates at the thirds, times cos(superior endplate angle). AP depth is the length of the superior
  endplate.
- **Width.** 3-D lateral extent (99.5th minus 0.5th percentile of the pooled points), needs at least 20 mm of
  slice span.
- **Canal.** Canal pixels of all slices pooled in 5 mm z bins; within a bin each slice counts only if it holds
  at least 30 % of the bin's largest slice count (lateral strays); a bin needs two slices. AP diameter at a
  level is the median AP chord of the canal on the sagittal slice nearest the centreline at that height, over
  rows within 2 mm of the level.
- **Posterior elements (approximations).** Lamina proxy: the posterior canal edge on the body's mid-height
  row. Spinous tip: the posterior-most window-normalised intensity edge between 3 and 45 mm behind that edge.

## Calibration (fitted on SPIDER training only)

Training subsample: 60 of 360 training volumes, evenly spaced over the sorted list. Validation was not used for any choice.

- `vert_width_slice_fraction = 0.0`. Width gate sweep on the training subsample: median absolute error 6.7 mm
  with no gate, 9.6 mm at 0.25, 13.2 mm at 0.5. The gate that minimises the median is no gate.
- `canal_ap_bias_mm = 0.5`. The mid-sagittal chord overstates the truth chord by a median 0.5 mm (one
  pixel of boundary). Subtracting 0.5 mm halves the median absolute AP error on training (1.0 to 0.5 mm).
  On validation the signed median is then 0.0 mm, so the correction carried over.

Other defaults (`vert_core_fraction`, `vert_min_ap_mm`, `canal_z_bin_mm`, spinous-tip parameters) are
judgement values, documented in `config.py`.

## Validation (SPIDER, masks as truth)

Truth uses the same functions as the labeller, applied to the SPIDER label maps (vertebra label `j` = body
with `j` discs below it, canal 100, disc `200+i` below vertebra `i`), on the same plane grid as the
prediction. Only bodies whose two bounding discs carry the correct level name are scored. Errors are
prediction minus truth. The scored volumes are the 87 validation volumes (held out) and a 60-volume training
subsample (calibration, reported for reference).

Run: `python tools/eval_vertebrae.py --split validation --jobs 3 --out <dir>` (same with `--split training --n 60`).

| Quantity | Validation (87 vol.) median / p90 | Training subsample (60) median / p90 |
|---|---|---|
| Body centre, 3-D | 2.8 / 5.2 mm (n = 596) | 2.7 / 6.5 mm |
| Body centre, in-plane | 2.6 / 4.9 mm | 2.4 / 5.3 mm |
| Superior endplate angle | 1.4 / 5.1 deg | 1.2 / 4.0 deg |
| Inferior endplate angle | 1.3 / 5.7 deg | 1.3 / 5.5 deg |
| Body height, mean of thirds | 0.34 / 1.43 mm | 0.38 / 1.0 mm |
| Body height, anterior / posterior | 0.57 / 2.2, 0.56 / 2.2 mm | 0.5 / 1.7 mm |
| Body AP depth | 0.95 / 2.1 mm | 0.87 / 1.9 mm |
| Body width (3-D, low confidence) | 3.3 signed, 6.6 absolute median; p90 24 mm | 4.4 signed, 6.7 absolute median; p90 28 mm |
| Canal centreline (per z bin) | 0.83 / 3.3 mm (n = 3631) | 0.63 / 7.0 mm |
| Canal AP diameter, absolute | 0.5 / 2.0 mm (n = 1097) | 0.5 / 1.5 mm |
| Disc anterior margin | 0.93 / 2.6 mm (n = 613) | 0.82 / 1.8 mm |
| Disc posterior margin | 0.94 / 2.7 mm | 0.84 / 1.9 mm |

Signed biases (validation): body mean height -0.05 mm (median), canal AP 0.0 mm after correction, and body
centre z +1.3 mm (predicted centres sit about 1.3 mm superior of the mask centroid; the same on training).
The centre bias is not corrected.

Groups (validation, body centre 3-D median / superior endplate angle median): Philips 2.9 mm / 1.0 deg
(n = 32 volumes), Siemens 2.8 mm / 1.6 deg (n = 55); t1 2.7 / 1.3, t2 2.9 / 1.3, t2_SPACE 2.6 / 1.9.
The `vendor`, `field` and `sequence` groups are in `summary.json`. SPIDER's `num_vertebrae` counts all
labelled vertebrae (6 to 9), so it does not identify transitional anatomy, and no transitional split is made.

## Sanity on non-SPIDER data (no body truth)

Six Sudirman and six OSF qx5rt sessions through the public `label_spine` path (hook included): no
exceptions and no geometry warnings. Body mean heights 16 to 28 mm, endplate angles within about plus or
minus 20 deg, canal AP 5 to 21 mm, with the canal centreline present in every case. Two cases to note:
one Sudirman body measures 4 mm (a likely segmentation failure), and one Sudirman folder is a cervical
study that the lumbar labeller accepts without a warning.

## Limits

- **Width** is not reliable (median absolute 6.6 mm, p90 about 24 mm on validation). It has no separate
  confidence; treat it as indicative only.
- **Posterior elements are unvalidated.** No training labels exist for them. The lamina proxy is the
  posterior canal edge, not the lamina surface. The spinous tip is the posterior-most intensity edge in the
  search window and often lands near the search limit (about 40 to 45 mm behind the canal), which suggests the
  skin-fat boundary rather than the tip. Confidence is fixed at 0.2 for tips and 0.5 for laminae.
- **Canal** is measured on the labeller's canal class. Its AP diameter is a sagittal dural-sac chord, not the
  bony canal. Tops and bottoms of the lumbar column can have sparse bins.
- **Cost.** The hook runs the network once more and then the geometry. Measured on SPIDER volumes, the geometry
  step takes about 2.6 to 3.6 s per volume on top of the labeller's own run.
- **Classical backend:** no body, canal or posterior fields.
- **Disc margins** are the existing mid-plane values from the labeller, validated as they stand. A disc far
  from the mid-plane has margins on the mid-plane, not on its own centre.

## Tests

`tests/test_anatomy_vertebrae.py` (14 tests) uses synthetic label maps with known geometry: a 10 degree
superior endplate, 32 mm body height, 10 mm canal AP, a posterior intensity step, stray lateral canal
pixels, relabelling, JSON round trip and a 1.0-document load. One test runs the labeller and hook on the
phantom (out of distribution, so it checks the code path only).
