# lab29_correspondence: which slices show each labelled level, in every stack

Branch `lab/lab29_correspondence` (from main dccbf9a). Owner: `src/spinescope/anatomy/correspondence.py`,
`src/spinescope/anatomy/_validate_correspondence.py`, `tests/test_anatomy_correspondence*.py`, this file.
Shared-file edits (additive only): `ai/briefs.py` (a per-level slice table in section 3 of the brief, and the slice
lines in each level task's scope) and `ai/mcp_server.py` (`SpineTools.level_slices` and its entry in `_TOOL_NAMES`).
`anatomy/pipeline.py` is untouched; `config.py` is untouched (the tunables live in `CorrespondenceConfig`).

## What it does

`level_map(study, anchors, config=None, *, measure_geometry=True) -> LevelMap` takes the labelled anchors and the
study's stacks and says, for each disc and each vertebra, where it is in every stack:

- For every stack (sagittal, axial, coronal, slabs, localizers): the nearest slice, the signed out-of-plane distance,
  whether the level is inside the stack's slab and field of view, the in-plane pixel of the level's centre, and the
  slices whose plane cuts the body (for axials: those within the disc's thickness along its normal; oblique stacks
  use the same cylinder test, see below).
- For every disc: the axial slab angled to it. Slabs are scored against the disc normal and the disc centre's position
  in each slab block. The choice is one-to-one (a slab goes to one disc). Among slabs with the same geometry the
  preferred contrast wins (`t2`, `t1`, `stir`, `pd` by default).
- For every disc: the canal point, the left and right foramen and facet points, and which sagittal stacks cover each
  of them (`covered_by`, with the slice index), or that none does (`no_parasagittal_stack`, with the distance to the
  nearest sagittal in the text lines).
- Quality flags and a quality per level (`ok`, `check`, `poor`).
- `slice_lines(lm, label)` and `slice_table(lm, label)` give the same content as plain text and rows.

`LevelMap.to_dict()` is JSON-serialisable and contains no paths, study identifiers or description text (tested).

### Definitions

- `out_of_plane_mm`: target minus the slice plane, along the stack normal. Positive means the target lies toward higher
  slice indices.
- Cylinder test (`intersects`): a slice cuts a body if its plane is within
  `(thickness/2)|cos θ| + R sin θ` of the body centre, where θ is the angle between the slice normal and the body normal
  and R is the in-plane half width of the body (or disc). For a slab parallel to the disc this is exactly "within the
  disc's thickness". For a tilted disc the in-plane extent contributes, which is the geometry of the cut.
- `tilt_deg`: angle between the stack normal and the target's reference axis: the disc normal for axial stacks,
  the disc's lateral axis for sagittal stacks, the disc's AP axis for coronal stacks. `tilt_mismatch` is raised when
  this exceeds `tilt_tol_deg` and the stack shows the target.
- A sagittal stack covers a foramen or facet when the target lies within half a slice thickness of its plane, inside
  the slab and the field of view.
- Lateral distances (all derived, none fixed in mm; defaults only where a measure is missing, and flagged):
  - body half width R from the vertebral body width (`Vertebra.body['width_mm']`, measured by the learned labeller);
  - canal half width = `canal_transverse_to_ap` x (canal AP / 2) when the AP diameter is measured, else
    `canal_to_body_half_width` x R;
  - foramen = canal half width + `foramen_fraction` x (R - canal half width);
  - facet = canal half width + `facet_fraction` x (R - canal half width).
- Canal centre: the measured centreline (`anchors.canal['points']`, interpolated at the disc's height). Without it, the
  posterior body margin plus half the canal AP diameter, at the midline.
- Foramen height: `foramen_z_fraction` x disc height above the disc centre; AP: `foramen_ap_fraction` of the half depth
  (posterior third).

### Parameters (judgement, in `CorrespondenceConfig`, each with its reason)

| parameter | default | why |
|---|---|---|
| `tilt_tol_deg` | 12 | same as the axial slab matcher (`AxialConfig.slab_match_angle_deg`) |
| `slab_fallback_deg` | 24 (2 x tolerance) | see "Position fallback" |
| `default_body_width_mm` | 44 | adult lumbar body width, anatomy texts (about 40 to 50 mm); used only when unmeasured |
| `body_depth_to_width` | 0.75 | lumbar body AP depth is slightly less than its width |
| `canal_transverse_to_ap` | 1.3 | the lumbar canal is wider than it is deep |
| `canal_to_body_half_width` | 0.4 | canal half width as a share of body half width, when no AP diameter |
| `foramen_fraction`, `facet_fraction` | 0.5, 0.2 | foramen midway between canal wall and body edge; facet near the canal |
| `foramen_ap_fraction`, `foramen_z_fraction` | 2/3, 0.25 | posterior third of the disc; upper half of the disc |
| `slab_geometry_tol_deg`, `slab_geometry_tol_mm` | 1.0, 2.0 | slabs with the same geometry are ranked by contrast |
| `low_confidence` | 0.5 | anchor confidence below this: `low_confidence`, quality `poor` |

Fallback disc height, body height and canal AP are the midpoints of the ranges in `spinescope.config.Config`.

### Flags

Per stack and level: `off_slab`, `off_fov`, `no_intersection`, `slice_far` (the target is more than R from the stack's
plane and no slice cuts it), `tilt_mismatch`.
Per disc: `slab_tilt_mismatch` (the slab angled to the disc is off by more than the tolerance, or the only slab
through the disc centre is), `slab_conflict` (another disc took the slab), `no_sagittal`, `low_confidence`,
`geometry_default`, `midline_unknown`, `tilt_mismatch`, `slice_far`. Coverage facts that do not lower quality:
`no_axial_slab`, `no_axial_slab_for_level`, `no_parasagittal_stack`.

`quality`: `poor` when `low_confidence` or `no_sagittal`; `check` when any other quality flag is set; else `ok`.

## Position fallback (a rule added after the first Sudirman pilot; both variants are reported)

On the first six Sudirman patients, the labeller's L5-S1 normal was 13 to 14 degrees off the annotated slab normal in
two cases, just outside the 12 degree gate, while the slab's block held the disc centre. The map then gave no slab.
The rule was changed to: a slab whose block holds the disc centre is taken when its normal is within
`slab_fallback_deg` (24 degrees), and flagged `slab_tilt_mismatch`. The reason is geometric: a slab through the disc
centre is that disc's slab, and the disc normal is the less reliable input. The rule was chosen after seeing those six
patients, so the strict (angle-only) variant is reported next to it everywhere
(`sudirman_slab_agreement` returns both). Strict mode is `CorrespondenceConfig(slab_fallback_deg=12)`.

## Validation

### Phantom truth (`tests/test_anatomy_correspondence.py`, 29 tests)

- Slab per level equals the truth for all five oblique slabs, including the 28-degree sacral-slope slab for L5-S1.
  Each disc gets a different slab (`test_slab_for_each_level_matches_the_truth_including_the_tilted_one`).
- Pixels map back: for every hit in every stack, `pix_to_lps(slice, row, col)` lands on the target's plane point,
  within 0.3 mm (`test_pixels_map_back_to_the_target_in_every_stack`).
- Flipped acquisition: every stack transposed in-plane, so its normal is reversed (asserted), gives the same slab
  assignment, the same in/out flags, the same |out-of-plane| and the same cut counts.
- Tilt: a slab 20 degrees off the disc, through its centre, is assigned and flagged; at 30 degrees, or in strict mode,
  it is not assigned. A slab 5 degrees off is assigned unflagged.
- Two discs claiming one slab: one keeps it and the other is flagged `slab_conflict`.
- Paramedian stack at x = +15 mm covers the left foramen; with the thin midline stack alone it does not, and the
  text says how far the nearest sagittal is.
- A 48 mm field of view flags the levels outside it (`off_fov`).
- Sagittal-only study: no slab for any level, a warning, no exception.
- Lateral distances scale with the measured body width (doubling it doubles the body term).
- `measure_geometry=True` leaves the input anchors unchanged.
- The brief and the MCP tool carry the table (`test_brief_carries_the_per_level_slice_table`,
  `test_mcp_level_slices_tool`).

Visual check: `/tmp/spinescope-lab29/map_sag_t2_L4L5.png` (the mid-sagittal slice with the L4-L5 centre marked) and
`map_slab_L4L5.png` (the slab the map chose for L4-L5, with the centre marked). The marker sits on the disc in both.

The foramen and facet points are not checked against anatomy. No public dataset here locates them, and the phantom
has none. They are checked for geometry only (symmetry, lateral position from the measures, coverage by the stacks).

### Sudirman (public, T1 annotations; `sudirman_slab_agreement`)

Ground truth: each annotated disc (D3 L3-L4, D4 L4-L5, D5 L5-S1) is the axial T1 slab that the dataset annotated
(`LevelTruth.stack_key`). The map is built on the default labeller's anchors with `slab_contrasts=("t1",)`, because the
annotated slabs are T1 and the same geometry also exists as T2 slabs. The labeller is not changed by this module.

All 515 labelled patients, 1,545 annotated levels, one labeller run per patient. Run time 490 s on 6 workers.

| variant | slab equals the annotated slab | same, given the labeller's disc is within 12 mm |
|---|---|---|
| default (with the position fallback) | 1,397 / 1,545 = 90.4 % | 1,397 / 1,416 = 98.7 % |
| strict (angle only) | 1,341 / 1,545 = 86.8 % | 1,341 / 1,416 = 94.7 % |

Reasons, default variant: `agree` 1,397; `disc_mismatch` 123 (the labeller's disc centre is more than 12 mm from the
annotated centre, so the disc itself is wrong); `no_slab` 10; `other_slab` 9; `no_disc` 6.
Labeller disc centre against the annotated IVD centroid: median 2.47 mm, p90 6.29 mm. Per level the agreement is
456 / 515 (L3-L4), 465 / 515 (L4-L5), 476 / 515 (L5-S1).

Reading: most of the remaining gap is the labeller's disc, not the slab choice. Given a correctly placed disc the map
picks the annotated slab in 98.7 % of levels. The `disc_mismatch` cases are a labelling problem and are reported
as such. This is not a number for the labeller.

### SPIDER (sagittal only; `spider_graceful`)

SPIDER has no axial slabs. The check is that the map builds on every validation case, reports no slab, says so
(`no_axial_slab`, warning "no axial slabs in the study") and produces a level entry for each labelled disc.

Only 39 of the 87 validation cases were on disk when this ran (218 SPIDER studies were extracted; the rest
were still arriving), so the numbers are for those 39 (38 with a T2 sagittal; the rest used their default sequence).

| measure | result |
|---|---|
| studies run / failed | 39 / 0 |
| studies with any slab | 0 (sagittal only, as expected) |
| levels with a labelled disc | 275 |
| levels inside a sagittal stack's slab and field of view | 275 of 275 |
| quality ok / check / poor | 237 / 16 / 22 |

All 22 `poor` levels carry only `low_confidence` (the labeller's own disc confidence below 0.5). None is a coverage
problem. The map builds on every case and reports the missing axial slabs as a warning, not an error.

## API

```python
from spinescope.anatomy.correspondence import CorrespondenceConfig, level_map, level_slices, slice_lines

lm = level_map(study, anchors)  # measures body and canal geometry on a copy if it is missing
lm.slab_for_level["L4-L5"]  # e.g. "ax_t2_slab4", or None
lm.discs["L4-L5"].stacks["sag_t2"].slice  # nearest sagittal slice index
lm.discs["L4-L5"].targets["left_foramen"]["covered_by"]  # {"sag_t2_2": 6, ...}
level_slices(lm, "L4-L5")  # JSON-ready dict with plain-text "lines"
lm.to_dict()  # JSON
```

MCP: `level_slices(study_id, level)` returns the same dict. The brief (`spinescope.ai.briefs.brief`) has a "Slices per
level" block in section 3, and each level task's scope lists its slices.

## Limitations

- The foramen and facet positions are approximations from the canal and body measures. They are not validated against
  anatomy (no ground truth here), so a reader should use them to choose a sagittal to look at, not as a measurement.
- Without a measured body width or canal, the defaults in `CorrespondenceConfig` are used and the entry is flagged
  `geometry_default`. The learned labeller measures both; the classical labeller does not.
  `measure_geometry=True` runs the learned labeller's segmentation once on a copy of the anchors.
- Only multi-slab series give slabs (`role.slab_index`, as in `Study.slabs`). A single oblique axial acquisition is
  treated as an axial stack, not as a slab.
- The slab rule depends on the labeller's disc normal, which is the weakest input at L5-S1 (see the Sudirman numbers).
  The fallback reduces the loss but flags it.
- The learned labeller found no discs on the seed-3 phantom (out of distribution, as the wave brief notes). The map
  then returns no entries, with the warning "no discs in the anchors", rather than inventing levels.
- The cylinder model treats the body as a cylinder of its measured width; the real body is not a cylinder, so the
  cut counts are approximate near the edges.
- The phantom is a cartoon of anatomy at a few fixed angles; the Sudirman numbers are the stronger evidence.

## Reproduce

- Phantom tests: `.venv/bin/python -m pytest -q tests/test_anatomy_correspondence.py`
- Dataset tests (skip without the data): `.venv/bin/python -m pytest -q tests/test_anatomy_correspondence_dataset.py`
- Full test suite on this branch: 1384 passed, 1 skipped, 43 xfailed (pre-existing labeller xfails), 38 deselected (slow).
- Full Sudirman numbers: `sudirman_slab_agreement(contrast="t1")` in `spinescope.anatomy._validate_correspondence`
  (run it under `if __name__ == "__main__":`; the default labeller is used).
- SPIDER: `spider_graceful(subset="validation")`.
