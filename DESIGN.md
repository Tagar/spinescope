# spinescope design

## Principles

1. **Exact geometry.** Everything lives in DICOM patient LPS mm. Pixel <-> patient transforms come
   from each slice's ImagePositionPatient / ImageOrientationPatient / PixelSpacing. Never assume a
   fixed slice index, image size, pixel size, orientation or slice order.
2. **Nothing hand-picked.** No per-study constants, no hard-coded slice indices, coordinates, offsets
   tuned on one scan, or manual anchor files. Anatomy (levels, midline, canal, foramina, facets,
   muscles...) is detected from the data by documented algorithms. Tunable numbers live in
   `spinescope.config` with a comment saying where the default comes from (literature, anatomy, or
   validation on public data), and are expressed in mm or relative to measured anatomy/intensity,
   never in pixels or absolute signal units. Every detected anchor carries a confidence and the
   evidence used, and fails loudly (warning + low confidence) rather than silently.
3. **Privacy by construction.** Readers keep an allowlist of technical DICOM attributes
   (`spinescope.io.KEEP_TAGS`). Names, IDs, dates, institutions, physicians, UIDs and file paths never
   reach outputs; free-text attributes are reduced to protocol vocabulary in outputs (`spinescope.privacy`,
   `docs/privacy.md`) (private `meta` keys start with `_` and are never serialized). Tests use the phantom
   (which carries fake PHI precisely to test this) or public datasets.
4. **Honest outputs.** The toolkit measures and flags; it does not diagnose. Automated grades are
   "suggestions" with confidence; reading sheets and structured findings are for a reader (human or
   AI) to confirm. Every number can be traced to an image.
5. **AI-native, not AI-dependent.** Every capability is a Python function first, then exposed through
   the CLI and the MCP server, with JSON + PNG outputs an agent can consume. Classical algorithms are
   the default; ML backends are optional extras.

## Package layout

```
spinescope/
  geometry.py        Stack, plane maths, intersections, plane resampling
  series.py          series classification (plane, contrast, fat-sat, slabs...), display orientation, keys
  study.py           Study container, load_study()
  io/dicom.py        DICOM directory reader (any layout, multi-slab, multi-echo, enhanced multi-frame)
  io/volumes.py      NIfTI / MetaImage / NRRD readers
  config.py          all tunable defaults, documented
  anatomy/           automatic anatomy: midline, canal, vertebra/disc detection & labelling -> Anchors
  analysis/          per-structure measurements: discs, endplates/Modic, STIR survey, canal, foramina,
                     facets/posterior elements, muscles, alignment, field-of-view survey
  render/            windowing, contact sheets, zoom with rulers, annotations, linked views, MPR,
                     per-level reading sheets, enhancement
  viewer/            offline HTML viewer (3-plane, cross-reference, in-browser MPR)
  report/            findings JSON schema, validation, multi-reader merge/consensus, md/html rendering
  ai/                reading briefs, task scopes, MCP server
  testing/           synthetic phantom + DICOM writer for tests and examples
  cli.py             `spinescope` command
plugins/claude-code/ Claude Code plugin: skills for AI-assisted reads
```

## Core data model

- `Stack` (geometry.py): parallel slices, one orientation, `vol[k, r, c]`, exact `pix_to_lps` /
  `lps_to_pix` (no clamping; use `contains`). After `load_study` every stack is in display orientation:
  sagittal anterior-left/head-up, axial and coronal patient-left on screen-right.
- `Study` (study.py): `stacks` keyed by role (`sag_t2`, `sag_t1`, `sag_stir`, `ax_t2`, `ax_t2_slab3`,
  `loc_sag`, ...), `info` (technical, PHI-free), `find()`, `best()`, `slabs()`.
- `Anchors` (anatomy/anchors.py, JSON schema `data/anchors.schema.json`): detected anatomy:
  `midline_x`, `discs{label: {center, anterior, posterior, normal, height_mm, confidence, evidence}}`,
  `vertebrae{label: {center, confidence}}`, `conus{z, level, confidence}`,
  `numbering{method, confidence, notes}`, `method`, `warnings`.
- Findings (report/): a JSON schema for a structured read (per level + global sections), every field
  `{value, confidence, evidence[], notes}`; merge produces a consensus with disagreements.

## Analysis module contract

Each analysis module exposes `analyze(study, anchors, config=None) -> Result` where `Result` has
`rows` (list of dicts, one per measured item), `flags` (notable items with confidence), `warnings`,
and `figures` (name -> RGB array), plus `save(result, outdir)`. Modules must work when a sequence
is missing (e.g. no STIR, no axials) by returning warnings, not raising.

## Testing

- Unit tests run on the synthetic phantom (`spinescope.testing.phantom`), which has ground truth.
- Dataset tests (`@pytest.mark.dataset`) run on public data found under `$SPINESCOPE_DATA`
  (e.g. SPIDER, CC BY 4.0) and are skipped otherwise. Validation numbers go to `docs/validation.md`.
- No test or doc may use real patient data that isn't from a public, licensed dataset.
