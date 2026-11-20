# Changelog

## Unreleased

- Findings panel in the offline viewer (dx18, `docs/analysis/README.md`, "Findings in the viewer"):
  `spinescope viewer STUDY OUT --findings` runs the registered analyses on the labels (or `--findings FILE` loads the
  JSON of `spinescope analyze --out`). A table per level with value, confidence, experimental badge and validation
  status, a structure filter, a row click or Enter moves all planes to the level, markers on the planes (colour by
  kind, opacity by confidence, dashed when experimental), and level names with their numbering status. Only the
  documented fields reach the page (`viewer/findings_view.py`; the canary scan covers it). Keyboard: `F` opens the
  panel, arrows and Enter in the table. Browser smoke test with Playwright when a Chromium is available.

- AI reader surface for measured findings (`docs/ai_reader.md`): per disc level, each registered field with its
  confidence and validation status (`spinescope.ai.reading.FIELD_STATUS`); a reading sheet per field (`render/reading_sheet.py`,
  MCP `reading_sheet`, `spinescope findings --outdir`); review tasks that confirm or reject each measured value on its sheet
  (`submit_reader_answer`, `spinescope review`). An AI review can only lower trust, is refused while the numbering is
  unconfirmed, and converts to a findings file that the merge reads as one reader. Plugin skill `spine-findings`; `spine-read`
  gains the review step. Pilot on 10 SPIDER validation discs (blind): the AI reader was within one Pfirrmann grade on all
  ten, but missed all seven radiologist-positive bulges; a pilot, not a validation.
- Labelling integration review (lab37, `docs/labeling/pipeline.md`): one map of the pipeline stage order and of every
  place that changes the numbering status. Fixes: a human answer to the verification packet no longer sets `confirmed`
  on its own (it is `probable`, as a human correction without a confirm; the import already downgraded it); the
  labeller's "numbering unconfirmed" warning no longer stands under a probable or confirmed status (one status summary,
  written after the last cap); the conus confidence and its status note are derived after the image-quality and
  transitional-vertebra caps; the second sagittal series is no longer run through the network twice; an AI answer on
  anchors with no status can no longer raise trust; a suspected transitional vertebra keeps its message when the kind is
  unresolved. Faster: the image-quality ghosting correlation is about twice as fast (same values). Output changes are
  wording and order of warnings, conus confidence after a quality or transitional cap, and the verification-packet
  status above.

- Human review of the level names, end to end (`docs/guide/correcting-labels.md`): `spinescope label --out`, then
  `spinescope viewer --anchors`, correct in the viewer (shift, rename, add, drop, each with a reason), export, then
  `spinescope apply-labels STUDY corrected.json --out reviewed.json` (also `label --anchors`). Corrected files are
  bound to their study by a geometry fingerprint (`numbering.study_fingerprint`), checked against the schema and for
  paths, and their status is set by provenance: an AI reader cannot confirm the names, and `confirmed` needs a human
  confirm with the sacrum in view. Brief, tasks, viewer, MCP `open_study(anchors_path)` and the analysis modules use
  the reviewed file. The viewer's rename, delete and confirm now carry canal, posterior-element and body names too.
- Privacy audit (lab33): a canary phantom (`tests/_canary.py`) with identifying text in every text attribute, a
  private tag, the folder name and the UIDs, scanned through every output surface (`tests/test_privacy_canaries.py`,
  own CI job). Fixes: free-text series descriptions are scrubbed to protocol vocabulary in every output
  (`spinescope.privacy.public_text`; the classifier still reads the raw text), exception messages and warnings no
  longer quote input paths or file names, merge and evidence-image labels are positional or hashed, benchmark errors
  drop path-like tokens, the MCP server withholds messages of unexpected exceptions, and DERIVED series are flagged.
  See `docs/privacy.md` and `SECURITY.md`.
- One-command labelling benchmark (`tools/benchmark.py`, `spinescope benchmark`): SPIDER validation, Sudirman, qx5rt and
  the phantom robustness suites, cached and parallel, with a golden file (`benchmarks/golden.json`) and a regression
  check (`--check`). Numbers and protocol in `docs/benchmarks.md`.
- Numbering evidence before the calibrated decision (lab22): the S1 body below the lowest disc is measured before
  numbering, so `confirmed` is reachable (shipped calibration v2 on the new order: precision 1.00 on held-out sets).
  Geometry and the conus tip are set before the fusion votes; applicability no longer undoes the status caps.
  Calibration v3 candidate is not shipped (`docs/labeling/lab22_confirmed.md`).
- Labelling speed and memory (`docs/labeling/lab14_perf.md`): one network run per study instead of three, the classical
  second opinion is opt-in (`Config.classical_second_opinion`), ONNX threads follow `SPINESCOPE_THREADS` and pool
  workers use one thread each, and `label_spine_cli` reads only the stacks the labeller uses. `import spinescope` no
  longer loads scipy or pydicom.
- Lumbosacral check: a packaged patch CNN (`lumbosacral_v1.onnx`, 448 KB) removes sacral disc candidates below L5-S1
  before numbering (`spinescope.anatomy.lumbosacral`, recipe `tools/lumbosacral.py`, report `docs/labeling/lab13_lumbosacral.md`).
  `label_spine_details(..., lumbosacral=False)` turns it off.
- Initial package: geometry, DICOM and volume loading, series classification.
- Sudirman lumbar DICOM dataset: loader and disc level-name ground truth (`spinescope.datasets.sudirman`),
  scorer for `label_spine` (`spinescope.anatomy._validate_sudirman`), baseline in `docs/labeling/lab02_sudirman.md`.
- Series classification: Siemens POSDISP position-display copies are localizers, not acquisitions.
