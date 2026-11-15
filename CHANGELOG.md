# Changelog

## Unreleased

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
