# Changelog

## Unreleased

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
