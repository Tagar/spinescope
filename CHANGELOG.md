# Changelog

## Unreleased

- Lumbosacral check: a packaged patch CNN (`lumbosacral_v1.onnx`, 448 KB) removes sacral disc candidates below L5-S1
  before numbering (`spinescope.anatomy.lumbosacral`, recipe `tools/lumbosacral.py`, report `docs/labeling/lab13_lumbosacral.md`).
  `label_spine_details(..., lumbosacral=False)` turns it off.
- Initial package: geometry, DICOM and volume loading, series classification.
- Sudirman lumbar DICOM dataset: loader and disc level-name ground truth (`spinescope.datasets.sudirman`),
  scorer for `label_spine` (`spinescope.anatomy._validate_sudirman`), baseline in `docs/labeling/lab02_sudirman.md`.
- Series classification: Siemens POSDISP position-display copies are localizers, not acquisitions.
