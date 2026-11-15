# Changelog

## Unreleased

- Initial package: geometry, DICOM and volume loading, series classification.
- Sudirman lumbar DICOM dataset: loader and disc level-name ground truth (`spinescope.datasets.sudirman`),
  scorer for `label_spine` (`spinescope.anatomy._validate_sudirman`), baseline in `docs/labeling/lab02_sudirman.md`.
- Series classification: Siemens POSDISP position-display copies are localizers, not acquisitions.
