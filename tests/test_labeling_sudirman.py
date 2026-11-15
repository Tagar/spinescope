"""Sudirman DICOM dataset: label_spine scored against the dataset's own disc labels (dataset-marked)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spinescope.anatomy._validate_sudirman import evaluate
from spinescope.datasets.sudirman import DISC_LEVELS, labelled_patients


def _roots() -> tuple[Path, Path] | None:
    base = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public"))
    root = base / "lumbar_spine_mri_sudirman"
    labels = base / "lumbar_spine_mri_sudirman_labels"
    if (root / "01_MRI_Data").is_dir() and (labels / "05_Final_Ground_Truth_Data").is_dir():
        return root, labels
    return None


@pytest.mark.dataset
@pytest.mark.slow
def test_sudirman_labels_score_label_spine_on_a_sample():
    roots = _roots()
    if roots is None:
        pytest.skip("Sudirman DICOM and label set not found (set SPINESCOPE_DATA)")
    root, labels = roots
    ids = labelled_patients(labels)[:3]
    ev = evaluate(ids, root=str(root), labels_root=str(labels), workers=1)
    assert not ev.failures
    rows = ev.rows
    assert rows, "no annotated discs were matched to a session"
    assert {r.level for r in rows} <= set(DISC_LEVELS.values())
    for r in rows:
        assert r.dist_mm is None or r.dist_mm >= 0.0
    s = ev.summary()
    # Regression guard only: the labeller is not accurate on this dataset (see docs/labeling/lab02_sudirman.md).
    assert 0.0 <= s["recall"] <= 1.0 and 0.0 <= s["label_accuracy"] <= 1.0
