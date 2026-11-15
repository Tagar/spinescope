"""Level correspondence on public data (dataset-marked; skipped when the data is not on disk)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spinescope.anatomy._validate_correspondence import spider_graceful, sudirman_slab_agreement


def _data() -> Path:
    return Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public"))


def _has_sudirman() -> bool:
    base = _data()
    return (base / "lumbar_spine_mri_sudirman" / "01_MRI_Data").is_dir() and (
        base / "lumbar_spine_mri_sudirman_labels" / "05_Final_Ground_Truth_Data"
    ).is_dir()


@pytest.mark.dataset
@pytest.mark.skipif(not _has_sudirman(), reason="Sudirman lumbar dataset not on disk")
def test_sudirman_slab_for_level_agrees_with_the_annotation_on_a_sample():
    from spinescope.datasets.sudirman import labelled_patients

    ids = labelled_patients()[:4]
    out = sudirman_slab_agreement(ids, workers=1)
    assert out["patients_failed"] == 0
    assert out["default"]["levels"] > 0
    # the position-first rule must never be worse than the angle-only rule on the same labelled discs
    assert out["default"]["slab_agreement"] >= out["strict_angle_only"]["slab_agreement"]


@pytest.mark.dataset
def test_spider_sagittal_only_is_graceful_on_a_sample():
    if not (_data() / "spider" / "images").is_dir():
        pytest.skip("SPIDER not on disk")
    out = spider_graceful(limit=2)
    assert out["failed"] == 0
    assert out["studies_with_any_slab"] == 0
    assert out["levels"] > 0
