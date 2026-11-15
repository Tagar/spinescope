"""SPIDER dataset tests for label_spine (skipped unless the public data is on disk)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from spinescope import load_study
from spinescope.anatomy import label_spine
from spinescope.anatomy._validate import aggregate, score_anchors, truth_from_mask


def _spider_root() -> Path | None:
    base = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public"))
    root = base / "spider"
    return root if (root / "images").is_dir() and (root / "masks").is_dir() else None


@pytest.mark.dataset
@pytest.mark.slow
def test_spider_validation_sample_runs_and_scores():
    root = _spider_root()
    if root is None:
        pytest.skip("SPIDER not found (set SPINESCOPE_DATA)")
    with open(root / "overview.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = [r["new_file_name"] for r in rows if r["subset"] == "validation"][:5]
    scores = []
    for nm in names:
        truth, mx = truth_from_mask(str(root / "masks" / f"{nm}.mha"))
        anchors = label_spine(load_study(str(root / "images" / f"{nm}.mha")))
        scores.append(score_anchors(anchors, truth, mx))
    m = aggregate(scores)
    # Regression guard only: the labeller is not yet accurate on SPIDER (see docs/validation.md).
    assert m["volumes"] == len(names)
    assert 0.0 <= m["recall"] <= 1.0
