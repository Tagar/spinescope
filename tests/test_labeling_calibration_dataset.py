"""SPIDER dataset test for the calibrated labelling confidence (skipped unless the public data is on disk).

Runs the default labeller with the packaged calibration on a few SPIDER validation studies. Checks the contract
(probabilities in [0, 1], a status on every numbering, calibrated values replace the raw ones) and that calibrated
confidence separates right from wrong labels on the sample when both occur. The full before/after numbers are in
docs/labeling/lab08_fusion.md (from tools/calibrate_labeling.py); this test is a regression guard, not the metric.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from _spider import spider_root
from spinescope import load_study
from spinescope.anatomy import label_spine
from spinescope.anatomy._validate import score_anchors, truth_from_mask
from spinescope.anatomy.calibration import load_calibration
from spinescope.anatomy.numbering import (
    STATUS_CONFIRMED,
    STATUS_PROBABLE,
    STATUS_UNCONFIRMED,
)
from spinescope.anatomy.pipeline import label_spine_details
from spinescope.geometry import Stack
from spinescope.study import Study, from_stacks

SAMPLE = 6  # validation studies (T2 files, with their T1 sibling as second series)


def _study_for(root: Path, pid: str) -> Study | None:
    """The T2 file through the package loader, with its T1 sibling as second sagittal series (as the tool does)."""
    t2 = root / "images" / f"{pid}_t2.mha"
    if not t2.is_file():
        return None
    st2 = load_study(str(t2)).best("sag")
    t1 = root / "images" / f"{pid}_t1.mha"
    st1: Stack | None = load_study(str(t1)).best("sag") if t1.is_file() else None
    if st2 is None:
        return None
    return from_stacks([st2, st1]) if st1 is not None else from_stacks([st2])


def _validation_t2_ids(root: Path) -> list[str]:
    with open(root / "overview.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["new_file_name"].strip() for r in rows if r["subset"].strip() == "validation"]
    t2 = sorted((n[: -len("_t2")] for n in ids if n.endswith("_t2")), key=lambda s: int(s.split("_")[0]))
    return t2[:SAMPLE]


@pytest.mark.dataset
def test_packaged_calibration_is_loaded_and_consistent() -> None:
    cal = load_calibration()
    if cal is None:
        pytest.skip("no packaged calibration (run tools/calibrate_labeling.py fit --write)")
    assert cal.thresholds[STATUS_CONFIRMED] >= cal.thresholds[STATUS_PROBABLE] > 0.0
    assert cal.disc.n_train > 0 and cal.volume.n_train > 0
    # probabilities stay inside (0, 1) for any feature vector in a plausible range
    x = np.array([[v] * len(cal.disc.names) for v in (-6.0, 0.0, 6.0)])
    p = cal.disc.predict(x)
    assert np.all((p > 0) & (p < 1))


@pytest.mark.dataset
@pytest.mark.slow
def test_spider_validation_sample_is_calibrated_and_labelled() -> None:
    root = spider_root()
    if root is None:
        pytest.skip("SPIDER not found (set SPINESCOPE_DATA)")
    if load_calibration() is None:
        pytest.skip("no packaged calibration")
    from spinescope.anatomy.ml import model_path

    if not model_path().exists():
        pytest.skip("packaged labeller weights not found")
    right: list[float] = []
    wrong: list[float] = []
    n_truth = n_matched = n_ok = 0
    for pid in _validation_t2_ids(root):
        study = _study_for(root, pid)
        if study is None:
            continue
        truth, mx = truth_from_mask(str(root / "masks" / f"{pid}_t2.mha"))
        det = label_spine_details(study, backend="ml", calibrate=True)
        anc = det.anchors
        assert anc.numbering["status"] in {STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED}
        assert anc.numbering["calibrated"] is True
        assert 0.0 <= float(anc.numbering["confidence"]) <= 1.0
        score = score_anchors(anc, truth, mx)
        n_truth += score.n_truth
        n_matched += score.matched
        n_ok += score.label_correct
        numbering_ok = score.matched > 0 and score.label_correct == score.matched
        if not numbering_ok:
            # the policy invariant: a study whose numbering is wrong is never reported as confirmed
            assert anc.numbering["status"] != STATUS_CONFIRMED, pid
        for m in score.matches:
            c = float(m.pred_confidence)
            assert 0.0 <= c <= 1.0
            (right if m.label_correct else wrong).append(c)
    assert n_truth > 0
    # Six studies include one whose lowest discs are sacral discs read as lumbar (a two-level shift, validation 38);
    # the sample accuracy is therefore lower than the full-set 0.96. Regression guard only.
    assert n_ok / max(n_matched, 1) >= 0.75
    if right and wrong:
        assert np.mean(wrong) < np.mean(right)


@pytest.mark.dataset
@pytest.mark.slow
def test_calibration_applies_only_with_the_learned_backend() -> None:
    root = spider_root()
    if root is None:
        pytest.skip("SPIDER not found (set SPINESCOPE_DATA)")
    if load_calibration() is None:
        pytest.skip("no packaged calibration")
    study = _study_for(root, _validation_t2_ids(root)[0])
    if study is None:
        pytest.skip("validation study could not be loaded")
    anc = label_spine(study, backend="classical")
    assert anc.numbering["calibrated"] is False
