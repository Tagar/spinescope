"""Scorer additions for labelling evaluation (level errors, per-level recall, calibration), phantom-based."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from spinescope.anatomy._validate import (
    TruthDisc,
    aggregate,
    aggregate_extended,
    group_scores,
    level_index,
    level_offset,
    numbering_error_histogram,
    per_level_recall,
    reliability,
    score_anchors,
    truth_from_mask,
    truth_from_phantom,
)
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.testing.phantom import SpinePhantom


@pytest.fixture(scope="module")
def truth_and_anchors(phantom: SpinePhantom):
    truth, mx = truth_from_phantom(phantom.truth)
    return truth, Anchors.from_truth(phantom.truth), mx


def test_level_index_and_offset():
    assert level_index("L5-S1") == 0
    assert level_index("L4-L5") == 1
    assert level_index("T12-L1") == 5
    assert level_index("nonsense") is None
    assert level_offset("L3-L4", "L4-L5") == 1  # predicted one level too cranial
    assert level_offset("L4-L5", "L3-L4") == -1
    assert level_offset("L4-L5", "L4-L5") == 0
    assert level_offset("L5", "L5-S1") is None  # not a disc name
    assert level_offset("V9-S1", "L5-S1") == 0  # levels are keyed on the lower vertebra, as disc_names does


def test_perfect_prediction_has_no_level_errors(truth_and_anchors):
    truth, anc, mx = truth_and_anchors
    s = score_anchors(anc, truth, mx)
    assert s.matched == s.n_truth == s.label_correct
    assert s.missed == []
    assert all(m.level_offset == 0 and m.label_correct for m in s.matches)
    assert numbering_error_histogram([s]) == {0: s.n_truth}
    rel = per_level_recall([s])
    assert all(v["recall"] == 1.0 and v["label_accuracy"] == 1.0 for v in rel.values())
    ext = aggregate_extended([s])
    assert ext["off_by_one_of_matched"] == 0.0
    assert ext["volumes_all_truth_labelled"] == 1.0


def _shift_names(anc: Anchors, steps: int) -> Anchors:
    """Relabel every disc ``steps`` levels up (a numbering shift), keeping the geometry."""
    shifted = copy.deepcopy(anc)
    names = list(shifted.discs)
    order = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12", "T10-T11"]
    mapping = {}
    for n in names:
        i = order.index(n)
        mapping[n] = order[min(i + steps, len(order) - 1)]
    shifted.discs = {mapping[n]: d for n, d in shifted.discs.items()}
    return shifted


def test_shifted_numbering_is_counted_as_off_by_one(truth_and_anchors):
    truth, anc, mx = truth_and_anchors
    s = score_anchors(_shift_names(anc, 1), truth, mx)
    assert s.label_correct == 0
    assert s.matched == s.n_truth  # geometry is unchanged, so every disc still matches
    hist = numbering_error_histogram([s])
    assert hist == {1: s.n_truth}
    ext = aggregate_extended([s])
    assert ext["off_by_one_of_matched"] == 1.0
    assert ext["level_error_of_matched"] == 1.0


def test_missed_and_false_positive_discs_are_reported(truth_and_anchors):
    truth, anc, mx = truth_and_anchors
    bad = copy.deepcopy(anc)
    dropped = "L3-L4"
    del bad.discs[dropped]
    bad.discs["L9-T1"] = Disc(center=[0.0, 500.0, 500.0], normal=[0.0, 0.0, 1.0], confidence=0.2)
    s = score_anchors(bad, truth, mx)
    assert s.missed == [dropped]
    assert sum(1 for m in s.matches if m.truth is None) == 1
    assert aggregate_extended([s])["false_positives_per_volume"] == 1.0
    rel = per_level_recall([s])
    assert rel[dropped]["recall"] == 0.0
    assert rel["L4-L5"]["recall"] == 1.0


def test_reliability_separates_correct_from_wrong(truth_and_anchors):
    """Correct discs at high confidence and false positives at low confidence give AUROC 1."""
    truth, anc, mx = truth_and_anchors
    good = copy.deepcopy(anc)
    for d in good.discs.values():
        d.confidence = 0.9
    for i in range(3):  # far from every truth disc in the sagittal plane: false positives
        good.discs[f"FP{i}"] = Disc(
            center=[0.0, 400.0 + 10 * i, 400.0], normal=[0.0, 0.0, 1.0], confidence=0.1
        )
    rel = reliability([score_anchors(good, truth, mx)])
    assert rel["auroc"] == pytest.approx(1.0)
    assert rel["n"] == len(good.discs)
    assert sum(b["n"] for b in rel["bins"]) == rel["n"]


def test_reliability_perfect_confidence_has_zero_ece(truth_and_anchors):
    truth, anc, mx = truth_and_anchors
    for d in anc.discs.values():
        d.confidence = 1.0
    rel = reliability([score_anchors(anc, truth, mx)])
    assert rel["ece"] == pytest.approx(0.0)
    assert rel["auroc"] is None  # one class only


def test_group_scores_and_aggregate_on_empty_inputs():
    assert group_scores([], lambda n: n) == {}
    assert aggregate([]) and aggregate([])["volumes"] == 0.0


def test_truth_from_mask_canal_uses_full_transform(tmp_path):
    """Regression: the canal midline must go through the image-to-LPS transform (sitk x can run along y).

    The old formula used only the sitk x index, which here gives 10.0 (the origin); the correct LPS x of
    the canal voxels is -32.0 (sitk z runs along -x with spacing 3 mm from origin x = 10; the median
    canal slice is z = 14, so x = 10 - 3 * 14).
    """
    sitk = pytest.importorskip("SimpleITK")
    arr = np.zeros((24, 40, 6), dtype=np.uint8)  # numpy [z, y, x] of a (x=6, y=40, z=24) image
    arr[5:24, 5:35, 2:4] = 100  # canal voxels: sitk z 5..23, sitk x 2..3
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 1.0, 3.0))
    img.SetOrigin((10.0, -4.0, -20.0))
    # column-major axes: sitk x -> LPS +y, sitk y -> LPS -z, sitk z -> LPS -x
    img.SetDirection((0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    path = str(tmp_path / "mask.mha")
    sitk.WriteImage(img, path)
    _discs, canal_x = truth_from_mask(path)
    assert canal_x == pytest.approx(-32.0)


def test_truth_from_phantom_round_trips_names(phantom: SpinePhantom):
    truth, mx = truth_from_phantom(phantom.truth)
    assert {t.name for t in truth} == set(phantom.truth["discs"])
    assert mx == pytest.approx(phantom.truth["midline_x"])
    assert all(isinstance(t, TruthDisc) for t in truth)
