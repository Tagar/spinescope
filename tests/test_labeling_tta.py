"""Test-time augmentation of the learned labeller (lab35, ``anatomy/ml/tta.py``): variant maths, agreement and contract.

The variant maths is tested on synthetic arrays and on the phantom's analytic probability maps (an oracle), so it does
not depend on the network. The end-to-end contract test needs the packaged weights and ONNX Runtime and is skipped when
they are missing. Nothing here is calibrated: the numbers are checked in docs/labeling/lab35_uncertainty.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.calibration import (
    TTA_STUDY_FEATURES,
    VOLUME_FEATURES_V2,
    load_tta_calibration,
)
from spinescope.anatomy.ml.grid import plane_grid
from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SegmentedStack
from spinescope.anatomy.ml.postprocess import anchors_from_segmentation
from spinescope.anatomy.ml.tta import (
    TTA_ORDER,
    _probs_of,
    _rotation_maps,
    _warp,
    aggregate,
    disc_records,
    variant_names,
)
from spinescope.config import DEFAULT, Config
from spinescope.geometry import Stack
from spinescope.study import from_stacks
from spinescope.testing.phantom import (
    ANNULUS,
    CORD,
    CORTEX,
    CSF,
    MARROW,
    MODIC1,
    NUCLEUS,
    PhantomSpec,
    SpinePhantom,
)

_TO_CLASS = {MARROW: BODY, CORTEX: BODY, MODIC1: BODY, ANNULUS: DISC, NUCLEUS: DISC, CSF: CANAL, CORD: CANAL}


def _oracle_segmentation(stack: Stack, phantom: SpinePhantom, res: float = 1.0) -> SegmentedStack:
    """One-hot network probabilities from the phantom's analytic tissue labels on each slice's grid."""
    grids = [plane_grid(stack, k, res) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 4, h, w), np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for k, g in enumerate(grids):
        pts = g.lps_of_index(ii, jj).reshape(-1, 3)
        tissue = phantom.labels(pts).reshape(h, w)
        cls = np.zeros((h, w), np.int64)
        for t, c in _TO_CLASS.items():
            cls[tissue == t] = c
        for c in range(4):
            probs[k, c] = (cls == c).astype(np.float32)
    window = (0.0, 1.0)
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=window)


def _disc(label: str, z: float, y: float = 0.0, conf: float = 0.9) -> tuple[str, Disc]:
    return label, Disc(center=[0.0, y, z], normal=[0.0, 0.0, 1.0], height_mm=10.0, confidence=conf)


def _anchors(*discs: tuple[str, Disc]) -> Anchors:
    a = Anchors(method="test")
    a.discs = dict(discs)
    return a


# ------------------------------------------------------------------ configuration and order


def test_config_defaults_keep_the_augmentation_off() -> None:
    assert DEFAULT.ml_tta is False
    assert DEFAULT.ml_tta_variants == 4
    assert len(variant_names(4)) == 4
    assert variant_names(0) == ()


def test_variant_order_and_bounds() -> None:
    assert TTA_ORDER[:4] == ("rot_pos", "rot_neg", "gamma", "plane_up")
    assert variant_names(len(TTA_ORDER)) == TTA_ORDER
    with pytest.raises(ValueError):
        variant_names(len(TTA_ORDER) + 1)
    with pytest.raises(ValueError):
        variant_names(-1)


# ------------------------------------------------------------------ rotation maps


def test_rotation_round_trip_and_direction() -> None:
    """A blob rotated by +/- 4 degrees moves to the predicted pixel, and bringing it back recovers it."""
    h, w = 200, 160
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.exp(-(((ii - 100) ** 2 + (jj - 60) ** 2) / 50.0)).astype(np.float32)
    c = np.array([(h - 1) / 2, (w - 1) / 2])
    for deg in (4.0, -4.0):
        inv, fwd = _rotation_maps((h, w), deg)
        var = _warp(img, inv)
        back = _warp(var, fwd)
        inner = (slice(20, 180), slice(20, 140))
        assert float(np.abs(back[inner] - img[inner]).max()) < 0.05
        th = np.radians(deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        expected = rot @ (np.array([100.0, 60.0]) - c) + c
        peak = np.unravel_index(int(np.argmax(var)), var.shape)
        assert np.abs(np.asarray(peak) - expected).max() <= 1.0


def test_rotated_oracle_maps_recover_the_base_discs() -> None:
    """Rotate the oracle probability maps as a rotation variant does and bring them back: the discs are unchanged."""
    ph = SpinePhantom(PhantomSpec())
    stack = ph.render("sag", "t2", rows=320)
    seg = _oracle_segmentation(stack, ph)
    base = anchors_from_segmentation(seg)
    inv, fwd = _rotation_maps(seg.grids[0].shape, 4.0)
    rotated = np.stack([np.stack([_warp(seg.probs[k, c], inv) for c in range(4)]) for k in range(stack.n)])
    back = np.stack([np.stack([_warp(rotated[k, c], fwd) for c in range(4)]) for k in range(stack.n)])
    again = anchors_from_segmentation(_probs_of(seg, back))
    assert set(again.discs) == set(base.discs)
    for lab in base.discs:
        err = np.linalg.norm(np.asarray(again.discs[lab].center) - np.asarray(base.discs[lab].center))
        assert err < 1.5, (lab, err)


def test_plane_shift_moves_the_analysed_plane() -> None:
    """``plane_shift`` analyses the plane one slice away; 0 leaves the labeller's plane (default behaviour)."""
    ph = SpinePhantom(PhantomSpec())
    stack = ph.render("sag", "t2", rows=320)
    seg = _oracle_segmentation(stack, ph)
    base = anchors_from_segmentation(seg)
    same = anchors_from_segmentation(seg, DEFAULT, plane_shift=0)
    assert same.midline_x == base.midline_x
    plus = anchors_from_segmentation(seg, DEFAULT, plane_shift=1)
    minus = anchors_from_segmentation(seg, DEFAULT, plane_shift=-1)
    assert base.midline_x is not None and plus.midline_x is not None and minus.midline_x is not None
    assert plus.midline_x != base.midline_x
    assert minus.midline_x != base.midline_x
    assert plus.midline_x != minus.midline_x


def test_plane_shift_is_clipped_to_the_stack() -> None:
    ph = SpinePhantom(PhantomSpec())
    stack = ph.render("sag", "t2", rows=320)
    seg = _oracle_segmentation(stack, ph)
    far = anchors_from_segmentation(seg, DEFAULT, plane_shift=10_000)
    assert far.midline_x is not None
    last = float(stack.ipp[-1][0])
    assert far.midline_x == pytest.approx(last)


# ------------------------------------------------------------------ aggregation (synthetic)


def test_identical_variants_agree_perfectly() -> None:
    base = _anchors(_disc("L5-S1", 0.0), _disc("L4-L5", 30.0), _disc("L3-L4", 60.0))
    variants = [("rot_pos", base), ("gamma", base)]
    res = aggregate(base, variants)
    assert res.variants == ["rot_pos", "gamma"]
    assert res.study["tta_count_agree"] == 1.0
    assert res.study["tta_chain_agree"] == 1.0
    assert res.study["tta_presence_min"] == 1.0
    assert res.study["tta_name_min"] == 1.0
    assert res.study["tta_spread_max"] == 0.0
    assert set(res.study) == set(TTA_STUDY_FEATURES)


def test_missing_bottom_disc_and_extra_disc_show_as_chain_disagreement() -> None:
    base = _anchors(_disc("L5-S1", 0.0), _disc("L4-L5", 30.0), _disc("L3-L4", 60.0))
    # The variant misses the lowest disc: the rest are renamed one level down (count from the bottom).
    missing = _anchors(_disc("L5-S1", 30.0), _disc("L4-L5", 60.0))
    # The variant has an extra disc below: every name shifts.
    extra = _anchors(_disc("L5-S1", -30.0), _disc("L4-L5", 0.0), _disc("L3-L4", 30.0), _disc("L2-L3", 60.0))
    res = aggregate(base, [("rot_pos", missing), ("rot_neg", extra)])
    assert res.study["tta_count_agree"] == 0.0
    assert res.study["tta_chain_agree"] == 0.0
    assert res.study["tta_presence_min"] < 1.0
    assert res.study["tta_name_min"] == 0.0
    assert res.study["tta_bottom_name"] == 0.0
    assert res.discs[0].label == "L5-S1"


def test_centre_spread_is_the_largest_displacement() -> None:
    base = _anchors(_disc("L5-S1", 0.0), _disc("L4-L5", 30.0, y=0.0))
    shifted = _anchors(_disc("L5-S1", 0.0), _disc("L4-L5", 30.0, y=3.0))
    res = aggregate(base, [("rot_pos", shifted)])
    by = {d.label: d for d in res.discs}
    assert by["L4-L5"].spread_mm == pytest.approx(3.0)
    assert by["L5-S1"].spread_mm == pytest.approx(0.0)
    assert res.study["tta_spread_max"] == pytest.approx(3.0)
    assert res.study["tta_chain_agree"] == 1.0  # a shift of the centre alone keeps the chain


def test_no_variants_or_no_discs_gives_neutral_features() -> None:
    base = _anchors(_disc("L5-S1", 0.0))
    res = aggregate(base, [])
    assert res.discs == []
    assert res.study["tta_chain_agree"] == 1.0 and res.study["tta_spread_max"] == 0.0
    empty = aggregate(_anchors(), [("rot_pos", base)])
    assert empty.discs == [] and empty.study["tta_name_min"] == 1.0


def test_disc_records_follow_the_final_labels() -> None:
    base = _anchors(_disc("L5-S1", 0.0), _disc("L4-L5", 30.0))
    res = aggregate(base, [("rot_pos", _anchors(_disc("L5-S1", 0.0), _disc("L4-L5", 30.0, y=2.0)))])
    # After a numbering step the same discs carry other names; the records follow the centres.
    final = _anchors(_disc("L4-L5", 0.0), _disc("L3-L4", 30.0))
    rec = disc_records(res, final)
    assert set(rec) == {"L4-L5", "L3-L4"}
    assert rec["L3-L4"]["spread_mm"] == pytest.approx(2.0)
    assert rec["L4-L5"]["spread_mm"] == pytest.approx(0.0)
    far = _anchors(_disc("L1-L2", 500.0))
    assert disc_records(res, far) == {}


# ------------------------------------------------------------------ calibration file and the pipeline


def test_tta_calibration_contract_when_shipped() -> None:
    cal = load_tta_calibration()
    if cal is None:
        pytest.skip("calibration_tta_v1.json is not shipped (the augmentation is off by default)")
    allowed = set(VOLUME_FEATURES_V2) | set(TTA_STUDY_FEATURES)
    assert set(cal.volume.names) <= allowed
    assert set(cal.thresholds) == {"confirmed", "probable"}


def test_pipeline_records_the_augmentation_only_when_on() -> None:
    pytest.importorskip("onnxruntime")
    from spinescope.anatomy.ml import model_path
    from spinescope.anatomy.pipeline import label_spine_details

    if not model_path().is_file():
        pytest.skip("labeller weights not packaged")
    ph = SpinePhantom(PhantomSpec())
    study = from_stacks([ph.render("sag", "t2", rows=320)])
    off = label_spine_details(study, backend="ml", calibrate=False)
    assert off.tta == {} and off.tta_discs == {}
    assert "tta" not in off.anchors.numbering
    on = label_spine_details(study, Config(ml_tta=True), backend="ml", calibrate=False)
    assert on.tta["variants"] == list(variant_names(DEFAULT.ml_tta_variants))
    assert set(on.tta["study"]) == set(TTA_STUDY_FEATURES)
    assert set(on.tta_discs) == set(on.anchors.discs)
    for lab in on.anchors.discs:
        assert any("test-time variants" in e for e in on.anchors.discs[lab].evidence)
    assert "tta" in on.anchors.numbering
