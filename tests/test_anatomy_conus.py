"""Conus locator (spinescope.anatomy.conus): level naming, low-lying flag and phantom recovery.

The phantom tests use an oracle canal segmentation built from the phantom's analytic labels (the learned
labeller is not trained on the phantom; see docs/labeling/lab16_conus.md). No network, no weights needed.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.conus import (
    ConusResult,
    _longest_run,
    conus_from_segmentation,
    level_at,
    low_lying,
)
from spinescope.anatomy.ml.grid import intensity_window, plane_grid
from spinescope.anatomy.ml.labeller import SegmentedStack
from spinescope.config import DEFAULT
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

_TO_CLASS = {MARROW: 1, CORTEX: 1, MODIC1: 1, ANNULUS: 2, NUCLEUS: 2, CSF: 3, CORD: 3}


def _oracle(phantom: SpinePhantom, contrast: str) -> SegmentedStack:
    """Canal (thecal sac incl. cord) and body/disc classes from the phantom's labels, on the 1 mm grids."""
    stack = phantom.render("sag", contrast)
    grids = [plane_grid(stack, k, 1.0) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 4, h, w), np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for k, g in enumerate(grids):
        tissue = phantom.labels(g.lps_of_index(ii, jj).reshape(-1, 3)).reshape(h, w)
        cls = np.zeros((h, w), np.int64)
        for t, c in _TO_CLASS.items():
            cls[tissue == t] = c
        for c in range(4):
            probs[k, c] = cls == c
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=intensity_window(stack.vol))


def _disc(z: float, h: float = 10.0) -> Disc:
    return Disc(center=[0.0, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=h)


def _anchors(discs: dict[str, float], numbering: dict | None = None) -> Anchors:
    return Anchors(
        discs={k: _disc(v) for k, v in discs.items()},
        numbering=numbering or {"status": "confirmed"},
    )


# ------------------------------------------------------------------------------ helpers
def test_longest_run_picks_longest_and_respects_minimum() -> None:
    flags = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert _longest_run(flags, 2) == (3, 5)
    assert _longest_run(flags, 4) is None
    assert _longest_run(np.zeros(5, dtype=bool), 1) is None


# ------------------------------------------------------------------------------ levels
ANATOMY = {
    "T12-L1": 80.0,
    "L1-L2": 50.0,
    "L2-L3": 22.0,
    "L3-L4": -6.0,
    "L4-L5": -34.0,
    "L5-S1": -62.0,
}


def test_level_at_disc_margin_and_vertebra_thirds() -> None:
    a = _anchors(ANATOMY)
    assert level_at(80.0 + 3.0, a)[0:2] == ("T12-L1", "at disc")
    # L1 body spans from the T12-L1 inferior margin (75) to the L1-L2 superior margin (55).
    lvl, pos, _ = level_at(70.0, a)
    assert lvl == "L1" and pos == "upper third"
    lvl, pos, _ = level_at(65.0, a)
    assert lvl == "L1" and pos == "middle third"
    lvl, pos, _ = level_at(57.0, a)
    assert lvl == "L1" and pos == "lower third"


def test_level_at_outside_discs_names_end_vertebra_without_third() -> None:
    a = _anchors(ANATOMY)
    lvl, pos, _ = level_at(95.0, a)
    assert lvl == "T12" and pos is None
    lvl, pos, _ = level_at(-80.0, a)
    assert lvl == "S1" and pos is None


def test_level_at_gap_in_disc_labels_is_not_named() -> None:
    gap = {"L1-L2": 50.0, "L3-L4": -6.0}  # L2-L3 missing: the interval spans two bodies
    lvl, pos, ev = level_at(20.0, _anchors(gap))
    assert lvl is None and pos is None
    assert "gap" in ev[0]


def test_level_at_without_discs() -> None:
    lvl, pos, ev = level_at(10.0, Anchors())
    assert lvl is None and pos is None and ev


# ------------------------------------------------------------------------------ low-lying flag
def test_low_lying_uses_the_l2_l3_disc_margin() -> None:
    a = _anchors(ANATOMY)  # L2-L3 centre 22, height 10 -> inferior margin 17
    assert low_lying(10.0, "L3", a, DEFAULT) is True
    assert low_lying(20.0, "L2-L3", a, DEFAULT) is False  # inside the disc: not below it
    assert low_lying(30.0, "L2", a, DEFAULT) is False


def test_low_lying_name_fallback_without_the_reference_disc() -> None:
    a = _anchors({"L1-L2": 50.0, "L4-L5": -34.0})
    assert low_lying(0.0, "L4", a, DEFAULT) is True
    assert low_lying(0.0, "L3-L4", a, DEFAULT) is True
    assert low_lying(0.0, "L2-L3", a, DEFAULT) is False
    assert low_lying(0.0, "L2", a, DEFAULT) is False
    assert low_lying(0.0, None, a, DEFAULT) is None


# ------------------------------------------------------------------------------ phantom recovery
@pytest.mark.parametrize(
    "spec",
    [
        PhantomSpec(),
        PhantomSpec(segment_lordosis=(10.0, 12.0, 14.0, 16.0, 18.0, 22.0)),  # strong lordosis
        PhantomSpec(noise=0.06, seed=3),
        PhantomSpec(conus_body="T12"),
    ],
    ids=["default", "lordosis", "noisy", "conus_T12"],
)
def test_phantom_t2_tip_matches_truth(spec: PhantomSpec) -> None:
    ph = SpinePhantom(spec)
    seg = _oracle(ph, "t2")
    res = conus_from_segmentation(seg, Anchors.from_truth(ph.truth), contrast="t2")
    assert res.found
    assert res.z is not None
    assert abs(res.z - ph.truth["conus_z"]) <= 3.0
    assert res.level == ph.truth["conus_level"]
    assert res.low_lying is False
    assert res.xyz is not None and abs(float(res.xyz[2]) - res.z) < 1e-6


def test_phantom_t1_polarity_also_locates_the_tip() -> None:
    ph = SpinePhantom(PhantomSpec())
    seg = _oracle(ph, "t1")
    res = conus_from_segmentation(seg, Anchors.from_truth(ph.truth), contrast="t1")
    assert res.found
    assert res.z is not None and abs(res.z - ph.truth["conus_z"]) <= 3.0


def test_phantom_low_lying_conus_is_flagged_with_indeterminate_l3_warning() -> None:
    ph = SpinePhantom(PhantomSpec(conus_body="L3"))
    res = conus_from_segmentation(_oracle(ph, "t2"), Anchors.from_truth(ph.truth), contrast="t2")
    assert res.found and res.level == "L3"
    assert res.low_lying is True
    assert any("low-lying conus" in w and "L3 is an indeterminate" in w for w in res.warnings)


def test_low_confidence_low_lying_is_worded_as_unconfirmed() -> None:
    ph = SpinePhantom(PhantomSpec(conus_body="L3"))
    seg = _oracle(ph, "t2")
    anc = Anchors.from_truth(ph.truth)
    anc.numbering = {"status": "unconfirmed"}  # numbering factor pulls the confidence below the threshold
    strict = dataclasses.replace(DEFAULT, conus_low_lying_min_confidence=0.95)
    res = conus_from_segmentation(seg, anc, strict, contrast="t2")
    assert res.low_lying is True
    assert any("too low to call" in w for w in res.warnings)


def test_no_cord_in_the_sac_is_not_located(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = SpinePhantom._table

    def without_cord(self: SpinePhantom, contrast: str) -> dict[int, float]:
        table = orig(self, contrast)
        table[CORD] = table[CSF]  # the cord looks like CSF on the image
        return table

    monkeypatch.setattr(SpinePhantom, "_table", without_cord)
    ph = SpinePhantom(PhantomSpec())
    res = conus_from_segmentation(_oracle(ph, "t2"), Anchors.from_truth(ph.truth), contrast="t2")
    assert not res.found
    assert res.to_dict() == {}
    assert any("no cord signal" in w for w in res.warnings)


def test_to_dict_is_plain_and_has_the_contract_keys() -> None:
    ph = SpinePhantom(PhantomSpec())
    res = conus_from_segmentation(_oracle(ph, "t2"), Anchors.from_truth(ph.truth), contrast="t2")
    d = res.to_dict()
    assert {"z", "xyz", "level", "position", "confidence", "low_lying", "evidence"} <= set(d)
    json.dumps(d)  # JSON-safe
    assert 0.0 <= d["confidence"] <= 1.0
    assert isinstance(ConusResult(found=False).to_dict(), dict)


def test_tips_without_consensus_are_not_located(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sac-slice tips 60 mm apart disagree with the interpolated tip: not located, no NaN position."""
    from spinescope.anatomy import conus as conus_module

    def tip(z: float) -> conus_module.SliceTip:
        return conus_module.SliceTip(
            k=0,
            z=z,
            xyz=np.array([0.0, 0.0, z]),
            confidence=0.9,
            run_mm=20.0,
            reached_fov_end=False,
            score=0.5,
        )

    def fake_locate(seg, cfg, contrast, mid_x=None):  # type: ignore[no-untyped-def]
        located = conus_module._Located(
            tip=tip(0.0), slice_tips=[tip(0.0), tip(60.0)], n_sac_slices=2, polarity=1
        )
        return located, ""

    monkeypatch.setattr(conus_module, "_locate_on_stack", fake_locate)
    ph = SpinePhantom(PhantomSpec())
    res = conus_from_segmentation(_oracle(ph, "t2"), Anchors.from_truth(ph.truth), contrast="t2")
    assert not res.found
    assert res.xyz is None
    assert any("disagree in z" in w for w in res.warnings)


# ------------------------------------------------------------------------------ v2: tracking, brackets, cross-contrast
def test_track_bridges_short_gaps_and_stops_at_a_long_gap() -> None:
    from spinescope.anatomy.conus import _track

    res = DEFAULT.ml_resolution_mm
    flags = np.array(
        [True] * 10 + [False] * 2 + [True] * 10 + [False] * 12 + [True] * 5, dtype=bool
    )  # gap of 2 mm is bridged, the 12 mm gap ends the cord
    tr = _track(flags, DEFAULT, res)
    assert tr is not None
    assert (tr.start, tr.tip) == (0, 21)
    assert tr.reached_end is False


def test_track_needs_a_minimum_run_and_reports_the_field_end() -> None:
    from spinescope.anatomy.conus import _track

    res = DEFAULT.ml_resolution_mm
    assert _track(np.array([True] * 3 + [False] * 20), DEFAULT, res) is None
    tr = _track(np.array([False] * 4 + [True] * 12), DEFAULT, res)  # cord to the last row of the field
    assert tr is not None and tr.tip == 15 and tr.reached_end is True


def test_bracket_weights_interpolate_linearly_between_sac_slices() -> None:
    from spinescope.anatomy.conus import _bracket_weights

    got = _bracket_weights([0.0, 4.0, 8.0], 1.0)
    assert [i for i, _ in got] == [0, 1]
    assert got[0][1] == pytest.approx(0.75) and got[1][1] == pytest.approx(0.25)
    assert _bracket_weights([0.0, 4.0], -3.0) == [(0, 1.0)]


def test_second_contrast_disagreement_halves_the_confidence() -> None:
    from spinescope.anatomy import conus as conus_module

    ph = SpinePhantom(PhantomSpec())
    seg = _oracle(ph, "t2")
    anchors = Anchors.from_truth(ph.truth)
    base = conus_from_segmentation(seg, anchors, DEFAULT, contrast="t2")
    assert base.found and base.confidence > 0
    far = conus_module.SliceTip(
        k=0,
        z=base.z + 60.0,
        xyz=np.array([0.0, 0.0, base.z + 60.0]),
        confidence=0.0,
        run_mm=20.0,
        reached_fov_end=False,
        score=0.5,
    )
    cross = conus_module._Located(tip=far, slice_tips=[], n_sac_slices=1, polarity=1)
    dis = conus_from_segmentation(seg, anchors, DEFAULT, contrast="t2", cross=cross, cross_contrast="t1")
    assert dis.found
    ratio = dis.confidence / base.confidence
    assert ratio == pytest.approx(DEFAULT.conus_xc_disagree_factor / DEFAULT.conus_xc_absent_factor, rel=1e-6)
    assert dis.cross_z == pytest.approx(far.z) and dis.cross_contrast == "t1"
