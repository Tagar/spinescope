"""Lumbosacral junction module (lab13): unit tests on synthetic anchors and phantom class maps.

Nothing here needs SPIDER, Sudirman or the ONNX weights. The phantom segmentation is built from its analytic labels
(as in test_anatomy_ml_sacrum.py), with a thin sacral disc painted below the lowest true disc.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spinescope.anatomy import lumbosacral as L
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.calibration import Logistic
from spinescope.anatomy.ml.grid import plane_grid
from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SegmentedStack
from spinescope.anatomy.ml.postprocess import anchors_from_segmentation, mid_sagittal
from spinescope.anatomy.numbering import disc_names
from spinescope.config import DEFAULT
from spinescope.testing.phantom import ANNULUS, CORD, CORTEX, CSF, MARROW, MODIC1, NUCLEUS, SpinePhantom

_TO_CLASS = {MARROW: BODY, CORTEX: BODY, MODIC1: BODY, ANNULUS: DISC, NUCLEUS: DISC, CSF: CANAL, CORD: CANAL}


def _disc(z: float) -> Disc:
    return Disc(center=[0.0, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=10.0, confidence=0.9)


def _anchors_at(zs: list[float]) -> Anchors:
    """Discs at heights ``zs`` (bottom-up), named by count from the bottom, with bodies between them."""
    a = Anchors(method="test")
    names = disc_names(len(zs))
    for name, z in zip(names, zs, strict=True):
        a.discs[name] = _disc(z)
    bodies = [zs[0] - 20.0] + [(zs[i] + zs[i + 1]) / 2 for i in range(len(zs) - 1)] + [zs[-1] + 20.0]
    for i, z in enumerate(bodies):
        a.vertebrae[f"V{i}"] = Vertebra(center=[0.0, 0.0, z])
    return a


def _cand(label: str, rank: int, **features: float) -> L.Candidate:
    base = {name: 0.0 for name in L.FEATURES}
    base.update(features)
    return L.Candidate(label=label, rank=rank, features=base)


def _linear_model(config: L.LumbosacralConfig | None = None) -> L.LumbosacralModel:
    """p(L5-S1) = expit(tilt_deg): each candidate's tilt is set to logit(p) in the tests."""
    w = [1.0 if n == "tilt_deg" else 0.0 for n in L.FEATURES]
    lg = Logistic(
        names=L.FEATURES,
        mean=[0.0] * len(L.FEATURES),
        scale=[1.0] * len(L.FEATURES),
        weights=w,
        bias=0.0,
    )
    return L.LumbosacralModel(lg, config or L.DEFAULT_LUMBOSACRAL)


def _cands_for(ps: list[float]) -> list[L.Candidate]:
    out = []
    for i, p in enumerate(ps):
        out.append(_cand(f"D{i}", i, tilt_deg=float(np.log(p / (1 - p)))))
    return out


# ------------------------------------------------------------------ dropping sacral discs


def test_drop_lowest_removes_sacral_disc_and_counts_up() -> None:
    a = _anchors_at([-80.0, -50.0, -15.0, 20.0])  # -80 is the S1-S2 remnant, -50 is the true L5-S1
    mapping = L.drop_lowest(a, 1)
    # Before the drop the names were L5-S1 (-80), L4-L5 (-50), L3-L4 (-15), L2-L3 (20); after it -50 is L5-S1.
    assert mapping == {"L4-L5": "L5-S1", "L3-L4": "L4-L5", "L2-L3": "L3-L4"}
    assert a.disc_labels() == ["L3-L4", "L4-L5", "L5-S1"]
    assert a.discs["L5-S1"].center[2] == -50.0
    assert any("lumbosacral" in e for e in a.discs["L5-S1"].evidence)


def test_drop_lowest_renames_bodies_and_keeps_one_s1() -> None:
    a = _anchors_at([-80.0, -50.0, -15.0, 20.0])
    L.drop_lowest(a, 1)
    names = {name: round(v.center[2]) for name, v in a.vertebrae.items()}
    # Bodies: below -50 (S1 after the drop), -50..-15 (L5), -15..20 (L4), above 20 (L3). The old S1 below -80 and the
    # body between -80 and -50 both count as "below the new lowest disc": the one nearest to it is kept as S1.
    assert names["S1"] == -65  # nearest body below -50 is the one at -65
    assert "L5" in names and names["L5"] == -32
    assert "L4" in names and names["L4"] == 2
    assert len(a.vertebrae) == len(set(a.vertebrae))


def test_drop_zero_or_empty_is_identity() -> None:
    a = _anchors_at([-50.0, -15.0])
    before = a.disc_labels()
    L.drop_lowest(a, 0)
    assert a.disc_labels() == before
    empty = Anchors(method="test")
    assert L.drop_lowest(empty, 2) == {}


def test_drop_lowest_follows_conus_level() -> None:
    a = _anchors_at([-80.0, -50.0, -15.0])  # names L5-S1 (-80), L4-L5 (-50), L3-L4 (-15)
    a.conus = {"z": -40.0, "level": "L4-L5", "confidence": 0.5}
    L.drop_lowest(a, 1)
    assert a.conus["level"] == "L5-S1"  # the kept disc at -50 is renamed in place
    b = _anchors_at([-80.0, -50.0, -15.0])
    b.conus = {"z": -70.0, "level": "L5-S1", "confidence": 0.5}  # the removed disc
    L.drop_lowest(b, 1)
    assert b.conus["level"] is None


# ------------------------------------------------------------------ decision rules


def test_decide_keep_when_lowest_is_clearly_l5s1() -> None:
    dec = L.decide(_cands_for([0.95, 0.05, 0.02]), _linear_model())
    assert dec.status == "keep" and dec.drop == 0


def test_decide_drops_sacral_lowest_disc_when_next_is_l5s1() -> None:
    dec = L.decide(_cands_for([0.05, 0.97, 0.02]), _linear_model())
    assert dec.status == "drop" and dec.drop == 1
    assert dec.to_dict()["p_l5s1"][1] == pytest.approx(0.97, abs=1e-3)


def test_decide_unsure_when_second_candidate_is_plausible_but_not_strong() -> None:
    dec = L.decide(_cands_for([0.2, 0.7, 0.05]), _linear_model())
    assert dec.status == "unsure" and dec.drop == 0 and dec.warnings


def test_decide_unsure_when_no_candidate_is_clearly_l5s1() -> None:
    assert L.decide(_cands_for([0.3, 0.4, 0.2]), _linear_model()).status == "unsure"
    assert L.decide(_cands_for([0.4, 0.2]), _linear_model()).status == "unsure"


def test_decide_not_assessed_without_model_or_candidates() -> None:
    assert L.decide(_cands_for([0.9]), None).status == "not_assessed"
    assert L.decide([], _linear_model()).status == "not_assessed"


def test_drop_threshold_is_configurable() -> None:
    cfg = L.LumbosacralConfig(drop_min_p=0.99)
    assert L.decide(_cands_for([0.05, 0.97]), _linear_model(cfg)).status == "unsure"


# ------------------------------------------------------------------ packaged model


def test_model_round_trip_and_feature_guard() -> None:
    m = _linear_model(L.LumbosacralConfig(drop_min_p=0.8))
    d = json.loads(json.dumps(m.to_dict()))
    m2 = L.LumbosacralModel.from_dict(d)
    assert m2.config.drop_min_p == 0.8
    assert m2.logistic.weights == m.logistic.weights
    bad = dict(d, features=list(d["features"])[:-1])
    with pytest.raises(ValueError):
        L.LumbosacralModel.from_dict(bad)


def test_load_model_missing_file_is_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert L.load_model(str(tmp_path / "absent.json")) is None


# ------------------------------------------------------------------ features on a phantom segmentation


def _phantom_seg(phantom: SpinePhantom) -> SegmentedStack:
    stack = phantom.render("sag", "t2")
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
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=(0.0, 1.0))


def _paint_sacral_remnant(seg: SegmentedStack, anchors: Anchors) -> None:
    """Paint a thin disc 35 mm below the lowest disc, on the mid-sagittal slice (the labeller's job; here analytic)."""
    mid = mid_sagittal(seg, DEFAULT)
    assert mid is not None
    k = mid.k
    grid = seg.grids[k]
    low = anchors.discs[anchors.disc_labels()[-1]]
    down, post = (float(v) for v in grid.mm_of(np.asarray(low.center))[:2])
    row = int(round(float(grid.index_of_mm(down + 35.0, post)[0])))
    col = int(round(float(grid.index_of_mm(down, post)[1])))
    rows = slice(row - 2, row + 2)
    cols = slice(col - 12, col + 13)
    seg.probs[k, DISC, rows, cols] = 0.95
    seg.probs[k, BODY, rows, cols] = 0.0


def test_candidate_features_have_every_feature_and_bottom_order(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    an = anchors_from_segmentation(seg, DEFAULT)
    cands = L.candidate_features(seg, an, DEFAULT)
    assert len(cands) == min(L.K_CANDIDATES, len(an.discs))
    assert [c.rank for c in cands] == list(range(len(cands)))
    assert cands[0].label == an.disc_labels()[-1]
    for c in cands:
        assert set(c.features) == set(L.FEATURES)
        assert all(np.isfinite(v) for v in c.features.values())
    rows = L.feature_rows(cands)
    assert rows.shape == (len(cands), len(L.FEATURES))


def test_thin_sacral_disc_is_the_lowest_candidate_and_is_dropped(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    an = anchors_from_segmentation(seg, DEFAULT)
    n_true = len(an.discs)
    _paint_sacral_remnant(seg, an)
    an2 = anchors_from_segmentation(seg, DEFAULT)
    assert len(an2.discs) == n_true + 1
    cands = L.candidate_features(seg, an2, DEFAULT)
    remnant = cands[0]
    l5s1 = cands[1]
    # The painted disc is thin (4 mm) and sits 35 mm below the true lowest disc: thinner and lower than L5-S1.
    assert remnant.features["log_height"] < l5s1.features["log_height"]
    assert remnant.features["spacing_mm"] == pytest.approx(35.0, abs=3.0)
    assert remnant.label != l5s1.label
    remnant_z = float(an2.discs[remnant.label].center[2])
    dropped = L.drop_lowest(an2, 1)
    assert all(abs(float(d.center[2]) - remnant_z) > 1.0 for d in an2.discs.values())  # the remnant is gone
    assert len(an2.discs) == n_true
    assert dropped[l5s1.label] == "L5-S1"  # the true L5-S1 is now the lowest disc


def test_no_discs_gives_no_candidates(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    assert L.candidate_features(seg, Anchors(method="test"), DEFAULT) == []
