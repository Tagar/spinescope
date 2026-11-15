"""Lumbosacral junction module (lab13): unit tests, network-free except the packaged-model smoke test.

Synthetic anchors test the drop and rename; a fake scorer tests the decision rules; the phantom segmentation is built
from its analytic labels (as in test_anatomy_ml_sacrum.py), with a thin sacral disc painted below the lowest disc.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinescope.anatomy import lumbosacral as L
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.ml.grid import plane_grid
from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SegmentedStack
from spinescope.anatomy.ml.postprocess import anchors_from_segmentation, mid_sagittal
from spinescope.anatomy.numbering import disc_names
from spinescope.anatomy.pipeline import _apply_status_caps
from spinescope.config import DEFAULT
from spinescope.testing.phantom import ANNULUS, CORD, CORTEX, CSF, MARROW, MODIC1, NUCLEUS, SpinePhantom

_TO_CLASS = {MARROW: BODY, CORTEX: BODY, MODIC1: BODY, ANNULUS: DISC, NUCLEUS: DISC, CSF: CANAL, CORD: CANAL}


class _Fixed:
    """A scorer with fixed P(L5-S1) per candidate (stands in for the packaged CNN)."""

    def __init__(self, probs: list[float], cfg: L.LumbosacralConfig | None = None) -> None:
        self._p = np.asarray(probs, dtype=np.float64)
        self.config = cfg or L.DEFAULT_LUMBOSACRAL

    def predict(self, patches: object) -> np.ndarray:
        return self._p


def _cands(n: int) -> list[L.Candidate]:
    return [L.Candidate(label=f"D{i}", rank=i) for i in range(n)]


def _patches(n: int) -> np.ndarray:
    return np.zeros((n, L.PATCH_MM, L.PATCH_MM), np.float32)


def _decide(probs: list[float], cfg: L.LumbosacralConfig | None = None) -> L.LumbosacralDecision:
    return L.decide(_cands(len(probs)), _patches(len(probs)), _Fixed(probs, cfg))


def _disc(z: float) -> Disc:
    return Disc(center=[0.0, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=10.0, confidence=0.9)


def _anchors_at(zs: list[float]) -> Anchors:
    """Discs at heights ``zs`` (bottom-up), named by count from the bottom, with bodies between them."""
    a = Anchors(method="test")
    for name, z in zip(disc_names(len(zs)), zs, strict=True):
        a.discs[name] = _disc(z)
    bodies = [zs[0] - 20.0] + [(zs[i] + zs[i + 1]) / 2 for i in range(len(zs) - 1)] + [zs[-1] + 20.0]
    for i, z in enumerate(bodies):
        a.vertebrae[f"V{i}"] = Vertebra(center=[0.0, 0.0, z])
    return a


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
    # Bodies below -50 (S1 after the drop): the nearest one (at -65) keeps the name; the one at -100 is dropped.
    assert names["S1"] == -65
    assert names["L5"] == -32
    assert names["L4"] == 2


def test_drop_zero_or_empty_is_identity() -> None:
    a = _anchors_at([-50.0, -15.0])
    before = a.disc_labels()
    L.drop_lowest(a, 0)
    assert a.disc_labels() == before
    assert L.drop_lowest(Anchors(method="test"), 2) == {}


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
    dec = _decide([0.95, 0.05, 0.02])
    assert dec.status == "keep" and dec.drop == 0


def test_decide_drops_sacral_lowest_disc_when_next_is_l5s1() -> None:
    dec = _decide([0.05, 0.97, 0.02])
    assert dec.status == "drop" and dec.drop == 1
    assert dec.to_dict()["p_l5s1"][1] == pytest.approx(0.97, abs=1e-3)


def test_decide_drops_two_sacral_discs_when_third_is_l5s1() -> None:
    dec = _decide([0.01, 0.02, 0.93])
    assert dec.status == "drop" and dec.drop == 2


def test_decide_unsure_when_lower_candidate_is_plausible_but_not_strong() -> None:
    dec = _decide([0.2, 0.7, 0.05])  # default drop threshold 0.8
    assert dec.status == "unsure" and dec.drop == 0 and dec.warnings


def test_decide_unsure_when_no_candidate_is_clearly_l5s1() -> None:
    assert _decide([0.3, 0.4, 0.2]).status == "unsure"
    assert _decide([0.4, 0.2]).status == "unsure"


def test_decide_not_assessed_without_model_or_candidates() -> None:
    assert L.decide(_cands(1), _patches(1), None).status == "not_assessed"
    assert L.decide([], _patches(0), _Fixed([])).status == "not_assessed"


def test_drop_threshold_is_configurable() -> None:
    assert _decide([0.05, 0.97], L.LumbosacralConfig(drop_min_p=0.99)).status == "unsure"


def test_record_has_decision_and_applied_drop() -> None:
    rec = L.lumbosacral_record(_decide([0.05, 0.97]), 1)
    assert rec["status"] == "drop" and rec["applied_drop"] == 1 and rec["candidates"] == ["D0", "D1"]


# ------------------------------------------------------------------ packaged model and patches


def test_packaged_model_loads_and_scores_patches() -> None:
    model = L.load_model()
    assert model is not None, "lumbosacral_v1.json and .onnx must be packaged"
    assert model.onnx_path.stat().st_size <= 2 * 1024 * 1024
    assert 0.5 <= model.config.drop_min_p <= 1.0
    rng = np.random.default_rng(0)
    p = model.predict(rng.random((3, L.PATCH_MM, L.PATCH_MM)).astype(np.float32))
    assert p.shape == (3,) and np.all((p > 0) & (p < 1))


def test_model_missing_files_gives_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert L.load_model(str(tmp_path)) is None


def test_patch_grid_puts_disc_at_the_documented_position(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    g = seg.grids[mid_sagittal(seg, DEFAULT).k]  # type: ignore[union-attr]
    pg = L.patch_grid(g, 10.0, -5.0)
    assert pg.shape == (L.PATCH_MM, L.PATCH_MM) and pg.res_mm == 1.0
    i, j = pg.index_of_mm(10.0, -5.0)
    assert float(i) == pytest.approx(L.PATCH_ABOVE_MM - 0.5, abs=1e-6)
    assert float(j) == pytest.approx(L.PATCH_ANTERIOR_MM - 0.5, abs=1e-6)


def test_candidate_patches_bottom_up_and_normalised(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    an = anchors_from_segmentation(seg, DEFAULT)
    cands, patches = L.candidate_patches(seg, an, DEFAULT)
    assert len(cands) == min(L.K_CANDIDATES, len(an.discs))
    assert [c.rank for c in cands] == list(range(len(cands)))
    assert cands[0].label == an.disc_labels()[-1]
    assert patches.shape == (len(cands), L.PATCH_MM, L.PATCH_MM)
    assert patches.min() >= 0.0 and patches.max() <= 1.0 + 1e-6


def test_no_discs_gives_no_candidates(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    cands, patches = L.candidate_patches(seg, Anchors(method="test"), DEFAULT)
    assert cands == [] and patches.shape[0] == 0


def test_thin_sacral_disc_is_lowest_candidate_and_drop_removes_it(phantom: SpinePhantom) -> None:
    seg = _phantom_seg(phantom)
    an = anchors_from_segmentation(seg, DEFAULT)
    n_true = len(an.discs)
    _paint_sacral_remnant(seg, an)
    an2 = anchors_from_segmentation(seg, DEFAULT)
    assert len(an2.discs) == n_true + 1
    cands, patches = L.candidate_patches(seg, an2, DEFAULT)
    remnant_z = float(an2.discs[cands[0].label].center[2])
    l5s1_label = cands[1].label
    dec = L.decide(cands, patches, _Fixed([0.0, 0.99, 0.0]))
    assert dec.status == "drop" and dec.drop == 1
    mapping = L.drop_lowest(an2, dec.drop)
    assert all(abs(float(d.center[2]) - remnant_z) > 1.0 for d in an2.discs.values())
    assert len(an2.discs) == n_true
    assert mapping[l5s1_label] == "L5-S1"


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
    """Paint a thin disc 35 mm below the lowest disc on the mid-sagittal slice (the labeller's job; here analytic)."""
    mid = mid_sagittal(seg, DEFAULT)
    assert mid is not None
    k = mid.k
    grid = seg.grids[k]
    low = anchors.discs[anchors.disc_labels()[-1]]
    down, post = (float(v) for v in grid.mm_of(np.asarray(low.center))[:2])
    row = round(float(grid.index_of_mm(down + 35.0, post)[0]))
    col = round(float(grid.index_of_mm(down, post)[1]))
    seg.probs[k, DISC, row - 2 : row + 2, col - 12 : col + 13] = 0.95
    seg.probs[k, BODY, row - 2 : row + 2, col - 12 : col + 13] = 0.0


# ------------------------------------------------------------------ status caps


def test_status_caps_for_the_lumbosacral_decision() -> None:
    kw = {"sacrum_below": True, "applied": False, "second_extra": False}
    assert _apply_status_caps("confirmed", **kw, lumbo_status="keep") == "confirmed"
    assert _apply_status_caps("probable", **kw, lumbo_status="unsure") == "unconfirmed"
    assert _apply_status_caps("confirmed", **kw, lumbo_status="unsure") == "unconfirmed"
    assert _apply_status_caps("confirmed", **kw, lumbo_status="drop") == "probable"
    assert _apply_status_caps("probable", **kw, lumbo_status="not_assessed") == "probable"
    assert _apply_status_caps("unconfirmed", **kw, lumbo_status="unsure") == "unconfirmed"


def test_pipeline_switch_off_gives_no_lumbosacral_record(phantom_study) -> None:  # type: ignore[no-untyped-def]
    """``lumbosacral=False`` is the labeller-only "before"; the default records the decision."""
    from spinescope.anatomy.pipeline import label_spine_details

    off = label_spine_details(phantom_study, backend="ml", calibrate=False, lumbosacral=False)
    assert "lumbosacral" not in off.anchors.numbering
    on = label_spine_details(phantom_study, backend="ml", calibrate=False)
    rec = on.anchors.numbering.get("lumbosacral")
    assert rec is not None and rec["status"] in ("keep", "drop", "unsure", "not_assessed")
    assert rec["applied_drop"] == (rec["drop"] if rec["status"] == "drop" else 0)
