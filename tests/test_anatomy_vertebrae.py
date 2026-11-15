"""Vertebral geometry (anatomy/vertebrae.py, anatomy/canal_ml.py): synthetic label maps with known geometry.

A synthetic sagittal stack (9 slices, 1 mm grid, rows run inferior, columns posterior) carries three discs, three
bodies (the middle one with a 10-degree superior endplate), a canal tube of 10 mm AP and a posterior intensity step.
Every expected value is known analytically, so the tests check the measurements, not the labeller. A phantom test
checks the hook on a real labeller run.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.canal_ml import (
    ap_chord_mm,
    canal_geometry,
    centreline_from_points,
    largest_component,
    posterior_canal_edge,
    row_heights_mm,
)
from spinescope.anatomy.ml.grid import plane_grid
from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SegmentedStack
from spinescope.anatomy.vertebrae import (
    attach_geometry,
    body_name,
    body_plane_geometry,
    measure_geometry,
    pooled_body,
)
from spinescope.config import DEFAULT
from spinescope.geometry import Stack

N_SL = 9
ROWS = 140
COLS = 110
#: Disc bands (down coordinate, mm), bottom-up: D0 (L5-S1), D1 (L4-L5), D2 (L3-L4).
DISCS = {"L5-S1": (100.0, 108.0), "L4-L5": (60.0, 68.0), "L3-L4": (20.0, 28.0)}
CANAL_Y = (30.0, 39.0)  # posterior canal edges (mm), AP chord 10 mm
TILT = np.radians(10.0)


def _synthetic_seg(step_posterior: bool = False) -> SegmentedStack:
    """Segmentation of a synthetic sagittal stack with the layout described in the module docstring."""
    row_dir = np.array([0.0, 1.0, 0.0])  # columns increase posterior
    col_dir = np.array([0.0, 0.0, -1.0])  # rows increase inferior
    xs = np.array([-12.0 + 3.0 * k for k in range(N_SL)])
    ipp = np.stack([np.array([x, 0.0, 0.0]) for x in xs])
    vol = np.zeros((N_SL, ROWS, COLS), dtype=np.float32)
    if step_posterior:
        vol[:, :, 70:] = 0.8  # posterior soft tissue brighter than the bone-like column in front of it
        vol[:, :, :70] = 0.2
    stack = Stack(key="sag_synth", vol=vol, ipp=ipp, row_dir=row_dir, col_dir=col_dir, row_mm=1.0, col_mm=1.0)
    grids = [plane_grid(stack, k, 1.0) for k in range(N_SL)]
    probs = np.zeros((N_SL, 4, *grids[0].shape), dtype=np.float32)  # grids are padded to 16-pixel multiples
    for k, g in enumerate(grids):
        ii, jj = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]), indexing="ij")
        down, post = g.mm_of_index(ii, jj)
        x = float(stack.ipp[k, 0])
        body = np.zeros(g.shape, dtype=bool)
        d2 = DISCS["L3-L4"][1]
        # body 1: between D0 and D1, superior endplate tilted 10 degrees, posterior edge at the canal's
        top1 = DISCS["L4-L5"][1] + np.tan(TILT) * (post - 45.0)
        body[(post >= 30) & (post <= 60) & (down >= top1) & (down <= DISCS["L5-S1"][0])] = True
        # body 2: between D1 and D2, flat
        body[(post >= 30) & (post <= 60) & (down >= d2) & (down <= DISCS["L4-L5"][0])] = True
        # body 3: above D2 (top of the field), inferior endplate only
        body[(post >= 30) & (post <= 60) & (down >= 0) & (down <= DISCS["L3-L4"][0])] = True
        probs[k, BODY] = body.astype(np.float32)
        disc = np.zeros(g.shape, dtype=bool)
        for lo, hi in DISCS.values():
            disc |= (down >= lo) & (down <= hi) & (post >= 30) & (post <= 60)
        probs[k, DISC] = disc.astype(np.float32)
        canal = (np.abs(x) <= 8.0) & (post >= CANAL_Y[0]) & (post <= CANAL_Y[1]) & (down >= 0) & (down <= 120)
        probs[k, CANAL] = canal.astype(np.float32)
    seg = SegmentedStack(stack=stack, grids=grids, probs=probs, window=(0.0, 1.0))
    return seg


def _synthetic_anchors(seg: SegmentedStack) -> Anchors:
    a = Anchors(method="synthetic")
    for lab, (lo, hi) in DISCS.items():
        down_c = 0.5 * (lo + hi)
        a.discs[lab] = Disc(
            center=[0.0, 45.0, -down_c], normal=[0.0, 0.0, 1.0], height_mm=hi - lo, confidence=0.9
        )
    return a


# ----------------------------------------------------------------------------------------- body geometry
def test_body_plane_geometry_recovers_tilt_height_and_ap():
    seg = _synthetic_seg()
    grid = seg.grids[4]  # x = 0, the middle slice
    ii, jj = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
    down, _ = grid.mm_of_index(ii, jj)
    # body 1 (the tilted one): the body pixels between the two discs that bound it
    b1 = (seg.probs[4, BODY] > 0.5) & (down >= 62.0) & (down <= DISCS["L5-S1"][0])
    g = body_plane_geometry(largest_component(b1), grid, sup=True, inf=True, cfg=DEFAULT)
    assert g is not None
    assert g["sup"]["angle_deg"] == pytest.approx(10.0, abs=1.5)
    assert abs(g["inf"]["angle_deg"]) < 1.0
    # middle height ~ body height (32 mm) times cos(tilt)
    assert g["height_mm"]["middle"] == pytest.approx(32.0 * np.cos(TILT), abs=1.5)
    assert 28.0 < g["height_mm"]["anterior"] < 36.0 and 28.0 < g["height_mm"]["posterior"] < 36.0
    assert g["ap_mm"] == pytest.approx(31.0, abs=2.0)


def test_body_plane_geometry_rejects_fragments():
    seg = _synthetic_seg()
    grid = seg.grids[4]
    tiny = np.zeros(grid.shape, dtype=bool)
    tiny[50:60, 40:45] = True  # 5 mm AP
    assert body_plane_geometry(tiny, grid, sup=True, inf=True, cfg=DEFAULT) is None


def test_body_naming_follows_final_disc_labels():
    order = ["L5-S1", "L4-L5", "L3-L4"]  # bottom-up
    assert body_name(0, order) == "S1"
    assert body_name(1, order) == "L5"
    assert body_name(2, order) == "L4"
    assert body_name(3, order) == "L3"
    # after an offset correction the labels shift and so do the body names
    shifted = ["L4-L5", "L3-L4", "L2-L3"]
    assert body_name(1, shifted) == "L4"


def test_pooled_body_centre_and_width_from_all_slices():
    seg = _synthetic_seg()
    a = _synthetic_anchors(seg)
    from spinescope.anatomy.vertebrae import _slice_zones

    disc_z = np.sort([a.discs[lab].center[2] for lab in a.discs])
    zones = _slice_zones(seg, disc_z)
    pts, probs, _ids, counts = pooled_body(seg, zones, 1, DEFAULT)
    assert pts.shape[0] > 0 and probs.min() > 0.5
    assert len(counts) == N_SL
    # body 1 is symmetric in x, so the pooled centre sits at x = 0 and at its mid-height (z = -84)
    assert abs(pts[:, 0].mean()) < 0.5
    assert pts.mean(axis=0)[2] == pytest.approx(-84.0, abs=3.0)


# ----------------------------------------------------------------------------------------------- canal
def test_ap_chord_and_posterior_edge_on_a_tube():
    seg = _synthetic_seg()
    g = seg.grids[4]
    canal_mid = largest_component(seg.probs[4, CANAL] > 0.5)
    ap = ap_chord_mm(canal_mid, g, z_mm=-84.0, band_mm=2.0)
    assert ap == pytest.approx(10.0, abs=0.6)
    row = int(np.argmin(np.abs(row_heights_mm(g) + 84.0)))
    edge = posterior_canal_edge(canal_mid, row)
    assert edge is not None and abs(g.mm_of_index(row, edge)[1] - CANAL_Y[1]) < 1.0


def test_canal_geometry_levels_and_points():
    seg = _synthetic_seg()
    disc_pairs = [(lab, 0.5 * (lo + hi) * -1.0) for lab, (lo, hi) in DISCS.items()]
    body_mids = [("L5", -84.0, 4), ("L4", -44.0, 4)]
    c = canal_geometry(seg, 4, disc_pairs, [(n, z) for n, z, _ in body_mids], DEFAULT, confidence=1.0)
    assert len(c["points"]) >= 5
    xs = np.array([p[0] for p in c["points"]])
    ys = np.array([p[1] for p in c["points"]])
    assert np.all(np.abs(xs) < 1.0)
    assert np.all(np.abs(ys - 34.5) < 1.5)
    for lab in DISCS:
        assert c["per_level"][lab]["ap_mm"] == pytest.approx(10.0, abs=1.0)
        assert c["per_level"][lab]["kind"] == "disc"
    assert c["per_level"]["L5"]["kind"] == "mid_body"
    assert c["confidence"] > 0.9


def test_centreline_ignores_stray_lateral_pixels():
    # 30 canal points on slice 0 at x=0, and 2 stray points on slice 1 at x=20, same z bin
    pts = np.array([[0.0, 30.0 + i * 0.3, -50.0] for i in range(30)] + [[20.0, 90.0, -50.0]] * 2)
    ids = np.array([0] * 30 + [1] * 2, dtype=np.intp)
    cen = centreline_from_points(pts, -60.0, -40.0, DEFAULT, ids)
    # the stray slice holds 2/30 of the count (below the 0.3 fraction) and is dropped; but a bin needs two slices
    assert cen == []  # only one slice passes: the bin is not reliable
    pts2 = np.concatenate([pts, np.array([[1.0, 31.0, -50.0]] * 28)])
    ids2 = np.concatenate([ids, np.array([2] * 28, dtype=np.intp)])
    cen2 = centreline_from_points(pts2, -60.0, -40.0, DEFAULT, ids2)
    assert len(cen2) == 1 and abs(cen2[0][0] - 0.5) < 0.6 and cen2[0][1] < 60.0


def test_canal_geometry_without_points_returns_empty_centreline():
    seg = _synthetic_seg()
    seg.probs[:, CANAL] = 0.0
    c = canal_geometry(seg, 4, [("L5-S1", -104.0)], [], DEFAULT)
    assert c["points"] == []
    assert c["per_level"]["L5-S1"]["ap_mm"] is None


# ----------------------------------------------------------------------------------------- the hook
def test_measure_geometry_fills_bodies_canal_and_posterior():
    seg = _synthetic_seg(step_posterior=True)
    a = _synthetic_anchors(seg)
    measure_geometry(seg, a, DEFAULT)
    assert {"L5", "L4", "L3"} <= set(a.vertebrae)
    assert "S1" not in a.vertebrae  # no body class below the lowest disc in this layout
    b1 = a.vertebrae["L5"].body
    assert b1 is not None and b1["discs_below"] == 1
    assert b1["superior_endplate"] is not None and b1["inferior_endplate"] is not None
    assert b1["complete"] is True
    assert b1["superior_endplate"]["angle_deg"] == pytest.approx(10.0, abs=1.5)
    assert b1["centre"] is not None and abs(b1["centre"][0]) < 0.5
    assert b1["width_mm"] == pytest.approx(24.0, abs=1.5)  # body spans the nine slices, x from -12 to 12
    # the top body (above the highest disc) has one endplate only
    assert a.vertebrae["L3"].body["complete"] is False
    assert a.canal is not None and a.canal["per_level"]["L4-L5"]["ap_mm"] == pytest.approx(10.0, abs=1.0)
    pe = a.posterior_elements
    assert pe is not None and pe["limits"] and "lamina_proxy" in pe and "spinous_tip" in pe
    assert "L5" in pe["lamina_proxy"]
    tip = pe["spinous_tip"].get("L5")
    assert tip is not None and 26.0 <= tip["depth_behind_canal_mm"] <= 36.0
    assert tip["confidence"] <= 0.3


def test_measure_geometry_without_discs_warns_and_changes_nothing():
    seg = _synthetic_seg()
    a = Anchors(method="synthetic")
    measure_geometry(seg, a, DEFAULT)
    assert a.canal is None and a.posterior_elements is None and a.vertebrae == {}
    assert any("no discs" in w for w in a.warnings)


def test_attach_geometry_no_sagittal_is_a_no_op():
    from spinescope.study import Study

    a = Anchors(method="x")
    out = attach_geometry(Study(stacks={}), a)
    assert out is a and a.canal is None


def test_geometry_round_trips_through_json_and_old_documents_load():
    seg = _synthetic_seg()
    a = _synthetic_anchors(seg)
    measure_geometry(seg, a, DEFAULT)
    b = Anchors.from_json(a.to_json())
    assert b.vertebrae["L5"].body["superior_endplate"] == a.vertebrae["L5"].body["superior_endplate"]
    assert b.canal == a.canal and b.posterior_elements == a.posterior_elements
    # a 1.0 document (no geometry keys) still loads, with the new fields empty
    old = Anchors(method="old", discs={}, vertebrae={"L5": Vertebra(center=[0.0, 0.0, 0.0])})
    d = old.to_dict()
    for key in ("canal", "posterior_elements"):
        d.pop(key)
    d["schema_version"] = "1.0"
    d["vertebrae"]["L5"].pop("body")
    loaded = Anchors.from_dict(d)
    assert loaded.canal is None and loaded.vertebrae["L5"].body is None
    assert Anchors().schema_version == "1.1"


def test_hook_on_phantom_study_serialises(phantom_study):
    """The labeller runs on the phantom (out of distribution, so only the code path is checked, not the numbers)."""
    from _labels import label_spine_ml

    a = label_spine_ml(phantom_study)
    attach_geometry(phantom_study, a)
    d = Anchors.from_json(a.to_json())
    assert d.schema_version == "1.1"
    if a.canal is not None:
        assert isinstance(a.canal["points"], list)
    for v in a.vertebrae.values():
        if v.body is not None:
            assert v.body["level"] in a.vertebrae


def test_relabelling_carries_geometry_names():
    seg = _synthetic_seg(step_posterior=True)
    a = _synthetic_anchors(seg)
    measure_geometry(seg, a, DEFAULT)
    a.shift_labels(1, who="test", source="programmatic", why="shift")  # every level one cranial
    assert "L4-L5" in a.canal["per_level"] and "L5-S1" not in a.canal["per_level"]
    assert "L5" not in a.vertebrae and a.vertebrae["L4"].body["level"] == "L4"
    assert a.vertebrae["L3"].body["level"] == "L3"
    assert set(a.canal["per_level"]) <= set(a.discs) | set(a.vertebrae)
    assert set(a.posterior_elements["lamina_proxy"]) <= set(a.discs) | set(a.vertebrae)
