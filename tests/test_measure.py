from __future__ import annotations

import numpy as np
import pytest

from spinescope.geometry import Stack
from spinescope.measure import angle_deg, dist_lps, dist_mm, edge_to_edge, poly_stats, profile, roi_stats


def _stack(vol: np.ndarray, row_mm: float = 1.0, col_mm: float = 1.0) -> Stack:
    return Stack(
        key="t",
        vol=vol,
        ipp=np.zeros((vol.shape[0], 3)),
        row_dir=np.array([0.0, 1.0, 0.0]),
        col_dir=np.array([0.0, 0.0, -1.0]),
        row_mm=row_mm,
        col_mm=col_mm,
        thickness=1.0,
    )


def test_dist_mm_uses_anisotropic_pixels():
    st = _stack(np.zeros((2, 10, 10)), row_mm=0.5, col_mm=2.0)
    assert dist_mm(st, (0, 0), (0, 3)) == pytest.approx(6.0)  # 3 columns of 2 mm
    assert dist_mm(st, (0, 0), (4, 0)) == pytest.approx(2.0)  # 4 rows of 0.5 mm
    assert dist_mm(st, (1, 1), (5, 4)) == pytest.approx(np.hypot(4 * 0.5, 3 * 2.0))


def test_dist_lps():
    assert dist_lps([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_angle_deg_perpendicular_parallel_and_mm_scaling():
    assert angle_deg(((0, 0), (0, 1)), ((0, 0), (1, 0))) == pytest.approx(90.0)
    assert angle_deg(((0, 0), (0, 1)), ((5, 5), (5, 9))) == pytest.approx(0.0)
    # with a stack, the pixel axes are scaled to mm first: a 45 deg line stays 45 deg only if spacing is equal
    st = _stack(np.zeros((1, 4, 4)), row_mm=1.0, col_mm=2.0)
    assert angle_deg(((0, 0), (1, 1)), ((0, 0), (0, 1)), st) == pytest.approx(
        np.degrees(np.arctan2(1.0, 2.0))
    )
    with pytest.raises(ValueError):
        angle_deg(((1, 1), (1, 1)), ((0, 0), (0, 1)))


def test_profile_positions_and_values_on_a_ramp():
    vol = np.tile(np.arange(20, dtype=float), (1, 20, 1))  # intensity = column index
    st = _stack(vol)
    s, vals, xyz = profile(st, 0, (5, 2), (5, 12), step_mm=1.0)
    assert s[0] == 0 and s[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(vals, np.linspace(2, 12, len(s)), atol=1e-9)
    assert xyz.shape == (len(s), 3)
    with pytest.raises(ValueError):
        profile(st, 0, (3, 3), (3, 3))


def test_profile_width_averages_across_the_band():
    vol = np.zeros((1, 20, 20))
    vol[0, 9:11, :] = 100.0  # a bright two-row stripe
    st = _stack(vol)
    _, narrow, _ = profile(st, 0, (10, 2), (10, 17), width_mm=0.0, step_mm=1.0)
    _, wide, _ = profile(st, 0, (10, 2), (10, 17), width_mm=3.0, step_mm=1.0)
    assert narrow.max() == pytest.approx(100.0)
    assert wide.max() < narrow.max()


def test_roi_stats_circle_area_and_mean():
    vol = np.full((1, 40, 40), 10.0)
    vol[0, 20, 20] = 50.0
    st = _stack(vol, row_mm=0.5, col_mm=0.5)
    s = roi_stats(st, 0, (20, 20), radius_mm=2.0)
    assert s["n"] > 0 and s["median"] == pytest.approx(10.0)
    assert s["area_mm2"] == pytest.approx(s["n"] * 0.25)
    assert s["area_mm2"] == pytest.approx(np.pi * 4.0, rel=0.1)
    assert s["mean"] > 10.0  # the bright centre pixel is inside


def test_poly_stats_rectangle_area_and_mean():
    vol = np.zeros((1, 30, 30))
    vol[0, 5:15, 5:25] = 7.0
    st = _stack(vol, row_mm=2.0, col_mm=1.0)
    s = poly_stats(st, 0, [(4.5, 4.5), (4.5, 24.5), (14.5, 24.5), (14.5, 4.5)])
    assert s["mean"] == pytest.approx(7.0)
    assert s["area_mm2"] == pytest.approx(s["n"] * 2.0)
    assert s["n"] == pytest.approx(10 * 20, rel=0.05)


def test_edge_to_edge_finds_the_two_margins_of_a_dark_band():
    vol = np.full((1, 10, 60), 100.0)
    vol[0, :, 20:40] = 10.0  # dark band 20 mm wide at 1 mm per column
    st = _stack(vol)
    res = edge_to_edge(st, 0, (5, 5), (5, 55), min_sep_mm=2.0)
    assert res["ok"]
    assert res["dist_mm"] == pytest.approx(20.0, abs=1.0)
    e = res["edges"]
    assert e[0]["s_mm"] < e[1]["s_mm"]


def test_edge_to_edge_reports_missing_edges():
    st = _stack(np.full((1, 10, 30), 5.0))
    res = edge_to_edge(st, 0, (5, 2), (5, 27))
    assert not res["ok"] and np.isnan(res["dist_mm"]) and res["edges"] == []


def test_phantom_profile_along_disc_normal_shows_nucleus(phantom, phantom_study):
    d = phantom.truth["discs"]["L2-L3"]
    st = phantom_study["sag_t2"]
    c, n = np.asarray(d["center"]), np.asarray(d["normal"])
    k, _, _ = st.lps_to_pix(c)
    p0 = st.lps_to_pix(c - 12 * n)[1:]
    p1 = st.lps_to_pix(c + 12 * n)[1:]
    s, vals, _ = profile(st, round(k), p0, p1, step_mm=0.5)
    centre = vals[np.argmin(np.abs(s - 12.0))]
    ends = np.r_[vals[:3], vals[-3:]].mean()  # marrow, 12 mm either side of the disc centre
    assert centre > 1.5 * ends
