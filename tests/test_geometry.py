from __future__ import annotations

import numpy as np
import pytest

from spinescope.geometry import Stack, intersection_segment, obliquity_deg, plane_of, sample_plane, unit


def _random_stack(rng, n=7, rows=40, cols=50, key="s"):
    a = unit(rng.normal(size=3))
    b = unit(np.cross(a, rng.normal(size=3)))
    normal = np.cross(a, b)
    origin = rng.normal(scale=50, size=3)
    spacing = rng.uniform(2, 6)
    ipp = np.array([origin + i * spacing * normal for i in range(n)])
    vol = rng.normal(size=(n, rows, cols)).astype(np.float32)
    return Stack(
        key=key,
        vol=vol,
        ipp=ipp,
        row_dir=a,
        col_dir=b,
        row_mm=rng.uniform(0.3, 1.5),
        col_mm=rng.uniform(0.3, 1.5),
        thickness=spacing * 0.8,
    )


def test_round_trip_random_orientations():
    rng = np.random.default_rng(0)
    for _ in range(20):
        st = _random_stack(rng)
        for _ in range(20):
            k, r, c = rng.uniform(0, st.n - 1), rng.uniform(0, 39), rng.uniform(0, 49)
            xyz = st.pix_to_lps(k, r, c)
            k2, r2, c2 = st.lps_to_pix(xyz)
            assert np.allclose([k, r, c], [k2, r2, c2], atol=1e-6)


def test_slices_sorted_along_normal_and_positions():
    rng = np.random.default_rng(1)
    st = _random_stack(rng)
    shuffled = Stack(
        key="x",
        vol=st.vol[::-1],
        ipp=st.ipp[::-1],
        row_dir=st.row_dir,
        col_dir=st.col_dir,
        row_mm=st.row_mm,
        col_mm=st.col_mm,
    )
    assert np.all(np.diff(shuffled.positions) > 0)
    assert np.allclose(shuffled.vol, st.vol)


def test_lps_to_pix_does_not_clamp_and_contains():
    rng = np.random.default_rng(2)
    st = _random_stack(rng)
    far = st.pix_to_lps(st.n - 1, 10, 10) + 5 * st.spacing * st.normal
    k, _, _ = st.lps_to_pix(far)
    assert k == pytest.approx(st.n - 1 + 5, abs=1e-6)
    assert not st.contains(far)
    assert st.contains(st.pix_to_lps(2, 5, 5))


def test_nearest_slice_and_plane_helpers():
    st = Stack(
        key="ax",
        vol=np.zeros((5, 10, 10)),
        ipp=[[0, 0, z] for z in (0, 4, 8, 12, 16)],
        row_dir=(1, 0, 0),
        col_dir=(0, 1, 0),
        row_mm=1,
        col_mm=1,
    )
    assert st.plane == "ax" and st.obliquity == pytest.approx(0)
    k, d = st.nearest_slice([3, 3, 9.0])
    assert (k, d) == (2, pytest.approx(1.0))
    assert plane_of((0.99, 0.1, 0)) == "sag"
    assert obliquity_deg((0, np.sin(np.radians(20)), np.cos(np.radians(20)))) == pytest.approx(20)


def test_intersection_axial_on_sagittal_is_horizontal():
    sag = Stack(
        key="sag",
        vol=np.zeros((3, 100, 100)),
        ipp=[[x, -50, 50] for x in (-5, 0, 5)],
        row_dir=(0, 1, 0),
        col_dir=(0, 0, -1),
        row_mm=1,
        col_mm=1,
    )
    ax = Stack(
        key="ax",
        vol=np.zeros((3, 100, 100)),
        ipp=[[-50, -50, z] for z in (0, 10, 20)],
        row_dir=(1, 0, 0),
        col_dir=(0, 1, 0),
        row_mm=1,
        col_mm=1,
    )
    seg = intersection_segment(sag, 1, ax, 1)  # axial z=10 -> sagittal row 40
    assert seg is not None
    (r0, c0), (r1, c1) = seg
    assert r0 == pytest.approx(40) and r1 == pytest.approx(40)
    assert {round(c0, 3), round(c1, 3)} == {-0.5, 99.5}
    assert intersection_segment(sag, 0, sag, 1) is None  # parallel


def test_sample_plane_reproduces_native_slice():
    rng = np.random.default_rng(3)
    vol = rng.normal(size=(4, 30, 30)).astype(np.float32)
    st = Stack(
        key="s",
        vol=vol,
        ipp=[[0, 0, z] for z in (0, 3, 6, 9)],
        row_dir=(1, 0, 0),
        col_dir=(0, 1, 0),
        row_mm=1,
        col_mm=1,
        thickness=3,
    )
    centre = st.pix_to_lps(2, 14.5, 14.5)
    img = sample_plane(st, centre, (1, 0, 0), (0, 1, 0), (30, 30), 1.0)
    assert np.allclose(img, vol[2], atol=1e-5)
    off = sample_plane(st, centre + np.array([0, 0, 100]), (1, 0, 0), (0, 1, 0), (10, 10), 1.0)
    assert np.isnan(off).all()
