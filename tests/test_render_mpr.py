from __future__ import annotations

import logging

import numpy as np
import pytest

from spinescope.anatomy import Anchors
from spinescope.anatomy.anchors import Disc
from spinescope.config import Config
from spinescope.render import (
    curved_coronal,
    disc_plane,
    effective_resolution,
    reformat,
    render_reformat,
    resample_onto,
    sample_points,
    series_window,
)
from spinescope.render.mpr import curved_centreline
from spinescope.testing.phantom import PhantomSpec, SpinePhantom


@pytest.fixture(scope="module")
def clean():
    """Noise-free phantom, so sampled values can be compared with the analytic tissue intensity."""
    return SpinePhantom(PhantomSpec(noise=0.0, degenerated={"L4-L5": 0.2}))


@pytest.fixture(scope="module")
def clean_stacks(clean):
    """In-memory renders with the standard study's geometry. (The DICOM writer rescales intensities to
    uint16, so analytic comparisons must use the phantom's own units.)"""
    lumbar_mid = (clean.discs["L1-L2"]["center"] + clean.discs["L5-S1"]["center"]) / 2
    ax = clean.render(
        "ax",
        "t2",
        n=36,
        spacing=5.0,
        thickness=4.0,
        rows=128,
        cols=128,
        px=1.4,
        center=lumbar_mid + np.array([0, 10.0, 0]),
        key="ax_t2",
    )
    sag = clean.render(
        "sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2, key="sag_t2"
    )
    return {"ax_t2": ax, "sag_t2": sag}


@pytest.fixture(scope="module")
def clean_anchors(clean):
    return Anchors.from_truth(clean.truth)


def test_sample_points_matches_analytic_phantom(clean, clean_stacks):
    st = clean_stacks["ax_t2"]
    body = np.asarray(clean.bodies["L3"]["center"])
    k, _ = st.nearest_slice(body)
    pts = st.pix_to_lps_many(k, np.array([[60.0, 60.0], [64.0, 70.0]])).reshape(2, 1, 3)
    vals = sample_points(st, pts)
    assert vals.shape == (2, 1)
    expect = clean.intensity(clean.labels(pts.reshape(-1, 3)), "t2", pts.reshape(-1, 3))
    assert np.allclose(vals.ravel(), expect, atol=1e-3)


def test_sample_points_nan_outside_fov(clean_stacks):
    st = clean_stacks["ax_t2"]
    far = st.corners(0).mean(0) + np.array([500.0, 0.0, 0.0])
    assert np.isnan(sample_points(st, far))


def test_coronal_from_axial_reads_the_marrow(clean, clean_stacks):
    st = clean_stacks["ax_t2"]
    y = float(clean.bodies["L3"]["center"][1])
    ref = reformat(st, "cor", y)
    assert ref.img.shape[0] > 0 and ref.img.shape[1] > 0
    target = np.asarray(clean.bodies["L3"]["center"])
    x, yy, _ = ref.frame.to_px(target)
    row, col = round(float(yy)), round(float(x))
    val = float(ref.img[row, col])
    expect = float(clean.intensity(clean.labels(target[None]), "t2")[0])
    assert val == pytest.approx(expect, abs=1e-3)
    assert ref.frame.inside(x, yy)


def test_reformat_effective_resolution_is_source_limited(clean_stacks):
    ax = clean_stacks["ax_t2"]  # 1.4 mm in plane, 4 mm slices 5 mm apart
    cor = reformat(ax, "cor", 0.0)
    h, v = cor.eff_res_mm
    assert h == pytest.approx(ax.row_mm, abs=0.05)  # in plane, along x
    assert v == pytest.approx(ax.thickness, abs=0.05)  # through-plane blur along z
    sag = reformat(ax, "sag", 0.0)
    assert sag.eff_res_mm[1] == pytest.approx(ax.thickness, abs=0.05)
    assert "eff. res" in sag.note


def test_effective_resolution_never_better_than_source_or_sampling(clean_stacks):
    st = clean_stacks["sag_t2"]
    h, v = effective_resolution(st, (1, 0, 0), (0, 0, -1), 0.1)
    assert h >= st.col_mm - 1e-9 or h >= 0.1
    assert v >= 0.1


def test_disc_plane_centre_and_orientation(clean_anchors, clean_stacks):
    st = clean_stacks["sag_t2"]
    ref = disc_plane(st, clean_anchors, "L4-L5")
    assert ref is not None
    x, y, off = ref.frame.to_px(clean_anchors.disc("L4-L5").center)
    assert (x, y) == pytest.approx(((ref.img.shape[1] - 1) / 2, (ref.img.shape[0] - 1) / 2), abs=1e-6)
    assert abs(off) < 1e-6
    assert ref.frame.u[0] > 0.9  # image right is patient left (+x)
    assert ref.frame.v[1] > 0.5  # image down is posterior (+y)
    assert abs(ref.frame.normal @ np.asarray(clean_anchors.disc("L4-L5").normal)) == pytest.approx(
        1.0, abs=1e-6
    )


def test_disc_plane_missing_label_is_none(clean_anchors, clean_stacks, caplog):
    with caplog.at_level(logging.WARNING):
        assert disc_plane(clean_stacks["sag_t2"], clean_anchors, "L9-S9") is None
    assert "L9-S9" in caplog.text


def test_disc_plane_uses_config_size(clean_anchors, clean_stacks):
    ref = disc_plane(clean_stacks["sag_t2"], clean_anchors, "L3-L4", size_mm=30.0, px_mm=1.0)
    assert ref is not None and ref.img.shape == (30, 30)


def test_curved_centreline_is_ordered_and_posterior_to_discs(clean_anchors):
    pts = curved_centreline(clean_anchors, Config())
    assert pts is not None
    assert np.all(np.diff(pts[:, 2]) > 0)  # inferior to superior
    cfg = Config()
    # the centre sits posterior to the posterior disc margins by the configured offsets
    lowest = clean_anchors.disc_labels()[-1]
    d = clean_anchors.disc(lowest)
    assert (pts[0] - np.asarray(d.posterior)) @ d.ap_dir == pytest.approx(
        cfg.sacral_canal_offset_mm, abs=1e-6
    )


def test_curved_coronal_follows_canal(clean_anchors, clean_stacks):
    st = clean_stacks["ax_t2"]
    ref = curved_coronal(st, clean_anchors, half_width_mm=25.0)
    assert ref is not None
    img = ref.img
    centre_col = img[:, img.shape[1] // 2]
    lateral = img[:, 0]
    # CSF is the brightest T2 tissue; the centre column follows the thecal sac
    assert np.nanmean(centre_col) > 1000.0
    assert np.nanmean(centre_col) > np.nanmean(lateral) * 1.5


def test_curved_coronal_needs_two_discs(clean_stacks, clean):
    one = Anchors(discs={"L4-L5": Disc(center=[0, 0, 0], normal=[0, 0, 1], posterior=[0, 10, 0])})
    assert curved_coronal(clean_stacks["ax_t2"], one) is None


def test_render_reformat_is_rgb_with_title(clean_stacks):
    st = clean_stacks["sag_t2"]
    ref = reformat(st, "ax", float(st.ipp[0][2]))
    out = render_reformat(ref, series_window(st))
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    assert out.shape[0] > ref.img.shape[0]  # title band and rulers added


def test_resample_onto_matches_source_geometry(clean_stacks):
    ax = clean_stacks["ax_t2"]
    sag = clean_stacks["sag_t2"]
    k = sag.n // 2
    on = resample_onto(ax, sag, k)
    assert on.shape == (sag.vol.shape[1], sag.vol.shape[2])
    r, c = 60, 70
    p = sag.pix_to_lps(k, r, c)
    assert on[r, c] == pytest.approx(float(sample_points(ax, p)), abs=1e-3)


def test_reformat_rejects_unknown_plane(clean_stacks):
    with pytest.raises(ValueError):
        reformat(clean_stacks["sag_t2"], "oblique", 0.0)


def test_reformat_outside_stack_warns(clean_stacks, caplog):
    with caplog.at_level(logging.WARNING):
        ref = reformat(clean_stacks["sag_t2"], "cor", 500.0)
    assert np.isnan(ref.img).all()
    assert "outside the stack" in caplog.text


def test_config_offsets_change_centreline(clean_anchors):
    base = curved_centreline(clean_anchors, Config())
    shifted = curved_centreline(clean_anchors, Config(thecal_sac_offset_mm=6.0))
    assert base is not None and shifted is not None
    assert not np.allclose(base, shifted)
