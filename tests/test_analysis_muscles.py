from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _alignment_truth import with_body_geometry
from _studies import flip_rows
from spinescope.analysis import muscles as M
from spinescope.analysis.base import Result, available, run
from spinescope.anatomy import Anchors
from spinescope.anatomy.correspondence import level_map
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.study import Study, from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

LEVELS = ("L3-L4", "L4-L5", "L5-S1")


def _axial_study(spec: PhantomSpec, rows: int = 128) -> tuple[SpinePhantom, Study, Anchors]:
    """Phantom with sagittal T1/T2 (for the labelling geometry) and axial T1/T2 covering the lumbar discs."""
    ph = SpinePhantom(spec)
    sag = [
        ph.render("sag", c, n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
        for c in ("t2", "t1")
    ]
    centre = (ph.discs["L1-L2"]["center"] + ph.discs["L5-S1"]["center"]) / 2 + np.array([0.0, 10.0, 0.0])
    ax = [
        ph.render("ax", c, n=36, spacing=5.0, thickness=4.0, rows=rows, cols=rows, px=1.4, center=centre)
        for c in ("t2", "t1")
    ]
    for st, key in zip(ax, ("ax_t2", "ax_t1"), strict=True):
        st.key = key
    study = from_stacks(sag + ax)
    anchors = with_body_geometry(ph, study, Anchors.from_truth(ph.truth))
    return ph, study, anchors


@pytest.fixture(scope="module")
def normal() -> tuple[SpinePhantom, Study, Anchors, Result]:
    ph, study, anchors = _axial_study(PhantomSpec())
    return ph, study, anchors, run("muscles", study, anchors)


@pytest.fixture(scope="module")
def fatty_left() -> tuple[SpinePhantom, Study, Anchors, Result]:
    spec = PhantomSpec(paraspinal_detail=True, multifidus_fat={"L": 0.4})
    ph, study, anchors = _axial_study(spec)
    return ph, study, anchors, run("muscles", study, anchors)


def _rows(
    res: Result, level: str, series: str = "ax_t2", scope: str = "muscle"
) -> dict[tuple[str, str], dict]:
    return {
        (r["muscle"], r["side"]): r
        for r in res.rows
        if r.get("scope") == scope and r.get("level") == level and r.get("series") == series
    }


# ------------------------------------------------------------------------------------ pure helpers
@pytest.mark.parametrize(
    ("fat", "grade"),
    [(0.0, 0), (0.0999, 0), (0.10, 1), (0.30, 1), (0.50, 1), (0.5001, 2), (0.95, 2)],
)
def test_kjaer_grade_boundaries(fat: float, grade: int) -> None:
    assert M.kjaer_grade(fat) == grade


def test_kjaer_bounds_come_from_config() -> None:
    assert DEFAULT.muscles_kjaer_bounds == (0.10, 0.50)


def test_perpendicular_area_is_cosine_of_the_slab_tilt() -> None:
    assert M.perpendicular_area(100.0, 0.0) == pytest.approx(100.0)
    assert M.perpendicular_area(100.0, 60.0) == pytest.approx(50.0)
    assert M.perpendicular_area(100.0, -60.0) == pytest.approx(50.0)  # sign of the tilt does not matter


def test_asymmetry_is_percent_of_mean_and_nan_safe() -> None:
    assert M.asymmetry_pct(110.0, 90.0) == pytest.approx(20.0)
    assert M.asymmetry_pct(90.0, 110.0) == pytest.approx(-20.0)
    assert math.isnan(M.asymmetry_pct(float("nan"), 90.0))
    assert math.isnan(M.asymmetry_pct(0.0, 0.0))


def _frame(lat: np.ndarray, ap: np.ndarray, hw: float = 22.0, hd: float = 16.0) -> M.LevelFrame:
    return M.LevelFrame(
        lat=lat, ap=ap, row_mm=1.0, col_mm=1.0, half_width_mm=hw, half_depth_mm=hd, measured=True
    )


def test_windows_put_multifidus_on_the_patient_left_for_positive_lateral_offsets() -> None:
    yy, xx = np.mgrid[0:120, 0:120].astype(float)
    frame = _frame(lat=xx - 60.0, ap=yy - 40.0)  # lat > 0 is patient left
    win = M.muscle_windows(frame, DEFAULT)
    assert win[("MF", "L")].any() and win[("MF", "R")].any()
    assert not (win[("MF", "L")] & win[("MF", "R")]).any()
    # the left window lies at positive lateral offsets only
    assert (frame.lat[win[("MF", "L")]] > 0).all()
    assert (frame.lat[win[("MF", "R")]] < 0).all()
    # the multifidus is medial to the erector spinae, the psoas is anterior to the erector spinae
    lat_mf = np.abs(frame.lat[win[("MF", "L")]]).mean()
    lat_es = np.abs(frame.lat[win[("ES", "L")]]).mean()
    assert lat_mf < lat_es
    ap_ps = frame.ap[win[("PS", "L")]].mean()
    ap_es = frame.ap[win[("ES", "L")]].mean()
    assert ap_ps < ap_es


def test_body_outline_is_the_tissue_ellipse_not_the_air() -> None:
    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    ell = ((xx - 100) / 60.0) ** 2 + ((yy - 100) / 45.0) ** 2 <= 1.0
    img = np.where(ell, 400.0, 0.0) + 2.0 * np.random.default_rng(0).standard_normal(ell.shape)
    body = M.body_outline(img, DEFAULT.muscles_body_threshold_rel)
    assert abs(int(body.sum()) - int(ell.sum())) / ell.sum() < 0.12  # edge blur: about 10 percent larger


def test_largest_component_drops_small_islands() -> None:
    m = np.zeros((20, 20), dtype=bool)
    m[2:10, 2:10] = True
    m[15, 15] = True
    out = M.largest_component(m)
    assert out[5, 5] and not out[15, 15]


def test_slice_measure_splits_fat_from_lean_on_a_synthetic_slice() -> None:
    """Synthetic slice: air outside r=70 mm, subcutaneous fat ring 60-70 mm, soft tissue inside, a fat band in the
    lateral multifidus. The band's share of the multifidus outline is the fat fraction."""
    yy, xx = np.mgrid[0:160, 0:160].astype(float)
    frame = _frame(lat=(xx - 80.0) * 1.0, ap=(yy - 80.0) * 1.0, hw=22.0, hd=16.0)
    r = np.hypot(frame.lat, frame.ap)
    img = np.zeros((160, 160))
    img[r < 70.0] = 420.0  # soft tissue
    img[(r >= 60.0) & (r < 70.0)] = 760.0  # subcutaneous fat ring
    mf_l = M.muscle_windows(frame, DEFAULT)[("MF", "L")]
    band = mf_l & (frame.lat > 14.0) & (r < 60.0)
    img[band] = 760.0
    meas = M.measure_slice(img, frame, DEFAULT)
    assert meas.reference in ("psoas", "erector spinae")
    roi = meas.masks[("MF", "L")]
    assert roi.sum() > 100
    frac = meas.fat[("MF", "L")].sum() / roi.sum()
    assert frac == pytest.approx((band & roi).sum() / roi.sum(), abs=0.02)
    assert 0.1 < frac < 0.9
    assert meas.fat[("MF", "R")].sum() == 0


def test_overlay_is_rgb_uint8() -> None:
    img = np.random.default_rng(0).uniform(0, 255, (60, 60))
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    out = M._overlay(img, {("MF", "L"): mask})
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3


# ------------------------------------------------------------------------------------ registration and warnings
def test_registered_as_builtin() -> None:
    assert "muscles" in available()


def test_no_axial_stack_is_a_warning_not_an_error() -> None:
    from spinescope.testing.phantom import SpinePhantom as SP

    ph = SP(PhantomSpec())
    from _studies import phantom_study

    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study, Anchors.from_truth(ph.truth))
    res = run("muscles", study, anchors)
    assert res.rows == []
    assert any("no axial stack" in w for w in res.warnings)


def test_no_discs_is_a_warning(normal: tuple) -> None:
    _ph, study, _anchors, _res = normal
    res = M.analyze(study, Anchors())
    assert any("no discs" in w or "no axial" in w for w in res.warnings)


# ------------------------------------------------------------------------------------ phantom measurements
def test_every_level_with_an_axial_slab_is_measured(normal: tuple) -> None:
    _ph, _study, _anchors, res = normal
    assert not [w for w in res.warnings if "not assessed" in w]
    for lev in LEVELS:
        rows = _rows(res, lev)
        assert set(rows) == {(m, s) for m in M.MUSCLES for s in M.SIDES}
        assert all(r["value"] == "measured" for r in rows.values())


def test_normal_phantom_has_little_fat_and_symmetric_muscles(normal: tuple) -> None:
    _ph, _study, _anchors, res = normal
    for lev in LEVELS:
        rows = _rows(res, lev)
        for mu in M.MUSCLES:
            left, right = rows[(mu, "L")], rows[(mu, "R")]
            assert left["fat_fraction_pct"] < 5.0
            assert right["fat_fraction_pct"] < 5.0
            assert abs(M.asymmetry_pct(left["csa_mm2"], right["csa_mm2"])) < 5.0
        assert rows[("MF", "L")]["kjaer_grade"] == 0


def test_fatty_left_multifidus_is_graded_on_the_left_only(fatty_left: tuple) -> None:
    _ph, _study, _anchors, res = fatty_left
    rows = _rows(res, "L4-L5")
    left, right = rows[("MF", "L")], rows[("MF", "R")]
    assert left["fat_fraction_pct"] > 20.0
    assert right["fat_fraction_pct"] < 5.0
    assert left["kjaer_grade"] == 1
    assert right["kjaer_grade"] == 0
    assert any(f["kind"] == "multifidus_fat" and f["side"] == "L" for f in res.flags)
    assert not any(f["kind"] == "multifidus_fat" and f["side"] == "R" for f in res.flags)


def test_fat_fraction_tracks_the_phantom_band(fatty_left: tuple) -> None:
    """The phantom draws a lateral band that covers a stated share of the multifidus window; the estimate follows it."""
    ph, study, anchors, res = fatty_left
    stack = study.stacks["ax_t2"]
    lm = level_map(study, anchors, measure_geometry=False)
    ls = M.select_axial_slices(lm, "L4-L5")[0]
    centre = np.asarray(lm.discs["L4-L5"].centre)
    hw, hd, meas_ok = M._body_dims(lm.discs["L4-L5"])
    frame = M.level_frame(stack, ls.k, centre, anchors.midline_x, hw, hd, meas_ok)
    win = M.muscle_windows(frame, DEFAULT)[("MF", "L")]
    truth_band = _truth_fat_band(ph, stack, ls.k, side="L")
    band = truth_band & win
    est = _rows(res, "L4-L5")[("MF", "L")]["fat_fraction_pct"] / 100.0
    real = band.sum() / max(win.sum(), 1)
    assert abs(est - real) < 0.15


def _truth_fat_band(ph: SpinePhantom, stack: Stack, k: int, side: str) -> np.ndarray:
    """Pixels of slice k that the phantom draws as the fatty band of the multifidus (from the phantom's own labels)."""
    h, w = stack.vol.shape[1:]
    rr, cc = np.mgrid[0:h, 0:w]
    pts = stack.pix_to_lps_many(float(k), np.column_stack([rr.ravel(), cc.ravel()]).astype(float))
    cz = ph.canal_line
    y_canal = np.interp(pts[:, 2], cz[::-1, 2], cz[::-1, 1])
    sx = 1.0 if side == "L" else -1.0
    mf = ((pts[:, 0] - sx * 12.0) / 9.0) ** 2 + ((pts[:, 1] - (y_canal + 26.0)) / 20.0) ** 2 <= 1.0
    u = sx * (pts[:, 0] - sx * 12.0) / 9.0
    return (mf & (u > 1.0 - 2.0 * ph.spec.multifidus_fat[side])).reshape(h, w)


def test_erector_and_psoas_are_not_graded(normal: tuple) -> None:
    _ph, _study, _anchors, res = normal
    for r in res.rows:
        if r.get("scope") == "muscle" and r.get("muscle") != "MF":
            assert "kjaer_grade" not in r


def test_oblique_slab_corrects_csa_and_lowers_confidence(normal: tuple) -> None:
    _ph, study, anchors, res = normal
    lm = level_map(study, anchors, measure_geometry=False)
    rows = {lev: _rows(res, lev) for lev in LEVELS}
    tilts = {lev: lm.discs[lev].stacks["ax_t2"].tilt_deg for lev in LEVELS}
    steep = max(LEVELS, key=lambda lev: tilts[lev])
    flat = min(LEVELS, key=lambda lev: tilts[lev])
    assert tilts[steep] > DEFAULT.muscles_oblique_tilt_max_deg
    r_steep = rows[steep][("ES", "L")]
    assert r_steep["csa_mm2"] == pytest.approx(
        r_steep["csa_slab_mm2"] * math.cos(math.radians(tilts[steep])), rel=1e-3
    )
    assert "oblique" in r_steep["reason"]
    assert r_steep["confidence"] < rows[flat][("ES", "L")]["confidence"]


def test_normalised_csa_is_the_slab_area_ratio(normal: tuple) -> None:
    _ph, _study, _anchors, res = normal
    for lev in LEVELS:
        r = _rows(res, lev)[("ES", "R")]
        assert r["normalised_csa"] == pytest.approx(
            r["csa_slab_mm2"] / (r["body_csa_mm2"] / math.cos(math.radians(r["tilt_deg"]))), rel=2e-3
        )
        assert 0.0 < r["normalised_csa"] < 2.0


def test_confidence_is_capped_and_no_meta_key_is_private(normal: tuple) -> None:
    _ph, _study, _anchors, res = normal
    assert all(
        r["confidence"] <= DEFAULT.muscles_max_confidence + 1e-9 for r in res.rows if r.get("confidence")
    )
    assert not any(str(k).startswith("_") for k in res.meta)


def test_result_is_deterministic(normal: tuple) -> None:
    _ph, study, anchors, res = normal
    again = run("muscles", study, anchors)
    assert [r.get("fat_fraction_pct") for r in again.rows] == [r.get("fat_fraction_pct") for r in res.rows]


def test_figures_are_saved_per_level(normal: tuple) -> None:
    _ph, _study, _anchors, res = normal
    assert {f"ax_{lev}" for lev in LEVELS} <= set(res.figures)


# ------------------------------------------------------------------------------------ side conventions
def _flip_cols(stack: Stack) -> Stack:
    cols = stack.vol.shape[2]
    return Stack(
        key=stack.key,
        vol=stack.vol[:, :, ::-1].copy(),
        ipp=stack.ipp + (cols - 1) * stack.col_mm * stack.row_dir,
        row_dir=-stack.row_dir,
        col_dir=stack.col_dir,
        row_mm=stack.row_mm,
        col_mm=stack.col_mm,
        thickness=stack.thickness,
        description=stack.description,
        meta=dict(stack.meta),
    )


def _flipped(study: Study, how: str) -> Study:
    out = {}
    for key, st in study.stacks.items():
        if st.plane != "ax":
            out[key] = st
        elif how == "rows":
            out[key] = flip_rows(st)
        else:
            out[key] = _flip_cols(st)
    return Study(stacks=out, info=dict(study.info))


@pytest.mark.parametrize("how", ["rows", "cols"])
def test_flipped_acquisition_keeps_patient_sides(fatty_left: tuple, how: str) -> None:
    """Reversing the row or column order of the axial stacks changes pixels, not patient geometry. Same answers."""
    ph, study, _anchors, res = fatty_left
    flipped_study = _flipped(study, how)
    anchors_f = with_body_geometry(ph, flipped_study, Anchors.from_truth(ph.truth))
    res_f = run("muscles", flipped_study, anchors_f)
    a = _rows(res, "L4-L5")
    b = _rows(res_f, "L4-L5")
    for key in a:
        assert b[key]["fat_fraction_pct"] == pytest.approx(a[key]["fat_fraction_pct"], abs=2.0), key
        assert b[key]["csa_mm2"] == pytest.approx(a[key]["csa_mm2"], rel=0.05), key
    assert b[("MF", "L")]["fat_fraction_pct"] > b[("MF", "R")]["fat_fraction_pct"] + 15.0


def test_side_labels_follow_the_patient_not_the_image(fatty_left: tuple) -> None:
    """'L' is patient left (+x in LPS) whatever the pixel order: the window and the phantom's fat band are both at +x."""
    ph, study, anchors, res = fatty_left
    lm = level_map(study, anchors, measure_geometry=False)
    stack = study.stacks["ax_t2"]
    ls = M.select_axial_slices(lm, "L4-L5")[0]
    hw, hd, measured = M._body_dims(lm.discs["L4-L5"])
    frame = M.level_frame(
        stack, ls.k, np.asarray(lm.discs["L4-L5"].centre), anchors.midline_x, hw, hd, measured
    )
    win = M.muscle_windows(frame, DEFAULT)
    xs_left = frame.lat[win[("MF", "L")]]
    assert xs_left.mean() > 0.0 and xs_left.min() > -1e-9
    band = _truth_fat_band(ph, stack, ls.k, side="L")
    assert band.any()
    _rr, cc = np.nonzero(band)
    x_band = stack.pix_to_lps_many(float(ls.k), np.column_stack([_rr, cc]).astype(float))[:, 0]
    assert x_band.mean() > float(anchors.midline_x)
    assert _rows(res, "L4-L5")[("MF", "L")]["fat_fraction_pct"] > 15.0
