"""Facet joints: sites, fluid measurement, side conventions, fallbacks and report wiring (phantom-based)."""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _studies import flip_rows
from spinescope.analysis import facets as F
from spinescope.analysis.base import available
from spinescope.analysis.findings import level_findings
from spinescope.anatomy import Anchors
from spinescope.anatomy.correspondence import level_map
from spinescope.config import DEFAULT
from spinescope.study import from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

CENTRE = np.array([0.0, 20.0, -30.0])  # covers L1..S1 with 24 slices of 3 mm
PX = 0.9
ROWS = 160


def _phantom(**kw: object) -> SpinePhantom:
    return SpinePhantom(PhantomSpec(realism="realistic", **kw))  # type: ignore[arg-type]


def _ax(ph: SpinePhantom, contrast: str = "stir", key: str = "ax_stir"):  # type: ignore[no-untyped-def]
    return ph.render(
        "ax", contrast, n=24, spacing=3.0, thickness=3.0, rows=ROWS, cols=ROWS, px=PX, center=CENTRE, key=key
    )


@cache
def _run(effusion: tuple[tuple[str, str, float], ...] = ()) -> tuple[SpinePhantom, F.Result]:
    """Phantom with axial STIR (fat-suppressed) and the given injected effusions; analysed once per setting."""
    ph = _phantom(facet_effusion={d: (s, w) for d, s, w in effusion})
    st = from_stacks([_ax(ph)])
    return ph, F.analyze(st, Anchors.from_truth(ph.truth))


def _row(res: F.Result, level: str, side: str) -> dict:  # type: ignore[type-arg]
    return next(
        r for r in res.rows if r.get("scope") == "joint" and r["level"] == level and r["side"] == side
    )


# ------------------------------------------------------------------------ registry and pure helpers
def test_registered_and_builtin():
    assert "facets" in available()


def test_caliper_of_a_disc_is_its_diameter():
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pts = np.stack([3.0 * np.cos(t), 3.0 * np.sin(t)], axis=1)
    cmax, cmin = F._calipers(pts)
    assert cmax == pytest.approx(6.0, abs=0.05)
    assert cmin == pytest.approx(6.0, abs=0.05)


def test_caliper_of_an_ellipse_shows_roundness():
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    pts = np.stack([8.0 * np.cos(t), 2.0 * np.sin(t)], axis=1)
    cmax, cmin = F._calipers(pts)
    assert cmax == pytest.approx(16.0, abs=0.1)
    assert cmin / cmax == pytest.approx(0.25, abs=0.02)


def test_tropism_flag_uses_the_threshold():
    assert F.tropism_flag(40.0, 48.0)["tropism"] is True  # 8 deg > 7 deg
    assert F.tropism_flag(40.0, 45.0)["tropism"] is False  # 5 deg
    assert F.tropism_flag(40.0, 45.0, threshold_deg=4.0)["tropism"] is True
    assert F.tropism_flag(50.0, 40.0)["difference_deg"] == pytest.approx(10.0)


# ------------------------------------------------------------------------ sites
def test_sites_are_mirror_symmetric_and_lateral():
    ph = _phantom()
    anc = Anchors.from_truth(ph.truth)
    sites = F.joint_sites(level_map(from_stacks([_ax(ph)]), anc), anc, DEFAULT)
    assert set(sites) >= {"L4-L5", "L3-L4"}
    for label in ("L3-L4", "L4-L5"):
        left, right = sites[label]["L"], sites[label]["R"]
        assert left.xyz[0] > 0 > right.xyz[0]  # L is patient left (+x in LPS)
        assert left.xyz[0] == pytest.approx(-right.xyz[0], abs=1e-6)
        assert left.xyz[1:] == pytest.approx(right.xyz[1:], abs=1e-6)
        assert left.lateral_mm > left.canal_half_mm


def test_sites_lie_near_the_phantom_joints():
    ph = _phantom()
    anc = Anchors.from_truth(ph.truth)
    sites = F.joint_sites(level_map(from_stacks([_ax(ph)]), anc), anc, DEFAULT)
    for label in ("L3-L4", "L4-L5", "L5-S1"):
        for side in ("L", "R"):
            truth = np.asarray(ph.truth["facets"][label][side]["center"])
            dist = float(np.linalg.norm(sites[label][side].xyz - truth))
            assert dist < 6.0, (label, side, dist)  # site error is a few mm (docs/analysis/facets.md)


# ------------------------------------------------------------------------ fluid measurement
def test_healthy_joints_have_no_effusion():
    _, res = _run()
    calls = [r for r in res.rows if r.get("scope") == "joint" and r["effusion_present"] is not None]
    assert calls, res.warnings
    assert all(r["effusion_present"] is False for r in calls)
    assert not [f for f in res.flags if f["kind"] == "facet_effusion"]


def test_injected_effusion_is_found_on_its_side_only():
    _, res = _run((("L3-L4", "R", 4.0),))
    right = _row(res, "L3-L4", "R")
    left = _row(res, "L3-L4", "L")
    assert right["effusion_present"] is True
    assert left["effusion_present"] is False
    assert (
        4.0 <= right["effusion_width_mm"] <= 6.0
    )  # phantom bias is +0.5 to +1.5 mm (docs/analysis/facets.md)
    assert right["fluid_ratio"] is not None and right["fluid_ratio"] > 0.5  # CSF-like fluid
    assert right["stack"] == "ax_stir"
    flags = [f for f in res.flags if f["kind"] == "facet_effusion" and f["level"] == "L3-L4"]
    assert [f["side"] for f in flags] == ["R"]


def test_width_grows_with_injected_extent():
    widths = []
    for w in (2.0, 4.0, 6.0):
        _, res = _run((("L4-L5", "L", w),))
        widths.append(_row(res, "L4-L5", "L")["effusion_width_mm"])
    assert widths == sorted(widths)
    assert widths[0] > 0.0 and widths[-1] - widths[0] > 2.0


def test_round_collection_of_cyst_size_is_flagged_possible():
    _, res = _run((("L3-L4", "R", 4.0),))
    row = _row(res, "L3-L4", "R")
    assert row["effusion_width_mm"] >= DEFAULT.facets_cyst_min_mm
    assert row["roundness"] >= DEFAULT.facets_cyst_roundness_min
    assert row["synovial_cyst_possible"] is True
    assert any(f["kind"] == "synovial_cyst_possible" and "possible" in f["wording"] for f in res.flags)


def test_elongated_collection_is_not_called_a_cyst():
    _, res = _run((("L4-L5", "L", 6.0),))
    row = _row(res, "L4-L5", "L")
    assert row["effusion_width_mm"] >= DEFAULT.facets_cyst_min_mm
    assert row["roundness"] < DEFAULT.facets_cyst_roundness_min
    assert row["synovial_cyst_possible"] is False


def test_side_convention_holds_for_a_flipped_acquisition():
    ph = _phantom(facet_effusion={"L3-L4": ("R", 4.0)})
    st = _ax(ph)
    plain = F.analyze(from_stacks([st]), Anchors.from_truth(ph.truth))
    flipped = F.analyze(from_stacks([flip_rows(st)]), Anchors.from_truth(ph.truth))
    for side in ("L", "R"):
        a, b = _row(plain, "L3-L4", side), _row(flipped, "L3-L4", side)
        assert a["effusion_present"] == b["effusion_present"]
    assert _row(flipped, "L3-L4", "R")["effusion_present"] is True
    assert _row(flipped, "L3-L4", "L")["effusion_present"] is False
    assert _row(flipped, "L3-L4", "R")["effusion_width_mm"] == pytest.approx(
        _row(plain, "L3-L4", "R")["effusion_width_mm"], abs=0.5
    )


def test_plain_t2_is_not_called_because_fat_is_as_bright():
    ph = _phantom(facet_effusion={"L3-L4": ("R", 4.0)})
    res = F.analyze(from_stacks([_ax(ph, "t2", "ax_t2")]), Anchors.from_truth(ph.truth))
    rows = [r for r in res.rows if r.get("scope") == "joint" and r.get("plane") == "ax"]
    assert rows
    assert all(r["effusion_present"] is None for r in rows)
    assert all(r["confidence"] == 0.0 for r in rows)
    assert not [f for f in res.flags if f["kind"] == "facet_effusion"]
    assert any("fat not suppressed" in r["notes"] for r in rows)


def test_t1_only_study_gives_warnings_not_calls():
    ph = _phantom()
    res = F.analyze(from_stacks([_ax(ph, "t1", "ax_t1")]), Anchors.from_truth(ph.truth))
    assert res.warnings
    assert all(r["effusion_present"] is None for r in res.rows if r.get("scope") == "joint")


def test_parasagittal_fallback_is_low_confidence():
    ph = _phantom(facet_effusion={"L4-L5": ("L", 6.0)})
    sag = ph.render(
        "sag", "stir", n=15, spacing=4.0, thickness=3.0, rows=ROWS, cols=ROWS, px=PX, key="sag_stir"
    )
    res = F.analyze(from_stacks([sag]), Anchors.from_truth(ph.truth))
    rows = [r for r in res.rows if r.get("scope") == "joint" and r.get("plane") == "sag"]
    assert rows, res.warnings
    assert all(
        r["confidence"] <= DEFAULT.facets_sagittal_max_confidence + 1e-9 for r in rows if r["confidence"]
    )
    assert any("no axial stack" in w for w in res.warnings)


def test_missing_anchors_and_sequences_warn_and_do_not_raise():
    ph = _phantom()
    empty = F.analyze(from_stacks([_ax(ph)]), Anchors())
    assert empty.rows == [] and empty.warnings
    nothing = F.analyze(from_stacks([_ax(ph)]), Anchors.from_truth(ph.truth))
    assert nothing.rows  # sanity: rows exist with a stack
    no_stack_rows = [r for r in nothing.rows if r.get("stack") is None]
    assert no_stack_rows and all(r["effusion_present"] is None for r in no_stack_rows)


def test_cartoon_phantom_without_facets_runs():
    ph = SpinePhantom()  # cartoon: no facet joints are drawn
    res = F.analyze(
        from_stacks(
            [
                ph.render(
                    "ax",
                    "stir",
                    n=24,
                    spacing=3.0,
                    thickness=3.0,
                    rows=ROWS,
                    cols=ROWS,
                    px=PX,
                    center=CENTRE,
                    key="ax_stir",
                )
            ]
        ),
        Anchors.from_truth(ph.truth),
    )
    assert isinstance(res, F.Result)


# ------------------------------------------------------------------------ figures and findings
def test_figure_is_rgb_uint8_per_level():
    _, res = _run((("L3-L4", "R", 4.0),))
    assert "joints_L3_L4" in res.figures
    fig = res.figures["joints_L3_L4"]
    assert fig.dtype == np.uint8 and fig.ndim == 3 and fig.shape[2] == 3
    assert fig.shape[0] > 0 and fig.shape[1] > fig.shape[0]  # two panels side by side


def test_findings_report_effusion_per_side_with_image_evidence():
    ph, res = _run((("L3-L4", "R", 4.0),))
    anc = Anchors.from_truth(ph.truth)
    lf = level_findings({"facets": res}, "L3-L4", anc, figures={"facets": {"joints_L3_L4": "/tmp/x.png"}})
    assert lf["facet_R"]["effusion"]["value"] == "present"
    assert lf["facet_L"]["effusion"]["value"] == "absent"
    assert lf["facet_R"]["effusion"]["evidence"][0]["series"] == "ax_stir"
    # without an image the field is skipped, never written without evidence
    bare = level_findings({"facets": res}, "L3-L4", anc)
    assert "facet_R" not in bare and bare.get("_skipped")


def test_level_not_on_the_series_is_not_reported():
    ph, res = _run((("L3-L4", "R", 4.0),))
    res.meta["series_key"] = "some_other_series"
    assert "facet_R" not in level_findings({"facets": res}, "L3-L4", Anchors.from_truth(ph.truth))


# ------------------------------------------------------------------------ phantom ground truth
def test_realistic_phantom_records_facet_truth():
    ph = _phantom(facet_effusion={"L4-L5": ("L", 3.0)})
    assert ph.truth["facets"]["L4-L5"]["L"]["effusion_mm"] == pytest.approx(3.0)
    assert ph.truth["facets"]["L4-L5"]["R"]["effusion_mm"] == 0.0
    assert ph.truth["facets"]["L4-L5"]["L"]["center"][0] > 0 > ph.truth["facets"]["L4-L5"]["R"]["center"][0]


# ------------------------------------------------------------------------ dx21: joint localisation, angle, tropism
def _dark_line_roi(orientation: str, n: int = 48, px: float = 0.5) -> np.ndarray:
    """Bright ROI with one dark line through the centre (horizontal in (u, v) if 'h', vertical if 'v')."""
    img = np.full((n, n), 1000.0)
    c = (n - 1) // 2
    if orientation == "v":  # line along v (AP): a sagittal joint line
        img[:, c - 1 : c + 2] = 100.0
    else:  # line along u (lateral): a coronal joint line
        img[c - 1 : c + 2, :] = 100.0
    return img


@pytest.mark.parametrize(
    ("orientation", "expected_angle"),
    [("v", 0.0), ("h", 90.0)],
)
def test_localise_finds_the_dark_line_and_its_angle(orientation: str, expected_angle: float):
    loc = F.localise_joint(_dark_line_roi(orientation), 0.5)
    assert loc.ok, loc.reason
    assert loc.centre_uv is not None
    assert abs(loc.centre_uv[0]) < 0.6 and abs(loc.centre_uv[1]) < 0.6  # centred on the line
    assert F.joint_angle_deg(loc.direction) == pytest.approx(expected_angle, abs=3.0)


def test_localise_reports_no_ridge_on_a_flat_image():
    loc = F.localise_joint(np.full((48, 48), 500.0), 0.5)
    assert not loc.ok
    assert loc.centre_uv is None and loc.direction is None


def test_localise_reports_no_data_on_an_empty_window():
    loc = F.localise_joint(np.full((48, 48), np.nan), 0.5)
    assert not loc.ok and "no image data" in loc.reason


def test_joint_angle_convention():
    assert F.joint_angle_deg((0.0, 1.0)) == pytest.approx(0.0)  # along AP: sagittal
    assert F.joint_angle_deg((1.0, 0.0)) == pytest.approx(90.0)  # along lateral: coronal
    assert F.joint_angle_deg((1.0, 1.0)) == pytest.approx(45.0)
    assert F.joint_angle_deg((-1.0, -1.0)) == pytest.approx(45.0)  # sign-free
    assert np.isnan(F.joint_angle_deg((0.0, 0.0)))


def test_tropism_flags_report_both_thresholds():
    small = F.tropism_flags(40.0, 44.0)
    assert small["tropism_7deg"] is False and small["tropism_10deg"] is False
    mid = F.tropism_flags(40.0, 48.0)
    assert mid["tropism_7deg"] is True and mid["tropism_10deg"] is False
    big = F.tropism_flags(40.0, 53.0)
    assert big["tropism_7deg"] is True and big["tropism_10deg"] is True
    assert big["difference_deg"] == pytest.approx(13.0)


def test_sample_label_plane_addresses_the_patient_grid():
    ph = _phantom()
    st = from_stacks([_ax(ph)]).stacks["ax_stir"]
    k = 3
    label2d = np.arange(st.vol.shape[1] * st.vol.shape[2]).reshape(st.vol.shape[1:])
    origin = st.pix_to_lps(float(k), 80.0, 80.0)
    shape = (48, 48)
    lab = F.sample_label_plane(
        st, k, label2d, origin, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), shape, 0.5
    )
    pts = F.roi_grid_mm(origin, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), shape, 0.5)
    for i, j in ((0, 0), (10, 30), (47, 47)):
        _kk, r, c = st.lps_to_pix(pts[i, j])
        assert lab[i, j] == label2d[round(r), round(c)]
    assert (lab >= 0).all()


def test_geometry_rows_are_capped_and_never_graded():
    _, res = _run()
    rows = [r for r in res.rows if r.get("scope") == "joint_geometry"]
    assert rows, "no joint geometry rows on the phantom"
    for r in rows:
        assert r["grade_status"] == "not assessed"
        assert r["weishaupt_grade"] is None
        assert r["experimental"] is True
        assert r["confidence"] <= DEFAULT.facets_morph_max_confidence + 1e-9
        if r["loc_ok"]:
            assert 0.0 <= r["joint_angle_deg"] <= 90.0
            assert r["joint_line_width_mm"] <= r["joint_line_length_mm"] + 1e-6
        else:
            assert r["joint_angle_deg"] is None and r["joint_uv_mm"] is None


def test_dx08_effusion_rows_still_present_with_geometry_added():
    _, res = _run()
    assert any(r.get("scope") == "joint" for r in res.rows)
    assert all(r.get("scope") in ("joint", "joint_geometry") for r in res.rows)
