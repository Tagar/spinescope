from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _studies import flip_rows
from spinescope.analysis import base, foramina
from spinescope.anatomy import Anchors
from spinescope.anatomy.anchors import Disc
from spinescope.config import DEFAULT
from spinescope.geometry import Stack

PITCH = 0.5  # mm per pixel in the synthetic sagittal stacks
PLANTED_X = (13.0, 19.0)  # planes that cut the planted foramen (LPS x, mm); midline at x = 0
DISC_CENTRE_Y = 40.0
POST_Y = 52.0  # posterior disc margin
FORAMEN_Y = 58.0  # foramen centre, 6 mm behind the posterior margin
FORAMEN_AP_HALF = 5.0  # AP width 10 mm
FORAMEN_SI_HALF = 8.0  # height 16 mm


def _grid_yz() -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(240)
    rows = np.arange(160)
    y = cols * PITCH  # 0..120 mm
    z = 40.0 - rows * PITCH  # +40 (top) .. -40 mm
    Y, Z = np.meshgrid(y, z)  # (rows, cols)
    return Y, Z


def _slice(x: float, mode: str) -> np.ndarray:
    """Intensity index of one parasagittal plane at lateral position ``x``."""
    Y, Z = _grid_yz()
    img = np.full(Y.shape, 0.15)  # cortex, dark bone
    img[Z >= 36.0] = (
        1.0  # subcutaneous fat: the series 99th percentile (the fat reference) is set by real fat
    )
    disc = (np.abs(Z) <= 5.0) & (Y >= 28.0) & (Y <= POST_Y)
    img[disc] = 0.35  # intermediate disc
    if PLANTED_X[0] <= x <= PLANTED_X[1] and mode != "none":
        ell = ((Y - FORAMEN_Y) / FORAMEN_AP_HALF) ** 2 + (Z / FORAMEN_SI_HALF) ** 2 <= 1.0
        fat = ell.copy()
        if mode == "sup_obliterated":
            fat &= Z <= 0.0  # no fat above the centre line
        img[fat] = 1.0  # fat-bright
        root = ((Y - 54.0) ** 2 + (Z + 3.0) ** 2) <= 2.0**2
        img[root & ell] = 0.5  # exiting root, intermediate
    return img


def synthetic_stack(xs: np.ndarray, mode: str = "planted") -> Stack:
    """Sagittal T1-like stack: planes at ``xs`` (LPS x), 0.5 mm pixels, rows = -z, columns = +y (posterior)."""
    vol = np.stack([_slice(float(x), mode) for x in xs]).astype(np.float32)
    ipp = np.stack([[float(x), 0.0, 40.0] for x in xs])
    return Stack(
        key="sag_t1",
        vol=vol,
        ipp=ipp,
        row_dir=np.array([0.0, 1.0, 0.0]),
        col_dir=np.array([0.0, 0.0, -1.0]),
        row_mm=PITCH,
        col_mm=PITCH,
    )


def _disc() -> Disc:
    return Disc(
        center=[0.0, DISC_CENTRE_Y, 0.0],
        normal=[0.0, 0.0, 1.0],
        anterior=[0.0, 28.0, 0.0],
        posterior=[0.0, POST_Y, 0.0],
        height_mm=10.0,
        confidence=0.9,
    )


def _anchors(discs: dict[str, Disc] | None = None, midline: float | None = 0.0) -> Anchors:
    a = Anchors(discs=discs if discs is not None else {"L4-L5": _disc()}, midline_x=midline)
    a.numbering = {"status": "confirmed", "confidence": 0.99}
    return a


XS = np.arange(0.0, 31.0, 2.0)  # planes 0..30 mm: covers the left window (10-25 mm), not the right


@pytest.fixture()
def planted() -> Stack:
    return synthetic_stack(XS)


def _frame() -> foramina.DiscFrame:
    frame = foramina.disc_frame(_disc())
    assert frame is not None
    return frame


def _site(stack: Stack) -> foramina.Site:
    site = foramina.search_site(stack, _frame(), "L", 0.0, float(np.percentile(stack.vol, 99)), DEFAULT)
    assert site is not None
    return site


def _measure(stack: Stack) -> foramina.Measure:
    scale = float(np.percentile(stack.vol, 99))
    site = _site(stack)
    return foramina.measure_site(stack, _frame(), site, scale, DEFAULT)


def _run(monkeypatch: pytest.MonkeyPatch, stack: Stack | None, anchors: Anchors | None = None) -> base.Result:
    monkeypatch.setattr(foramina, "_study_sagittal_t1", lambda study: stack)
    return foramina.analyze(None, anchors or _anchors(), DEFAULT)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------- pure functions
def test_registered_on_import():
    assert "foramina" in base.available()  # registered by importing the module (auto-discovery on main)


@pytest.mark.parametrize(
    ("dirs", "status", "lee", "wild"),
    [
        ((), "preserved", 0, 0),
        (("sup", "inf"), "partially obliterated", 1, 2),
        (("ant", "post"), "partially obliterated", 1, 2),
        (("sup", "ant"), "partially obliterated", None, 2),  # non-opposing pair: outside Lee's scheme
        (("ant",), "partially obliterated", None, 2),
        (("ant", "post", "sup"), "partially obliterated", None, 2),
        (("ant", "post", "sup", "inf"), "fully obliterated", 2, 3),
    ],
)
def test_grade_mapping(dirs, status, lee, wild):
    g = foramina.grade(dirs)
    assert g["perineural_status"] == status
    assert g["lee_grade"] == lee
    assert g["wildermuth_grade"] == wild
    assert g["n_obliterated"] == len(dirs)
    assert g["root_morphology"] == "not assessed"  # grade 3 of Lee is never assigned from the ray test


def test_side_convention():
    assert foramina.side_sign("L") == 1.0 and foramina.side_sign("R") == -1.0  # patient left = +x (LPS)
    assert foramina.lateral_mm(16.0, 0.0, "L") == pytest.approx(16.0)
    assert foramina.lateral_mm(-16.0, 0.0, "R") == pytest.approx(16.0)
    assert foramina.lateral_mm(-16.0, 0.0, "L") == pytest.approx(-16.0)
    with pytest.raises(ValueError):
        foramina.side_sign("X")


def test_normal_height_table():
    assert foramina.normal_height("L4-L5", DEFAULT) == pytest.approx((20.47, 2.73))
    assert foramina.normal_height("T12-L1", DEFAULT) is None


def test_disc_frame_requires_posterior_margin():
    d = _disc()
    d.posterior = None
    assert foramina.disc_frame(d) is None


# ----------------------------------------------------------------------------------- synthetic geometry
def test_search_finds_planted_plane_and_centre(planted):
    site = _site(planted)
    x = float(planted.ipp[site.k, 0])
    assert PLANTED_X[0] <= x <= PLANTED_X[1]
    assert site.lateral_mm == pytest.approx(x)  # left side: lateral = x - midline
    assert site.ap_off_mm == pytest.approx(FORAMEN_Y - POST_Y, abs=1.5)
    assert abs(site.si_off_mm) <= 1.5
    assert site.score_margin > 0


def test_measure_planted_foramen(planted):
    m = _measure(planted)
    assert m.centre_fat
    assert m.height_mm == pytest.approx(2 * FORAMEN_SI_HALF, abs=1.0)
    assert m.ap_width_mm == pytest.approx(2 * FORAMEN_AP_HALF, abs=1.0)
    assert m.area_mm2 == pytest.approx(np.pi * FORAMEN_AP_HALF * FORAMEN_SI_HALF, rel=0.15)
    assert m.obliterated == ()  # fat all round the foramen: preserved


def test_ray_test_sees_fat_only_on_one_side(planted):
    """The perineural ray is the fat-bright extent from the centre in one direction (a geometric proxy, see the doc).
    Fat exists only below the centre line in this stack: the superior ray finds none, the inferior one finds fat."""
    stack = synthetic_stack(XS, mode="sup_obliterated")
    scale = float(np.percentile(stack.vol, 99))
    k = int(np.argmin(np.abs(stack.ipp[:, 0] - 16.0)))
    F = np.array([stack.ipp[k, 0], 58.0, 0.0])
    thr = 0.5 * (1.0 + 0.15)
    up = np.array([0.0, 0.0, 1.0])
    assert not foramina._ray_fat(stack, k, F, up, 6.0, scale, thr, DEFAULT)
    assert foramina._ray_fat(stack, k, F, -up, 6.0, scale, thr, DEFAULT)


def test_flipped_acquisition_gives_same_measurement(planted):
    flipped = flip_rows(planted)
    a = _measure(planted)
    b = _measure(flipped)
    assert b.height_mm == pytest.approx(a.height_mm, abs=0.5)
    assert b.ap_width_mm == pytest.approx(a.ap_width_mm, abs=0.5)
    assert b.obliterated == a.obliterated


def test_no_foramen_is_not_assessed():
    stack = synthetic_stack(XS, mode="none")
    scale = float(np.percentile(stack.vol, 99))
    site = foramina.search_site(stack, _frame(), "L", 0.0, scale, DEFAULT)
    assert site is not None  # planes are there; the centre is not fat
    m = foramina.measure_site(stack, _frame(), site, scale, DEFAULT)
    assert not m.centre_fat


def test_narrow_coverage_returns_none():
    stack = synthetic_stack(np.arange(0.0, 6.0, 2.0))  # planes only 0-4 mm: no plane in the 10-25 mm window
    scale = float(np.percentile(stack.vol, 99))
    assert foramina.search_site(stack, _frame(), "L", 0.0, scale, DEFAULT) is None


# ------------------------------------------------------------------------------------ analysis entry point
def test_analyze_rows_flags_and_coverage(monkeypatch, planted):
    anchors = _anchors({"L4-L5": _disc()}, midline=0.0)
    res = _run(monkeypatch, planted, anchors)
    rows = {r["side"]: r for r in res.rows}
    assert set(rows) == {"L", "R"}
    left = rows["L"]
    assert left["status"] == "assessed"
    assert left["root"] == "L4"  # the L4-L5 foramen carries the L4 root (anatomy, not a compression claim)
    assert left["numbering"] == "confirmed"
    assert left["foraminal_height_mm"] == pytest.approx(16.0, abs=1.0)
    right = rows["R"]  # the synthetic series has no plane on the right: narrow coverage
    assert right["status"] == "not_assessed" and "coverage" in right["reason"]
    assert right.get("foraminal_height_mm") is None
    assert res.figures["foramina_grid"].ndim == 3 and res.figures["foramina_grid"].shape[2] == 3
    assert res.figures["foramina_grid"].dtype == np.uint8


def test_flags_carry_low_confidence_cap(monkeypatch):
    stack = synthetic_stack(XS, mode="sup_obliterated")
    res = _run(monkeypatch, stack)
    assert res.flags, "a partially obliterated foramen should be flagged"
    assert all(f["confidence"] <= DEFAULT.foramen_flag_max_confidence for f in res.flags)


def test_relative_to_other_levels(monkeypatch, planted):
    d1 = _disc()
    d2 = _disc()
    d2.center = [0.0, DISC_CENTRE_Y, -25.0]
    d2.posterior = [0.0, POST_Y, -25.0]
    d2.anterior = [0.0, 28.0, -25.0]
    anchors = _anchors({"L4-L5": d1, "L5-S1": d2}, midline=0.0)
    res = _run(monkeypatch, planted, anchors)
    left = [r for r in res.rows if r["side"] == "L" and r["status"] == "assessed"]
    assert left, "at least one assessed left foramen"
    assert all("height_minus_side_median_mm" in r for r in left)


def test_missing_inputs_warn_not_raise(monkeypatch, planted):
    res = _run(monkeypatch, None)
    assert not res.rows and any("no sagittal T1" in w for w in res.warnings)
    res = _run(monkeypatch, planted, _anchors(midline=None))
    assert not res.rows and any("no midline" in w for w in res.warnings)
    res = _run(monkeypatch, planted, _anchors({}, midline=0.0))
    assert not res.rows and any("no discs" in w for w in res.warnings)


def test_bad_disc_label_is_skipped(monkeypatch, planted):
    anchors = _anchors({"XX": _disc()}, midline=0.0)
    res = _run(monkeypatch, planted, anchors)
    assert not res.rows and any("not a disc label" in w for w in res.warnings)


def test_run_all_contains_failures(monkeypatch, planted):
    def boom(study, anchors, config=None):  # pragma: no cover - exercised through run_all
        raise RuntimeError("synthetic failure")

    monkeypatch.setitem(base._REGISTRY, "foramina", boom)
    out = base.run_all(None, _anchors(), DEFAULT, names=["foramina"])  # type: ignore[arg-type]
    assert "analysis failed" in out["foramina"].warnings[0]
