"""Unit tests for the robustness options of the phantom. Defaults must keep the original behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from spinescope.geometry import Stack
from spinescope.study import from_stacks
from spinescope.testing.phantom import (
    BLOOM,
    CORTEX,
    HARDWARE,
    MARROW,
    OEDEMA,
    SCHMORL,
    PhantomSpec,
    SpinePhantom,
)

SMALL = {"n": 3, "spacing": 5.0, "thickness": 4.0, "rows": 128, "cols": 128, "px": 1.2}


def _sag(ph: SpinePhantom, contrast: str = "t2", **kw: object) -> Stack:
    return ph.render("sag", contrast, **{**SMALL, **kw})  # type: ignore[arg-type]


def _body_points(ph: SpinePhantom, name: str, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(0)
    c = ph.bodies[name]["center"]
    return c + rng.uniform(-1, 1, size=(n, 3)) * np.array([20.0, 12.0, 12.0])


def test_default_levels_and_truth_are_unchanged() -> None:
    ph = SpinePhantom(PhantomSpec())
    assert list(ph.truth["discs"]) == ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12"]
    assert set(ph.bodies) == {"T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1"}
    assert ph.truth["transitional"] is None
    assert all(v["height_mm"] == 10.0 for v in ph.truth["discs"].values())


def test_default_render_is_deterministic() -> None:
    a = _sag(SpinePhantom(PhantomSpec())).vol
    b = _sag(SpinePhantom(PhantomSpec())).vol
    assert np.array_equal(a, b)


def test_thoracic_top_adds_a_level() -> None:
    ph = SpinePhantom(PhantomSpec(thoracic_top=10))
    assert list(ph.truth["discs"])[-1] == "T10-T11"
    assert "T10" in ph.bodies


def test_lumbarised_s1_adds_l6_and_the_s1_s2_disc() -> None:
    ph = SpinePhantom(PhantomSpec(transitional="lumbarised_s1"))
    discs = list(ph.truth["discs"])
    assert len(discs) == 8 and discs[0] == "L6-S1" and discs[1] == "L5-L6"
    assert ph.truth["discs"]["L6-S1"]["anatomical"] == "S1-S2"
    assert ph.truth["findings"]["lumbar_levels"] == ["L1", "L2", "L3", "L4", "L5", "L6"]
    assert "S2" in ph.bodies


def test_sacralised_l5_has_a_partial_disc_and_bone_bridge() -> None:
    ph = SpinePhantom(PhantomSpec(transitional="sacralised_l5"))
    assert ph.truth["discs"]["L5-S1"]["partial"] is True
    disc_c = ph.discs["L5-S1"]["center"]
    pts = disc_c + np.array([[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    lab = ph.labels(pts)
    assert lab[0] != lab[1]  # one side is disc, the other is bone


def test_collapsed_disc_loses_exactly_the_requested_height() -> None:
    ph = SpinePhantom(PhantomSpec(collapsed={"L4-L5": 0.6}))
    assert ph.discs["L4-L5"]["height_mm"] == pytest.approx(4.0)
    assert ph.discs["L3-L4"]["height_mm"] == pytest.approx(10.0)


def test_fracture_gives_oedema_and_anterior_wedge() -> None:
    plain = SpinePhantom(PhantomSpec())
    wedge = SpinePhantom(PhantomSpec(fracture={"L2": 0.45}))
    assert np.count_nonzero(wedge.labels(_body_points(wedge, "L2")) == OEDEMA) > 0
    assert np.count_nonzero(plain.labels(_body_points(plain, "L2")) == OEDEMA) == 0
    # the anterior body loses height: a point that is inside the plain body is outside the wedged one
    anterior = wedge.bodies["L2"]["center"] + wedge.bodies["L2"]["ap"] * -10.0
    up = np.array([anterior + wedge.bodies["L2"]["normal"] * 11.0])
    assert plain.labels(up)[0] == MARROW
    assert wedge.labels(up)[0] not in (MARROW, CORTEX, OEDEMA)


def test_schmorl_node_is_placed_at_the_endplate() -> None:
    ph = SpinePhantom(PhantomSpec(schmorl=[("L3", "sup")]))
    assert np.count_nonzero(ph.labels(_body_points(ph, "L3")) == SCHMORL) > 0


def test_hardware_gives_voids_and_blooming() -> None:
    ph = SpinePhantom(PhantomSpec(hardware=("cage:L4-L5", "screws:L4", "rod")))
    lab = ph.labels(_body_points(ph, "L4"))
    assert np.count_nonzero(lab == HARDWARE) > 0
    assert np.count_nonzero(lab == BLOOM) > 0
    img = ph.intensity(lab, "t2")
    assert np.all(img[lab == HARDWARE] == 0.0)


def test_unknown_hardware_is_rejected_when_built() -> None:
    with pytest.raises(ValueError, match="unknown hardware"):
        SpinePhantom(PhantomSpec(hardware=("screws:L9",)))


def test_unknown_transitional_and_field_strength_are_rejected() -> None:
    with pytest.raises(ValueError):
        SpinePhantom(PhantomSpec(transitional="cervicalised"))
    with pytest.raises(ValueError):
        SpinePhantom(PhantomSpec(field_strength=7.0))


def test_ghosting_changes_the_image_and_keeps_the_geometry() -> None:
    ph = SpinePhantom(PhantomSpec())
    clean, ghosted = _sag(ph), _sag(ph, ghost=0.3)
    assert not np.array_equal(clean.vol, ghosted.vol)
    assert np.array_equal(clean.ipp, ghosted.ipp) and clean.vol.shape == ghosted.vol.shape


def test_wrap_keeps_the_shape_and_folds_content() -> None:
    ph = SpinePhantom(PhantomSpec())
    clean, wrapped = _sag(ph), _sag(ph, wrap_rows=30)
    assert wrapped.vol.shape == clean.vol.shape
    assert not np.array_equal(clean.vol, wrapped.vol)
    assert wrapped.vol[:, :30].mean() > clean.vol[:, :30].mean()  # folded anatomy adds signal near the edge


def test_bias_field_is_multiplicative_and_off_by_default() -> None:
    ph = SpinePhantom(PhantomSpec(bias_field=0.6))
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 120.0, 0.0]])
    b = ph._bias(pts)
    assert b[1] > b[0]  # brighter near the posterior coil
    assert not np.array_equal(_sag(SpinePhantom(PhantomSpec())).vol, _sag(ph).vol)


def test_pd_contrast_is_classified_as_pd() -> None:
    st = _sag(SpinePhantom(PhantomSpec()), "pd", n=5)
    study = from_stacks([st])
    assert "sag_pd" in study.stacks


def test_three_tesla_variant_writes_the_field_strength() -> None:
    st = _sag(SpinePhantom(PhantomSpec(field_strength=3.0)))
    assert st.meta["dicom"]["MagneticFieldStrength"] == 3.0


def test_ribs_add_cortex_lateral_to_the_thoracic_bodies() -> None:
    plain = SpinePhantom(PhantomSpec())
    ribs = SpinePhantom(PhantomSpec(ribs=True))
    c = ribs.bodies["T11"]["center"]
    # points along the lateral course of a rib, outside the body outline
    course = c + np.array([[x, -10.0, 0.0] for x in np.linspace(30.0, 60.0, 30)])
    assert np.count_nonzero(ribs.labels(course) == CORTEX) > np.count_nonzero(plain.labels(course) == CORTEX)


def test_iso_and_thick_geometry_render() -> None:
    ph = SpinePhantom(PhantomSpec())
    iso = ph.render("sag", "t2", n=3, spacing=1.0, thickness=1.0, rows=96, cols=64, px=1.0)
    thick = ph.render("sag", "t2", n=3, spacing=6.0, thickness=6.0, rows=96, cols=96, px=1.2)
    assert iso.spacing == pytest.approx(1.0) and thick.spacing == pytest.approx(6.0)
