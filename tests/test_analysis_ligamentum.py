from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _studies import flip_rows, phantom_study
from spinescope.analysis import ligamentum as L
from spinescope.analysis.base import available, run
from spinescope.analysis.findings import level_findings
from spinescope.anatomy import Anchors
from spinescope.config import DEFAULT
from spinescope.study import Study, from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

LEVEL = "L4-L5"


# ------------------------------------------------------------------------------------------ pure functions
def _profile(
    csf: float, lf: float, marrow: float, n_csf: int, n_lf: int, n_tail: int, step: float
) -> np.ndarray:
    """Synthetic ray: bright sac, dark ligament band, then brighter tissue behind it (samples from the dural edge)."""
    return np.concatenate([np.full(n_csf, csf), np.full(n_lf, lf), np.full(n_tail, marrow)]).astype(float)


def test_band_thickness_of_a_clean_dark_band() -> None:
    step = 0.25
    prof = _profile(1450, 200, 470, n_csf=8, n_lf=16, n_tail=40, step=step)  # 4 mm band
    thick, cnr, a, b, reason = L.band_from_profile(prof, step, noise=10.0, cfg=DEFAULT)
    assert reason == "ok"
    assert thick == pytest.approx(4.0, abs=0.5)
    assert cnr > DEFAULT.ligamentum_min_cnr
    assert a is not None and b is not None and b - a == pytest.approx(thick)


def test_band_not_closed_when_the_dark_run_reaches_the_window_end() -> None:
    step = 0.25
    prof = np.concatenate(
        [np.full(40, 200.0), np.full(20, 470.0)]
    )  # dark from the dural edge: no closing boundary
    thick, _, _, _, reason = L.band_from_profile(prof, step, noise=10.0, cfg=DEFAULT)
    assert thick is None
    assert reason == "band not closed within the search window"


def test_monotonic_fall_has_no_contrast_behind_the_minimum() -> None:
    prof = np.linspace(1450.0, 200.0, 80)  # nothing brighter behind the minimum
    thick, _, _, _, reason = L.band_from_profile(prof, 0.25, noise=10.0, cfg=DEFAULT)
    assert thick is None
    assert reason == "low contrast"


def test_flat_profile_is_low_contrast_not_a_band() -> None:
    rng = np.random.default_rng(0)
    prof = 400.0 + rng.normal(0, 10.0, 80)
    thick, _, _, _, reason = L.band_from_profile(prof, 0.25, noise=10.0, cfg=DEFAULT)
    assert thick is None and reason == "low contrast"


def test_band_rejects_non_finite_and_short_profiles() -> None:
    assert L.band_from_profile(np.array([1.0, np.nan, 2.0, 3.0]), 0.25, 1.0)[4] == "no profile"
    assert L.band_from_profile(np.array([1.0, 2.0]), 0.25, 1.0)[4] == "no profile"


def test_fat_from_profile_run_above_midpoint() -> None:
    step = 0.25
    prof = np.concatenate([np.full(24, 950.0), np.full(60, 100.0)])  # 6 mm of fat, then dark
    fat, reason = L.fat_from_profile(prof, step, csf_level=160.0, fat_level=950.0, cfg=DEFAULT)
    assert reason == "ok"
    assert fat == pytest.approx(6.0, abs=0.5)


def test_fat_not_assessed_when_fat_is_not_brighter_than_the_sac() -> None:
    fat, reason = L.fat_from_profile(np.full(60, 500.0), 0.25, csf_level=1450.0, fat_level=760.0, cfg=DEFAULT)
    assert fat is None
    assert "not brighter" in reason


def test_robust_noise_recovers_the_sigma() -> None:
    rng = np.random.default_rng(0)
    img = rng.normal(300.0, 5.0, (200, 200))
    assert L.robust_noise(img) == pytest.approx(5.0, rel=0.1)
    assert L.robust_noise(np.zeros((4, 1))) == 0.0


def test_ray_angles_are_symmetric_and_include_the_midline() -> None:
    angles = L.ray_angles(DEFAULT)
    assert 0.0 in angles
    assert angles == sorted(angles)
    assert all(-a in angles for a in angles)


# ------------------------------------------------------------------------------------------ phantom helpers
def _axial_study(
    spec: PhantomSpec, contrast: str = "t2", flip: bool = False
) -> tuple[Study, Anchors, SpinePhantom]:
    """One axial slice through the disc, plus anchors with the phantom's canal centreline (learned labeller not used)."""
    ph = SpinePhantom(spec)
    d = ph.discs[LEVEL]
    st = ph.render(
        "ax",
        contrast,
        n=1,
        spacing=5.0,
        thickness=4.0,
        rows=160,
        cols=160,
        px=0.6,
        center=d["center"],
        normal=[0, 0, 1],
        row_dir=[0, 1, 0],
    )
    if flip:
        st = flip_rows(st)
    study = from_stacks([st])
    anchors = Anchors.from_truth(ph.truth)
    anchors.canal = {
        "points": ph.canal_line.tolist(),
        "per_level": {
            lab: {"z_mm": float(v["center"][2]), "ap_mm": 2 * spec.canal_radius, "kind": "truth"}
            for lab, v in ph.discs.items()
        },
    }
    return study, anchors, ph


def _level_values(res: object, level: str = LEVEL) -> dict[str, tuple[float | None, str]]:
    rows = res.rows  # type: ignore[attr-defined]
    return {r["side"]: (r["value_mm"], r["status"]) for r in rows if r.get("disc") == level}


# ------------------------------------------------------------------------------------------ phantom measurements
@pytest.mark.parametrize("true_mm", [2.0, 4.0, 6.0, 8.0])
def test_left_ligamentum_thickness_tracks_the_phantom(true_mm: float) -> None:
    study, anchors, _ = _axial_study(PhantomSpec(lf={LEVEL: ("L", true_mm)}))
    res = L.analyze(study, anchors)
    vals = _level_values(res)
    assert vals["L"][1] == "measured"
    # the half-maximum band reads 0.25 to 0.5 mm low on the phantom (sampling and smoothing)
    assert vals["L"][0] == pytest.approx(true_mm - 0.4, abs=0.6)


def test_right_ligamentum_is_on_the_right_and_not_on_the_left() -> None:
    study, anchors, _ = _axial_study(PhantomSpec(lf={LEVEL: ("R", 6.0)}))
    vals = _level_values(L.analyze(study, anchors))
    assert vals["R"][1] == "measured" and vals["R"][0] == pytest.approx(5.5, abs=0.6)
    assert vals["L"][1] == "not_assessed"


def test_flipped_acquisition_gives_the_same_sides() -> None:
    spec = PhantomSpec(lf={LEVEL: ("L", 6.0)})
    plain = _level_values(L.analyze(*_axial_study(spec)[:2]))
    flipped = _level_values(L.analyze(*_axial_study(spec, flip=True)[:2]))
    assert plain["L"] == flipped["L"]
    assert flipped["R"][1] == "not_assessed"


def test_both_sides_when_the_ligament_is_symmetric() -> None:
    vals = _level_values(L.analyze(*_axial_study(PhantomSpec(lf={LEVEL: ("both", 5.0)}))[:2]))
    assert vals["L"][1] == "measured" and vals["R"][1] == "measured"
    assert vals["L"][0] == pytest.approx(vals["R"][0], abs=0.3)


def test_no_ligamentum_is_not_called_thick() -> None:
    res = L.analyze(*_axial_study(PhantomSpec())[:2])
    vals = _level_values(res)
    assert all(v[0] is None or v[0] < DEFAULT.ligamentum_thick_mm for v in vals.values())


def test_thick_flag_follows_the_threshold() -> None:
    res = L.analyze(*_axial_study(PhantomSpec(lf={LEVEL: ("L", 6.0)}))[:2])
    row = next(r for r in res.rows if r["disc"] == LEVEL and r["side"] == "L")
    assert row["suggests_thickening"] is True
    assert row["confidence"] == DEFAULT.ligamentum_max_confidence


def test_dorsal_epidural_fat_on_t1_with_a_midline_ligament() -> None:
    spec = PhantomSpec(epidural_fat_mm=6.0, lf={LEVEL: ("both", 2.0)})
    res = L.analyze(*_axial_study(spec, contrast="t1")[:2])
    fat = next(r for r in res.rows if r["disc"] == LEVEL and r["metric"] == "dorsal_epidural_fat")
    assert fat["status"] == "measured"
    assert fat["value_mm"] == pytest.approx(6.0, abs=1.2)


def test_dorsal_fat_is_not_assessed_on_t2() -> None:
    res = L.analyze(*_axial_study(PhantomSpec(epidural_fat_mm=6.0), contrast="t2")[:2])
    fat = next(r for r in res.rows if r["disc"] == LEVEL and r["metric"] == "dorsal_epidural_fat")
    assert fat["status"] == "not_assessed"
    assert fat["value_mm"] is None


# ------------------------------------------------------------------------------------------ missing inputs
def test_sagittal_only_study_is_not_assessed_with_a_warning() -> None:
    ph = SpinePhantom(PhantomSpec(lf={LEVEL: ("L", 6.0)}))
    study = phantom_study(ph)
    anchors = Anchors.from_truth(ph.truth)
    anchors.canal = {
        "points": ph.canal_line.tolist(),
        "per_level": {},
    }  # canal present, so the axial check runs
    res = L.analyze(study, anchors)
    assert res.rows == []
    assert any("no axial stack" in w for w in res.warnings)


def test_missing_canal_geometry_is_a_warning_not_a_crash() -> None:
    study, anchors, _ = _axial_study(PhantomSpec(lf={LEVEL: ("L", 6.0)}))
    anchors.canal = None
    res = L.analyze(study, anchors)
    assert res.rows == []
    assert any("no canal geometry" in w for w in res.warnings)


def test_level_outside_the_canal_is_not_assessed() -> None:
    study, anchors, _ = _axial_study(PhantomSpec(lf={LEVEL: ("L", 6.0)}))
    anchors.canal["per_level"][LEVEL]["ap_mm"] = None
    res = L.analyze(study, anchors)
    assert all(r["status"] == "not_assessed" for r in res.rows if r["disc"] == LEVEL)


# ------------------------------------------------------------------------------------------ registry and findings
def test_analysis_is_registered_and_its_field_reaches_the_findings() -> None:
    assert "ligamentum" in available()
    study, anchors, _ = _axial_study(PhantomSpec(lf={LEVEL: ("L", 6.0)}))
    res = run("ligamentum", study, anchors)
    figs = {"ligamentum": {f"lf_{LEVEL}": f"lf_{LEVEL}.png"}}
    level = level_findings({"ligamentum": res}, LEVEL, anchors, figures=figs)
    field = level["ligamentum_flavum"]["thickening"]
    assert field["value"] == "mild"
    assert field["confidence"] == "low"
    assert field["evidence"][0]["image"] == f"lf_{LEVEL}.png"
    assert "not confirmed" in field["notes"] or "convention" in field["notes"]


def test_result_is_phi_free_and_has_figures() -> None:
    res = L.analyze(*_axial_study(PhantomSpec(lf={LEVEL: ("L", 6.0)}))[:2])
    assert f"lf_{LEVEL}" in res.figures
    d = res.to_dict()
    assert all(not str(k).startswith("_") for k in d["meta"])
