"""Central canal analysis (spinescope.analysis.canal): pure functions, grading, phantom end-to-end, findings wiring."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from spinescope.analysis import canal
from spinescope.analysis.base import Result, run
from spinescope.analysis.findings import level_findings
from spinescope.anatomy.anchors import Anchors
from spinescope.config import DEFAULT
from spinescope.testing.phantom import PhantomSpec, SpinePhantom


# ------------------------------------------------------------------ synthetic axial planes
def _disc_plane(radius_mm: float, *, bright: bool, px: float = 0.5, size_mm: float = 60.0) -> np.ndarray:
    """A square plane with a CSF-like disc of ``radius_mm`` at its centre on a tissue background."""
    n = int(size_mm / px)
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    inside = np.hypot((yy - c) * px, (xx - c) * px) <= radius_mm
    background, csf = 200.0, 900.0
    # small intensity noise, deterministic
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 8.0, (n, n))
    plane = np.full((n, n), background)
    plane = np.where(inside, csf, plane) if bright else np.where(inside, 60.0, 600.0)
    return plane + noise


def test_sac_area_of_a_bright_disc_is_its_area() -> None:
    radius = 7.0
    plane = _disc_plane(radius, bright=True)
    out = canal.sac_outline(plane, (plane.shape[0] / 2, plane.shape[1] / 2), 0.5, 0.5, bright_csf=True)
    assert out is not None
    assert out.area_mm2 == pytest.approx(math.pi * radius**2, rel=0.08)
    assert not out.truncated


def test_sac_area_of_a_dark_disc_on_t1_polarity() -> None:
    radius = 6.0
    plane = _disc_plane(radius, bright=False)
    out = canal.sac_outline(plane, (plane.shape[0] / 2, plane.shape[1] / 2), 0.5, 0.5, bright_csf=False)
    assert out is not None
    assert out.area_mm2 == pytest.approx(math.pi * radius**2, rel=0.08)


def test_sac_polarity_matters() -> None:
    plane = _disc_plane(6.0, bright=True)
    centre = (plane.shape[0] / 2, plane.shape[1] / 2)
    right = canal.sac_outline(plane, centre, 0.5, 0.5, bright_csf=True)
    wrong = canal.sac_outline(plane, centre, 0.5, 0.5, bright_csf=False)
    assert right is not None
    # the wrong polarity selects the dark tissue around the disc, which is much larger or absent
    assert wrong is None or wrong.area_mm2 > 2 * right.area_mm2 or wrong.truncated


def test_sac_touching_the_window_is_truncated() -> None:
    cfg = dataclasses.replace(DEFAULT, canal_axial_halfwindow_mm=5.0)
    plane = _disc_plane(7.0, bright=True)
    out = canal.sac_outline(plane, (plane.shape[0] / 2, plane.shape[1] / 2), 0.5, 0.5, True, cfg)
    assert out is not None
    assert out.truncated


def test_sac_seed_outside_the_slice_or_flat_image_gives_none() -> None:
    plane = _disc_plane(6.0, bright=True)
    assert canal.sac_outline(plane, (-5.0, 3.0), 0.5, 0.5, True) is None
    flat = np.full((80, 80), 300.0)
    assert canal.sac_outline(flat, (40.0, 40.0), 0.5, 0.5, True) is None


def test_sac_seed_far_from_csf_is_not_taken() -> None:
    plane = _disc_plane(4.0, bright=True)
    # seed 12 mm away from the disc: outside the 3 mm centre tolerance
    out = canal.sac_outline(plane, (plane.shape[0] / 2 + 24, plane.shape[1] / 2), 0.5, 0.5, True)
    assert out is None


def test_obliquity_and_correction_factor() -> None:
    assert canal.obliquity_deg([0, 0, 1], [0, 0, 1]) == pytest.approx(0.0)
    assert canal.obliquity_deg([0, 0, 1], [0, math.sin(math.radians(30)), math.cos(math.radians(30))]) == (
        pytest.approx(30.0)
    )
    # sign of a normal does not matter
    assert canal.obliquity_deg([0, 0, -1], [0, 0, 1]) == pytest.approx(0.0)
    assert canal.obliquity_factor(60.0) == pytest.approx(0.5)
    assert canal.obliquity_factor(0.0) == pytest.approx(1.0)


def test_ap_grade_boundaries_use_the_configured_cut_offs() -> None:
    assert canal.ap_grade(None) is None
    assert canal.ap_grade(9.99) == "absolute"
    assert canal.ap_grade(10.0) == "relative"
    assert canal.ap_grade(12.99) == "relative"
    assert canal.ap_grade(13.0) == "none"


def test_csa_grade_boundaries_use_the_configured_cut_offs() -> None:
    assert canal.csa_grade(None) is None
    assert canal.csa_grade(74.9) == "absolute"
    assert canal.csa_grade(75.0) == "relative"
    assert canal.csa_grade(99.9) == "relative"
    assert canal.csa_grade(100.0) == "none"


@pytest.mark.parametrize(
    ("csa", "ap", "expected"),
    [
        ("absolute", "absolute", "severe"),
        ("absolute", "relative", "moderate"),
        ("absolute", None, "moderate"),
        (None, "absolute", "moderate"),
        ("relative", "relative", "mild"),
        ("relative", "none", "mild"),
        ("none", "none", "none"),
        (None, None, None),
    ],
)
def test_stenosis_grade_table(csa: str | None, ap: str | None, expected: str | None) -> None:
    assert canal.stenosis_grade(csa, ap) == expected


def test_reference_area_needs_enough_levels_and_uses_the_quantile() -> None:
    assert canal.reference_area([100.0, 120.0]) is None
    areas = [80.0, 100.0, 120.0, 160.0]
    assert canal.reference_area(areas) == pytest.approx(np.quantile(areas, DEFAULT.canal_reference_quantile))


def test_grading_range_is_lumbar_only() -> None:
    assert canal.in_grading_range("L2-L3")
    assert canal.in_grading_range("L5-S1")
    assert canal.in_grading_range("L4")
    assert not canal.in_grading_range("T12-L1")
    assert not canal.in_grading_range("T11-T12")
    assert not canal.in_grading_range("T12")
    assert not canal.in_grading_range("not-a-level")


def test_module_registers_the_canal_analysis() -> None:
    """Registration by import (the built-in list in base.py is left alone on this branch; see the wave brief)."""
    from spinescope.analysis import base

    assert base._REGISTRY["canal"] is canal.analyze


def test_lumbar_grading_and_confidence_caps() -> None:
    """Absolute criteria give the suggested grade; unvalidated area carries the cap."""
    cfg = DEFAULT
    assert cfg.canal_axial_validated is False
    assert canal._axial_confidence(0.0, False, cfg) == pytest.approx(0.8)
    assert canal._axial_confidence(20.0, False, cfg) == pytest.approx(0.6)
    assert canal._axial_confidence(0.0, True, cfg) == pytest.approx(0.4)


def test_level_confidence_rules() -> None:
    assert canal._level_confidence(True, 0.9, 0.4) == pytest.approx(0.4)
    assert canal._level_confidence(False, 0.9, 0.4) == pytest.approx(0.4)
    assert canal._level_confidence(True, 0.8, None) == pytest.approx(0.4)
    assert canal._level_confidence(False, 0.8, None) is None


# ------------------------------------------------------------------ analysis on synthetic anchors
def _anchors_with_canal(ap_by_level: dict[str, float], *, confidence: float = 0.9) -> Anchors:
    a = Anchors()
    per_level = {}
    for i, (lab, ap) in enumerate(ap_by_level.items()):
        kind = "disc" if "-" in lab else "mid_body"
        per_level[lab] = {"z_mm": 100.0 - 30.0 * i, "ap_mm": ap, "kind": kind, "slice": 0}
    a.canal = {"points": [[0.0, 0.0, 0.0]], "per_level": per_level, "confidence": confidence}
    a.source_stack = "sag_t2"
    return a


def test_sagittal_only_analysis_grades_on_ap_with_lower_confidence() -> None:
    """No axial stack: sagittal AP only, warned, lower confidence than the canal confidence."""
    from spinescope.study import Study

    study = Study(stacks={})
    anchors = _anchors_with_canal({"L4-L5": 9.0, "L3-L4": 12.0, "L5": 14.0, "L4": 11.0})
    res = canal.analyze(study, anchors)
    assert isinstance(res, Result)
    assert any("no axial stack" in w for w in res.warnings)
    rows = {r["level"]: r for r in res.rows}
    assert rows["L4-L5"]["grade"] == "moderate"  # AP below 10 mm: one absolute criterion
    assert rows["L4-L5"]["confidence"] == pytest.approx(0.45)  # half the canal confidence
    assert rows["L3-L4"]["grade"] == "mild"
    assert rows["L5"]["grade"] == "none"
    assert rows["L4"]["grade"] == "mild"
    assert rows["L4-L5"]["csa_mm2"] is None
    assert any(f["level"] == "L4-L5" for f in res.flags)


def test_no_canal_geometry_warns_and_grades_nothing() -> None:
    from spinescope.study import Study

    anchors = Anchors()
    anchors.canal = None
    res = canal.analyze(Study(stacks={}), anchors)
    assert any("sagittal AP not assessed" in w for w in res.warnings)
    assert res.flags == []


def test_thoracic_levels_are_measured_not_graded() -> None:
    from spinescope.study import Study

    anchors = _anchors_with_canal({"T11-T12": 8.0, "T10": 9.0})
    res = canal.analyze(Study(stacks={}), anchors)
    for row in res.rows:
        assert row["grade"] is None
        assert row["ap_mm"] is not None
    assert any("outside the lumbar" in (r.get("note") or "") for r in res.rows if r["level"] == "T11-T12")


def test_findings_fields_are_written_only_with_their_figure(tmp_path) -> None:
    """The stenosis field comes with the saved profile figure; with no figure it is skipped, never written blank."""
    from spinescope.study import Study

    anchors = _anchors_with_canal({"L4-L5": 9.0, "L3-L4": 15.0, "L2-L3": 14.0})
    res = canal.analyze(Study(stacks={}), anchors)
    assert "profile" in res.figures
    paths = res.save(tmp_path)
    profile = next(p for p in paths if p.endswith("canal_profile.png"))
    fields = level_findings({"canal": res}, "L4-L5", anchors, figures={"canal": {"profile": profile}})
    assert "area_mm2" not in fields.get("central_canal", {})  # unvalidated: not a report field
    stenosis = fields["central_canal"]["stenosis"]
    assert stenosis["value"] == "moderate"
    assert stenosis["confidence"] in ("low", "medium", "high")
    assert stenosis["evidence"]
    no_fig = level_findings({"canal": res}, "L4-L5", anchors, figures={})
    assert "stenosis" not in no_fig.get("central_canal", {})
    assert canal.in_grading_range("L4-L5")


def test_profile_and_save_write_pngs(tmp_path) -> None:
    from spinescope.study import Study

    anchors = _anchors_with_canal({"L4-L5": 9.0, "L3-L4": 15.0, "L5": 12.0})
    res = canal.analyze(Study(stacks={}), anchors)
    paths = res.save(tmp_path)
    assert any(p.endswith("canal_profile.png") for p in paths)
    assert any(p.endswith("canal.csv") for p in paths)
    assert not any(str(k).startswith("_") for k in res.meta)  # private keys are never serialised


def test_run_registry_entry_on_a_study_without_axials() -> None:
    from spinescope.study import Study

    res = run("canal", Study(stacks={}), _anchors_with_canal({"L4-L5": 11.0}), None)
    assert res.name == "canal"


# ------------------------------------------------------------------ phantom end to end
@pytest.fixture(scope="module")
def phantom_bundle(tmp_path_factory):
    """Phantom study with its axial slabs and the truth-based canal measures, for the analysis tests."""
    from spinescope.study import load_study
    from spinescope.testing.phantom import write_standard_study

    root = tmp_path_factory.mktemp("canal_phantom")
    (root / "study").mkdir()
    ph = SpinePhantom(PhantomSpec())
    write_standard_study(ph, root / "study")
    study = load_study(root / "study")
    anchors = Anchors.from_truth(ph.truth)
    return ph, study, anchors


def test_phantom_axial_areas_are_within_the_plausible_range(phantom_bundle) -> None:
    ph, study, anchors = phantom_bundle
    anchors.canal = {"points": ph.canal_line.tolist(), "per_level": {}, "confidence": 0.9}
    anchors.canal["per_level"] = {
        lab: {
            "z_mm": float(d.center[2]),
            "ap_mm": 16.0 - DEFAULT.canal_ap_bias_mm,
            "kind": "disc",
            "slice": 0,
        }
        for lab, d in anchors.discs.items()
    }
    cfg = dataclasses.replace(DEFAULT, canal_axial_contrasts=("t2", "t1"))
    res = canal.analyze(study, anchors, cfg)
    areas = [r["csa_mm2"] for r in res.rows if r.get("scope") == "disc" and r.get("csa_mm2") is not None]
    for area in areas:
        assert DEFAULT.canal_csa_min_mm2 <= area <= DEFAULT.canal_csa_max_mm2
    # the phantom sac is a canal of about 8 mm radius: well above the absolute cut-off
    for r in res.rows:
        if r.get("csa_mm2") is not None:
            assert r["grade"] in ("none", "mild", "moderate", "severe")
            assert r["confidence"] <= cfg.canal_axial_unvalidated_cap + 1e-9
            assert r["axial_slab"] is not None


def test_narrow_sagittal_canal_gives_a_worse_grade() -> None:
    """A narrower sagittal canal (AP 16 mm against 8 mm) moves the suggested grade from none to moderate."""
    from spinescope.study import Study

    wide = canal.analyze(Study(stacks={}), _anchors_with_canal({"L4-L5": 16.0}))
    narrow = canal.analyze(Study(stacks={}), _anchors_with_canal({"L4-L5": 8.0}))
    assert wide.rows[0]["grade"] == "none"
    assert narrow.rows[0]["grade"] == "moderate"
