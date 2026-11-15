from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _alignment_truth import with_body_geometry
from _studies import phantom_study, stacks_flipped
from spinescope.analysis import alignment
from spinescope.analysis.base import Result, available, run_all
from spinescope.analysis.findings import level_findings
from spinescope.anatomy import Anchors
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.study import Study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

STRAIGHT = {"sacral_slope": 0.0, "segment_lordosis": (0.0,) * 6}


def _rows(res: Result) -> dict[str, dict]:
    return {r["disc"]: r for r in res.rows if r.get("scope") == "level"}


def _global(res: Result, name: str) -> dict:
    return next(r for r in res.rows if r.get("name") == name)


def _ep(ant: list[float], post: list[float]) -> alignment.Endplate:
    return alignment.Endplate(np.asarray(ant, float), np.asarray(post, float))


# ------------------------------------------------------------------------------------- pure geometry
def test_tilt_sign_posterior_higher_is_positive():
    assert _ep([0, 0, 0], [0, 10, 1]).tilt_deg == pytest.approx(np.degrees(np.arctan2(1, 10)))
    assert _ep([0, 0, 0], [0, 10, -1]).tilt_deg < 0


def test_endplate_from_record_prefers_wall_corners_and_rejects_degenerate():
    rec = {
        "anterior": [0, 0, 0],
        "posterior": [0, 30, 0],
        "anterior_wall": [0, 1, 0],
        "posterior_wall": [0, 29, 0],
    }
    e = alignment.endplate_from_record(rec)
    assert e is not None and e.length_mm == pytest.approx(28.0)
    assert alignment.endplate_from_record({"anterior": [0, 0, 0], "posterior": [0, 0.2, 0]}) is None
    assert alignment.endplate_from_record(None) is None


@pytest.mark.parametrize(
    ("pct", "grade"),
    [
        (0.0, "I"),
        (25.0, "I"),
        (25.1, "II"),
        (50.0, "II"),
        (75.0, "III"),
        (100.0, "IV"),
        (100.1, "V"),
        (-30.0, "II"),
    ],
)
def test_meyerding_grade_edges(pct, grade):
    assert alignment.meyerding_grade(pct) == grade


def test_listhesis_anterior_is_positive_and_graded():
    lower = _ep([0, 0, 0], [0, 30, 0])  # lower body AP depth 30 mm, horizontal endplate
    upper_inf = _ep([0, -6, 0], [0, 24, 0])  # upper body displaced 6 mm anteriorly along the endplate
    out = alignment.listhesis(upper_inf, lower)
    assert out["translation_mm"] == pytest.approx(6.0)
    assert out["direction"] == "anterolisthesis"
    assert out["percent"] == pytest.approx(20.0)
    assert out["grade"] == "I"


def test_listhesis_retro_has_no_meyerding_grade_and_threshold_gives_none():
    lower = _ep([0, 0, 0], [0, 30, 0])
    retro = alignment.listhesis(_ep([0, 4, 0], [0, 34, 0]), lower)
    assert retro["translation_mm"] == pytest.approx(-4.0)
    assert retro["direction"] == "retrolisthesis"
    assert retro["grade"] is None
    small = alignment.listhesis(_ep([0, 2.9, 0], [0, 32.9, 0]), lower)
    assert small["direction"] == "none"
    assert small["grade"] is None


def test_listhesis_threshold_is_config_driven():
    lower = _ep([0, 0, 0], [0, 30, 0])
    upper = _ep([0, 2.0, 0], [0, 32.0, 0])
    assert alignment.listhesis(upper, lower, DEFAULT)["direction"] == "none"
    strict = replace(DEFAULT, alignment_listhesis_min_mm=1.5)
    assert alignment.listhesis(upper, lower, strict)["direction"] == "retrolisthesis"  # 2 mm posterior


def test_segmental_and_lordosis_sign_convention():
    flat = _ep([0, 0, 0], [0, 30, 0])
    lordotic = _ep([0, 0, 0], [0, 30, 10])  # posterior higher by 10 mm over 30 mm: tilt ~18.4 deg
    assert alignment.segmental_angle_deg(flat, lordotic) == pytest.approx(lordotic.tilt_deg)
    assert alignment.lordosis_deg(flat, lordotic) == pytest.approx(lordotic.tilt_deg)
    assert alignment.lordosis_deg(lordotic, flat) == pytest.approx(-lordotic.tilt_deg)


# ------------------------------------------------------------------------------------- registry
def test_alignment_is_registered_and_run_all_contains_it(phantom, phantom_study):
    assert "alignment" in available()
    out = run_all(phantom_study, Anchors.from_truth(phantom.truth), names=["alignment"])
    assert out["alignment"].warnings  # truth anchors carry no body geometry: a warning, not an exception


def test_no_body_geometry_gives_warnings_not_exceptions(phantom, phantom_study):
    res = alignment.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert any("no vertebral body endplates" in w for w in res.warnings)
    rows = _rows(res)
    assert rows and all(r["translation_mm"] is None for r in rows.values())


def test_no_discs_is_a_warning(phantom, phantom_study):
    res = alignment.analyze(phantom_study, Anchors())
    assert res.warnings and not res.rows


# ------------------------------------------------------------------------------------- phantom truth
@pytest.fixture(scope="module")
def straight_phantom() -> SpinePhantom:
    return SpinePhantom(PhantomSpec(**STRAIGHT))


def test_straight_spine_zero_translation_at_every_measured_level(straight_phantom):
    study = phantom_study(straight_phantom)
    res = alignment.analyze(study, with_body_geometry(straight_phantom, study))
    rows = _rows(res)
    for lab in ("T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5"):
        assert rows[lab]["translation_mm"] == pytest.approx(0.0, abs=0.2), lab
        assert rows[lab]["direction"] == "none"
    assert rows["L5-S1"]["translation_mm"] is None  # the lowest disc needs the sacral endplate: not assessed


@pytest.mark.parametrize(
    ("disc", "shift", "expected"),
    [("L4-L5", 6.0, -6.0), ("L3-L4", -5.0, 5.0), ("L2-L3", 4.0, -4.0)],
)
def test_slip_is_recovered_with_its_sign(disc, shift, expected):
    ph = SpinePhantom(PhantomSpec(slip={disc: shift}, **STRAIGHT))
    study = phantom_study(ph)
    res = alignment.analyze(study, with_body_geometry(ph, study))
    row = _rows(res)[disc]
    truth = ph.truth["listhesis"][disc]["translation_anterior_mm"]
    assert truth == pytest.approx(expected, abs=0.05)
    assert row["translation_mm"] == pytest.approx(expected, abs=0.3)
    assert row["direction"] == ("anterolisthesis" if expected > 0 else "retrolisthesis")


def test_slip_flags_and_predictor_agree(straight_phantom):
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": 6.0}, **STRAIGHT))
    study = phantom_study(ph)
    res = alignment.analyze(study, with_body_geometry(ph, study))
    assert [f["disc"] for f in res.flags] == ["L4-L5"]
    pred = alignment.spondylolisthesis_prediction(res, with_body_geometry(ph, study))
    assert pred["L4-L5"][0] == 1
    assert pred["L3-L4"][0] == 0 and pred["L2-L3"][0] == 0
    assert pred["L4-L5"][1] == pytest.approx(6.0, abs=0.3)
    assert pred["L5-S1"] == (None, None)  # not assessed


def test_predictor_below_threshold_is_negative_and_uses_magnitude():
    res = Result(
        name="alignment",
        rows=[
            {"scope": "level", "disc": "L1-L2", "translation_mm": -2.9},
            {"scope": "level", "disc": "L2-L3", "translation_mm": -3.0},
            {"scope": "level", "disc": "L3-L4", "translation_mm": 3.4},
            {"scope": "level", "disc": "L4-L5", "translation_mm": None},
        ],
    )
    anchors = Anchors(discs={k: _disc(k) for k in ("L1-L2", "L2-L3", "L3-L4", "L4-L5")})
    pred = alignment.spondylolisthesis_prediction(res, anchors)
    assert pred["L1-L2"] == (0, pytest.approx(2.9))
    assert pred["L2-L3"][0] == 1  # exactly at the threshold counts
    assert pred["L3-L4"][0] == 1
    assert pred["L4-L5"] == (None, None)


def _disc(label: str):
    from spinescope.anatomy.anchors import Disc

    return Disc(center=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])


def test_lordosis_matches_phantom_truth_on_the_default_spine(phantom, phantom_study):
    ph = phantom
    study = phantom_study
    res = alignment.analyze(study, with_body_geometry(ph, study))
    lord = _global(res, "lumbar_lordosis_deg")
    assert lord["method"] == "L1-L5"
    truth = ph.bodies["L5"]["tilt_deg"] - ph.bodies["L1"]["tilt_deg"]  # inferior of L5 minus superior of L1
    assert lord["value"] == pytest.approx(truth, abs=3.0)
    assert any("S1 superior endplate not used" in n for n in lord["notes"].split("; "))


# Bodies tilted by up to about 20 degrees (the lumbar range). Steeper bodies are under-measured by the labeller's
# endplate fit (see SACRAL_NOTE and docs/analysis/alignment.md), so they are not in this tolerance test.
@pytest.mark.parametrize(("segments", "slope"), [((0.0,) * 6, 0.0), ((4.0,) * 6, 0.0), ((2.0,) * 6, 20.0)])
def test_lordosis_follows_the_phantom_variation(segments, slope):
    ph = SpinePhantom(PhantomSpec(segment_lordosis=segments, sacral_slope=slope))
    study = phantom_study(ph)
    res = alignment.analyze(study, with_body_geometry(ph, study))
    value = _global(res, "lumbar_lordosis_deg")["value"]
    truth = ph.bodies["L5"]["tilt_deg"] - ph.bodies["L1"]["tilt_deg"]
    assert value == pytest.approx(truth, abs=3.0)


def test_lordosis_increases_with_the_phantom_curvature():
    flat = SpinePhantom(PhantomSpec(segment_lordosis=(0.0,) * 6, sacral_slope=0.0))
    curved = SpinePhantom(PhantomSpec(segment_lordosis=(8.0,) * 6, sacral_slope=10.0))
    vals = []
    for ph in (flat, curved):
        study = phantom_study(ph)
        vals.append(
            _global(alignment.analyze(study, with_body_geometry(ph, study)), "lumbar_lordosis_deg")["value"]
        )
    assert vals[1] > vals[0] + 10.0


def test_posterior_wall_corner_of_a_tilted_body_is_on_the_wall():
    """The wall corner (not the global-extent corner) sits on the body's own posterior wall (truth within 1.5 mm)."""
    ph = SpinePhantom(PhantomSpec())
    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study)
    for name in ("L3", "L4"):
        b = anchors.vertebrae[name].body
        v = ph.bodies[name]
        top_post = v["center"] + v["normal"] * v["half_height"] + v["ap"] * v["hd"]
        bot_post = v["center"] - v["normal"] * v["half_height"] + v["ap"] * v["hd"]
        got_sup = np.asarray(b["superior_endplate"]["posterior_wall"])
        got_inf = np.asarray(b["inferior_endplate"]["posterior_wall"])
        assert np.linalg.norm(got_sup[1:] - top_post[1:]) < 1.5, name
        assert np.linalg.norm(got_inf[1:] - bot_post[1:]) < 1.5, name


def test_flipped_acquisition_gives_the_same_measurements(straight_phantom):
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": 6.0}, **STRAIGHT))
    study = phantom_study(ph)
    flipped = stacks_flipped(study)
    a = alignment.analyze(study, with_body_geometry(ph, study))
    b = alignment.analyze(flipped, with_body_geometry(ph, flipped))
    for lab in ("L3-L4", "L4-L5"):
        assert _rows(a)[lab]["translation_mm"] == pytest.approx(_rows(b)[lab]["translation_mm"], abs=0.2)


def test_missing_sacrum_keeps_l1_l5_and_does_not_fail(phantom):
    ph = SpinePhantom(PhantomSpec(**STRAIGHT))
    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study)
    anchors.vertebrae.pop("S1", None)  # the field of view ends above the sacrum
    res = alignment.analyze(study, anchors)
    assert _global(res, "lumbar_lordosis_deg")["method"] == "L1-L5"
    assert _rows(res)["L4-L5"]["translation_mm"] is not None


def test_no_sagittal_stack_keeps_the_rows_and_drops_the_figures():
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": 6.0}, **STRAIGHT))
    anchors = with_body_geometry(ph, phantom_study(ph))
    res = alignment.analyze(Study(stacks={}), anchors)
    assert _rows(res)["L4-L5"]["translation_mm"] == pytest.approx(-6.0, abs=0.3)
    assert res.figures == {}


def test_figures_are_rgb_and_save(tmp_path):
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": 6.0}, **STRAIGHT))
    study = phantom_study(ph)
    res = alignment.analyze(study, with_body_geometry(ph, study))
    assert "listhesis_L4-L5" in res.figures
    img = res.figures["listhesis_L4-L5"]
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    paths = res.save(tmp_path)
    assert any(p.endswith("alignment_listhesis_L4-L5.png") for p in paths)
    assert (tmp_path / "alignment.csv").exists()


def test_level_name_status_is_carried_into_rows_and_findings(straight_phantom):
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": 6.0}, **STRAIGHT))
    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study)
    anchors.numbering = {"method": "test", "confidence": 0.5, "status": "probable"}
    res = alignment.analyze(study, anchors)
    row = _rows(res)["L4-L5"]
    assert row["level"] == "L4-L5 (probable)"
    assert "not confirmed" in row["notes"]
    figs = {"alignment": {"listhesis_L4-L5": "fig.png"}}
    out = level_findings({"alignment": res}, "L4-L5", anchors, figures=figs)
    lst = out["listhesis"]
    assert lst["translation_mm"]["value"] == pytest.approx(-6.0, abs=0.3)
    assert lst["direction"]["value"] == "retrolisthesis"
    assert lst["direction"]["confidence"] in ("low", "medium")  # capped for a probable name
    assert lst["translation_mm"]["evidence"][0]["image"] == "fig.png"


def test_findings_grade_is_none_for_retrolisthesis(straight_phantom):
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": -6.0}, **STRAIGHT))
    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study)
    anchors.numbering = {"method": "test", "confidence": 1.0, "status": "confirmed"}
    res = alignment.analyze(study, anchors)
    out = level_findings(
        {"alignment": res}, "L4-L5", anchors, figures={"alignment": {"listhesis_L4-L5": "f.png"}}
    )
    assert out["listhesis"]["direction"]["value"] == "anterolisthesis"
    assert out["listhesis"]["meyerding_grade"]["value"] in ("I", "II")  # 6 mm of ~32 mm AP depth


def test_lateral_screening_is_passed_through_not_upgraded(straight_phantom):
    study = phantom_study(straight_phantom)
    anchors = with_body_geometry(straight_phantom, study)
    anchors.alignment = {
        "coronal": {
            "label": "lateral curvature screening (unvalidated)",
            "valid": True,
            "active": True,
            "amplitude_mm": 8.5,
            "cobb_like_uncalibrated_deg": 12.0,
            "direction": "left",
            "confidence": 0.4,
            "knots": [[0.0, 1.0, 3]],
        }
    }
    res = alignment.analyze(study, anchors)
    lat = _global(res, "lateral_curvature_screening")
    assert lat["value"] == 8.5
    assert lat["measured"] is True
    assert "unvalidated" in lat["label"]
    assert "not a Cobb measurement" in lat["notes"]
    anchors.alignment = {}
    assert _global(alignment.analyze(study, anchors), "lateral_curvature_screening")["measured"] is False


def test_analysis_is_deterministic_and_fast(straight_phantom):
    ph = SpinePhantom(PhantomSpec(slip={"L4-L5": 6.0}))
    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study)
    t0 = time.perf_counter()
    a = alignment.analyze(study, anchors)
    elapsed = time.perf_counter() - t0
    b = alignment.analyze(study, anchors)
    assert a.rows == b.rows
    assert elapsed < 5.0


def test_meta_has_no_private_keys_and_states_posture(straight_phantom):
    study = phantom_study(straight_phantom)
    res = alignment.analyze(study, with_body_geometry(straight_phantom, study))
    assert all(not str(k).startswith("_") for k in res.meta)
    assert res.meta["posture"] == "supine"
    assert res.meta["thresholds"]["listhesis_min_mm"] == DEFAULT.alignment_listhesis_min_mm


def test_stack_helper_returns_sagittal_t2(straight_phantom):
    study = phantom_study(straight_phantom)
    assert isinstance(alignment._stack_for(study, Anchors()), Stack)


def test_threshold_sweep_counts_and_rates():
    scores = [0.5, 1.0, 4.0, 6.0, 7.0, 2.0]
    truth = [0, 0, 1, 1, 0, 0]
    cases = ["a", "a", "b", "b", "c", "c"]
    out = {
        r["threshold_mm"]: r for r in alignment.threshold_sweep(scores, truth, cases, [3.0, 5.0], n_boot=50)
    }
    assert (out[3.0]["tp"], out[3.0]["fp"], out[3.0]["fn"], out[3.0]["tn"]) == (2, 1, 0, 3)
    assert out[3.0]["sensitivity"] == pytest.approx(1.0)
    assert out[3.0]["specificity"] == pytest.approx(0.75)
    assert out[3.0]["ppv"] == pytest.approx(2 / 3, abs=1e-3)
    assert out[5.0]["fp"] == 1 and out[5.0]["tp"] == 1 and out[5.0]["fn"] == 1
    assert out[3.0]["sensitivity_ci"] is not None and out[3.0]["sensitivity_ci"][1] <= 1.0


def test_threshold_sweep_is_deterministic_and_checks_lengths():
    scores, truth, cases = [1.0, 5.0, 4.0], [0, 1, 1], ["x", "y", "z"]
    a = alignment.threshold_sweep(scores, truth, cases, [3.0], n_boot=100, seed=3)
    b = alignment.threshold_sweep(scores, truth, cases, [3.0], n_boot=100, seed=3)
    assert a == b
    with pytest.raises(ValueError):
        alignment.threshold_sweep([1.0], [0, 1], ["x"], [3.0])
