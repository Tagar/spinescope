from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _alignment_truth import with_body_geometry
from _studies import keep_contrasts, phantom_study
from spinescope.analysis import herniation as H
from spinescope.analysis.base import Result, available, run
from spinescope.analysis.evaluate import predictors
from spinescope.analysis.findings import level_findings
from spinescope.anatomy import Anchors
from spinescope.config import DEFAULT
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

HERNIATED = "L4-L5"


def _disc_rows(res: Result) -> dict[str, dict]:
    return {r["disc"]: r for r in res.rows if r.get("scope") == "disc"}


@pytest.fixture(scope="module")
def herniated_study_and_anchors():
    """Phantom with a central posterior herniation (8 mm beyond the body margin) at L4-L5, truth body geometry."""
    ph = SpinePhantom(PhantomSpec(herniation={HERNIATED: ("central", 8.0)}, herniation_displaces_sac=True))
    study = phantom_study(ph)
    return study, with_body_geometry(ph, study, Anchors.from_truth(ph.truth))


@pytest.fixture(scope="module")
def normal_study_and_anchors():
    ph = SpinePhantom(PhantomSpec())
    study = phantom_study(ph)
    return study, with_body_geometry(ph, study, Anchors.from_truth(ph.truth))


# ------------------------------------------------------------------------------------- pure geometry
def test_line_through_corners_is_linear_and_extrapolates() -> None:
    h = np.array([-5.0, 0.0, 5.0, 10.0])
    s = H.line_s_at(h, s_up=15.0, h_up=5.0, s_low=17.0, h_low=-5.0)
    assert np.allclose(s, [17.0, 16.0, 15.0, 14.0])


def test_line_is_undefined_for_corners_at_one_height() -> None:
    s = H.line_s_at(np.array([0.0, 1.0]), 15.0, 2.0, 16.0, 2.4)
    assert np.isnan(s).all()


def test_posterior_edge_needs_two_samples_above_threshold() -> None:
    prof = np.array([100.0, 120.0, 900.0, 110.0, 950.0, 980.0, 990.0])
    assert H.posterior_edge_index(prof, 500.0) == 4  # the lone spike at index 2 is noise
    assert H.posterior_edge_index(np.array([1.0, 2.0]), 500.0) is None


def test_crossing_is_interpolated_between_samples() -> None:
    s = np.arange(5, dtype=float)
    prof = np.array([0.0, 0.0, 100.0, 300.0, 300.0])
    # threshold 200 lies halfway between samples 2 and 3
    assert H.interp_crossing(s, prof, 200.0, 3) == pytest.approx(2.5)
    assert H.interp_crossing(s, prof, 200.0, 0) == 0.0


def test_contiguous_run_contains_the_maximum() -> None:
    v = np.array([0.2, 1.5, 2.0, 1.2, 0.4, 3.0, 1.1])
    assert H.contiguous_run(v, 1.0) == (5, 6)  # the run around the maximum (3.0)
    assert H.contiguous_run(np.array([0.1, 0.2]), 1.0) is None
    assert H.contiguous_run(np.array([np.nan, 2.0, np.nan]), 1.0) == (1, 1)


@pytest.mark.parametrize(
    "e, base, expected",
    [
        (0.5, 7.0, "normal"),  # below the 1 mm tolerance
        (1.0, 7.0, "bulge"),  # tolerance reached, under 3 mm: bulge
        (2.9, 7.0, "bulge"),
        (3.0, 6.0, "protrusion"),  # beyond 3 mm, base wider than the displacement
        (5.0, 6.0, "protrusion"),
        (5.0, 3.0, "extrusion"),  # displacement wider than its base
    ],
)
def test_sagittal_classes_follow_fardon_thresholds(e: float, base: float, expected: str) -> None:
    assert H.classify_sagittal(e, base, DEFAULT) == expected


def test_nan_excursion_is_not_classified() -> None:
    with pytest.raises(ValueError):
        H.classify_sagittal(float("nan"), 5.0)


def test_corner_consistency_rejects_a_line_far_from_the_disc() -> None:
    assert H.corners_consistent(15.0, 16.0, s_post=16.0)
    assert not H.corners_consistent(15.0, 16.0, s_post=40.0)
    assert not H.corners_consistent(np.nan, 16.0, 16.0)


# ------------------------------------------------------------------------------------ analysis on phantom
def test_registered_and_listed() -> None:
    assert "herniation" in available()
    assert predictors().get("disc_herniation") == "herniation"
    assert predictors().get("disc_bulging") == "herniation"


def test_herniation_adds_excursion_to_the_same_disc(
    herniated_study_and_anchors, normal_study_and_anchors
) -> None:
    """Same disc, with and without a herniation: the herniation adds several mm of posterior excursion."""
    study_h, anchors_h = herniated_study_and_anchors
    study_n, anchors_n = normal_study_and_anchors
    e_h = _disc_rows(run("herniation", study_h, anchors_h, DEFAULT))[HERNIATED]["e_raw_mm"]
    e_n = _disc_rows(run("herniation", study_n, anchors_n, DEFAULT))[HERNIATED]["e_raw_mm"]
    assert e_h - e_n >= 3.0


def test_herniated_disc_is_called_central_protrusion(herniated_study_and_anchors) -> None:
    study, anchors = herniated_study_and_anchors
    rows = _disc_rows(run("herniation", study, anchors, DEFAULT))
    assert rows[HERNIATED]["contour"] in ("protrusion", "extrusion")
    assert rows[HERNIATED]["herniation_suggestion"] == 1
    assert rows[HERNIATED]["herniation_location"] == "central"
    assert "possibly L5" in rows[HERNIATED]["possible_roots"]


def test_herniation_is_sagittal_only_and_says_so(phantom, phantom_dir) -> None:
    """With axial stacks present the analysis says it does not measure the axial contour, side or lateral zones."""
    from spinescope import load_study

    study = load_study(phantom_dir["root"])
    anchors = with_body_geometry(phantom, study, Anchors.from_truth(phantom.truth))
    res = run("herniation", study, anchors, DEFAULT)
    assert any("axial contour" in w for w in res.warnings)
    assert all("side" not in r for r in res.rows if r.get("scope") == "disc")


def test_slice_rows_and_figures_are_written(herniated_study_and_anchors) -> None:
    study, anchors = herniated_study_and_anchors
    res = run("herniation", study, anchors, DEFAULT)
    slices = [r for r in res.rows if r.get("scope") == "slice" and r["disc"] == HERNIATED]
    assert slices and all(0.0 <= r["valid_fraction"] <= 1.0 for r in slices)
    assert f"sag_{HERNIATED}" in res.figures
    assert res.figures[f"sag_{HERNIATED}"].dtype == np.uint8 and res.figures[f"sag_{HERNIATED}"].ndim == 3


def test_confidence_is_bounded_and_drops_with_each_weak_input(herniated_study_and_anchors) -> None:
    study, anchors = herniated_study_and_anchors
    conf = _disc_rows(run("herniation", study, anchors, DEFAULT))[HERNIATED]["confidence"]
    assert 0.0 <= conf <= 1.0
    corners = H.posterior_corners(anchors, "L4", "L5")
    assert corners is not None
    good = H.SliceMeasure(
        e_max=5.0,
        e_adj=2.0,
        h_at_max=0.0,
        base_mm=6.0,
        valid_fraction=1.0,
        thr=0.0,
        csf=1400.0,
        ann=150.0,
        contrast_sigma=20.0,
        geometry="traced",
    )
    global_corners = H.LineCorners(corners.up, corners.low, "global")
    assert H._confidence(good, global_corners, 2) < H._confidence(good, corners, 2)
    anchored = replace(good, geometry="anchor")
    assert H._confidence(anchored, corners, 2) < H._confidence(good, corners, 2)
    one_slice = H._confidence(good, corners, 1)
    assert one_slice < H._confidence(good, corners, 2)


def test_normal_phantom_discs_are_not_called_herniated(normal_study_and_anchors) -> None:
    """Without a herniation, the mid discs are not called herniated. L5-S1 is excluded: the phantom sacrum is tilted and
    its posterior corners are not a clean reference there (a known phantom limitation, see the doc)."""
    study, anchors = normal_study_and_anchors
    rows = _disc_rows(run("herniation", study, anchors, DEFAULT))
    for lab in ("L2-L3", "L3-L4", "L4-L5"):
        assert rows[lab]["herniation_suggestion"] == 0, lab


def test_t1_only_study_is_not_assessed() -> None:
    ph = SpinePhantom(PhantomSpec())
    study = keep_contrasts(phantom_study(ph, contrasts=("t1",)), ("t1",))
    res = run("herniation", study, Anchors.from_truth(ph.truth), DEFAULT)
    assert res.rows == []
    assert any("not assessed" in w for w in res.warnings)


def test_missing_body_records_gives_not_assessed_rows(herniated_study_and_anchors) -> None:
    study, _anchors = herniated_study_and_anchors
    bare = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)  # no body geometry
    res = run("herniation", study, bare, DEFAULT)
    rows = _disc_rows(res)
    assert rows and all(r["contour"] is None for r in rows.values())
    assert all(r["herniation_suggestion"] is None and r["bulging_suggestion"] is None for r in rows.values())
    assert all("body records" in r["reason"] for r in rows.values())


def test_no_metadata_keys_are_serialised(herniated_study_and_anchors) -> None:
    study, anchors = herniated_study_and_anchors
    res = run("herniation", study, anchors, DEFAULT)
    d = res.to_dict()
    assert not any(str(k).startswith("_") for k in d["meta"])


def test_harness_predictors_return_flags_and_scores(herniated_study_and_anchors) -> None:
    from spinescope.analysis.evaluate import _PREDICTORS

    study, anchors = herniated_study_and_anchors
    res = run("herniation", study, anchors, DEFAULT)
    herniation_fn = _PREDICTORS[("herniation", "disc_herniation")]
    bulge_fn = _PREDICTORS[("herniation", "disc_bulging")]
    h = herniation_fn(res, anchors)
    b = bulge_fn(res, anchors)
    assert h[HERNIATED][0] == 1
    assert b[HERNIATED][0] == 1 and b[HERNIATED][1] is not None
    assert set(h) == set(anchors.discs)


def test_findings_fields_are_written_with_evidence(herniated_study_and_anchors) -> None:
    study, anchors = herniated_study_and_anchors
    res = run("herniation", study, anchors, DEFAULT)
    figs = {"herniation": {f"sag_{HERNIATED}": "figs/sag.png"}}
    obj = level_findings({"herniation": res}, HERNIATED, anchors, figures=figs)
    contour = obj["disc"]["contour"]
    assert contour["value"] in ("protrusion", "extrusion")
    assert contour["evidence"] and contour["evidence"][0]["image"] == "figs/sag.png"
    assert obj["disc"]["herniation_location"]["value"] == "central"
    assert obj["disc"]["herniation_size_mm"]["value"] >= 2.5
