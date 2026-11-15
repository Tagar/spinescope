"""Scoliosis-aware labelling (spinescope.anatomy.scoliosis, lab30): lateral curve of the canal, Cobb-like angle and
apex, and disc measurements on the sagittal plane through each disc's own lateral centre.

Most tests need no network: the curve and the disc planes are driven by the phantom's analytic tissue labels (the
oracle of test_anatomy_ml), so they test the geometry, not the learned model. The thresholds are Config defaults.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from importlib import resources

import numpy as np
import pytest

from spinescope.anatomy import scoliosis
from spinescope.anatomy._validate import score_anchors
from spinescope.anatomy.anchors import SCHEMA_VERSION, Anchors
from spinescope.anatomy.ml import postprocess
from spinescope.anatomy.ml.labeller import CANAL, SegmentedStack
from spinescope.anatomy.ml.postprocess import anchors_from_segmentation
from spinescope.config import DEFAULT
from spinescope.testing.robustness import build_case, scenario
from test_anatomy_ml import _oracle_segmentation


def _segmentation_with_canal(rows: int, cols: int, canal: np.ndarray) -> SegmentedStack:
    """A synthetic stack whose canal class is ``canal`` (slices, rows, cols); no discs. For the edge cases."""
    from spinescope.geometry import Stack

    n = canal.shape[0]
    ipp = np.stack([np.linspace(30.0, -30.0, n), np.zeros(n), np.zeros(n)], axis=1)
    stack = Stack(
        key="synthetic",
        vol=np.zeros((n, rows, cols)),
        ipp=ipp,
        row_dir=np.array([0.0, 1.0, 0.0]),
        col_dir=np.array([0.0, 0.0, 1.0]),
        row_mm=1.0,
        col_mm=1.0,
    )
    from spinescope.anatomy.ml.grid import plane_grid

    grids = [plane_grid(stack, k, 1.0) for k in range(n)]
    probs = np.zeros((n, 4, rows, cols), np.float32)
    probs[:, CANAL] = canal.astype(np.float32)
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=(0.0, 1.0))


def _gaussian_curve(amp: float, sigma: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    z = np.arange(-100.0, 100.0 + 5.0, 5.0)
    return z, amp * np.exp(-((z / sigma) ** 2))


# ----------------------------------------------------------------------------- Cobb-like angle (pure geometry)


def test_straight_canal_has_no_cobb_angle_and_no_amplitude():
    z = np.arange(-100.0, 105.0, 5.0)
    res = scoliosis.cobb_from_knots(z, np.full(z.shape, 3.0), DEFAULT)
    assert res is not None
    assert res["cobb_deg"] == pytest.approx(0.0, abs=1e-9)
    assert res["amplitude_mm"] == pytest.approx(0.0, abs=1e-9)


def test_gaussian_curve_gives_the_analytic_cobb_angle_apex_and_direction():
    z, x = _gaussian_curve(10.0)
    res = scoliosis.cobb_from_knots(z, x, DEFAULT)
    assert res is not None
    # tangent maximum of A exp(-(z/s)^2) is 2A/(s sqrt 2) exp(-1/2) = 8.1 deg each side: spread about 16 deg
    assert 13.0 <= res["cobb_deg"] <= 18.0
    assert abs(res["apex_z"]) <= 10.0
    # Amplitude is from the chord of the trimmed interior: 10 mm less the curve's end height (1.4 mm) and smoothing.
    assert 7.0 <= res["amplitude_mm"] <= 9.0
    assert res["direction"] == "left"  # positive LPS x is the patient's left


def test_negative_curve_points_right():
    z, x = _gaussian_curve(-10.0)
    res = scoliosis.cobb_from_knots(z, x, DEFAULT)
    assert res is not None and res["direction"] == "right"


def test_cobb_needs_four_knots():
    assert scoliosis.cobb_from_knots(np.array([0.0, 5.0, 10.0]), np.zeros(3), DEFAULT) is None


# ----------------------------------------------------------------------------- canal knots


def test_knot_is_the_weighted_centroid_over_slices():
    # One height bin, two slices at x=-5 (weight 3) and x=+5 (weight 1): centroid -2.5.
    z = np.array([1.0, 1.0, 1.0, 1.0])
    x = np.array([-5.0, -5.0, -5.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    k = np.array([0, 0, 0, 1], dtype=np.intp)
    zk, xk, sup = scoliosis.knots_from_samples(z, x, w, k, DEFAULT)
    assert zk.size == 1
    assert xk[0] == pytest.approx(-2.5)
    assert int(sup[0]) == 2


def test_bins_seen_on_one_slice_are_dropped():
    # Bin 0 (z near 1) is seen on two slices and kept; bin 4 (z near 20) only on slice 0 and is dropped.
    z = np.array([1.0, 1.0, 20.0, 20.0])
    x = np.array([-5.0, 5.0, 3.0, 4.0])
    w = np.ones(4)
    k = np.array([0, 1, 0, 0], dtype=np.intp)
    zk, xk, _sup = scoliosis.knots_from_samples(z, x, w, k, DEFAULT)
    assert zk.size == 1
    assert zk[0] == pytest.approx(2.5) and xk[0] == pytest.approx(0.0)


def test_no_canal_gives_an_invalid_inactive_curve():
    seg = _segmentation_with_canal(20, 20, np.zeros((5, 20, 20)))
    curve = scoliosis.measure_curve(seg, DEFAULT)
    assert curve.valid is False and curve.active is False
    rec = curve.to_dict()
    assert rec["valid"] is False and rec["cobb_like_uncalibrated_deg"] is None
    assert np.all(np.isnan(curve.x_at(np.array([0.0, 10.0]))))


def test_slice_spacing_is_the_median_gap():
    seg = _segmentation_with_canal(20, 20, np.zeros((5, 20, 20)))
    assert scoliosis.slice_spacing_mm(seg) == pytest.approx(15.0)  # five slices from x=30 to x=-30


# ----------------------------------------------------------------------------- the phantom (oracle canal)


@pytest.fixture(scope="module")
def oracle_scoliosis():
    case = build_case(scenario("scoliosis"), "realistic")
    stack = case.study.best("sag")
    return case, _oracle_segmentation(stack, case.phantom)


def test_straight_phantom_has_a_negligible_curve(phantom, phantom_study):
    seg = _oracle_segmentation(phantom_study.best("sag"), phantom)
    curve = scoliosis.measure_curve(seg, DEFAULT)
    assert curve.valid
    assert curve.active is False
    assert curve.amplitude_mm < DEFAULT.scoliosis_min_amplitude_mm


def test_scoliosis_phantom_curve_follows_the_canal_truth(oracle_scoliosis):
    case, seg = oracle_scoliosis
    curve = scoliosis.measure_curve(seg, DEFAULT)
    assert curve.valid and curve.active
    # The phantom's canal moves about 8 mm at the apex (not the full 14 mm of the discs): canal truth Cobb-like
    # 12 deg, amplitude 5.3 mm. The oracle estimate must land near it, and on the correct side.
    assert 8.0 <= curve.cobb_deg <= 16.0
    assert 3.5 <= curve.amplitude_mm <= 7.0
    assert curve.direction == "left"
    apex_truth = float(case.phantom.bodies["L2"]["center"][2])
    assert abs(curve.apex_z - apex_truth) <= 20.0
    assert curve.confidence > 0.5


def test_discs_on_their_own_planes_match_the_truth(oracle_scoliosis):
    case, seg = oracle_scoliosis
    cfg = dataclasses.replace(DEFAULT, scoliosis_relocate=True)
    anchors = anchors_from_segmentation(seg, cfg)
    assert anchors.alignment["coronal"]["active"] is True
    s = score_anchors(anchors, case.truth, midline_truth=case.phantom.truth["midline_x"])
    assert s.matched == s.n_truth
    assert s.label_correct == s.n_truth
    assert max(s.centre_err_mm) < 1.5
    assert any("measured on their own sagittal planes" in w for w in anchors.warnings)
    assert all("nearest its lateral centre" in " ".join(d.evidence) for d in anchors.discs.values())


def test_switch_off_keeps_the_mid_plane_path(oracle_scoliosis):
    _case, seg = oracle_scoliosis
    cfg = dataclasses.replace(DEFAULT, scoliosis_relocate=False)
    anchors = anchors_from_segmentation(seg, cfg)
    assert anchors.alignment["coronal"]["active"] is True  # still measured, just not used
    assert not any("own sagittal planes" in w for w in anchors.warnings)


def test_relocation_falls_back_to_the_mid_plane_when_no_disc_is_found(oracle_scoliosis, monkeypatch):
    _case, seg = oracle_scoliosis
    monkeypatch.setattr(postprocess, "discs_on_curve", lambda *a, **k: ([], []))
    anchors = anchors_from_segmentation(seg, dataclasses.replace(DEFAULT, scoliosis_relocate=True))
    assert anchors.discs  # the mid-sagittal discs are used
    assert any("mid-sagittal discs used" in w for w in anchors.warnings)
    assert not any("own sagittal planes" in w for w in anchors.warnings)


def test_relocated_discs_have_a_height_and_a_unit_superior_normal(oracle_scoliosis):
    _case, seg = oracle_scoliosis
    curve = scoliosis.measure_curve(seg, DEFAULT)
    discs, _warns = scoliosis.discs_on_curve(seg, curve, DEFAULT)
    assert discs
    for d in discs:
        assert d.height_mm is not None and d.height_mm > 0
        assert np.linalg.norm(d.normal) == pytest.approx(1.0)
        assert d.normal[2] > 0  # superior-pointing convention of the labeller


def test_3d_normal_flag_replaces_the_normal_and_the_default_keeps_it(oracle_scoliosis):
    _case, seg = oracle_scoliosis
    curve = scoliosis.measure_curve(seg, DEFAULT)
    flat, _ = scoliosis.discs_on_curve(seg, curve, DEFAULT)
    tilted, _ = scoliosis.discs_on_curve(seg, curve, dataclasses.replace(DEFAULT, scoliosis_normal_3d=True))
    assert [round(d.center[2], 3) for d in flat] == [round(d.center[2], 3) for d in tilted]
    assert any(not np.allclose(a.normal, b.normal, atol=1e-6) for a, b in zip(flat, tilted, strict=True))


# ----------------------------------------------------------------------------- anchors record and schema


def test_alignment_round_trips_and_validates_against_the_schema(oracle_scoliosis):
    jsonschema = pytest.importorskip("jsonschema")
    _case, seg = oracle_scoliosis
    anchors = anchors_from_segmentation(seg, DEFAULT)
    doc = json.loads(anchors.to_json())
    assert doc["schema_version"] == SCHEMA_VERSION == "1.2"
    assert doc["alignment"]["coronal"]["active"] in (True, False)
    back = Anchors.from_dict(copy.deepcopy(doc))
    assert back.alignment == anchors.alignment
    text = (resources.files("spinescope") / "data" / "anchors.schema.json").read_text(encoding="utf-8")
    schema = json.loads(text)
    jsonschema.Draft202012Validator.check_schema(schema)
    # The numbering record of the raw segmentation carries keys the full pipeline adds later; validate the rest.
    doc.pop("numbering", None)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(doc))
    assert errors == [], [e.message for e in errors]


def test_classical_backend_records_that_the_curve_was_not_measured(phantom_study):
    from spinescope.anatomy import label_spine

    anchors = label_spine(phantom_study, backend="classical")
    rec = anchors.alignment["coronal"]
    assert rec["valid"] is False and rec["active"] is False


def test_config_defaults_are_the_documented_ones():
    assert DEFAULT.scoliosis_min_cobb_deg == 10.0
    assert DEFAULT.scoliosis_relocate is False  # opt-in (lab30 decision)
    assert DEFAULT.scoliosis_disc_reach_mm == 20.0
    assert DEFAULT.scoliosis_normal_3d is False


def test_default_labels_are_the_mid_plane_path_even_with_a_large_curve(oracle_scoliosis):
    _case, seg = oracle_scoliosis
    default = anchors_from_segmentation(seg, DEFAULT)
    off = anchors_from_segmentation(seg, dataclasses.replace(DEFAULT, scoliosis_relocate=False))
    assert default.alignment["coronal"]["active"] is True  # measured and reported
    assert {k: (d.center, d.normal) for k, d in default.discs.items()} == {
        k: (d.center, d.normal) for k, d in off.discs.items()
    }
    assert not any("own sagittal planes" in w for w in default.warnings)


def test_record_leads_with_amplitude_and_marks_the_angle_uncalibrated(oracle_scoliosis):
    _case, seg = oracle_scoliosis
    anchors = anchors_from_segmentation(seg, DEFAULT)
    rec = anchors.alignment["coronal"]
    assert list(rec)[:2] == ["label", "amplitude_mm"]
    assert rec["label"] == "lateral curvature screening (unvalidated)"
    assert "cobb_like_uncalibrated_deg" in rec and "cobb_deg" not in rec


def test_large_curve_raises_a_screening_warning_not_a_finding(oracle_scoliosis):
    _case, seg = oracle_scoliosis
    anchors = anchors_from_segmentation(seg, DEFAULT)
    msgs = [w for w in anchors.warnings if "screening" in w]
    assert msgs and all("scoliosis" not in w.lower() for w in msgs)
