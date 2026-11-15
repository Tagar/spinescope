"""Robustness suite on the realistic phantom, labelled by the learned labeller (``label_spine(backend="ml")``).

The cartoon suite (``test_robustness.py``) grades the classical chain on the cartoon phantom. This suite grades the
default learned labeller on the realistic phantom (``PhantomSpec(realism="realistic")``: published morphometry and
spin-echo physics, see ``docs/labeling/lab21_phantom.md``). It needs no real images, so it runs in CI.

Two checks per scenario, as in the cartoon suite:

* ``test_labelled_correctly``: every in-view disc is found, named correctly and centred within 3 mm (p90).
* ``test_no_silent_error``: a wrong result must not come with numbering confidence at or above 0.5.

Every failing scenario is ``xfail(strict=True)`` with the numbers measured by
``tools/robustness_report.py --realism realistic``. A strict xfail that starts to pass fails the suite: remove its
entry from the dictionary below, so it becomes a real test. The labelled-correctly failures are mostly extra
disc-like detections on the posterior elements and above the top body, which shift the names below them.
"""

from __future__ import annotations

import functools

import pytest

from spinescope.anatomy.ml import model_path
from spinescope.testing.robustness import (
    SCENARIOS,
    SILENT_ERROR,
    CaseResult,
    cached_case,
    evaluate,
    scenario,
)


def _ml_ready() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path().exists()


requires_ml = pytest.mark.skipif(not _ml_ready(), reason="needs onnxruntime and the packaged labeller")

# Scenario -> reason, for the learned labeller on the realistic phantom (measured; see module docstring).
REALISTIC_XFAIL_LABELLED: dict[str, str] = {
    "baseline": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.8 mm, numbering confidence 0.93 "
    ),
    "t2_only": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.8 mm, numbering confidence 0.95 "
    ),
    "t1_only": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.5 mm, numbering confidence 0.95 "
    ),
    "stir_only": (
        "learned labeller on the realistic phantom: unsure: recall 0.67, precision 0.67, label "
        "accuracy 0.17, centre p90 8.9 mm, numbering confidence 0.31 "
    ),
    "pd_only": (
        "learned labeller on the realistic phantom: unsure: recall 1.00, precision 0.86, label "
        "accuracy 0.83, centre p90 4.5 mm, numbering confidence 0.33 "
    ),
    "iso_1mm_3d": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.71, "
        "label accuracy 0.60, centre p90 6.0 mm, numbering confidence 0.52 "
    ),
    "thick_6mm": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.60, "
        "label accuracy 0.50, centre p90 3.0 mm, numbering confidence 0.50 "
    ),
    "few_slices_7": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 2.0 mm, numbering confidence 0.96 "
    ),
    "field_3T": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.4 mm, numbering confidence 0.97 "
    ),
    "fov_T10_sacrum": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.64, "
        "label accuracy 0.44, centre p90 1.4 mm, numbering confidence 0.91 "
    ),
    "fov_L2_S1": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.80, "
        "label accuracy 0.75, centre p90 1.7 mm, numbering confidence 0.94 "
    ),
    "fov_L1_coccyx": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.62, "
        "label accuracy 0.60, centre p90 1.4 mm, numbering confidence 0.54 "
    ),
    "lumbarised_S1": (
        "learned labeller on the realistic phantom: unsure: recall 1.00, precision 0.64, label "
        "accuracy 0.00, centre p90 3.2 mm, numbering confidence 0.25 "
    ),
    "sacralised_L5": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.9 mm, numbering confidence 0.95 "
    ),
    "dark_discs": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 2.1 mm, numbering confidence 0.97 "
    ),
    "collapsed_discs": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 2.5 mm, numbering confidence 0.92 "
    ),
    "compression_fracture_L2": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.9 mm, numbering confidence 0.95 "
    ),
    "schmorl_nodes": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.8 mm, numbering confidence 0.93 "
    ),
    "scoliosis": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.86, "
        "label accuracy 0.83, centre p90 2.5 mm, numbering confidence 0.97 "
    ),
    "bias_field": (
        "learned labeller on the realistic phantom: silent_error: recall 1.00, precision 0.75, "
        "label accuracy 0.67, centre p90 1.8 mm, numbering confidence 0.97 "
    ),
    "motion_ghosting": (
        "learned labeller on the realistic phantom: unsure: recall 0.83, precision 0.83, label "
        "accuracy 0.83, centre p90 8.7 mm, numbering confidence 0.48 "
    ),
    "aliasing_wrap": (
        "learned labeller on the realistic phantom: unsure: recall 1.00, precision 0.67, label "
        "accuracy 0.50, centre p90 4.8 mm, numbering confidence 0.47 "
    ),
}

# Since numbering calibration v2 (lab18) every wrong result on the realistic phantom comes out "unconfirmed",
# so no scenario has a silent error any more; keep the table for future regressions.
REALISTIC_XFAIL_SILENT: dict[str, str] = {}


@functools.cache
def _ml_result(name: str) -> CaseResult:
    """Learned-labeller outcome on the realistic phantom, computed once per session."""
    return evaluate(cached_case(name, "realistic"), backend="ml")


def _params(xfails: dict[str, str]) -> list[object]:
    out: list[object] = []
    for sc in SCENARIOS:
        marks = [pytest.mark.xfail(strict=True, reason=xfails[sc.name])] if sc.name in xfails else []
        out.append(pytest.param(sc.name, marks=[requires_ml, *marks], id=sc.name))
    return out


def _describe(r: CaseResult) -> str:
    m = r.metrics
    return (
        f"{r.outcome}: recall {m['recall']:.2f}, precision {m['precision']:.2f}, "
        f"label accuracy {m['label_accuracy_all_truth']:.2f}, p90 centre {m['centre_err_p90_mm']:.1f} mm, "
        f"numbering confidence {r.numbering_confidence:.2f}; predicted {r.predicted}, truth {r.truth}"
    )


@pytest.mark.parametrize("name", _params(REALISTIC_XFAIL_LABELLED))
def test_labelled_correctly(name: str) -> None:
    r = _ml_result(name)
    assert r.passed, _describe(r)


@pytest.mark.parametrize("name", _params(REALISTIC_XFAIL_SILENT))
def test_no_silent_error(name: str) -> None:
    r = _ml_result(name)
    assert r.outcome != SILENT_ERROR, _describe(r)


@requires_ml
def test_cached_case_is_rendered_once() -> None:
    assert cached_case("baseline", "realistic") is cached_case("baseline", "realistic")
    assert cached_case("baseline", "realistic").realism == "realistic"
    assert scenario("baseline").name == "baseline"
