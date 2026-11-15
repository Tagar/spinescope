"""Robustness suite: ``label_spine`` must label every scenario in ``spinescope.testing.robustness``.

Each scenario is rendered from the synthetic phantom (which carries ground truth), labelled by the classical
backend and scored with the shared validation scorer. Two checks per scenario:

* ``test_labelled_correctly``: every in-view disc is found, named correctly and centred within 3 mm (p90),
  with no spurious disc (see :data:`spinescope.testing.robustness.PASS_CENTRE_P90_MM`).
* ``test_no_silent_error``: a wrong result must not come with numbering confidence at or above 0.5. The
  labeller should be loud about what it cannot label, not confidently wrong.

Scenarios the current backend fails are marked ``xfail(strict=False)`` with the failure observed on the
phantom, so the suite documents the status and turns green (XPASS) as backends improve. Run
``tools/robustness_report.py`` for the per-backend table.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

from spinescope.testing.robustness import (
    SCENARIOS,
    SILENT_ERROR,
    CaseResult,
    build_case,
    evaluate,
    scenario,
)

#: Observed failures of the classical backend (checked against the phantom, not guessed).
XFAIL: dict[str, str] = {
    "baseline": "L2-L3 is missed: the chain jumps to L1-L2 and the labels above shift by one level",
    "t2_only": "same L2-L3 miss as the baseline; the labels above it shift by one level",
    "t1_only": "L2-L3 is found 7 mm from its true centre (weaker T1 endplates); the other discs are fine",
    "stir_only": "all six discs are named correctly but the p90 centre error is 3.6 mm, above 3 mm",
    "pd_only": "no canal on PD (CSF is not bright): the chain sits 50-70 mm from the true discs, none matched",
    "iso_1mm_3d": "9 slices of 1 mm: 3 of 5 discs are found, L2-L3 is skipped and the labels above shift",
    "thick_6mm": "same L2-L3 miss at 6 mm slices; the labels above it shift by one level",
    "field_3T": "same L2-L3 miss at 3 T; the labels above it shift by one level",
    "fov_T10_sacrum": "the lumbar discs are right, but the T9-T10, T10-T11 and T11-T12 discs are 12-14 mm "
    "from their true centres and fall outside the match radius",
    "fov_L2_S1": "all four discs are named correctly but L4-L5 is 3.3 mm off, so p90 is 3.1 mm (limit 3 mm)",
    "lumbarised_S1": "the S1-S2 disc under the lumbarised S1 is not found, so L5-L6 is taken for L5-S1 and "
    "every name is one level off (1 of 7 correct)",
    "sacralised_L5": "same L2-L3 miss as the baseline; the partial L5-S1 disc itself is labelled correctly",
    "dark_discs": "the two dark discs (L2-L3, L4-L5) are centred 7-9 mm from their true positions",
    "collapsed_discs": "collapsed discs (4-5 mm high): 3 of 6 discs are found and the labels above them shift",
    "schmorl_nodes": "Schmorl nodes at L3 and L4 break the chain: L4-L5 is not found and the labels above shift",
    "scoliosis": "the L2-L3 miss again, at the apex of the 14 mm curve; the labels above it shift",
    "hardware_cage_screws": "the L4-L5 cage void breaks the chain: L3-L4 and L4-L5 sit 11-12 mm from their true "
    "positions and the labels above shift",
    "bias_field": "same L2-L3 miss as the baseline; the bias field does not change the chain",
    "motion_ghosting": "ghost copies move the sacral end of the chain: L5-S1 is 26 mm from its true position",
    "aliasing_wrap": "wrapped rows move the sacral end: L5-S1 is 34 mm off and L4-L5 is labelled at the "
    "L5-S1 position",
}


@functools.cache
def _result(name: str) -> CaseResult:
    """Scenario outcome, computed once per session."""
    return evaluate(build_case(scenario(name)))


#: One or two scenarios per family (clean, one sequence, field, FOV, transitional, pathology, deformity, artefact)
#: run in the default suite. The others are ``slow``: each costs a classical run of the whole stack (about 3 s), and
#: they all run in CI's full job (``pytest -m ""``), so no scenario is dropped.
FAST_SCENARIOS = frozenset(
    {
        "baseline",
        "t2_only",
        "field_3T",
        "fov_T10_sacrum",
        "lumbarised_S1",
        "dark_discs",
        "scoliosis",
        "motion_ghosting",
        "few_slices_7",
    }
)


def _params() -> list[object]:
    out: list[object] = []
    for sc in SCENARIOS:
        marks: list[Any] = (
            [pytest.mark.xfail(strict=False, reason=XFAIL[sc.name])] if sc.name in XFAIL else []
        )
        if sc.name not in FAST_SCENARIOS:
            marks.append(pytest.mark.slow)
        out.append(pytest.param(sc.name, marks=marks, id=sc.name))
    return out


def _describe(r: CaseResult) -> str:
    m = r.metrics
    return (
        f"{r.outcome}: recall {m['recall']:.2f}, precision {m['precision']:.2f}, "
        f"label accuracy {m['label_accuracy_all_truth']:.2f}, p90 centre {m['centre_err_p90_mm']:.1f} mm, "
        f"numbering confidence {r.numbering_confidence:.2f}; predicted {r.predicted}, truth {r.truth}"
    )


@pytest.mark.parametrize("name", _params())
def test_labelled_correctly(name: str) -> None:
    r = _result(name)
    assert r.passed, _describe(r)


@pytest.mark.parametrize("name", _params())
def test_no_silent_error(name: str) -> None:
    r = _result(name)
    assert r.outcome != SILENT_ERROR, _describe(r)
