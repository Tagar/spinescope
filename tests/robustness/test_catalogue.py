"""The scenario catalogue itself: valid specs, unique names, field-of-view truth and the outcome rules."""

from __future__ import annotations

import pytest

from spinescope.testing.phantom import PhantomSpec, SpinePhantom
from spinescope.testing.robustness import (
    PASS,
    PASS_CENTRE_P90_MM,
    SCENARIOS,
    SILENT_ERROR,
    UNSURE,
    UNSURE_CONFIDENCE,
    build_case,
    classify,
    scenario,
)

PERFECT = {"recall": 1.0, "precision": 1.0, "label_accuracy_all_truth": 1.0, "centre_err_p90_mm": 1.0}


def test_names_are_unique_and_specs_are_valid() -> None:
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))
    for sc in SCENARIOS:
        SpinePhantom(PhantomSpec(**sc.spec))  # raises on an unknown option or value
        assert sc.protocol.contrasts, sc.name


def test_default_protocol_matches_the_standard_study_slices() -> None:
    assert scenario("baseline").protocol.contrasts == ("t2", "stir", "t1")


def test_field_of_view_scenario_keeps_only_the_discs_in_view() -> None:
    case = build_case(scenario("fov_L2_S1"))
    assert [t.name for t in case.truth] == ["L5-S1", "L4-L5", "L3-L4", "L2-L3"]


def test_transitional_truth_uses_the_lumbar_count_convention() -> None:
    case = build_case(scenario("lumbarised_S1"))
    assert case.truth[0].name == "L6-S1"
    assert case.phantom.truth["discs"]["L6-S1"]["anatomical"] == "S1-S2"


@pytest.mark.parametrize(
    ("metrics", "confidence", "expected"),
    [
        (PERFECT, 0.0, PASS),
        (PERFECT, 1.0, PASS),
        ({**PERFECT, "recall": 0.9}, UNSURE_CONFIDENCE - 0.01, UNSURE),
        ({**PERFECT, "recall": 0.9}, UNSURE_CONFIDENCE, SILENT_ERROR),
        ({**PERFECT, "precision": 0.8}, 0.9, SILENT_ERROR),
        ({**PERFECT, "label_accuracy_all_truth": 0.99}, 0.2, UNSURE),
        ({**PERFECT, "centre_err_p90_mm": PASS_CENTRE_P90_MM + 0.1}, 0.9, SILENT_ERROR),
        ({**PERFECT, "centre_err_p90_mm": float("nan")}, 0.9, SILENT_ERROR),
    ],
)
def test_outcome_rules(metrics: dict[str, float], confidence: float, expected: str) -> None:
    assert classify(metrics, confidence) == expected
