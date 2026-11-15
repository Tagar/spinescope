"""Image quality and hardware flags (spinescope.anatomy.artifacts, lab31).

Phantom-based: the robustness scenarios hardware_cage_screws, motion_ghosting and aliasing_wrap must be flagged, the
baseline must not, and flagged levels must be capped with the raw value kept. Synthetic stacks cover the rules the
phantom does not reach (STIR fat rim, degenerate stacks). The realistic-look checks are marked slow.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spinescope.anatomy import artifacts as A
from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.pipeline import label_spine
from spinescope.anatomy.summary import quality_summary, summary_lines
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.study import from_stacks
from spinescope.testing.robustness import cached_case


@pytest.fixture(scope="module")
def labelled_hardware() -> Anchors:
    return label_spine(cached_case("hardware_cage_screws").study, backend="classical")


@pytest.fixture(scope="module")
def labelled_baseline() -> Anchors:
    return label_spine(cached_case("baseline").study, backend="classical")


@pytest.fixture(scope="module")
def labelled_ghost() -> Anchors:
    return label_spine(cached_case("motion_ghosting").study, backend="classical")


@pytest.fixture(scope="module")
def labelled_wrap() -> Anchors:
    return label_spine(cached_case("aliasing_wrap").study, backend="classical")


def _flagged_kinds(anchors: Anchors) -> set[str]:
    return {k for lv in anchors.quality.get("levels", {}).values() for k in lv.get("flagged", [])}


def test_hardware_reported_at_instrumented_levels_but_advisory(labelled_hardware: Anchors) -> None:
    q = labelled_hardware.quality
    metal = q["study"]["flags"]["metal"]
    # the screws sit in L4 and L5: the levels that touch those bodies carry the evidence ...
    assert metal["would_flag"] is True and metal["present"] is False and metal["advisory"] is True
    assert "metal" in q["levels"]["L4-L5"]["advisory"]
    assert "metal" in q["levels"]["L3-L4"]["advisory"]
    # ... but hardware is advisory until real-data review passes: it never caps or scores
    assert q["levels"]["L4-L5"]["flagged"] == []
    assert q["study"]["score"] == 1.0
    assert labelled_hardware.quality["caps"] == []


def test_baseline_has_no_hardware_or_motion_flags(labelled_baseline: Anchors) -> None:
    flags = _flagged_kinds(labelled_baseline)
    assert not flags & {"metal", "fusion_or_cage", "motion_ghost", "wrap"}
    assert labelled_baseline.quality["study"]["score"] == 1.0


def test_motion_ghosting_flagged(labelled_ghost: Anchors) -> None:
    ghost = labelled_ghost.quality["study"]["flags"]["motion_ghost"]
    assert ghost["present"] is True
    assert ghost["detail"]["lag_rows"] == 16  # the phantom's ghost spacing, found by search, not assumed
    assert "motion_ghost" in labelled_ghost.quality["levels"]["L4-L5"]["flagged"]


def test_aliasing_wrap_detected_at_outermost_superior_level(labelled_wrap: Anchors) -> None:
    wrap = labelled_wrap.quality["study"]["flags"]["wrap"]
    assert wrap["would_flag"] is True
    assert wrap["detail"]["edge"] == "superior"
    top = max(labelled_wrap.discs, key=lambda d: labelled_wrap.discs[d].center[2])
    assert "wrap" in labelled_wrap.quality["levels"][top]["advisory"]
    assert labelled_wrap.quality["levels"][top]["flagged"] == []


def test_quality_record_is_additive_and_serialises(labelled_hardware: Anchors) -> None:
    d = labelled_hardware.to_dict()
    assert "quality" in d
    json.dumps(d)  # plain JSON-safe values only
    back = Anchors.from_dict(d)
    assert back.quality["study"]["score"] == labelled_hardware.quality["study"]["score"]
    # the record carries no file paths or patient attributes
    text = json.dumps(labelled_hardware.quality)
    assert "/Users" not in text and ".mha" not in text and ".dcm" not in text


def test_motion_ghost_flagged_levels_end_at_or_below_the_cap(labelled_ghost: Anchors) -> None:
    flagged = [lab for lab, lv in labelled_ghost.quality["levels"].items() if "motion_ghost" in lv["flagged"]]
    assert flagged
    for lab in flagged:
        assert labelled_ghost.discs[lab].confidence <= DEFAULT.quality_disc_cap + 1e-9
    for cap in labelled_ghost.quality["caps"]:
        assert cap["raw"] > cap["capped"] and cap["reason"] == "motion_ghost"
        assert any("confidence capped" in e for e in labelled_ghost.discs[cap["level"]].evidence)


def test_summary_surfaces_quality(labelled_hardware: Anchors) -> None:
    s = quality_summary(labelled_hardware)
    assert s["assessed"] is True
    assert s["flagged"] == []
    assert "metal" in s["advisory"]
    assert any("image quality" in line for line in summary_lines(labelled_hardware))


def test_no_discs_is_a_warning_not_an_error() -> None:
    stack = _blank_stack()
    study = from_stacks([stack])
    record = A.assess_quality(study, Anchors(), DEFAULT)
    assert record["levels"] == {}
    assert record["warnings"]


def test_apply_quality_caps_and_keeps_raw_value() -> None:
    anchors = Anchors.from_truth(cached_case("baseline").phantom.truth)
    anchors.numbering = {"status": "confirmed", "confidence": 0.95}
    lab = "L4-L5"
    raw = anchors.discs[lab].confidence
    record: dict = {
        "levels": {lab: {"flags": {"motion_ghost": 0.9}, "flagged": ["motion_ghost"], "confidence": 0.9}},
        "study": {"score": 0.6, "flags": {}},
        "warnings": [],
    }
    A.apply_quality(anchors, record, DEFAULT)
    assert anchors.discs[lab].confidence == pytest.approx(DEFAULT.quality_disc_cap)
    assert any(f"{raw:.2f}" in e for e in anchors.discs[lab].evidence)
    assert anchors.numbering["status"] == "probable"
    assert anchors.numbering["quality_cap"]["from"] == "confirmed"
    # the numbering confidence number is not changed silently
    assert anchors.numbering["confidence"] == pytest.approx(0.95)


def test_apply_quality_never_caps_on_advisory_kinds() -> None:
    anchors = Anchors.from_truth(cached_case("baseline").phantom.truth)
    anchors.numbering = {"status": "confirmed", "confidence": 0.95}
    before = {k: d.confidence for k, d in anchors.discs.items()}
    record: dict = {
        "levels": {"L4-L5": {"flags": {"metal": 1.0}, "flagged": ["metal"], "confidence": 1.0}},
        "study": {"score": 0.9, "flags": {}},
        "warnings": [],
    }
    A.apply_quality(anchors, record, DEFAULT)
    assert {k: d.confidence for k, d in anchors.discs.items()} == before
    assert anchors.numbering["status"] == "confirmed"


def test_apply_quality_clean_record_changes_nothing() -> None:
    anchors = Anchors.from_truth(cached_case("baseline").phantom.truth)
    anchors.numbering = {"status": "confirmed", "confidence": 0.95}
    before = {k: d.confidence for k, d in anchors.discs.items()}
    A.apply_quality(anchors, {"levels": {}, "study": {"score": 1.0, "flags": {}}, "warnings": []}, DEFAULT)
    assert {k: d.confidence for k, d in anchors.discs.items()} == before
    assert anchors.numbering["status"] == "confirmed"
    assert "quality_cap" not in anchors.numbering


def test_very_low_study_score_caps_to_unconfirmed() -> None:
    anchors = Anchors.from_truth(cached_case("baseline").phantom.truth)
    anchors.numbering = {"status": "confirmed", "confidence": 0.95}
    A.apply_quality(anchors, {"levels": {}, "study": {"score": 0.1, "flags": {}}, "warnings": []}, DEFAULT)
    assert anchors.numbering["status"] == "unconfirmed"


def test_fat_rim_rule_is_advisory_and_separates_synthetic_stir() -> None:
    def stir(rim_value: float) -> Stack:
        vol = np.zeros((3, 120, 120), dtype=np.float32)  # air outside the body
        yy, xx = np.mgrid[0:120, 0:120]
        body = ((yy - 60) / 50.0) ** 2 + ((xx - 60) / 40.0) ** 2 <= 1.0
        inner = ((yy - 60) / 28.0) ** 2 + ((xx - 60) / 20.0) ** 2 <= 1.0
        for k in range(3):
            vol[k][body] = 300.0  # muscle
            vol[k][body & ~inner] = rim_value  # subcutaneous rim, 15 mm or more thick
        return Stack(
            key="sag_stir",
            vol=vol,
            ipp=np.tile([0.0, 0.0, 0.0], (3, 1)) + np.arange(3)[:, None] * np.array([0, 0, 5.0]),
            row_dir=np.array([0.0, 1.0, 0.0]),
            col_dir=np.array([0.0, 0.0, -1.0]),
            row_mm=1.0,
            col_mm=1.0,
        )

    dark = A.detect_fat_sat(stir(60.0), DEFAULT)  # suppressed fat: dark rim
    bright = A.detect_fat_sat(stir(330.0), DEFAULT)  # unsuppressed fat: rim as bright as muscle
    assert dark["present"] is False
    assert bright["present"] is True
    assert bright["ratio"] > dark["ratio"]


def test_degenerate_stack_gives_no_flags_and_no_crash() -> None:
    stack = _blank_stack()
    anchors = Anchors.from_truth(cached_case("baseline").phantom.truth)
    assert A.detect_snr(stack, anchors, DEFAULT)["present"] in (True, False)
    assert A.detect_wrap(stack, anchors, DEFAULT)["present"] is False
    assert A.detect_ghosting(stack, anchors, DEFAULT)["present"] is False
    assert A.detect_metal(stack, anchors, DEFAULT)["levels"] == {}


def test_quality_failure_never_sinks_the_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("detector broke")

    monkeypatch.setattr(A, "assess_quality", boom)
    anchors = label_spine(cached_case("baseline").study, backend="classical")
    assert anchors.discs
    assert any("image quality not assessed" in w for w in anchors.warnings)


@pytest.mark.slow
def test_realistic_hardware_flagged_and_baseline_clean() -> None:
    hw = label_spine(cached_case("hardware_cage_screws", "realistic").study, backend="ml")
    assert hw.quality["study"]["flags"]["metal"]["would_flag"] is True  # advisory: never capping
    base = label_spine(cached_case("baseline", "realistic").study, backend="ml")
    assert not _flagged_kinds(base) & {"motion_ghost"}
    assert base.quality["study"]["flags"]["metal"]["would_flag"] is False


def _blank_stack() -> Stack:
    return Stack(
        key="sag_t2",
        vol=np.zeros((2, 16, 16), dtype=np.float32),
        ipp=np.zeros((2, 3)),
        row_dir=np.array([0.0, 1.0, 0.0]),
        col_dir=np.array([0.0, 0.0, -1.0]),
        row_mm=1.0,
        col_mm=1.0,
    )
