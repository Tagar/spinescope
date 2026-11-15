"""Per-level report wiring: registry guards, confidence caps by level-name status, image evidence requirement."""

from __future__ import annotations

import pytest

from spinescope.analysis import findings as fd
from spinescope.analysis.base import Result
from spinescope.anatomy.anchors import Anchors, Disc


def _anchors(status: str | None) -> Anchors:
    a = Anchors(discs={"L4-L5": Disc(center=[0.0, 0.0, -50.0], normal=[0.0, 0.0, 1.0])})
    if status is not None:
        a.numbering = {"status": status}
    return a


def _discs_result(grade: int | None = 4, conf: float = 0.75) -> Result:
    res = Result(name="discs", meta={"series_key": "sag_t2"})
    res.rows.append(
        {"scope": "disc", "disc": "L4-L5", "pfirrmann_suggestion": grade, "pfirrmann_confidence": conf}
    )
    return res


def test_confidence_words_and_caps() -> None:
    assert fd.confidence_word(0.8) == "high"
    assert fd.confidence_word(0.5) == "medium"
    assert fd.confidence_word(0.1) == "low"
    assert fd.cap_confidence("high", "confirmed") == "high"
    assert fd.cap_confidence("high", "probable") == "medium"
    assert fd.cap_confidence("high", "unconfirmed") == "low"
    assert fd.cap_confidence("low", "confirmed") == "low"  # never raised


def test_registry_rejects_unknown_and_duplicate_fields() -> None:
    def nothing(res: Result, level: str) -> None:
        return None

    with pytest.raises(ValueError):
        fd.register_field(fd.FieldRule("disc", "not_a_field", "x", nothing))
    with pytest.raises(ValueError):
        fd.register_field(fd.FieldRule("disc", "pfirrmann", "discs", nothing))  # already a built-in rule


def test_level_finding_has_schema_shape_and_image_evidence() -> None:
    out = fd.level_findings(
        {"discs": _discs_result()},
        "L4-L5",
        _anchors("confirmed"),
        figures={"discs": {"disc_strip": "discs_disc_strip.png"}},
    )
    f = out["disc"]["pfirrmann"]
    assert f["value"] == 4
    assert (
        f["confidence"] == "medium"
    )  # 0.75 from the analysis -> high; the uncalibrated suggestion is capped at medium
    assert f["evidence"] == [{"image": "discs_disc_strip.png", "series": "sag_t2", "slices": []}]
    assert "model suggestion" in f["notes"]
    assert "_skipped" not in out


def test_unconfirmed_level_name_caps_confidence_and_says_so() -> None:
    out = fd.level_findings(
        {"discs": _discs_result(conf=0.9)},
        "L4-L5",
        _anchors("unconfirmed"),
        figures={"discs": {"disc_strip": "p.png"}},
    )
    f = out["disc"]["pfirrmann"]
    assert f["confidence"] == "low"
    assert "name is not confirmed" in f["notes"]


def test_field_without_image_is_skipped_not_written() -> None:
    out = fd.level_findings({"discs": _discs_result()}, "L4-L5", _anchors("confirmed"), figures={})
    assert "disc" not in out or "pfirrmann" not in out.get("disc", {})
    assert out["_skipped"] == [{"field": "disc.pfirrmann", "reason": "no image evidence saved"}]


def test_no_candidate_means_no_field() -> None:
    out = fd.level_findings(
        {"discs": _discs_result(grade=None)},
        "L4-L5",
        _anchors("confirmed"),
        figures={"discs": {"disc_strip": "p.png"}},
    )
    assert out == {}


def test_endplate_figure_depends_on_the_level() -> None:
    res = Result(name="endplates", meta={"series_key": "sag_t1"})
    res.rows.append(
        {
            "scope": "endplate",
            "vertebra": "L5",
            "endplate": "inferior",
            "adjacent_disc": "L4-L5",
            "side": "all",
            "pattern": "Modic 1",
            "confidence": 0.6,
        }
    )
    figs = {"endplates": {"triptych_L4_inferior": "t.png"}}
    out = fd.level_findings({"endplates": res}, "L4-L5", _anchors("confirmed"), figures=figs)
    assert out["endplate_above"]["modic"]["value"] == "1"
    assert out["endplate_above"]["modic"]["evidence"][0]["image"] == "t.png"
