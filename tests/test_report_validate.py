"""Validation: clear errors with JSON paths, warnings for missing images, built-in and jsonschema paths."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from report_factory import F, findings, l45_disc, minimal_valid
from spinescope.report import validate


def _errors(doc: dict[str, Any]) -> list[str]:
    return validate(doc).error_messages()


def _with_field(
    doc: dict[str, Any], level: str, group: str, name: str, field: dict[str, Any]
) -> dict[str, Any]:
    d = copy.deepcopy(doc)
    d["levels"][level].setdefault(group, {})[name] = field
    return d


def test_minimal_valid_document_passes() -> None:
    res = validate(minimal_valid())
    assert res.ok, res.error_messages()
    assert res.errors == []
    assert res.reader == "reader_a"


def test_out_of_range_grade_has_json_path_and_bounds() -> None:
    doc = findings("reader_a", levels={"L4-L5": {"disc": l45_disc(7)}})
    (msg,) = _errors(doc)
    assert msg.startswith('$.levels["L4-L5"].disc.pfirrmann.value:')
    assert "greater than the maximum 5" in msg


def test_unknown_enum_suggests_the_closest_value() -> None:
    doc = findings("reader_a", levels={"L4-L5": {"disc": {"contour": F("protrusoin")}}})
    (msg,) = _errors(doc)
    assert '$.levels["L4-L5"].disc.contour.value' in msg
    assert 'did you mean "protrusion"' in msg
    assert "allowed:" in msg


def test_missing_evidence_is_an_error_with_path() -> None:
    doc = findings("reader_a", levels={"L4-L5": {"disc": {"pfirrmann": {**F(3), "evidence": []}}}})
    msgs = _errors(doc)
    assert any('$.levels["L4-L5"].disc.pfirrmann.evidence' in m and "at least 1" in m for m in msgs), msgs


def test_missing_required_field_key() -> None:
    f = F(3)
    del f["confidence"]
    doc = findings("reader_a", levels={"L4-L5": {"disc": {"pfirrmann": f}}})
    assert any('missing required key "confidence"' in m for m in _errors(doc))


def test_unknown_key_inside_a_level_is_reported() -> None:
    doc = findings("reader_a", levels={"L4-L5": {"disk": {"pfirrmann": F(3)}}})
    (msg,) = _errors(doc)
    assert msg.startswith('$.levels["L4-L5"]:')
    assert 'unexpected key "disk"' in msg and 'did you mean "disc"' in msg


@pytest.mark.parametrize("bad_label", ["L4_L5", "L4-L4x", "L45"])
def test_bad_disc_label_is_rejected(bad_label: str) -> None:
    doc = findings("reader_a", levels={bad_label: {"disc": l45_disc(3)}})
    msgs = _errors(doc)
    assert msgs and all(m.startswith("$.levels:") for m in msgs), msgs
    assert any(f'unexpected key "{bad_label}"' in m for m in msgs)


@pytest.mark.parametrize("label", ["L5-S1", "L5-L6", "T12-L1", "T9-T10"])
def test_unusual_disc_labels_are_accepted(label: str) -> None:
    doc = findings("reader_a", levels={label: {"disc": l45_disc(3)}})
    assert validate(doc).ok


def test_root_label_pattern_accepts_any_root_and_unknown() -> None:
    for root in ["L5", "S1", "T12", "unknown"]:
        doc = findings("reader_a", levels={"L4-L5": {"nerve_root_L": {"root": F(root)}}})
        assert validate(doc).ok, root
    doc = findings("reader_a", levels={"L4-L5": {"nerve_root_L": {"root": F("five")}}})
    assert any("does not match the pattern" in m for m in _errors(doc))


def test_modic_is_string_and_null_is_indeterminate() -> None:
    ok = findings("reader_a", levels={"L4-L5": {"endplate_below": {"modic": F("mixed")}}})
    indet = findings("reader_a", levels={"L4-L5": {"endplate_below": {"modic": F(None, "low")}}})
    assert validate(ok).ok and validate(indet).ok
    bad = findings("reader_a", levels={"L4-L5": {"endplate_below": {"modic": F(1)}}})
    assert _errors(bad)


def test_global_section_errors_have_global_path() -> None:
    doc = findings("reader_a", global_={"paraspinal": {"multifidus_goutallier_L": F(5)}})
    (msg,) = _errors(doc)
    assert msg.startswith("$.global.paraspinal.multifidus_goutallier_L.value")


def test_item_errors_point_into_arrays() -> None:
    doc = minimal_valid()
    doc["focal_lesions"][0]["level"] = "hip"
    msgs = _errors(doc)
    assert any(m.startswith("$.focal_lesions[0].level:") for m in msgs), msgs
    doc = minimal_valid()
    doc["pain_generator_hypotheses"][0]["attributes"]["evidence_strength"]["value"] = "very strong"
    assert any(
        m.startswith("$.pain_generator_hypotheses[0].attributes.evidence_strength.value:")
        for m in _errors(doc)
    )


def test_top_level_problems() -> None:
    assert 'missing required key "reader"' in " ".join(_errors({"schema_version": "1.0", "scope": "x"}))
    assert any("schema_version" in m for m in _errors({**minimal_valid(), "schema_version": "2.0"}))
    assert any("unexpected key" in m for m in _errors({**minimal_valid(), "extra": 1}))


def test_top_level_must_be_object(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text("[1, 2]", encoding="utf-8")
    res = validate(p)
    assert not res.ok and res.errors[0].path == "$"


def test_unparseable_file_is_reported_not_raised(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    res = validate(p)
    assert not res.ok
    assert "cannot read JSON" in res.errors[0].message


def test_file_path_is_validated_and_reader_file_name_mismatch_warns(tmp_path: Path) -> None:
    p = tmp_path / "other_name.json"
    p.write_text(json.dumps(minimal_valid("reader_a")), encoding="utf-8")
    res = validate(p)
    assert res.ok
    assert any("differs from reader" in w for w in res.warnings)


def test_missing_evidence_images_are_warnings_not_errors(tmp_path: Path) -> None:
    doc = minimal_valid()
    res = validate(doc)
    assert res.ok
    assert any("evidence image not found: image " in w for w in res.warnings)
    assert not any("sag_t2.png" in w for w in res.warnings)  # the path is not echoed
    # aggregated: the same image cited many times gives one warning
    from spinescope.report.validate import _image_label

    assert sum(_image_label("out/reader/sag_t2.png") in w for w in res.warnings) == 1


def test_existing_relative_evidence_image_is_not_warned(tmp_path: Path) -> None:
    img = tmp_path / "shots" / "sag.png"
    img.parent.mkdir()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(img)
    doc = findings("reader_a", levels={"L4-L5": {"disc": {"pfirrmann": F(3, image="shots/sag.png")}}})
    p = tmp_path / "reader_a.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    res = validate(p)
    assert res.ok and res.warnings == []


def test_builtin_only_path_gives_the_same_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """With jsonschema unavailable the built-in validator alone still catches every error."""
    bad = findings("reader_a", levels={"L4-L5": {"disc": {"pfirrmann": F(9), "contour": F("bulgy")}}})
    with_js = validate(bad)
    monkeypatch.setitem(sys.modules, "jsonschema", None)  # makes "import jsonschema" raise ImportError
    builtin = validate(bad)
    assert not with_js.ok and not builtin.ok
    assert {e.path for e in builtin.errors} == {e.path for e in with_js.errors}
    assert validate(minimal_valid()).ok


def test_jsonschema_errors_are_added_when_available() -> None:
    pytest.importorskip("jsonschema")
    doc = minimal_valid()
    doc["levels"]["L4-L5"]["disc"]["pfirrmann"]["value"] = 42
    doc["levels"]["L4-L5"]["disc"]["contour"].pop("notes")
    res = validate(doc)
    paths = {e.path for e in res.errors}
    assert '$.levels["L4-L5"].disc.pfirrmann.value' in paths
    assert '$.levels["L4-L5"].disc.contour' in paths


def test_validate_does_not_mutate_input() -> None:
    doc = minimal_valid()
    before = copy.deepcopy(doc)
    validate(doc)
    assert doc == before
