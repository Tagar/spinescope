"""The findings JSON Schema: packaged, valid, and shaped as the spec asks."""

from __future__ import annotations

import json
from typing import Any

import pytest

from spinescope.report import SCHEMA_VERSION, load_schema
from spinescope.report.schema import (
    ITEM_SECTIONS,
    field_path,
    global_fields,
    global_group_names,
    is_disc_label,
    item_fields,
    level_fields,
    level_group_names,
    level_sort_key,
    normalise_kind,
)
from spinescope.report.validate import SUPPORTED_KEYWORDS


def _field_objects(node: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Every schema node that describes a field (an object with a 'value' property)."""
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        props = node.get("properties", {})
        if isinstance(props, dict) and "value" in props and "evidence" in props:
            out.append((path, node))
        for k, v in node.items():
            if k in ("properties", "patternProperties", "$defs") and isinstance(v, dict):
                for name, sub in v.items():
                    out += _field_objects(sub, f"{path}/{k}/{name}")
            elif isinstance(v, dict) and k in ("items", "additionalProperties"):
                out += _field_objects(v, f"{path}/{k}")
    return out


def _keywords(node: Any) -> set[str]:
    """Schema keywords used anywhere (property names are not keywords and are skipped)."""
    found: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("properties", "patternProperties", "$defs") and isinstance(v, dict):
                for sub in v.values():
                    found |= _keywords(sub)
                continue
            found.add(k)
            if isinstance(v, dict):
                found |= _keywords(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        found |= _keywords(x)
    return found


def test_schema_is_packaged_and_loads() -> None:
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"] == {"const": SCHEMA_VERSION}
    assert schema["required"] == ["schema_version", "reader", "scope"]


def test_load_schema_returns_an_independent_copy() -> None:
    a = load_schema()
    a["properties"]["scope"]["minLength"] = 999
    assert load_schema()["properties"]["scope"]["minLength"] == 1


def test_schema_is_valid_draft_2020_12() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(load_schema())


def test_schema_uses_only_keywords_the_builtin_validator_supports() -> None:
    used = _keywords(load_schema())
    allowed = SUPPORTED_KEYWORDS | {"title", "description"}
    assert used <= allowed, f"unsupported keywords: {sorted(used - allowed)}"


def test_level_keys_accept_any_disc_label_pattern() -> None:
    pattern = load_schema()["properties"]["levels"]["patternProperties"]
    (regex,) = pattern.keys()
    import re

    for ok in ["L4-L5", "L5-S1", "L5-L6", "T12-L1", "T9-T10", "C2-C3", "C7-T1"]:
        assert re.search(regex, ok), ok
    for bad in ["L4_L5", "L4-L4x", "L45", "L4-L5-S1", "l4-l5", "X1-L2", ""]:
        assert not re.search(regex, bad), bad


@pytest.mark.parametrize("label", ["L4-L5", "T12-L1", "L5-S1", "L5-L6", "C2-C3"])
def test_is_disc_label_accepts(label: str) -> None:
    assert is_disc_label(label)


@pytest.mark.parametrize("label", ["L4_L5", "L4", "global", "L4-L4x"])
def test_is_disc_label_rejects(label: str) -> None:
    assert not is_disc_label(label)


def test_every_spec_group_is_present() -> None:
    assert level_group_names() == (
        "disc",
        "central_canal",
        "lateral_recess_R",
        "lateral_recess_L",
        "foramen_R",
        "foramen_L",
        "facet_R",
        "facet_L",
        "ligamentum_flavum",
        "nerve_root_R",
        "nerve_root_L",
        "endplate_above",
        "endplate_below",
        "listhesis",
    )
    assert global_group_names() == (
        "alignment",
        "marrow",
        "conus_cord",
        "posterior_elements",
        "paraspinal",
        "sacrum_si",
    )
    assert ITEM_SECTIONS == ("focal_lesions", "incidental_findings", "pain_generator_hypotheses")


def test_disc_fields_match_the_spec() -> None:
    disc = {f.name: f for f in level_fields() if f.group == "disc"}
    assert set(disc) == {
        "pfirrmann",
        "height_loss",
        "contour",
        "herniation_location",
        "herniation_size_mm",
        "annular_fissure",
        "hiz",
        "migration",
    }
    assert disc["contour"].allowed == ("normal", "bulge", "protrusion", "extrusion", "sequestration")
    assert disc["pfirrmann"].minimum == 1 and disc["pfirrmann"].maximum == 5
    assert "central" in disc["herniation_location"].allowed  # type: ignore[operator]


def test_modic_and_root_fields_follow_the_spec() -> None:
    by = {(f.group, f.name): f for f in level_fields()}
    assert by[("endplate_above", "modic")].allowed == ("0", "1", "2", "3", "mixed")
    assert by[("endplate_below", "modic")].allowed == ("0", "1", "2", "3", "mixed")
    root = by[("nerve_root_R", "root")]
    assert root.allowed is None and root.pattern is not None
    assert by[("listhesis", "translation_mm")].is_continuous


def test_every_field_object_requires_value_confidence_evidence_notes() -> None:
    schema = load_schema()
    nodes = _field_objects(schema)
    # one node per shared definition (left and right groups share a $def): 30 + 14 + 11 fields in total
    assert len(nodes) >= 40
    for where, node in nodes:
        assert set(node["required"]) == {"value", "confidence", "evidence", "notes"}, where
        assert node["additionalProperties"] is False, where


def test_evidence_is_mandatory_and_has_the_three_keys() -> None:
    schema = load_schema()
    ev = schema["$defs"]["Evidence"]
    assert set(ev["required"]) == {"image", "series", "slices"}
    assert schema["$defs"]["EvidenceList"]["minItems"] == 1


def test_confidence_levels() -> None:
    assert load_schema()["$defs"]["Confidence"]["enum"] == ["low", "medium", "high"]


def test_item_sections_have_attributes_that_are_field_objects() -> None:
    for section in ITEM_SECTIONS:
        names = [f.name for f in item_fields(section)]
        assert names, section
    assert [f.name for f in item_fields("pain_generator_hypotheses")] == [
        "explains_chronic_lbp",
        "evidence_strength",
        "what_would_confirm",
    ]


def test_global_sections_cover_the_spec() -> None:
    names = {(f.group, f.name) for f in global_fields()}
    assert ("sacrum_si", "si_joint_R") in names and ("sacrum_si", "si_joint_L") in names
    assert ("paraspinal", "multifidus_goutallier_R") in names
    assert ("conus_cord", "conus_tip_level") in names


def test_side_aware_labels() -> None:
    by = {(f.group, f.name): f for f in level_fields()}
    assert by[("foramen_R", "lee_grade")].label == "Foramen Lee grade (right)"
    assert by[("foramen_L", "lee_grade")].label == "Foramen Lee grade (left)"
    assert by[("disc", "contour")].label == "Disc contour (Fardon 2014)"


def test_field_path_format() -> None:
    assert field_path("L4-L5", "disc", "pfirrmann") == 'levels["L4-L5"].disc.pfirrmann'
    assert field_path("global", "alignment", "scoliosis") == "global.alignment.scoliosis"


def test_level_sort_key_is_superior_to_inferior() -> None:
    labels = ["L5-S1", "L1-L2", "T12-L1", "L4-L5", "C7-T1", "global"]
    assert sorted(labels, key=level_sort_key) == ["C7-T1", "T12-L1", "L1-L2", "L4-L5", "L5-S1", "global"]


def test_normalise_kind() -> None:
    assert normalise_kind("L5 Root-Compression (L)") == "l5_root_compression_l"
    assert normalise_kind("__x__") == "x"


def test_schema_text_is_json_object_with_title() -> None:
    from importlib import resources

    text = (resources.files("spinescope") / "data" / "findings.schema.json").read_text(encoding="utf-8")
    doc = json.loads(text)
    assert doc["title"].startswith("spinescope findings")
