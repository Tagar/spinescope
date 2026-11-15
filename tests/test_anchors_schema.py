"""The anchors JSON schema (src/spinescope/data/anchors.schema.json, schema 1.1): every labeller output validates, every
field the code writes is declared, and bad documents are refused."""

from __future__ import annotations

import copy
import dataclasses
import json
from importlib import resources
from typing import Any

import pytest

jsonschema = pytest.importorskip("jsonschema")

from spinescope.anatomy import label_spine  # noqa: E402
from spinescope.anatomy.anchors import SCHEMA_VERSION, Anchors, Disc, Vertebra  # noqa: E402
from spinescope.anatomy.verify import apply_answer  # noqa: E402
from spinescope.config import DEFAULT  # noqa: E402


def load_schema() -> dict[str, Any]:
    text = (resources.files("spinescope") / "data" / "anchors.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def _validator() -> Any:
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _errors(doc: dict[str, Any]) -> list[str]:
    return [
        f"{list(e.path)}: {e.message}"
        for e in sorted(_validator().iter_errors(doc), key=lambda e: list(e.path))
    ]


@pytest.fixture(scope="module")
def learned_doc(phantom_study) -> dict[str, Any]:
    return json.loads(label_spine(phantom_study).to_json())


@pytest.fixture(scope="module")
def classical_doc(phantom_study) -> dict[str, Any]:
    return json.loads(label_spine(phantom_study, backend="classical").to_json())


def test_schema_is_draft_2020_12_and_declares_version_1_1():
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert SCHEMA_VERSION == "1.1"
    assert set(schema["properties"]["schema_version"]["enum"]) == {"1.0", "1.1"}


def test_learned_labeller_output_validates(learned_doc):
    assert learned_doc["schema_version"] == SCHEMA_VERSION
    assert _errors(learned_doc) == []


def test_classical_output_validates(classical_doc):
    assert _errors(classical_doc) == []


def test_reviewed_anchors_validate_with_provenance(classical_doc, phantom_study):
    a = Anchors.from_dict(copy.deepcopy(classical_doc))
    apply_answer(
        a,
        {"lowest_disc_is": "L5-S1", "missing": [], "extra": [], "transitional": False, "notes": "checked"},
        who="reader_x",
        source="ai_reader",
        cfg=DEFAULT,
    )
    doc = json.loads(a.to_json())
    assert _errors(doc) == []
    assert doc["numbering"]["corrections"][-1]["source"] == "ai_reader"


def test_unlabelled_anchors_validate():
    doc = json.loads(
        Anchors(numbering={"status": "unconfirmed", "confidence": 0.0, "labelled": False}).to_json()
    )
    assert _errors(doc) == []


def test_truth_anchors_from_the_phantom_validate(phantom):
    doc = json.loads(Anchors.from_truth(phantom.truth).to_json())
    assert _errors(doc) == []


def test_schema_1_0_documents_still_validate(learned_doc):
    old = copy.deepcopy(learned_doc)
    old["schema_version"] = "1.0"
    for k in ("canal", "posterior_elements"):
        old.pop(k, None)
    for v in old["vertebrae"].values():
        v.pop("body", None)
    assert _errors(old) == []


def test_every_field_the_code_writes_is_declared(learned_doc):
    """A new anchors field must be added to the schema (additionalProperties is false everywhere it matters)."""
    schema = load_schema()
    top = set(schema["properties"])
    assert {f.name for f in dataclasses.fields(Anchors)} <= top | {"schema_version"}
    disc_props = set(schema["$defs"]["Disc"]["properties"])
    assert {f.name for f in dataclasses.fields(Disc)} <= disc_props
    vert_props = set(schema["$defs"]["Vertebra"]["properties"])
    assert {f.name for f in dataclasses.fields(Vertebra)} <= vert_props
    numbering_props = set(schema["$defs"]["Numbering"]["properties"])
    assert set(learned_doc["numbering"]) <= numbering_props


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"surprise": 1}),
        lambda d: d["numbering"].update({"status": "maybe"}),
        lambda d: d["numbering"].update({"confidence": 1.5}),
        lambda d: d["discs"].update({"L9-X2": next(iter(d["discs"].values()))}),
        lambda d: d.update({"schema_version": "2.0"}),
        lambda d: d["applicability"].update({"verdict": "in_the_hospital"}),
        lambda d: d["conus"] and d["conus"].update({"unexpected": True}),
    ],
)
def test_bad_documents_are_refused(learned_doc, mutate):
    doc = copy.deepcopy(learned_doc)
    doc.setdefault("conus", None)
    if doc["conus"] is None:
        doc["conus"] = {"z": -10.0, "level": "L4", "confidence": 0.3}
    mutate(doc)
    assert _errors(doc), "the mutated document should not validate"


def test_packaged_schema_is_in_the_package_data():
    assert (resources.files("spinescope") / "data" / "anchors.schema.json").is_file()
