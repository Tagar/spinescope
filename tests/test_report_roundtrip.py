"""Round-trips (JSON in, consensus out, JSON back), reader instructions, and a many-reader merge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from report_factory import F, findings, item, minimal_valid
from spinescope.report import Consensus, merge, reader_instructions, render_html, render_markdown, validate
from spinescope.report.instructions import _example


def test_findings_json_roundtrip_validates_and_merges(tmp_path: Path) -> None:
    doc = minimal_valid("reader_a")
    path = tmp_path / "reader_a.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == doc
    res = validate(path)
    assert res.ok, res.error_messages()
    cons = merge([path])
    assert cons.reader_names == ["reader_a"]


def test_consensus_json_roundtrip_is_lossless(tmp_path: Path) -> None:
    docs = [minimal_valid("reader_a"), minimal_valid("reader_b")]
    docs[1]["levels"]["L4-L5"]["disc"]["pfirrmann"] = F(4)
    cons = merge(docs, weights={"reader_a": 2.0})
    out = tmp_path / "consensus.json"
    text = cons.to_json(out)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(text)
    back = Consensus.from_json(text)
    assert back.to_dict() == cons.to_dict()
    assert render_markdown(back) == render_markdown(cons)
    assert render_html(back, embed_images=False) == render_html(cons, embed_images=False)


def test_consensus_is_plain_json_data() -> None:
    cons = merge([minimal_valid("reader_a"), minimal_valid("reader_b")], use_confidence=True)
    json.dumps(cons.to_dict())  # no tuples, sets, dataclasses or numpy values left
    assert isinstance(cons.to_dict()["fields"], dict)


def test_from_dict_ignores_unknown_keys() -> None:
    d = merge([minimal_valid("reader_a")]).to_dict()
    d["something_new"] = 1
    assert Consensus.from_dict(d).reader_names == ["reader_a"]


def test_numeric_values_are_plain_python_floats_after_merge() -> None:
    doc = findings("reader_a", levels={"L4-L5": {"central_canal": {"area_mm2": F(float(np.float64(81.5)))}}})
    assert validate(doc).ok
    assert merge([doc]).fields['levels["L4-L5"].central_canal.area_mm2']["value"] == pytest.approx(81.5)


def test_reader_instructions_are_generated_from_the_schema() -> None:
    text = reader_instructions("radiologist_1")
    assert "Reader name: radiologist_1. Write ONE JSON file named radiologist_1.json" in text
    assert '"schema_version": "1.0"' in text
    assert 'levels["<LEVEL>"].disc.pfirrmann' in text
    assert 'one of "normal", "bulge", "protrusion", "extrusion", "sequestration", or null' in text
    assert 'levels["<LEVEL>"].endplate_above.modic' in text
    assert '"0", "1", "2", "3", "mixed"' in text
    assert "global.alignment.scoliosis" in text
    assert "focal_lesions[].attributes:" in text
    assert "evidence is required" in text
    assert "validate(" in text


def test_reader_instructions_example_is_valid_except_for_placeholder_images() -> None:
    text = reader_instructions()
    example_json = text.split("Example of the shape (placeholder values, not a read):\n", 1)[1]
    doc = json.loads(example_json)
    assert doc == _example("reader_a")
    res = validate(doc)
    assert res.ok, res.error_messages()
    assert all("evidence image not found" in w for w in res.warnings)


def test_reader_instructions_default_name() -> None:
    assert "reader_a.json" in reader_instructions()


def test_many_readers_merge_consistently() -> None:
    """Twenty synthetic readers with seeded noise: counts add up and nothing crashes."""
    rng = np.random.RandomState(0)
    docs = []
    for i in range(20):
        pf = int(rng.choice([2, 3, 3, 3, 4]))
        contour = str(rng.choice(["bulge", "bulge", "protrusion", "normal"]))
        area = float(80 + rng.normal(0, 3))
        conf = str(rng.choice(["low", "medium", "high"]))
        doc = findings(
            f"reader_{i:02d}",
            levels={
                "L4-L5": {
                    "disc": {"pfirrmann": F(pf, conf), "contour": F(contour, conf)},
                    "central_canal": {"area_mm2": F(round(area, 1))},
                },
                "L5-S1": {"disc": {"pfirrmann": F(4)}},
            },
            focal_lesions=[
                item("L4", "l4_lesion_a" if i % 2 else "l4_lesion_a2", attributes={"size_mm": F(5.0 + i % 3)})
            ],
        )
        docs.append(doc)
    cons = merge(docs)
    assert len(cons.readers) == 20
    pf = cons.fields['levels["L4-L5"].disc.pfirrmann']
    assert pf["n_definite"] == 20
    assert sum(len(d["readers"]) for d in pf["distribution"]) == 20
    if pf["status"] in ("unanimous", "majority", "plurality"):
        assert pf["agreement"] == max(len(d["readers"]) for d in pf["distribution"])
    assert cons.levels == ["L4-L5", "L5-S1"]
    (lesion_group,) = cons.items["focal_lesions"]  # l4_lesion_a / l4_lesion_a2 fuzzily grouped
    assert lesion_group["n_readers"] == 20 and len(lesion_group["reported_by"]) == 20
    assert lesion_group["attributes"]["size_mm"]["n_definite"] == 20
    md = render_markdown(cons)
    assert "Readers merged: 20" in md
    html = render_html(cons, embed_images=False)
    assert html.count("<table>") >= 3


def test_round_trip_through_directory_of_files(tmp_path: Path) -> None:
    for i, reader in enumerate(["alpha", "beta", "gamma"]):
        doc = minimal_valid(reader)
        doc["levels"]["L4-L5"]["disc"]["pfirrmann"] = F(3 + (i == 2))
        (tmp_path / f"{reader}.json").write_text(json.dumps(doc), encoding="utf-8")
    cons = merge(tmp_path)
    assert cons.reader_names == ["alpha", "beta", "gamma"]
    pf = cons.fields['levels["L4-L5"].disc.pfirrmann']
    assert pf["status"] == "majority" and pf["value"] == 3
    md = render_markdown(cons)
    assert "| **L4-L5: Pfirrmann grade** | 3 | 3: alpha, beta; 4: gamma |" in md
