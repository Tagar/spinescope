"""Multi-reader merge semantics: majority, plurality, splits, indeterminate, weights, arrays, sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from report_factory import F, findings, item, minimal_valid
from spinescope.report import Consensus, merge

PF = 'levels["L4-L5"].disc.pfirrmann'
CONTOUR = 'levels["L4-L5"].disc.contour'
AREA = 'levels["L4-L5"].central_canal.area_mm2'
MODIC_BELOW = 'levels["L4-L5"].endplate_below.modic'
FORAMEN_R = 'levels["L4-L5"].foramen_R.lee_grade'


def _reader(name: str, pf: Any = 3, contour: Any = "bulge", conf: str = "high", **kw: Any) -> dict[str, Any]:
    return findings(
        name, levels={"L4-L5": {"disc": {"pfirrmann": F(pf, conf), "contour": F(contour, conf)}}}, **kw
    )


def test_majority_with_agreement_counts() -> None:
    c = merge([_reader("a", 3), _reader("b", 3), _reader("c", 4)])
    rec = c.fields[PF]
    assert rec["status"] == "majority"
    assert rec["value"] == 3
    assert (rec["agreement"], rec["n_definite"], rec["n_reporting"]) == (2, 3, 3)
    assert rec["disagreement"] is True
    assert [d["readers"] for d in rec["distribution"]] == [["a", "b"], ["c"]]


def test_unanimous_is_not_a_disagreement() -> None:
    c = merge([_reader("a", 3), _reader("b", 3), _reader("c", 3)])
    rec = c.fields[PF]
    assert rec["status"] == "unanimous" and rec["value"] == 3 and rec["agreement"] == 3
    assert rec["disagreement"] is False
    assert all(d["path"] != PF for d in c.disagreements)


def test_single_reader_is_single() -> None:
    c = merge([_reader("only", 2)])
    rec = c.fields[PF]
    assert rec["status"] == "single" and rec["value"] == 2 and rec["disagreement"] is False


def test_two_way_split_has_no_consensus_value() -> None:
    c = merge([_reader("a", 3), _reader("b", 4)])
    rec = c.fields[PF]
    assert rec["status"] == "split"
    assert rec["value"] is None
    assert rec["disagreement"] is True
    assert any(d["path"] == PF for d in c.disagreements)


def test_plurality_without_majority_gives_the_top_value() -> None:
    # weights 2, 1.5, 1: top value holds 2 of 4.5, which is not more than half
    c = merge(
        [_reader("a", 1), _reader("b", 2), _reader("c", 5)],
        weights={"a": 2.0, "b": 1.5, "c": 1.0},
    )
    rec = c.fields[PF]
    assert rec["status"] == "plurality"
    assert rec["value"] == 1
    assert rec["disagreement"] is True


def test_weights_can_change_the_winner() -> None:
    readers = [_reader("a", 1), _reader("b", 5), _reader("c", 5)]
    assert merge(readers).fields[PF]["value"] == 5
    weighted = merge(readers, weights={"a": 5.0})
    assert weighted.fields[PF]["value"] == 1
    assert weighted.fields[PF]["status"] == "majority"
    assert weighted.settings["weights"]["a"] == 5.0


def test_negative_or_non_finite_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        merge([_reader("a")], weights={"a": -1.0})
    with pytest.raises(ValueError):
        merge([_reader("a")], weights={"a": float("nan")})


def test_zero_weight_reader_is_listed_but_does_not_vote() -> None:
    c = merge([_reader("a", 3), _reader("b", 5)], weights={"b": 0.0})
    rec = c.fields[PF]
    assert rec["value"] == 3
    assert rec["status"] == "majority"  # only the weighted reader votes
    assert rec["n_reporting"] == 2
    assert rec["agreement"] == 1


def test_confidence_weighting_is_optional() -> None:
    readers = [
        _reader("a", 3, conf="high"),
        _reader("b", 4, conf="low"),
        _reader("c", 4, conf="low"),
    ]
    assert merge(readers).fields[PF]["value"] == 4  # two equal votes
    weighted = merge(readers, use_confidence=True).fields[PF]
    assert weighted["value"] == 3  # 1.0 against 0.25 + 0.25
    assert weighted["status"] == "majority"


def test_missing_field_is_not_assessed_and_not_normal() -> None:
    a = _reader("a", 3)
    b = findings("b", levels={"L4-L5": {"disc": {"pfirrmann": F(3)}}})  # no contour
    c = merge([a, b])
    rec = c.fields[CONTOUR]
    assert rec["status"] == "single"
    assert rec["value"] == "bulge"
    assert rec["not_assessed_by"] == ["b"]
    assert rec["n_reporting"] == 1


def test_field_nobody_reported_is_explicit_not_assessed() -> None:
    c = merge([_reader("a"), _reader("b")])
    rec = c.fields[AREA]
    assert rec["status"] == "not_assessed"
    assert rec["value"] is None and rec["n_reporting"] == 0
    assert rec["not_assessed_by"] == ["a", "b"]


def test_levels_not_covered_by_any_reader_have_no_entries() -> None:
    c = merge([_reader("a")])
    assert c.levels == ["L4-L5"]
    assert not any('"L5-S1"' in p for p in c.fields)


def test_indeterminate_is_listed_but_not_disagreement() -> None:
    a = findings("a", levels={"L4-L5": {"endplate_below": {"modic": F("1")}}})
    b = findings("b", levels={"L4-L5": {"endplate_below": {"modic": F(None, "low")}}})
    c = merge([a, b])
    rec = c.fields[MODIC_BELOW]
    assert rec["status"] == "single"
    assert rec["value"] == "1"
    assert rec["disagreement"] is False
    assert rec["indeterminate_by"] == ["b"]
    assert any(row["path"] == MODIC_BELOW for row in c.indeterminate)
    assert all(row["path"] != MODIC_BELOW for row in c.disagreements)


def test_all_indeterminate_is_its_own_status() -> None:
    a = findings("a", levels={"L4-L5": {"endplate_below": {"modic": F(None, "low")}}})
    b = findings("b", levels={"L4-L5": {"endplate_below": {"modic": F(None, "low")}}})
    rec = merge([a, b]).fields[MODIC_BELOW]
    assert rec["status"] == "indeterminate"
    assert rec["value"] is None
    assert rec["disagreement"] is False
    assert rec["indeterminate_by"] == ["a", "b"]


def test_indeterminate_can_be_made_to_count_as_disagreement() -> None:
    a = findings("a", levels={"L4-L5": {"endplate_below": {"modic": F("1")}}})
    b = findings("b", levels={"L4-L5": {"endplate_below": {"modic": F(None, "low")}}})
    rec = merge([a, b], indeterminate_is_disagreement=True).fields[MODIC_BELOW]
    assert rec["disagreement"] is True


def test_continuous_values_use_median_and_tolerance() -> None:
    readers = [
        findings("a", levels={"L4-L5": {"central_canal": {"area_mm2": F(80.0)}}}),
        findings("b", levels={"L4-L5": {"central_canal": {"area_mm2": F(82.0)}}}),
        findings("c", levels={"L4-L5": {"central_canal": {"area_mm2": F(90.0)}}}),
    ]
    rec = merge(readers).fields[AREA]
    assert rec["status"] == "numeric"
    assert rec["value"] == 82.0
    assert rec["range"] == [80.0, 90.0]
    assert rec["disagreement"] is False  # spread 10 is within 20% of 82
    assert rec["agreement"] == 3


def test_continuous_values_far_apart_disagree() -> None:
    readers = [
        findings("a", levels={"L4-L5": {"central_canal": {"area_mm2": F(60.0)}}}),
        findings("b", levels={"L4-L5": {"central_canal": {"area_mm2": F(100.0)}}}),
    ]
    rec = merge(readers).fields[AREA]
    assert rec["value"] == 80.0  # even count: midpoint of the two
    assert rec["disagreement"] is True


def test_signed_continuous_near_zero_uses_absolute_tolerance() -> None:
    readers = [
        findings("a", levels={"L4-L5": {"listhesis": {"translation_mm": F(-0.5)}}}),
        findings("b", levels={"L4-L5": {"listhesis": {"translation_mm": F(0.2)}}}),
    ]
    rec = merge(readers).fields['levels["L4-L5"].listhesis.translation_mm']
    assert rec["disagreement"] is True  # spread 0.7 > 0.2 * max(|median|, 1)


def test_global_section_is_merged_and_listed_last() -> None:
    a = findings(
        "a", levels={"L4-L5": {"disc": {"pfirrmann": F(3)}}}, global_={"alignment": {"scoliosis": F("none")}}
    )
    b = findings(
        "b", levels={"L4-L5": {"disc": {"pfirrmann": F(3)}}}, global_={"alignment": {"scoliosis": F("mild")}}
    )
    c = merge([a, b])
    rec = c.fields["global.alignment.scoliosis"]
    assert rec["status"] == "split" and rec["disagreement"] is True
    assert rec["level"] == "global"
    assert list(c.fields)[-1].startswith("global.")


def test_side_labels_are_used_in_tables() -> None:
    c = merge([findings("a", levels={"L4-L5": {"foramen_R": {"lee_grade": F(2)}}})])
    assert c.fields[FORAMEN_R]["title"] == "Foramen Lee grade (right)"


def test_levels_are_sorted_superior_to_inferior() -> None:
    doc = findings(
        "a",
        levels={
            "L5-S1": {"disc": {"pfirrmann": F(4)}},
            "T12-L1": {"disc": {"pfirrmann": F(2)}},
            "L4-L5": {"disc": {"pfirrmann": F(3)}},
        },
    )
    assert merge([doc]).levels == ["T12-L1", "L4-L5", "L5-S1"]


def test_array_items_group_by_level_and_fuzzy_kind() -> None:
    a = findings("a", focal_lesions=[item("L4-L5", "l4_modic_1", attributes={"size_mm": F(5.0)})])
    b = findings("b", focal_lesions=[item("L4-L5", "l4_modic1", attributes={"size_mm": F(6.0)})])
    c = findings("c", focal_lesions=[item("L5-S1", "l4_modic_1")])
    res = merge([a, b, c])
    groups = res.items["focal_lesions"]
    assert [g["level"] for g in groups] == ["L4-L5", "L5-S1"]
    first = groups[0]
    assert first["kind"] in {"l4_modic_1", "l4_modic1"}  # tie in counts: deterministic pick
    assert groups[1]["kind"] == "l4_modic_1"
    assert first["reported_by"] == ["a", "b"]
    assert first["kind_variants"] == ["l4_modic1", "l4_modic_1"]
    assert first["kind"] == "l4_modic1" and first["kind_by"] == {"a": "l4_modic_1", "b": "l4_modic1"}
    assert first["partial"] is True  # c did not report this one
    assert first["attributes"]["size_mm"]["status"] == "numeric"


def test_different_kinds_are_not_merged() -> None:
    a = findings("a", focal_lesions=[item("L4", "l4_hemangioma"), item("L4", "l4_fracture_old")])
    res = merge([a])
    assert len(res.items["focal_lesions"]) == 2


def test_item_attribute_disagreement_is_listed() -> None:
    a = findings(
        "a",
        pain_generator_hypotheses=[
            item("L4-L5", "discogenic", attributes={"evidence_strength": F("weak")}),
        ],
    )
    b = findings(
        "b",
        pain_generator_hypotheses=[
            item("L4-L5", "discogenic", attributes={"evidence_strength": F("strong")}),
        ],
    )
    res = merge([a, b])
    (group,) = res.items["pain_generator_hypotheses"]
    assert group["disagreement"] is True
    assert group["attributes"]["evidence_strength"]["status"] == "split"
    rows = [d for d in res.disagreements if d["kind"] == "item_attribute"]
    assert rows and rows[0]["path"] == "pain_generator_hypotheses[L4-L5|discogenic].evidence_strength"


def test_item_summaries_and_evidence_are_kept_per_reader() -> None:
    a = findings("a", incidental_findings=[item("global", "renal_cyst", "Simple cyst", confidence="high")])
    b = findings("b", incidental_findings=[item("global", "renal_cyst", "Cyst, 1 cm", confidence="low")])
    (group,) = merge([a, b]).items["incidental_findings"]
    assert group["summaries"] == {"a": "Simple cyst", "b": "Cyst, 1 cm"}
    assert group["confidence"] == {"a": "high", "b": "low"}
    assert len(group["evidence"]) == 1  # both readers cite the same image and slices
    assert group["evidence"][0]["readers"] == ["a", "b"]


def test_evidence_is_unioned_and_keeps_readers() -> None:
    a = _reader("a", 3)
    b = _reader("b", 3)
    rec = merge([a, b]).fields[PF]
    assert rec["evidence"] == [
        {"image": "out/reader/sag_t2.png", "series": "sag_t2", "slices": [12], "readers": ["a", "b"]}
    ]


def test_empty_arrays_produce_empty_groups() -> None:
    res = merge([minimal_valid("a")])
    assert set(res.items) == {"focal_lesions", "incidental_findings", "pain_generator_hypotheses"}
    assert all(len(v) == 1 for v in res.items.values())


def test_sources_as_directory_with_rejects_and_skips(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_reader("a", 3)), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_reader("b", 3)), encoding="utf-8")
    bad = _reader("c", 9)  # out of range grade
    (tmp_path / "c.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "notes.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    res = merge(tmp_path)
    assert res.reader_names == ["a", "b"]
    assert [r["source"] for r in res.rejected] == ["broken.json", "c.json"]
    assert res.skipped == ["notes.json"]
    assert res.fields[PF]["status"] == "unanimous"
    assert any('$.levels["L4-L5"].disc.pfirrmann.value' in e for e in res.rejected[1]["errors"])


def test_duplicate_reader_names_are_rejected() -> None:
    res = merge([_reader("a", 3), _reader("a", 5)])
    assert res.reader_names == ["a"]
    assert res.rejected and "duplicate reader" in res.rejected[0]["errors"][0]
    assert res.fields[PF]["value"] == 3


def test_single_file_and_list_of_files(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text(json.dumps(_reader("a", 3)), encoding="utf-8")
    assert merge(f).reader_names == ["a"]
    assert merge([f]).reader_names == ["a"]


def test_merge_is_deterministic_and_order_independent() -> None:
    docs = [_reader("a", 3), _reader("b", 4), _reader("c", 3, contour="protrusion")]
    one = merge(docs).to_dict()
    two = merge(list(reversed(docs))).to_dict()
    assert one["fields"] == two["fields"]
    assert one["items"] == two["items"]
    assert one["disagreements"] == two["disagreements"]


def test_invalid_only_sources_give_empty_consensus() -> None:
    res = merge([{"schema_version": "1.0", "reader": "x"}])
    assert res.readers == [] and res.fields == {} and res.rejected


def test_warnings_from_validation_are_collected() -> None:
    res = merge([minimal_valid("a")])
    assert res.warnings and all("evidence image not found" in w for w in res.warnings)


def test_settings_record_how_the_merge_was_done() -> None:
    res = merge([_reader("a")], use_confidence=True)
    assert res.settings["use_confidence"] is True
    assert res.settings["numeric_rel_tol"] == pytest.approx(0.2)
    assert res.settings["confidence_weights"] == [0.25, 0.5, 1.0]


def test_consensus_type_and_dict_source() -> None:
    res = merge(_reader("a", 3))
    assert isinstance(res, Consensus)
    assert res.reader_names == ["a"]
    assert res.fields[PF]["value"] == 3


def test_disagreement_table_contents() -> None:
    res = merge([_reader("a", 3, contour="bulge"), _reader("b", 4, contour="bulge")])
    where = {d["path"]: d for d in res.disagreements}
    assert PF in where and CONTOUR not in where
    assert where[PF]["readers"] == ["a", "b"]
    assert where[PF]["status"] == "split"
