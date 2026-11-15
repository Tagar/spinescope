"""Label verification: Anchors corrections (shift, relabel, drop, add, provenance) and the reader loop.

Phantom-based. The ``truth`` anchors are the phantom's own disc positions; "automatic" anchors are made from
them with the labeller's typical counting error (a missed bottom disc, or a missed middle disc) so that the
reader's answer has something to fix.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.verify import (
    ANSWER_SCHEMA,
    AnswerError,
    apply_answer,
    build_packet,
    parse_answer,
    verify_labels,
)
from spinescope.cli import main

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PHI = ("PHANTOM^SPINE", "PHANTOM-0001", "DOE^JANE", "Phantom General Hospital")


def _disc(z: float, normal=(0.0, 0.0, 1.0), h: float = 9.0) -> Disc:
    return Disc(center=[0.0, 0.0, z], normal=list(normal), height_mm=h, confidence=0.8, evidence=["t"])


def _chain(names_z: list[tuple[str, float]]) -> Anchors:
    a = Anchors(method="test")
    for name, z in names_z:
        a.discs[name] = _disc(z)
    a.numbering = {"method": "test", "confidence": 0.7, "notes": ""}
    return a


#: Lowest five discs at 33 mm spacing; L5-S1 at the bottom.
LUMBAR = [("L1-L2", 132.0), ("L2-L3", 99.0), ("L3-L4", 66.0), ("L4-L5", 33.0), ("L5-S1", 0.0)]


# ----------------------------------------------------------------------------- Anchors corrections
def test_shift_labels_renames_discs_vertebrae_and_conus_and_records_provenance():
    a = _chain(LUMBAR)
    a.vertebrae = {"S1": Vertebra([0, 0, -10.0]), "L5": Vertebra([0, 0, 15.0])}
    a.conus = {"z": 150.0, "level": "L1", "confidence": 0.5}
    a.shift_labels(1, who="reader_a", source="ai_reader", why="lowest disc is L4-L5", confidence=0.85)
    assert a.disc_labels() == ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5"]  # every level moves up one
    assert set(a.vertebrae) == {"L4", "L5"}
    assert a.conus["level"] == "T12"  # cranial: L1 moves up to T12
    rec = a.numbering["corrections"][-1]
    assert rec["action"] == "shift" and rec["who"] == "reader_a" and rec["source"] == "ai_reader"
    assert rec["why"] == "lowest disc is L4-L5" and rec["detail"]["offset"] == 1
    assert a.numbering["confidence"] == 0.85
    assert a.disc_labels() == rec["labels_after"]


def test_shift_labels_conus_and_rejects_names_outside_the_count():
    a = _chain(LUMBAR[:4])  # lowest disc is L4-L5 here, so a caudal shift stays in range
    a.conus = {"z": 150.0, "level": "L2", "confidence": 0.5}
    a.shift_labels(-1)
    assert a.conus["level"] == "L3"  # caudal: L2 moves to L3
    b = _chain([("L5-S1", 0.0), ("L4-L5", 33.0)])
    with pytest.raises(ValueError):
        b.shift_labels(-1)  # S1 cannot move caudally
    assert b.disc_labels() == ["L4-L5", "L5-S1"]  # unchanged after the failed call


def test_relabel_validates_before_changing_and_records_mapping():
    a = _chain(LUMBAR)
    a.relabel({"L5-S1": "L5-L6"}, who="reader_b", source="human", why="lumbarized S1")
    assert a.disc_labels()[-1] == "L5-L6"
    assert a.numbering["corrections"][-1]["detail"]["mapping"] == {"L5-S1": "L5-L6"}
    with pytest.raises(KeyError):
        a.relabel({"L9-S9": "L1-L2"})
    with pytest.raises(ValueError):
        a.relabel({"L4-L5": "L3-L4"})  # would duplicate an existing label
    with pytest.raises(ValueError):
        a.relabel({"L4-L5": "not a disc"})
    assert a.disc_labels()[-1] == "L5-L6" and len(a.discs) == 5


def test_drop_and_add_disc_with_provenance_and_geometry_defaults():
    a = _chain(LUMBAR)
    a.drop_disc("L2-L3", who="p", why="thin interval")
    assert "L2-L3" not in a.discs
    with pytest.raises(KeyError):
        a.drop_disc("L2-L3")
    a.add_disc("L2-L3", [0.0, 0.0, 99.0], who="reader_a", source="ai_reader", why="missed")
    d = a.discs["L2-L3"]
    assert d.normal == [0.0, 0.0, 1.0]  # copied from the nearest disc
    assert d.height_mm == 9.0 and 0.0 < d.confidence <= 1.0
    rec = a.numbering["corrections"][-1]
    assert rec["action"] == "add" and rec["detail"]["label"] == "L2-L3"
    with pytest.raises(ValueError):
        a.add_disc("L2-L3", [0.0, 0.0, 99.0])  # already there
    with pytest.raises(ValueError):
        a.add_disc("L9-L8", [0.0, 0.0, 1.0])  # not a disc label
    with pytest.raises(ValueError):
        a.add_disc("L0-L1", [0.0, 0.0, 1.0], normal=[0.0, 0.0, 0.0])
    # a normal pointing inferior is flipped to superior
    a.add_disc("T12-L1", [0.0, 0.0, 160.0], normal=[0.0, 0.0, -1.0])
    assert a.discs["T12-L1"].normal[2] == 1.0


def test_correction_source_and_roundtrip_keep_the_record():
    a = _chain(LUMBAR)
    with pytest.raises(ValueError):
        a.record_correction("x", source="someone")
    a.shift_labels(1, who="reader_a", source="ai_reader", why="r")
    b = Anchors.from_dict(json.loads(a.to_json()))
    assert b.numbering["corrections"] == a.numbering["corrections"]
    # provenance never carries file paths or long free text
    long = "x" * 500
    a.record_correction("note", who="reader_a", source="human", why=long)
    assert len(a.numbering["corrections"][-1]["why"]) == 200


# ----------------------------------------------------------------------------- answers
@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        {"lowest_disc_is": "L6-S1"},
        {"lowest_disc_is": 5},
        {"lowest_disc_is": "L5-S1", "unexpected": 1},
        {"lowest_disc_is": "L5-S1", "missing": ["L9"]},
        {"lowest_disc_is": "L5-S1", "extra": "L2-L3"},
        {"lowest_disc_is": "L5-S1", "missing": ["L3-L4", "L3-L4"]},
        {"lowest_disc_is": "L5-S1", "transitional": "yes"},
        {"lowest_disc_is": "L5-S1", "notes": 3},
        {"missing": []},
        [1, 2],
    ],
)
def test_invalid_answers_are_rejected(bad):
    with pytest.raises(AnswerError):
        parse_answer(bad)


def test_valid_answers_are_normalised_and_fenced_json_is_accepted():
    ans = parse_answer(
        '```json\n{"lowest_disc_is": "l4/l5", "missing": ["l3/l4"], "transitional": null}\n```'
    )
    assert ans.lowest_disc_is == "L4-L5" and ans.missing == ("L3-L4",) and ans.transitional is None
    assert parse_answer({"lowest_disc_is": "UNSURE"}).lowest_disc_is == "unsure"
    assert ANSWER_SCHEMA["required"] == ["lowest_disc_is"]


def test_rejected_answer_changes_nothing():
    a = _chain(LUMBAR)
    before = a.to_dict()
    with pytest.raises(AnswerError):
        apply_answer(a, {"lowest_disc_is": "L4-L5", "missing": ["bad"]})
    assert a.to_dict() == before


# ----------------------------------------------------------------------------- reader loop on truth
def _truth_anchors(phantom) -> Anchors:
    return Anchors.from_truth(phantom.truth)


def test_sacralized_lowest_disc_is_fixed_by_shift(phantom):
    truth = _truth_anchors(phantom)
    # the labeller missed the true lowest disc (L5-S1) and counted the L4-L5 one as L5-S1
    auto = Anchors.from_dict(truth.to_dict())
    del auto.discs["L5-S1"]
    auto.shift_labels(-1)
    assert auto.disc_labels()[-1] == "L5-S1"
    fixed = apply_answer(auto, {"lowest_disc_is": "L4-L5", "transitional": True})
    assert fixed.disc_labels() == [n for n in truth.disc_labels() if n != "L5-S1"]
    assert fixed.numbering["transitional"] is True
    assert fixed.numbering["transitional_kind"] == "sacralized"
    assert fixed.numbering["confidence"] == pytest.approx(0.85)


def test_lumbarized_lowest_disc_renames_only_the_lowest(phantom):
    truth = _truth_anchors(phantom)
    auto = Anchors.from_dict(truth.to_dict())
    fixed = apply_answer(auto, {"lowest_disc_is": "L5-L6", "transitional": True})
    assert fixed.disc_labels()[-1] == "L5-L6"
    assert fixed.disc_labels()[:-1] == truth.disc_labels()[:-1]
    assert fixed.numbering["transitional_kind"] == "lumbarized"


def test_missing_middle_disc_is_placed_between_its_neighbours(phantom):
    truth = _truth_anchors(phantom)
    auto = Anchors.from_dict(truth.to_dict())
    del auto.discs["L3-L4"]
    fixed = apply_answer(auto, {"lowest_disc_is": "L5-S1", "missing": ["L3-L4"]})
    placed = fixed.discs["L3-L4"]
    assert placed.confidence == pytest.approx(0.3)
    assert placed.evidence[0].startswith("added L3-L4")
    err = np.linalg.norm(np.asarray(placed.center) - np.asarray(truth.discs["L3-L4"].center))
    assert err < 6.0  # interpolated between neighbours on a gently curved phantom spine
    assert fixed.numbering["corrections"][-1]["action"] == "review"


def test_extra_disc_is_dropped_and_unsure_changes_no_structure(phantom):
    truth = _truth_anchors(phantom)
    auto = Anchors.from_dict(truth.to_dict())
    auto.add_disc("T10-T11", [0.0, 0.0, 200.0], who="labeller", source="programmatic")
    fixed = apply_answer(auto, {"lowest_disc_is": "L5-S1", "extra": ["T10-T11"]})
    assert "T10-T11" not in fixed.discs
    unsure = apply_answer(
        Anchors.from_dict(truth.to_dict()), {"lowest_disc_is": "unsure", "missing": ["L3-L4"]}
    )
    assert unsure.disc_labels() == truth.disc_labels()
    assert unsure.numbering["confidence"] == pytest.approx(0.3)
    assert any("unsure" in w for w in unsure.warnings)


def test_unplaceable_missing_disc_is_reported_and_costs_confidence():
    a = _chain([("L4-L5", 33.0)])
    out = apply_answer(a, {"lowest_disc_is": "L5-S1", "missing": ["T11-T12"]})
    assert "T11-T12" not in out.discs
    assert out.numbering["confidence"] == pytest.approx(0.85 - 0.1)
    assert any("could not be placed" in w for w in out.warnings)


def test_human_source_uses_the_human_confidence(phantom):
    out = apply_answer(_truth_anchors(phantom), {"lowest_disc_is": "L5-S1"}, who="dr", source="human")
    assert out.numbering["confidence"] == pytest.approx(0.95)
    assert out.numbering["corrections"][-1]["source"] == "human"


def test_verify_labels_takes_any_callable_and_returns_packet_answer(phantom_study, phantom):
    auto = _truth_anchors(phantom)

    def ask(packet):
        assert "lowest mobile disc" in packet.question
        return json.dumps({"lowest_disc_is": "L5-S1", "missing": [], "extra": [], "transitional": False})

    out, _packet, answer = verify_labels(phantom_study, auto, ask, who="r")
    assert answer.lowest_disc_is == "L5-S1"
    assert out.numbering["corrections"][-1]["who"] == "r"
    with pytest.raises(AnswerError):
        verify_labels(phantom_study, auto, lambda p: "nope")


# ----------------------------------------------------------------------------- packet
def test_packet_has_overlays_zoom_question_and_no_phi(phantom_study, phantom):
    packet = build_packet(phantom_study, _truth_anchors(phantom))
    assert next(iter(packet.images)) == "sag_t2_midline"
    assert "sag_t1_midline" in packet.images
    assert "sacrum_zoom" in packet.images
    assert all(png.startswith(PNG_MAGIC) for png in packet.images.values())
    assert packet.predicted_levels == _truth_anchors(phantom).disc_labels()
    text = json.dumps(packet.to_dict())
    assert "Count from the sacrum" in text and "lowest_disc_is" in text
    assert not any(p in text for p in PHI)


def test_packet_without_discs_and_without_series_does_not_raise(phantom_study):
    from spinescope.study import Study

    packet = build_packet(phantom_study, Anchors())
    assert packet.images and "no discs" in " ".join(packet.notes)
    empty = build_packet(Study(stacks={}, info={}), Anchors())
    assert empty.images == {} and "no sagittal series" in empty.notes


def test_packet_write_and_cli_roundtrip(phantom_dir, tmp_path, capsys):
    packet_dir = tmp_path / "packet"
    assert main(["verify-labels", str(phantom_dir["root"]), "--outdir", str(packet_dir)]) == 0
    assert (packet_dir / "packet.json").is_file()
    assert any(p.suffix == ".png" for p in packet_dir.iterdir())
    assert str(phantom_dir["root"]) not in capsys.readouterr().out

    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"lowest_disc_is": "L4-L5", "transitional": True}), encoding="utf-8")
    out = tmp_path / "anchors.json"
    assert main(["verify-labels", str(phantom_dir["root"]), "--answer", str(answer), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["numbering"]["corrections"][-1]["action"] == "review"
    assert main(["verify-labels", str(phantom_dir["root"])]) == 2  # neither --outdir nor --answer
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["verify-labels", str(phantom_dir["root"]), "--answer", str(bad)]) == 2


def test_mcp_verify_tools_apply_once_and_show_corrected_labels(phantom_dir, tmp_path):
    pytest.importorskip("mcp")
    from spinescope.ai.mcp_server import PacketResult, SpineTools, ToolError, _as_content

    tools = SpineTools(output_dir=tmp_path / "mcp_out")
    root = str(phantom_dir["root"])
    sid = tools.open_study(root)["study_id"]
    res = tools.verify_labels(sid)
    assert isinstance(res, PacketResult) and res.images and res.saved_to
    assert len(_as_content(res)) == len(res.images) + 1  # one image per panel plus the JSON metadata
    assert "lowest mobile disc" in res.meta["question"] and root not in json.dumps(res.meta)

    first = tools.apply_label_verification(sid, json.dumps({"lowest_disc_is": "L4-L5", "transitional": True}))
    again = tools.apply_label_verification(sid, json.dumps({"lowest_disc_is": "L4-L5", "transitional": True}))
    assert first["disc_labels"] == again["disc_labels"]  # applied to the automatic labels, not twice
    assert set(tools.get_anchors(sid)["discs"]) == set(first["disc_labels"])
    with pytest.raises(ToolError):
        tools.apply_label_verification(sid, "not json")
    with pytest.raises(ToolError):
        tools.apply_label_verification(sid, '{"lowest_disc_is": "L9-S9"}')


def test_tasks_use_corrected_anchors_from_file(phantom_dir, tmp_path, capsys):
    """verify-labels --answer --out, then tasks --anchors: the level tasks carry the corrected names."""
    root = str(phantom_dir["root"])
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"lowest_disc_is": "L4-L5"}), encoding="utf-8")
    corrected = tmp_path / "corrected.json"
    assert main(["verify-labels", root, "--answer", str(answer), "--out", str(corrected)]) == 0
    capsys.readouterr()
    assert main(["tasks", root, "--anchors", str(corrected)]) == 0
    tasks = json.loads(capsys.readouterr().out)
    level_tasks = {lev for t in tasks if t["kind"] == "level" for lev in t["levels"]}
    corrected_labels = set(json.loads(corrected.read_text(encoding="utf-8"))["discs"])
    assert level_tasks == corrected_labels
    assert "L5-S1" not in level_tasks  # the uncorrected bottom name is gone


def test_open_study_with_verified_anchors_uses_their_levels(phantom_dir, tmp_path):
    pytest.importorskip("mcp")
    from spinescope.ai.mcp_server import SpineTools, ToolError

    tools = SpineTools()
    root = str(phantom_dir["root"])
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"lowest_disc_is": "L4-L5"}), encoding="utf-8")
    corrected = tmp_path / "verified.json"
    main(["verify-labels", root, "--answer", str(answer), "--out", str(corrected)])
    sid = tools.open_study(root, anchors_path=str(corrected))["study_id"]
    assert set(tools.get_anchors(sid)["discs"]) == set(json.loads(corrected.read_text())["discs"])
    with pytest.raises(ToolError):
        tools.open_study(root, anchors_path=str(tmp_path / "missing.json"))


def test_overcount_extra_bottom_disc_then_downward_shift():
    """The labeller counted one disc too many at the bottom: that disc is extra and every name moves down.

    True discs sit at 0 (L5-S1), 33 (L4-L5), 66, 99 and 132 mm (L1-L2), and 165 mm is T12-L1. The labeller
    also has a fake disc at -33 mm, so each true disc carries the name one level too cranial.
    """
    a = Anchors(method="test", numbering={"method": "test", "confidence": 0.7})
    for name, z in [
        ("L5-S1", -33.0),
        ("L4-L5", 0.0),
        ("L3-L4", 33.0),
        ("L2-L3", 66.0),
        ("L1-L2", 99.0),
        ("T12-L1", 132.0),
    ]:
        a.discs[name] = _disc(z)
    out = apply_answer(a, {"lowest_disc_is": "L5-S1", "extra": ["L5-S1"]})
    assert out.disc_labels() == ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    assert out.discs["L5-S1"].center[2] == pytest.approx(0.0)
    assert out.discs["L1-L2"].center[2] == pytest.approx(132.0)
    assert out.numbering["corrections"][-1]["detail"]["dropped"] == ["L5-S1"]


def test_inconsistent_lowest_answer_leaves_anchors_untouched():
    a = _chain([("L3-L4", 66.0), ("L4-L5", 33.0)])  # the displayed bottom is L4-L5
    before = a.to_dict()
    with pytest.raises(AnswerError):
        apply_answer(a, {"lowest_disc_is": "L5-L6"})  # a lumbarized S1 would keep the upper level L5
    assert a.to_dict() == before
