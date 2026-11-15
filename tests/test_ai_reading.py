"""AI-reader surface for measured findings (spinescope.ai.reading, render.reading_sheet, review tasks, CLI).

Phantom-based. The review rules are tested directly: a confirm never raises a confidence, a reject is a disagreement
that merge can count, an unconfirmed numbering accepts no review, and nothing is ever marked as human-confirmed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _labels import label_spine
from spinescope.ai.briefs import reading_tasks
from spinescope.ai.reading import (
    FIELD_STATUS,
    REVIEW_POLICY,
    field_status,
    measured_findings,
    render_sheet,
    review_errors,
    review_paths,
    review_to_findings,
    sheet_file_name,
)
from spinescope.analysis.base import run_all
from spinescope.anatomy.anchors import Anchors
from spinescope.cli import main as cli_main
from spinescope.render.reading_sheet import banner_lines, reading_sheet, sheet_series
from spinescope.report import merge, validate
from spinescope.report.instructions import review_instructions

NAMES = ["discs", "herniation", "endplates", "alignment"]


def _anchors_with(anchors: Anchors, status: str) -> Anchors:
    """A copy of the anchors with a numbering status set (tests only: the real status comes from the labeller)."""
    a = Anchors.from_dict(anchors.to_dict())
    a.numbering["status"] = status
    return a


@pytest.fixture(scope="module")
def anchors(phantom_study) -> Anchors:
    return _anchors_with(label_spine(phantom_study, backend="classical"), "probable")


@pytest.fixture(scope="module")
def results(phantom_study, anchors):
    return run_all(phantom_study, anchors, names=NAMES)


@pytest.fixture(scope="module")
def measured(results, anchors) -> dict:
    return measured_findings(results, anchors)


def _ev(series: str = "sag_t2") -> list[dict]:
    return [{"image": "out/sheet.png", "series": series, "slices": [6]}]


def _review(path: str, verdict: str, *, confidence: str = "high", value=None, notes: str = "") -> dict:
    chk: dict = {
        "path": path,
        "verdict": verdict,
        "confidence": confidence,
        "evidence": _ev(),
        "notes": notes,
    }
    if value is not None:
        chk["value"] = value
    return {"kind": "review", "reader": "review_L4_L5", "checks": [chk]}


def _path(measured: dict, level: str, group: str, name: str) -> str:
    return f'levels["{level}"].{group}.{name}'


# ----------------------------------------------------------------- status and registry


def test_field_status_defaults_to_experimental_and_cites_a_doc():
    unknown = field_status("nosuch", "field")
    assert unknown.status == "experimental"
    assert unknown.doc.startswith("docs/analysis/")
    assert field_status("disc", "pfirrmann").status == "validated"
    assert field_status("disc", "contour").status == "partly_validated"
    assert all(s.status in ("validated", "partly_validated", "experimental") for s in FIELD_STATUS.values())
    assert all(s.doc.startswith("docs/analysis/") for s in FIELD_STATUS.values())


def test_measured_findings_carry_status_and_no_paths(measured, phantom_dir):
    assert measured, "the phantom has measured disc fields"
    root = str(phantom_dir["root"])
    for obj in measured.values():
        for group, body in obj.items():
            if group.startswith("_"):
                continue
            for fld in body.values():
                assert fld["status"] in ("validated", "partly_validated", "experimental")
                assert fld["doc"].startswith("docs/analysis/")
                assert fld["confidence"] in ("low", "medium", "high")
                for ev in fld["evidence"]:
                    assert (
                        "/" in ev["image"] or ev["image"]
                    )  # a figure token: analysis/figure, not a file path
                    assert not ev["image"].startswith("/")
    text = json.dumps(measured, default=str)
    assert root not in text


def test_unconfirmed_numbering_caps_confidence_in_measured_findings(results, anchors):
    unconf = measured_findings(results, _anchors_with(anchors, "unconfirmed"))
    for obj in unconf.values():
        for group, body in obj.items():
            if group.startswith("_"):
                continue
            for fld in body.values():
                assert fld["confidence"] == "low"
                assert "not confirmed" in fld["notes"]


# ----------------------------------------------------------------- reading sheets


def test_reading_sheet_is_an_image_within_the_png_limit(phantom_study, anchors, measured):
    lab = next(lv for lv, obj in measured.items() if "pfirrmann" in obj.get("disc", {}))
    img = render_sheet(phantom_study, anchors, measured, lab, "disc", "pfirrmann")
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert max(img.shape[:2]) <= 2000
    blind = render_sheet(phantom_study, anchors, measured, lab, "disc", "pfirrmann", show_value=False)
    assert blind.dtype == np.uint8 and blind.shape[2] == 3


def test_sheet_series_puts_the_field_series_first(phantom_study, anchors):
    lab = next(iter(anchors.disc_labels()))
    keys = sheet_series(phantom_study, anchors, lab, "sag_t2")
    assert keys[0] == "sag_t2"
    assert len(keys) <= 3 and len(set(keys)) == len(keys)


def test_banner_hides_the_value_when_blind_and_shows_position_when_unconfirmed(anchors):
    lab = anchors.disc_labels()[0]
    shown = banner_lines(lab, anchors, field_label="disc.pfirrmann", value_text="3", confidence="medium")
    assert any(line.startswith("measured: 3") for line in shown)
    blind = banner_lines(lab, anchors, field_label="disc.pfirrmann", validation="validated: x")
    assert not any(line.startswith("measured:") for line in blind)
    unconf = banner_lines(lab, _anchors_with(anchors, "unconfirmed"), field_label="disc.pfirrmann")
    assert unconf[0].startswith("position 1 (automatic name ")
    assert "Research and education only" in unconf[-1]


def test_reading_sheet_rejects_an_unknown_level(phantom_study, anchors):
    with pytest.raises(KeyError):
        reading_sheet(phantom_study, anchors, "L9-L10", field_label="x")


def test_sheet_file_name_is_stable():
    assert sheet_file_name("L4-L5", "disc", "pfirrmann") == "sheet_L4-L5_disc.pfirrmann.png"


# ----------------------------------------------------------------- review: validation


def test_review_errors_accepts_a_valid_confirm(measured, anchors):
    lab = "L4-L5"
    path = _path(measured, lab, "disc", "pfirrmann")
    assert review_errors(_review(path, "confirm"), measured, status="probable") == []


def test_review_errors_refuses_unconfirmed_numbering(measured):
    path = _path(measured, "L4-L5", "disc", "pfirrmann")
    errs = review_errors(_review(path, "confirm"), measured, status="unconfirmed")
    assert errs and "unconfirmed" in errs[0]


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d["checks"][0].update(path='levels["L9-L10"].disc.pfirrmann'), "not a measured field"),
        (lambda d: d["checks"].append(dict(d["checks"][0])), "reviewed twice"),
        (lambda d: d["checks"][0].update(verdict="maybe"), "verdict must be"),
        (lambda d: d["checks"][0].update(confidence="certain"), "confidence must be"),
        (lambda d: d["checks"][0].update(evidence=[]), "non-empty list"),
        (lambda d: d["checks"][0].update(value=3), "only allowed with verdict reject"),
        (lambda d: d.update(kind="answer"), 'kind" must be "review"'),
        (lambda d: d.update(reader="bad name!"), "reader must be"),
        (lambda d: d.update(checks=[]), "non-empty list"),
    ],
)
def test_review_errors_rejects_malformed_answers(measured, mutate, needle):
    doc = _review(_path(measured, "L4-L5", "disc", "pfirrmann"), "confirm")
    mutate(doc)
    errs = review_errors(doc, measured, status="probable")
    assert errs and any(needle in e for e in errs), errs


def test_review_rejects_an_alternative_value_outside_the_schema(measured):
    doc = _review(_path(measured, "L4-L5", "disc", "pfirrmann"), "reject", value=9)
    errs = review_errors(doc, measured, status="probable")
    assert errs and any("converted" in e for e in errs)


# ----------------------------------------------------------------- review: trust rule


def test_confirm_never_raises_the_measured_confidence(measured):
    path = _path(measured, "L4-L5", "disc", "pfirrmann")
    lowered = json.loads(json.dumps(measured))
    lowered["L4-L5"]["disc"]["pfirrmann"]["confidence"] = "low"
    doc = _review(path, "confirm", confidence="high")
    out = review_to_findings(doc, lowered)
    assert out["levels"]["L4-L5"]["disc"]["pfirrmann"]["confidence"] == "low"
    assert (
        out["levels"]["L4-L5"]["disc"]["pfirrmann"]["value"] == lowered["L4-L5"]["disc"]["pfirrmann"]["value"]
    )


def test_reject_is_a_disagreement_and_unclear_is_indeterminate(measured):
    path = _path(measured, "L4-L5", "disc", "pfirrmann")
    rej = review_to_findings(_review(path, "reject", value=4), measured)
    fld = rej["levels"]["L4-L5"]["disc"]["pfirrmann"]
    assert fld["value"] == 4 and "reject" in fld["notes"]
    no_alt = review_to_findings(_review(path, "reject"), measured)
    assert no_alt["levels"]["L4-L5"]["disc"]["pfirrmann"]["value"] is None
    unc = review_to_findings(_review(path, "unclear", confidence="high"), measured)
    fld = unc["levels"]["L4-L5"]["disc"]["pfirrmann"]
    assert fld["value"] is None and fld["confidence"] == "low"


def test_every_converted_note_says_it_is_not_a_human_confirmation(measured):
    path = _path(measured, "L4-L5", "disc", "pfirrmann")
    for verdict in ("confirm", "reject", "unclear"):
        out = review_to_findings(_review(path, verdict), measured)
        note = out["levels"]["L4-L5"]["disc"]["pfirrmann"]["notes"]
        assert "not a human confirmation" in note
    assert "only lower trust" in REVIEW_POLICY and "No AI verdict is a human confirmation" in REVIEW_POLICY


def test_converted_review_validates_and_merges_as_a_reader(measured, anchors, tmp_path):
    lab = "L4-L5"
    path = _path(measured, lab, "disc", "pfirrmann")
    rev = _review(path, "reject", value=4, notes="sheet looks like grade 4")
    conv = review_to_findings(rev, measured)
    assert validate(conv).ok
    other = {
        "schema_version": conv["schema_version"],
        "reader": "reader_b",
        "scope": "other reader",
        "levels": {
            lab: {"disc": {"pfirrmann": {"value": 3, "confidence": "medium", "evidence": _ev(), "notes": ""}}}
        },
    }
    c = merge([conv, other])
    assert c.reader_names == ["reader_b", "review_L4_L5"]
    assert any(d.get("path") == path for d in c.disagreements), c.disagreements


def test_review_path_index_covers_every_measured_field(measured):
    paths = review_paths(measured)
    n = sum(len(body) for obj in measured.values() for g, body in obj.items() if not g.startswith("_"))
    assert len(paths) == n


# ----------------------------------------------------------------- review tasks and instructions


def test_reading_tasks_add_review_tasks_only_with_measured_findings(phantom_study, anchors, measured):
    plain = reading_tasks(phantom_study, anchors)
    assert not [t for t in plain if t.kind == "review"]
    tasks = reading_tasks(phantom_study, anchors, measured=measured)
    rev = [t for t in tasks if t.kind == "review"]
    assert rev and {t.levels[0] for t in rev} <= set(anchors.disc_labels())
    t = rev[0]
    assert t.depends_on == ()
    assert "Trust rule" in t.scope and "sheet" in t.scope
    assert "Answer format" in t.prompt and '"kind": "review"' in t.prompt
    assert "Findings format" not in t.prompt


def test_reading_tasks_skip_review_while_unconfirmed(phantom_study, results, anchors):
    un = _anchors_with(anchors, "unconfirmed")
    tasks = reading_tasks(phantom_study, un, measured=measured_findings(results, un))
    assert not [t for t in tasks if t.kind == "review"]


def test_review_instructions_list_allowed_values_and_the_trust_rule():
    text = review_instructions("review_L4_L5", "L4-L5", [("disc", "pfirrmann")])
    assert "1 to 5" in text or "from 1 to 5" in text
    assert "can only lower trust" in text
    assert "not a diagnosis" in text.lower() or "do not diagnose" in text.lower()


# ----------------------------------------------------------------- CLI


def test_cli_findings_writes_findings_and_sheets(phantom_dir, tmp_path, capsys):
    out = tmp_path / "findings"
    rc = cli_main(["findings", str(phantom_dir["root"]), "--outdir", str(out), "--blind"])
    assert rc == 0
    doc = json.loads((out / "findings.json").read_text(encoding="utf-8"))
    assert doc["review_policy"] == REVIEW_POLICY
    sheets = sorted((out / "sheets").glob("*.png"))
    assert sheets, "the phantom has measured fields with sheets"
    assert str(phantom_dir["root"]) not in (out / "findings.json").read_text(encoding="utf-8")
    assert "findings:" in capsys.readouterr().out


def test_cli_review_validates_an_answer(phantom_dir, tmp_path, capsys):
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"kind": "review", "reader": "r", "checks": []}), encoding="utf-8")
    rc = cli_main(["review", str(phantom_dir["root"]), str(answer)])
    assert rc == 1  # the phantom's numbering is unconfirmed, so no review names its levels
    assert "unconfirmed" in capsys.readouterr().err


# ----------------------------------------------------------------- MCP tools


def _first_measured(lf: dict) -> tuple[str, str] | None:
    for lab, obj in lf["levels"].items():
        for group, body in obj.items():
            if not group.startswith("_") and body:
                return lab, f"{group}.{next(iter(body))}"
    return None


def test_mcp_findings_sheet_and_answer_round_trip(phantom_dir, tmp_path):
    from spinescope.ai.mcp_server import SpineTools, ToolError

    tools = SpineTools(output_dir=tmp_path / "mcp_out")
    sid = tools.open_study(str(phantom_dir["root"]))["study_id"]
    tools._anchors_for(sid).numbering["status"] = "probable"  # tests only: the labeller sets this in real use
    lf = tools.list_findings(sid)
    assert lf["numbering"] == "probable" and lf["review_policy"] == REVIEW_POLICY
    assert str(phantom_dir["root"]) not in json.dumps(lf, default=str)
    first = _first_measured(lf)
    assert first is not None
    lab, field = first
    with pytest.raises(ToolError):
        tools.reading_sheet(sid, "L9-L10", field)
    with pytest.raises(ToolError):
        tools.list_findings(sid, level="L9-L10")
    rs = tools.reading_sheet(sid, lab, field)
    assert rs.png[:8] == b"\x89PNG\r\n\x1a\n" and rs.meta["shows_value"] is True and rs.saved_to
    blind = tools.reading_sheet(sid, lab, field, show_value=False)
    assert blind.meta["shows_value"] is False

    group, name = field.split(".")
    doc = _review(_path(tools._measured_for(sid)[0], lab, group, name), "confirm")
    doc["checks"][0]["evidence"] = [{"image": rs.saved_to, "series": "sag_t2", "slices": [6]}]
    res = tools.submit_reader_answer(sid, json.dumps(doc))
    assert res["ok"] and res["counts"] == {"confirm": 1, "reject": 0, "unclear": 0}
    assert res["findings"]["reader"] == "review_L4_L5"
    assert validate(res["findings"]).ok
    assert res["saved_to"] and (tmp_path / "mcp_out").exists()
    assert str(phantom_dir["root"]) not in json.dumps(res, default=str)
    assert tools.submit_reader_answer(sid, "not json") == {"ok": False, "errors": ["not valid JSON"]}


def test_mcp_tasks_include_review_when_numbering_allows(phantom_dir):
    from spinescope.ai.mcp_server import SpineTools

    tools = SpineTools()
    sid = tools.open_study(str(phantom_dir["root"]))["study_id"]
    tools._anchors_for(sid).numbering["status"] = "probable"
    kinds = {t["kind"] for t in tools.tasks(sid)}
    assert {"level", "review", "structure", "holistic", "verify"} <= kinds
