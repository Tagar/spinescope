"""User-facing surfaces of the labelling result: CLI (label, inspect --label), MCP tools, reading briefs and tasks,
the shared summary, review status consistency, and the Claude Code skills. Phantom-based; no PHI in any output."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from spinescope.ai.briefs import brief, reading_tasks
from spinescope.anatomy import label_spine
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.numbering import STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED
from spinescope.anatomy.summary import (
    ABSTAIN_POLICY,
    banner,
    labelling_summary,
    numbering_votes,
    one_line,
    status_of,
    summary_lines,
)
from spinescope.anatomy.verify import review_status
from spinescope.cli import main
from spinescope.testing.phantom import FAKE_PHI

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "claude-code" / "spinescope" / "skills"
STATUSES = (STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED)


#: The identifying fields of the phantom's fake PHI (short codes such as the sex "O" are not identifiers).
_IDENTIFYING_TAGS = (
    "PatientName",
    "PatientID",
    "AccessionNumber",
    "InstitutionName",
    "ReferringPhysicianName",
)


def _phi_strings(root: str) -> list[str]:
    return [str(FAKE_PHI[t]) for t in _IDENTIFYING_TAGS] + [root]


def _assert_no_phi(text: str, root: str) -> None:
    for s in _phi_strings(root):
        assert s not in text, "PHI or input path leaked into an output"


def _with_status(anchors: Anchors, status: str, *, sacrum: bool = True) -> Anchors:
    """A copy of ``anchors`` with the numbering status forced (for surface tests of each status)."""
    a = Anchors.from_dict(copy.deepcopy(anchors.to_dict()))
    a.numbering["status"] = status
    a.numbering["confidence"] = {STATUS_CONFIRMED: 0.95, STATUS_PROBABLE: 0.75, STATUS_UNCONFIRMED: 0.3}[
        status
    ]
    if sacrum:
        a.applicability["unconfirmed_reasons"] = []
        a.applicability["coverage"]["sacrum_covered"] = True
    return a


@pytest.fixture(scope="module")
def root(phantom_dir) -> str:
    return str(phantom_dir["root"])


@pytest.fixture(scope="module")
def classical(phantom_study) -> Anchors:
    return label_spine(phantom_study, backend="classical")


@pytest.fixture(scope="module")
def learned(phantom_study) -> Anchors:
    return label_spine(phantom_study)


# ----------------------------------------------------------------- summary


def test_status_policy_and_banner_wording_per_status(classical):
    assert set(ABSTAIN_POLICY) == set(STATUSES)
    assert ABSTAIN_POLICY[STATUS_CONFIRMED] == "use the level names"
    assert "positions only" in ABSTAIN_POLICY[STATUS_UNCONFIRMED]
    assert "verify_labels" in ABSTAIN_POLICY[STATUS_UNCONFIRMED]
    assert "to be confirmed" in ABSTAIN_POLICY[STATUS_PROBABLE]
    for st in STATUSES:
        a = _with_status(classical, st)
        assert status_of(a) == st
        text = banner(a)
        assert {
            STATUS_CONFIRMED: "CONFIRMED",
            STATUS_PROBABLE: "PROBABLE",
            STATUS_UNCONFIRMED: "UNCONFIRMED",
        }[st] in text
        assert labelling_summary(a)["policy"] == ABSTAIN_POLICY[st]


def test_unknown_or_missing_status_counts_as_unconfirmed(classical):
    a = _with_status(classical, STATUS_CONFIRMED)
    a.numbering["status"] = "bogus"
    assert status_of(a) == STATUS_UNCONFIRMED
    a.discs = {}
    assert status_of(a) == STATUS_UNCONFIRMED
    assert "NO LEVELS" in banner(a)


def test_summary_has_stable_keys_and_levels_superior_to_inferior(learned):
    s = labelling_summary(learned)
    assert {
        "status",
        "status_text",
        "policy",
        "tier",
        "confidence",
        "levels",
        "applicability",
        "fov",
        "conus",
    } <= set(s)
    z = [lv["z_mm"] for lv in s["levels"]]
    assert z == sorted(z, reverse=True)
    assert s["conus"]["experimental"] is True
    assert "EXPERIMENTAL" in s["conus"]["line"] or "not located" in s["conus"]["line"]


def test_summary_lines_and_one_line_mention_status_applicability_and_conus(learned):
    lines = summary_lines(learned)
    text = "\n".join(lines)
    assert "Policy:" in text and "applicability" in text and "conus" in text
    line = one_line(learned)
    assert line.startswith("labelling: ") and "conus" in line and "(experimental)" in line
    assert status_of(learned) in line


def test_votes_are_counted_from_numbering(learned):
    agree, counted = numbering_votes(learned)
    assert 0 <= agree <= counted


def test_review_status_rules():
    a = Anchors(discs={"L5-S1": Disc(center=[0, 0, -100], normal=[0, 0, 1])})
    a.numbering = {"confidence": 0.85}
    a.applicability = {"unconfirmed_reasons": []}
    assert review_status(a, "ai_reader") == STATUS_PROBABLE  # an AI answer is never confirmed
    a.numbering["confidence"] = 0.95
    assert review_status(a, "ai_reader") == STATUS_PROBABLE
    assert review_status(a, "human") == STATUS_CONFIRMED
    a.numbering["confidence"] = 0.3
    assert review_status(a, "human") == STATUS_UNCONFIRMED
    a.numbering["confidence"] = 0.95
    a.applicability = {"unconfirmed_reasons": ["sacrum not in the field of view"]}
    assert review_status(a, "human") == STATUS_UNCONFIRMED


# -------------------------------------------------------------------- CLI


def test_label_prints_table_and_writes_anchors_and_overlay(root, tmp_path, capsys):
    out_json, overlay = tmp_path / "anchors.json", tmp_path / "qa.png"
    rc = main(["label", root, "--backend", "classical", "--out", str(out_json), "--overlay", str(overlay)])
    text = capsys.readouterr().out
    assert rc == 0
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.1"
    status = doc["numbering"]["status"]
    assert banner_word(status) in text  # the banner matches the written numbering status
    assert "Policy:" in text and "applicability:" in text and "conus:" in text
    assert "Level" in text and "Confidence" in text and "Tier" in text
    assert "Warnings:" in text
    assert overlay.read_bytes()[:8] == PNG_MAGIC
    _assert_no_phi(text, root)
    _assert_no_phi(out_json.read_text(encoding="utf-8"), root)


def banner_word(status: str) -> str:
    return {
        "confirmed": "LEVELS CONFIRMED",
        "probable": "LEVELS PROBABLE",
        "unconfirmed": "LEVELS UNCONFIRMED",
    }[status]


def test_label_json_is_the_anchors_document(root, capsys):
    rc = main(["label", root, "--backend", "classical", "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema_version"] == "1.1" and doc["discs"]
    assert "labelling" not in doc  # the file form is pure anchors; the summary is in the table
    _assert_no_phi(json.dumps(doc), root)


def test_label_with_learned_backend_default(root, capsys):
    assert main(["label", root]) == 0
    text = capsys.readouterr().out
    assert re.search(r"LEVELS (CONFIRMED|PROBABLE|UNCONFIRMED)", text)
    _assert_no_phi(text, root)


def test_label_rejects_unknown_backend(root):
    with pytest.raises(SystemExit):
        main(["label", root, "--backend", "magic"])


def test_inspect_label_adds_one_line_and_json_block(root, capsys):
    assert main(["inspect", root, "--label"]) == 0
    text = capsys.readouterr().out
    assert "labelling:" in text and "(experimental)" in text
    assert main(["inspect", root, "--label", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["labelling"]["status"] in STATUSES
    assert "stacks" in doc and "info" in doc
    _assert_no_phi(json.dumps(doc, default=str), root)


def test_inspect_without_label_is_unchanged(root, capsys):
    assert main(["inspect", root, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert "labelling" not in doc


def test_cli_help_lists_label(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "label" in capsys.readouterr().out


# -------------------------------------------------------------------- MCP


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    pytest.importorskip("PIL")
    from spinescope.ai.mcp_server import SpineTools

    return SpineTools(output_dir=tmp_path_factory.mktemp("lab19_mcp"))


@pytest.fixture(scope="module")
def sid(tools, root) -> str:
    return tools.open_study(root)["study_id"]


def test_get_anchors_has_status_summary_and_new_fields(tools, sid, root):
    doc = tools.get_anchors(sid)
    assert doc["schema_version"] == "1.1"
    assert doc["labelling"]["status"] in STATUSES
    assert doc["labelling"]["policy"] == ABSTAIN_POLICY[doc["labelling"]["status"]]
    assert "LEVELS" in doc["summary"] and "Policy:" in doc["summary"]
    assert "applicability" in doc and doc["applicability"]["verdict"]
    assert "coverage" in doc["applicability"]
    _assert_no_phi(json.dumps(doc, default=str), root)


def test_verify_labels_metadata_carries_status_and_policy(tools, sid, root):
    res = tools.verify_labels(sid)
    assert res.meta["status"] in STATUSES
    assert res.meta["policy"] == ABSTAIN_POLICY[res.meta["status"]]
    assert res.meta["status_text"]
    _assert_no_phi(json.dumps(res.meta, default=str), root)


def test_apply_unsure_keeps_status_unconfirmed(tools, sid):
    answer = json.dumps(
        {
            "lowest_disc_is": "unsure",
            "missing": [],
            "extra": [],
            "transitional": None,
            "notes": "sacrum not visible",
        }
    )
    res = tools.apply_label_verification(sid, answer)
    assert res["status"] == STATUS_UNCONFIRMED
    assert res["policy"] == ABSTAIN_POLICY[STATUS_UNCONFIRMED]
    assert res["numbering"]["status"] == STATUS_UNCONFIRMED


def test_apply_named_answer_is_consistent_with_applicability(tools, sid, root):
    # The phantom's field of view does not hold the sacrum, so a reader cannot confirm the lowest disc: the
    # review is recorded, but the status stays unconfirmed (never higher than the applicability record allows).
    answer = json.dumps(
        {"lowest_disc_is": "L5-S1", "missing": [], "extra": [], "transitional": False, "notes": "ok"}
    )
    res = tools.apply_label_verification(sid, answer)
    assert res["numbering"]["corrections"][-1]["action"] == "review"
    assert res["status"] == STATUS_UNCONFIRMED
    assert res["status"] == res["numbering"]["status"]
    _assert_no_phi(json.dumps(res, default=str), root)


def test_mcp_status_fields_match_summary_for_each_status(classical):
    from spinescope.ai.mcp_server import _status_fields
    from spinescope.config import DEFAULT

    for st in STATUSES:
        a = _with_status(classical, st)
        fields = _status_fields(a, DEFAULT)
        assert fields["status"] == st
        assert fields["policy"] == ABSTAIN_POLICY[st]


# ----------------------------------------------------------------- briefs


def test_brief_states_status_policy_applicability_fov_and_conus(phantom_study, classical, root):
    unconf = brief(
        phantom_study, _with_status(classical, STATUS_UNCONFIRMED, sacrum=False), reader="reader_a"
    )
    assert "LEVEL NAMING: Level names are UNCONFIRMED" in unconf
    assert "Numbering status: UNCONFIRMED" in unconf
    assert "positions only" in unconf and "verify_labels" in unconf
    assert "Applicability:" in unconf or "applicability:" in unconf
    assert "Field of view:" in unconf and "sacrum NOT in view" in unconf
    assert "EXPERIMENTAL" in unconf and "Do not report the conus" in unconf
    assert "Why the names are not confirmed" in unconf
    _assert_no_phi(unconf, root)


def test_brief_probable_and_confirmed_wording(phantom_study, classical):
    prob = brief(phantom_study, _with_status(classical, STATUS_PROBABLE), reader="reader_a")
    assert "LEVEL NAMING: Level names are PROBABLE" in prob
    assert 'each marked "to be confirmed"' in prob
    assert "verify_labels" not in prob.split("3. Levels and anchors")[1].split("4. Tools")[0]
    conf = brief(phantom_study, _with_status(classical, STATUS_CONFIRMED), reader="reader_a")
    assert "LEVEL NAMING: Level names are CONFIRMED: use the level names" in conf


def test_reading_tasks_carry_the_policy_and_no_phi(phantom_study, classical, root):
    tasks = reading_tasks(phantom_study, _with_status(classical, STATUS_UNCONFIRMED, sacrum=False))
    level = [t for t in tasks if t.kind == "level"]
    assert level, "level tasks expected"
    for t in level:
        assert "Level naming: Level names are UNCONFIRMED" in t.prompt
        assert "verify_labels" in t.prompt
        assert "position" in t.prompt
        _assert_no_phi(t.prompt, root)
    conf_tasks = reading_tasks(phantom_study, _with_status(classical, STATUS_CONFIRMED))
    level_conf = [t for t in conf_tasks if t.kind == "level"]
    assert all("Level naming: Level names are CONFIRMED" in t.prompt for t in level_conf)
    struct = [t for t in conf_tasks if t.kind == "structure"]
    assert all("Level naming:" in t.scope for t in struct)
    holistic = [t for t in conf_tasks if t.kind == "holistic"]
    assert all("Level naming:" not in t.scope for t in holistic)


def test_brief_without_anchors_has_no_status_block(phantom_study):
    text = brief(phantom_study, None, reader="reader_a")
    assert "LEVEL NAMING" not in text and "Numbering status" not in text


# ----------------------------------------------------------------- skills


def _skill(name: str) -> str:
    return (PLUGIN / name / "SKILL.md").read_text(encoding="utf-8")


def test_spine_read_states_the_abstain_policy_and_verify_rule():
    text = _skill("spine-read")
    assert "Abstain policy for the level names" in text
    assert "`confirmed`: use the level names" in text
    assert "`probable`: use the level names" in text
    assert "`unconfirmed`: positions only" in text
    assert "`verify_labels`" in text
    assert "whenever the status is `probable` or `unconfirmed`" in text
    assert "numbering status" in text and "applicability" in text and "experimental" in text


def test_spine_inspect_and_viewer_describe_the_status():
    inspect = _skill("spine-inspect")
    assert "--label" in inspect and "unconfirmed" in inspect
    viewer = _skill("spine-viewer")
    assert "banner" in viewer and "unconfirmed" in viewer and "experimental" in viewer


# ----------------------------------------------------------------- viewer


def test_viewer_page_has_the_applicability_line_and_probable_style():
    from spinescope.viewer.build import _asset

    html = _asset("viewer.html")
    css = _asset("viewer.css")
    assert 'id="applic"' in html and 'id="banner"' in html
    assert "#banner.probable" in css and "#banner.unconfirmed" in css
    assert "L.applicabilityLine(model)" in _asset("app.js")
