"""Human correction of the level names, end to end on the phantom.

The flow under test (docs/guide/correcting-labels.md)::

    spinescope label STUDY --out anchors.json
    spinescope viewer STUDY outdir --anchors anchors.json
    (a reviewer corrects in the viewer and exports corrected.json; simulated here two ways: the page's own
     label logic under node, and the same steps through the Python Anchors API)
    spinescope apply-labels STUDY corrected.json --out reviewed.json --who NAME --why TEXT

Phantom: the default phantom's field of view stops above the sacrum, so the names can never be confirmed there
(by design). The phantom with 1.5 mm pixels shows the sacrum; the automatic labeller calls its lowest disc L5-S1
where it is really L4-L5 and misses the true L5-S1, so a correction is needed. The reviewer shifts all names up one
level, adds the missing L5-S1 at the disc's true position and confirms.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.review import ReviewError, apply_review, load_anchors_file, study_fingerprint
from spinescope.cli import main
from spinescope.config import DEFAULT
from spinescope.study import load_study
from spinescope.testing.phantom import FAKE_PHI, PhantomSpec, SpinePhantom, write_standard_study
from spinescope.viewer import build_viewer
from spinescope.viewer.build import _asset
from spinescope.viewer.export import REVIEWER

NODE = shutil.which("node")
PX_SACRUM_IN_VIEW = 1.5
REASON = "sacrum in view: lowest disc is L4-L5 and the true L5-S1 was missed; added"
WHEN = "2026-11-15T09:00:00.000Z"
#: What the reviewed labels must be (superior to inferior).
EXPECTED = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def _phantom() -> SpinePhantom:
    return SpinePhantom(
        PhantomSpec(
            degenerated={"L4-L5": 0.2},
            modic1=[("L4-L5", "inf", "L")],
            herniation={"L5-S1": ("central", 5.0)},
        )
    )


@pytest.fixture(scope="module")
def phantom_truth() -> dict[str, Any]:
    return _phantom().truth


@pytest.fixture(scope="module")
def sacrum_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("review_sacrum") / "study"
    write_standard_study(_phantom(), root, px=PX_SACRUM_IN_VIEW)
    return root


@pytest.fixture(scope="module")
def other_root(tmp_path_factory) -> Path:
    """The default phantom: a different geometry (1.2 mm pixels), so its labels must not be accepted for the other."""
    root = tmp_path_factory.mktemp("review_other") / "study"
    write_standard_study(_phantom(), root)
    return root


@pytest.fixture(scope="module")
def auto_file(sacrum_root, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("review_auto") / "anchors.json"
    assert main(["label", str(sacrum_root), "--out", str(out)]) == 0
    return out


def _review_point(truth: dict[str, Any]) -> list[float]:
    """Where the reviewer clicks the missing L5-S1 disc: the true centre (the phantom's ground truth)."""
    return [float(x) for x in truth["discs"]["L5-S1"]["center"]]


def _python_export(auto: Anchors, truth: dict[str, Any]) -> dict[str, Any]:
    """The viewer's steps through the Python correction API, with the page's provenance and confidence."""
    a = Anchors.from_dict(auto.to_dict())
    who = REVIEWER
    a.shift_labels(1, who=who, source="human", why=REASON, confidence=DEFAULT.review_confidence_human)
    a.add_disc("L5-S1", _review_point(truth), who=who, source="human", why=REASON, config=DEFAULT)
    a.record_correction(
        "confirm", who=who, source="human", why=REASON, confidence=DEFAULT.review_confidence_human
    )
    a.numbering["status"] = "confirmed"  # what the page writes on Confirm; the import does not trust it
    n = len(a.numbering["corrections"])
    a.record_correction("export", who=who, source="human", why=REASON, detail={"corrections": n})
    return a.to_dict()


#: Runs the page's own label logic (labels.js) on the auto labels: shift, add, confirm, export.
_NODE_RUNNER = r"""
const fs = require('fs');
const L = require(process.argv[2]);
const spec = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const o = { who: spec.who, why: spec.why, when: spec.when, source: 'human',
  numConfidence: spec.conf, confHuman: spec.conf, confAi: 0.85 };
const a = L.clone(spec.anchors);
L.shift(a, 1, Object.assign({}, o));
L.add(a, 'L5-S1', spec.center, Object.assign({}, o, { clicked: true }));
L.confirm(a, Object.assign({}, o));
fs.writeFileSync(process.argv[4], JSON.stringify(L.exportAnchors(a, Object.assign({}, o)), null, 1));
"""


def _node_export(auto: Anchors, truth: dict[str, Any], tmp: Path) -> dict[str, Any]:
    (tmp / "labels.js").write_text(_asset("labels.js"), encoding="utf-8")
    (tmp / "run.js").write_text(_NODE_RUNNER, encoding="utf-8")
    spec = {
        "anchors": json.loads(json.dumps(auto.to_dict(), default=float)),
        "who": REVIEWER,
        "why": REASON,
        "when": WHEN,
        "conf": DEFAULT.review_confidence_human,
        "center": _review_point(truth),
    }
    (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    subprocess.run(
        [NODE, str(tmp / "run.js"), str(tmp / "labels.js"), str(tmp / "spec.json"), str(tmp / "export.json")],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads((tmp / "export.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corrected(auto_file, phantom_truth, tmp_path_factory) -> dict[str, Path]:
    """Corrected files from the simulated viewer export, one per available path (node and Python)."""
    tmp = tmp_path_factory.mktemp("review_export")
    auto = Anchors.from_json(str(auto_file))
    out: dict[str, Path] = {}
    py = tmp / "corrected_python.json"
    py.write_text(json.dumps(_python_export(auto, phantom_truth), indent=1), encoding="utf-8")
    out["python"] = py
    if NODE is not None:
        js = tmp / "corrected_node.json"
        js.write_text(json.dumps(_node_export(auto, phantom_truth, tmp), indent=1), encoding="utf-8")
        out["node"] = js
    return out


@pytest.fixture(params=["python", "node"])
def export_kind(request: pytest.FixtureRequest, corrected: dict[str, Path]) -> str:
    if request.param not in corrected:
        pytest.skip("node not installed")
    return request.param


@pytest.fixture
def reviewed(sacrum_root, corrected, export_kind, tmp_path, capsys) -> Path:
    out = tmp_path / "reviewed.json"
    rc = main(
        [
            "apply-labels",
            str(sacrum_root),
            str(corrected[export_kind]),
            "--out",
            str(out),
            "--who",
            "Reviewer A",
            "--why",
            REASON,
        ]
    )
    assert rc == 0, capsys.readouterr().err
    return out


# ------------------------------------------------------------------------------------------------ the flow
def test_auto_labels_are_bound_to_the_study_and_flag_the_bottom_disc(auto_file, sacrum_root):
    doc = json.loads(auto_file.read_text(encoding="utf-8"))
    assert doc["numbering"]["study_fingerprint"] == study_fingerprint(
        load_study(sacrum_root), doc["source_stack"]
    )
    # the labeller's lowest disc is the true L4-L5 (the true L5-S1 is missed): the review must move it
    assert doc["numbering"]["status"] != "confirmed"
    assert doc["discs"]["L5-S1"]["center"][2] == pytest.approx(-47.9, abs=3.0)


def test_viewer_build_shows_the_auto_labels(sacrum_root, auto_file, tmp_path):
    out = tmp_path / "viewer"
    assert main(["viewer", str(sacrum_root), str(out), "--anchors", str(auto_file)]) == 0
    html = (out / "spinescope_viewer.html").read_text(encoding="utf-8")
    assert '"label":"L5-S1"' in html  # the compact JSON payload embedded in the page


def test_reviewed_names_status_and_provenance(reviewed, export_kind):
    doc = json.loads(reviewed.read_text(encoding="utf-8"))
    assert sorted(doc["discs"], key=lambda k: -doc["discs"][k]["center"][2]) == EXPECTED
    assert doc["numbering"]["status"] == "confirmed", doc["numbering"]
    recs = doc["numbering"]["corrections"]
    actions = [r["action"] for r in recs]
    assert actions[-3:] == ["confirm", "export", "apply"]
    confirm = [r for r in recs if r["action"] == "confirm"][-1]
    assert confirm["source"] == "human" and confirm["why"] == REASON
    applied = recs[-1]
    assert applied["source"] == "human" and applied["who"] == "Reviewer A" and applied["why"] == REASON
    assert applied["detail"]["status_before"] in ("confirmed", "probable", "unconfirmed")
    assert doc["numbering"]["confidence"] >= 0.9


def test_reviewed_positions_match_the_phantom_truth(reviewed, phantom_truth):
    doc = json.loads(reviewed.read_text(encoding="utf-8"))
    for name in EXPECTED:
        z = doc["discs"][name]["center"][2]
        assert z == pytest.approx(phantom_truth["discs"][name]["center"][2], abs=3.0), name


def test_reviewed_file_is_free_of_phi_and_paths(reviewed, sacrum_root):
    text = reviewed.read_text(encoding="utf-8")
    for token in FAKE_PHI:
        assert token not in text
    assert str(sacrum_root) not in text
    assert "/Users/" not in text and "/tmp/" not in text


def test_downstream_brief_and_tasks_use_the_reviewed_names(reviewed, sacrum_root):
    from spinescope.ai.briefs import brief, reading_tasks

    study = load_study(sacrum_root)
    anchors = load_anchors_file(study, str(reviewed), path=str(sacrum_root))
    text = brief(study, anchors)
    for name in EXPECTED:
        assert name in text
    assert "L5-S1" in text
    level_tasks = {lev for t in reading_tasks(study, anchors) if t.kind == "level" for lev in t.levels}
    assert level_tasks == set(EXPECTED)


def test_downstream_cli_tasks_and_viewer_use_the_reviewed_names(reviewed, sacrum_root, tmp_path, capsys):
    capsys.readouterr()  # the apply-labels table printed by the fixture is not part of this check
    assert main(["tasks", str(sacrum_root), "--anchors", str(reviewed)]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert {lev for t in tasks if t["kind"] == "level" for lev in t["levels"]} == set(EXPECTED)
    out = tmp_path / "viewer_reviewed"
    assert main(["viewer", str(sacrum_root), str(out), "--anchors", str(reviewed)]) == 0
    html = (out / "spinescope_viewer.html").read_text(encoding="utf-8")
    for name in EXPECTED:
        assert f'"{name}"' in html
    assert '"status":"confirmed"' in html.replace(" ", "")


def test_downstream_analysis_and_level_slices_use_the_reviewed_names(reviewed, sacrum_root):
    from spinescope.anatomy.correspondence import level_map, level_slices

    study = load_study(sacrum_root)
    anchors = load_anchors_file(study, str(reviewed), path=str(sacrum_root))
    lm = level_map(study, anchors)
    assert set(lm.discs) == set(EXPECTED)
    assert isinstance(level_slices(lm, "L5-S1"), dict)


def test_downstream_mcp_open_study_uses_the_reviewed_names(reviewed, sacrum_root):
    pytest.importorskip("mcp")
    from spinescope.ai.mcp_server import SpineTools

    tools = SpineTools()
    sid = tools.open_study(str(sacrum_root), anchors_path=str(reviewed))["study_id"]
    doc = tools.get_anchors(sid)
    assert set(doc["discs"]) == set(EXPECTED)
    assert doc["numbering"]["status"] == "confirmed"
    assert "L5-S1" in doc["summary"] or "confirmed" in doc["summary"].lower()


def test_build_viewer_api_accepts_the_reviewed_anchors(reviewed, sacrum_root, tmp_path):
    study = load_study(sacrum_root)
    anchors = load_anchors_file(study, str(reviewed), path=str(sacrum_root))
    path = build_viewer(study, tmp_path / "api", anchors=anchors, title="Phantom (synthetic)")
    assert path.exists()


# ------------------------------------------------------------------------------------ guard rails
def test_corrected_file_of_another_study_is_refused(other_root, auto_file, tmp_path, capsys):
    rc = main(
        ["apply-labels", str(other_root), str(auto_file), "--out", str(tmp_path / "x.json"), "--why", "r"]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "different study" in err and str(other_root) not in err


def test_file_without_a_fingerprint_is_refused(sacrum_root, auto_file, tmp_path, capsys):
    doc = json.loads(auto_file.read_text(encoding="utf-8"))
    del doc["numbering"]["study_fingerprint"]
    p = tmp_path / "unbound.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["apply-labels", str(sacrum_root), str(p), "--why", "r"]) == 2
    assert "not bound to a study" in capsys.readouterr().err


def test_schema_violations_are_refused_without_echoing_values(sacrum_root, auto_file, tmp_path):
    study = load_study(sacrum_root)
    doc = json.loads(auto_file.read_text(encoding="utf-8"))
    doc["patient_name"] = "SMITH^JANE"  # not a schema field
    with pytest.raises(ReviewError) as exc:
        apply_review(study, doc, why="r")
    assert "anchors schema" in str(exc.value) and "SMITH" not in str(exc.value)


def test_bad_confidence_is_refused(sacrum_root, auto_file):
    study = load_study(sacrum_root)
    doc = json.loads(auto_file.read_text(encoding="utf-8"))
    doc["discs"]["L4-L5"]["confidence"] = 1.7
    with pytest.raises(ReviewError, match="anchors schema"):
        load_anchors_file(study, doc, path=str(sacrum_root))


def test_path_in_a_reason_is_refused(sacrum_root, corrected, export_kind):
    study = load_study(sacrum_root)
    doc = json.loads(corrected[export_kind].read_text(encoding="utf-8"))
    with pytest.raises(ReviewError, match="file name or path"):
        apply_review(study, doc, why=f"checked {sacrum_root}", path=str(sacrum_root))


def test_path_shaped_text_is_refused(sacrum_root, corrected, export_kind):
    study = load_study(sacrum_root)
    doc = json.loads(corrected[export_kind].read_text(encoding="utf-8"))
    with pytest.raises(ReviewError, match="looks like a file path"):
        apply_review(study, doc, why="saw /Users/someone/scan in the notes")


def test_malformed_and_unreasoned_requests_are_refused(sacrum_root, corrected, export_kind, capsys):
    bad = sacrum_root.parent / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["apply-labels", str(sacrum_root), str(bad), "--why", "r"]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    study = load_study(sacrum_root)
    doc = json.loads(corrected[export_kind].read_text(encoding="utf-8"))
    doc["numbering"]["corrections"] = [{**c, "why": ""} for c in doc["numbering"]["corrections"]]  # no reason
    with pytest.raises(ReviewError, match="reason is needed"):
        apply_review(study, doc)


def test_confirm_on_a_bone_blind_study_stays_unconfirmed(other_root, tmp_path):
    """Default phantom: the sacrum is out of view. A human confirm is recorded but cannot confirm the names."""
    auto = tmp_path / "other_auto.json"
    assert main(["label", str(other_root), "--out", str(auto)]) == 0
    doc = json.loads(auto.read_text(encoding="utf-8"))
    assert doc["applicability"].get("unconfirmed_reasons")  # the bone is not in view
    doc["numbering"].setdefault("corrections", []).append(
        {
            "action": "confirm",
            "who": "r",
            "source": "human",
            "why": "x",
            "detail": {},
            "labels_after": [],
            "when": WHEN,
        }
    )
    doc["numbering"]["confidence"] = 0.95
    doc["numbering"]["status"] = "confirmed"
    study = load_study(other_root)
    assert load_anchors_file(study, doc, path=str(other_root)).numbering["status"] == "unconfirmed"
    reviewed = apply_review(study, doc, why="reviewed the bone; sacrum not in view")
    assert reviewed.numbering["status"] == "unconfirmed"


def test_ai_reader_answer_cannot_confirm_the_names(sacrum_root, auto_file, tmp_path):
    """An AI relabelling is unconfirmed until a human reviews it; an AI-only review is at most probable."""
    from spinescope.anatomy.verify import apply_answer

    study = load_study(sacrum_root)
    auto = Anchors.from_json(str(auto_file))
    ai = apply_answer(auto, {"lowest_disc_is": "L4-L5"}, who="reader", source="ai_reader")
    ai_doc = ai.to_dict()
    ai_doc["numbering"]["confidence"] = 0.99  # even a high confidence from the reader does not confirm
    ai_doc["numbering"]["status"] = "confirmed"
    anchors = load_anchors_file(study, ai_doc, path=str(sacrum_root))
    assert anchors.numbering["status"] == "unconfirmed"  # relabelled by the AI, no human since
    reviewed = apply_review(study, ai_doc, why="looked at it, not yet confirmed", who="Reviewer A")
    assert reviewed.numbering["status"] == "probable"  # a human looked; no confirm yet
    assert reviewed.numbering["status"] != "confirmed"


def test_apply_labels_cli_writes_nothing_on_refusal(other_root, auto_file, tmp_path):
    out = tmp_path / "must_not_exist.json"
    assert main(["apply-labels", str(other_root), str(auto_file), "--out", str(out), "--why", "r"]) == 2
    assert not out.exists()


def test_label_with_anchors_flag_is_the_same_review(sacrum_root, corrected, export_kind, tmp_path, capsys):
    out = tmp_path / "via_label.json"
    rc = main(
        ["label", str(sacrum_root), "--anchors", str(corrected[export_kind]), "--out", str(out), "--json"]
    )
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["numbering"]["status"] == "confirmed"
    assert json.loads(out.read_text(encoding="utf-8"))["discs"].keys() == doc["discs"].keys()


# ------------------------------------------------------------------- geometry follows the names
def test_drop_disc_removes_canal_and_posterior_entries():
    a = Anchors.from_truth(_phantom().truth)
    a.canal = {"per_level": {"L4-L5": {"z_mm": 1.0}, "L4": {"z_mm": 2.0}, "L3-L4": {"z_mm": 3.0}}}
    a.drop_disc("L4-L5", who="r", source="human", why="test")
    assert "L4-L5" not in a.canal["per_level"] and "L4" in a.canal["per_level"]


def test_shift_moves_canal_and_body_names_with_the_levels():
    a = Anchors.from_truth(_phantom().truth)
    a.canal = {"per_level": {"L4-L5": {"z_mm": 1.0}, "L4": {"z_mm": 2.0}}}
    a.vertebrae["L4"].body = {"level": "L4", "centre": [0.0, 0.0, 0.0]}
    a.shift_labels(1, who="r", source="human", why="test")
    assert set(a.canal["per_level"]) == {"L3-L4", "L3"}
    assert a.canal["per_level"]["L3"]["z_mm"] == 2.0  # the entry follows its level
    assert a.vertebrae["L3"].body["level"] == "L3"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_page_geometry_sync_matches_python(tmp_path):
    """labels.js renames canal/body/posterior names like Anchors does (shift and drop)."""
    base = Anchors.from_truth(_phantom().truth)
    base.canal = {"per_level": {"L4-L5": {"z_mm": 1.0}, "L3-L4": {"z_mm": 1.5}, "L4": {"z_mm": 2.0}}}
    base.vertebrae["L4"].body = {"level": "L4", "centre": [0.0, 0.0, 0.0]}
    base.posterior_elements = {"lamina_proxy": {"L4-L5": {"x": 1}, "L4": {"x": 2}}}
    spec = {"anchors": json.loads(json.dumps(base.to_dict(), default=float))}
    (tmp_path / "labels.js").write_text(_asset("labels.js"), encoding="utf-8")
    # after the shift, L2-L3 is the old L3-L4 (no canal entry of its own): drop it on both sides
    runner = (
        "const fs=require('fs');const L=require(process.argv[2]);const s=JSON.parse(fs.readFileSync(process.argv[3]));"
        "const a=L.clone(s.anchors);const o={who:'w',why:'y',when:'t',source:'human',numConfidence:0.9,confHuman:0.9,"
        "confAi:0.8};L.shift(a,1,o);L.drop(a,'L2-L3',o);"
        "fs.writeFileSync(process.argv[4],JSON.stringify(a));"
    )
    (tmp_path / "run.js").write_text(runner, encoding="utf-8")
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    subprocess.run(
        [
            NODE,
            str(tmp_path / "run.js"),
            str(tmp_path / "labels.js"),
            str(tmp_path / "spec.json"),
            str(tmp_path / "out.json"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    js = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    py = Anchors.from_dict(json.loads(json.dumps(base.to_dict())))
    py.shift_labels(1, who="w", source="human", why="y")
    py.drop_disc("L2-L3", who="w", source="human", why="y")
    assert set(js["canal"]["per_level"]) == set(py.canal["per_level"]) == {"L3-L4", "L3"}
    assert set(js["posterior_elements"]["lamina_proxy"]) == set(py.posterior_elements["lamina_proxy"])
    assert js["vertebrae"]["L3"]["body"]["level"] == "L3" == py.vertebrae["L3"].body["level"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_page_confirm_refused_when_the_bone_is_not_in_view(tmp_path):
    base = Anchors.from_truth(_phantom().truth)
    base.applicability = {"unconfirmed_reasons": ["sacrum not in the field of view"]}
    spec = {"anchors": json.loads(json.dumps(base.to_dict(), default=float))}
    (tmp_path / "labels.js").write_text(_asset("labels.js"), encoding="utf-8")
    runner = (
        "const fs=require('fs');const L=require(process.argv[2]);const s=JSON.parse(fs.readFileSync(process.argv[3]));"
        "const o={who:'w',why:'y',when:'t',source:'human',numConfidence:0.95,confHuman:0.95,confAi:0.8};"
        "try{L.confirm(L.clone(s.anchors),o);console.log('CONFIRMED')}catch(e){console.log('REFUSED '+e.message)}"
    )
    (tmp_path / "run.js").write_text(runner, encoding="utf-8")
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(tmp_path / "run.js"), str(tmp_path / "labels.js"), str(tmp_path / "spec.json")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.stdout.startswith("REFUSED") and "cannot be confirmed" in proc.stdout


def test_fingerprint_is_stable_across_loads(sacrum_root, other_root):
    a = study_fingerprint(load_study(sacrum_root))
    assert a == study_fingerprint(load_study(sacrum_root))
    assert a.startswith("sha256:") and len(a) == len("sha256:") + 16
    assert a != study_fingerprint(load_study(other_root))  # different geometry, different fingerprint
