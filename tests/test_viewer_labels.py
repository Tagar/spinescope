"""Viewer label review: payload, correction logic (labels.js under node) against Anchors, export and privacy.

The browser's correction logic (``assets/labels.js``) runs under node on the phantom's labels; the same
steps are replayed through the Python :class:`Anchors` API. The two must agree on levels, centres and
provenance, and the exported JSON must load with :meth:`Anchors.from_json`. Skipped without node.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors
from spinescope.config import DEFAULT
from spinescope.testing.phantom import FAKE_PHI
from spinescope.viewer import build_viewer
from spinescope.viewer.build import _asset
from spinescope.viewer.export import REVIEWER, build_payload, label_policy

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

WHO = REVIEWER
WHEN = "2026-11-14T10:00:00.000Z"
REASON = "lowest disc sits on the sacrum; second disc missed"

#: Runs the page's label logic on a spec (anchors, policy, steps, export, probes); writes the results.
NODE_RUNNER = r"""
const fs = require('fs');
const L = require(process.argv[2]);
const spec = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const P = spec.policy;
const out = { steps: [], export: null, probes: {} };
let model = L.clone(spec.anchors);

function opts(extra) {
  return Object.assign({ who: P.who, why: spec.reason, when: spec.when, source: 'human',
    numConfidence: P.confHuman, confHuman: P.confHuman, confAi: P.confAi }, extra || {});
}
function applyStep(a, st) {
  const o = opts(st.opts);
  if (st.op === 'shift') L.shift(a, st.n, o);
  else if (st.op === 'relabel') L.relabel(a, st.mapping, o);
  else if (st.op === 'drop') L.drop(a, st.label, o);
  else if (st.op === 'add') L.add(a, st.label, st.center, Object.assign(o, st.normal ? { normal: st.normal } : {}));
  else if (st.op === 'confirm') L.confirm(a, o);
  else throw new Error('unknown op ' + st.op);
}
spec.steps.forEach(function (st) {
  const next = L.clone(model);
  try {
    applyStep(next, st);
    model = next;
    out.steps.push({ ok: true, labels: L.discOrder(model), error: null });
  } catch (e) {
    out.steps.push({ ok: false, labels: L.discOrder(model), error: e.message });
  }
});
out.model = model;
if (spec.export) {
  try { out.export = JSON.stringify(L.exportAnchors(model, opts(spec.export)), null, 1); }
  catch (e) { out.export_error = e.message; }
}
const probes = {};
function probe(name, fn) { try { probes[name] = { value: fn(), error: null }; } catch (e) { probes[name] = { value: null, error: e.message }; } }
probe('tiers', function () { return [0.9, 0.8, 0.79, 0.5, 0.49, 0].map(function (c) { return L.tier(c, P); }); });
probe('banner_unconfirmed', function () { return L.banner({ numbering: { confidence: 0.39 }, discs: {}, vertebrae: {}, warnings: [], conus: null }, P).level; });
probe('banner_confirmed', function () { return L.banner({ numbering: { confidence: 0.2, status: 'confirmed' }, discs: {}, vertebrae: {}, warnings: [] }, P).level; });
probe('banner_ok', function () { return L.banner({ numbering: { confidence: 0.9 }, discs: {}, vertebrae: {}, warnings: [] }, P).level; });
probe('banner_none', function () { return L.banner(null, P).level; });
probe('disc_labels', function () { return ['L4-L5', 'L5-S1', 'L5-L6', 'S2-L1', 'L4', 'L4-L9', 'T1-T1', 'T13-L1', 'L6-S1'].map(function (s) { return L.isDiscLabel(s); }); });
probe('shift_l6_disc_refused', function () { return L.shiftLevel('L6-S1', 1); });
probe('relabel_unknown', function () { return L.relabel(L.clone(model), { 'L9-L10': 'L8-L9' }, opts()); });
probe('when_required', function () { return L.shift(L.clone(model), 1, { why: 'x' }); });
probe('export_needs_reason', function () { return L.exportAnchors(model, opts({ why: '   ' })); });
probe('add_inferior_normal_flipped', function () {
  const a = L.clone(model); const lab = L.discOrder(a)[0]; L.drop(a, lab, opts());
  L.add(a, lab, a.discs[L.discOrder(a)[0]].center, opts({ normal: [0.1, 0.0, -1.0] }));
  return a.discs[lab].normal;
});
probe('add_height_median', function () {
  const a = L.clone(model); const lab = L.discOrder(a)[0]; const h0 = a.discs[lab].height_mm; L.drop(a, lab, opts());
  const others = L.discOrder(a).map(function (x) { return a.discs[x].height_mm; }).filter(function (h) { return !!h; });
  L.add(a, lab, [0, 0, 0], opts({ normal: [0, 0, 1] }));
  return { height: a.discs[lab].height_mm, others: others, h0: h0 };
});
probe('add_existing_refused', function () { const a = L.clone(model); const lab = L.discOrder(a)[0]; return L.add(a, lab, [0, 0, 0], opts()); });
probe('add_bad_centre', function () { return L.add(L.clone(model), 'L5-L6', [0, 0], opts()); });
probe('confirm_sets_status', function () { return L.confirm(L.clone(model), opts()).numbering.status; });
out.probes = probes;
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def _run_node(spec: dict[str, Any], tmp: Path) -> dict[str, Any]:
    labels_path = tmp / "labels.js"
    labels_path.write_text(_asset("labels.js"), encoding="utf-8")
    runner = tmp / "run.js"
    runner.write_text(NODE_RUNNER, encoding="utf-8")
    spec_path = tmp / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out_path = tmp / "out.json"
    proc = subprocess.run(
        [NODE, str(runner), str(labels_path), str(spec_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_path.read_text(encoding="utf-8"))


def _policy() -> dict[str, Any]:
    return label_policy(DEFAULT)


# The steps of a realistic review on the phantom (shift, a refused shift, a refused rename, drop, add).
def _steps(missed_centre: list[float]) -> list[dict[str, Any]]:
    return [
        {"op": "shift", "n": 1},
        {"op": "shift", "n": 20},  # leaves the counted range: refused, nothing changes
        {"op": "relabel", "mapping": {"T10-T11": "L4-L5"}},  # duplicate: refused
        {"op": "drop", "label": "L1-L2"},
        {"op": "add", "label": "L1-L2", "center": missed_centre},
        {"op": "relabel", "mapping": {"T10-T11": "T9-T10"}},
    ]


@pytest.fixture(scope="module")
def truth_anchors(phantom) -> Anchors:
    return Anchors.from_truth(phantom.truth)


@pytest.fixture(scope="module")
def review(phantom, tmp_path_factory):
    """Run the scripted review in node and replay it through Anchors. Returns both sides."""
    tmp = tmp_path_factory.mktemp("labels_js")
    anchors = Anchors.from_truth(phantom.truth)
    spec_centre = np.array(phantom.truth["discs"]["L2-L3"]["center"], dtype=float)
    missed = (spec_centre + np.array([0.0, 2.0, 0.5])).tolist()
    steps = _steps(missed)
    spec = {
        "anchors": json.loads(json.dumps(anchors.to_dict(), default=float)),
        "policy": _policy(),
        "reason": REASON,
        "when": WHEN,
        "steps": steps,
        "export": {"why": REASON, "when": WHEN},
    }
    js = _run_node(spec, tmp)

    # the same steps through the Python API (confidence set the same way the page does)
    py = anchors
    py_steps: list[tuple[bool, list[str]]] = []
    for st in steps:
        try:
            if st["op"] == "shift":
                py.shift_labels(
                    st["n"], who=WHO, source="human", why=REASON, confidence=DEFAULT.review_confidence_human
                )
            elif st["op"] == "relabel":
                py.relabel(
                    st["mapping"],
                    who=WHO,
                    source="human",
                    why=REASON,
                    confidence=DEFAULT.review_confidence_human,
                )
            elif st["op"] == "drop":
                py.drop_disc(
                    st["label"],
                    who=WHO,
                    source="human",
                    why=REASON,
                    confidence=DEFAULT.review_confidence_human,
                )
            elif st["op"] == "add":
                py.add_disc(st["label"], st["center"], who=WHO, source="human", why=REASON, config=DEFAULT)
            py_steps.append((True, py.disc_labels()))
        except (ValueError, KeyError):
            py_steps.append((False, py.disc_labels()))
    return {"js": js, "py": py, "py_steps": py_steps, "steps": steps, "phantom": phantom, "tmp": tmp}


# ------------------------------------------------------------------ correction logic matches Anchors
def test_each_step_is_accepted_or_refused_like_anchors(review):
    js_steps = review["js"]["steps"]
    assert len(js_steps) == len(review["py_steps"]) == len(review["steps"])
    for got, (ok, labels) in zip(js_steps, review["py_steps"], strict=True):
        assert got["ok"] == ok, got
        assert got["labels"] == labels
    # the refused steps (range, duplicate) are the second and third, and leave the labels as before
    assert [s["ok"] for s in js_steps] == [True, False, False, True, True, True]
    assert js_steps[1]["labels"] == js_steps[0]["labels"]
    assert js_steps[2]["labels"] == js_steps[0]["labels"]


def test_exported_geometry_matches_python_labels(review):
    text = review["js"]["export"]
    loaded = Anchors.from_json(text)
    py = review["py"]
    assert loaded.disc_labels() == py.disc_labels()
    for lab in py.disc_labels():
        assert np.allclose(loaded.discs[lab].center, py.discs[lab].center, atol=1e-9), lab
        assert np.allclose(loaded.discs[lab].normal, py.discs[lab].normal, atol=1e-9), lab
        assert loaded.discs[lab].height_mm == pytest.approx(py.discs[lab].height_mm)
    assert sorted(loaded.vertebrae) == sorted(py.vertebrae)
    assert loaded.conus == py.conus
    assert loaded.numbering["confidence"] == pytest.approx(py.numbering["confidence"]) == pytest.approx(0.95)


def test_export_matches_to_dict_shape_and_provenance(review):
    text = review["js"]["export"]
    doc = json.loads(text)
    assert set(doc) == set(Anchors.from_truth(review["phantom"].truth).to_dict())
    corr = doc["numbering"]["corrections"]
    actions = [c["action"] for c in corr]
    assert actions == ["shift", "drop", "add", "relabel", "export"]
    assert "undefined" not in corr[0]["what"]
    assert corr[0]["what"] == "shift all labels up 1 level (T11-T12 becomes T10-T11)"
    for entry in corr:
        assert {"who", "what", "why", "when", "action", "source", "detail", "labels_after"} <= set(entry)
        assert entry["who"] == "viewer user"
        assert entry["why"] == REASON
        assert entry["when"] == WHEN
        assert entry["source"] == "human"
        assert entry["what"]
    # the export entry is the last one and counts the changes before it
    assert corr[-1]["detail"] == {"corrections": 4}
    # the Python record_correction keys (action/who/source/why/detail/labels_after) are all present
    assert all(isinstance(c["labels_after"], list) for c in corr)


def test_export_loads_with_from_json_from_path_and_text(review):
    path = review["tmp"] / "export.json"
    path.write_text(review["js"]["export"], encoding="utf-8")
    a = Anchors.from_json(str(path))
    b = Anchors.from_json(review["js"]["export"])
    assert a.disc_labels() == b.disc_labels() == review["py"].disc_labels()
    assert a.to_dict()["discs"].keys() == b.to_dict()["discs"].keys()


def test_add_disc_gets_nearest_normal_and_median_height(review):
    added = review["py"].discs["L1-L2"]
    assert added.evidence and "added L1-L2" in added.evidence[0]
    assert np.linalg.norm(added.normal) == pytest.approx(1.0)
    assert added.normal[2] > 0  # superior-pointing


def test_probes_match_python_rules(review, phantom):
    probes = review["js"]["probes"]
    assert probes["tiers"]["value"] == ["high", "high", "medium", "medium", "low", "low"]
    assert probes["banner_unconfirmed"]["value"] == "unconfirmed"
    assert probes["banner_confirmed"]["value"] == "confirmed"
    assert probes["banner_ok"]["value"] == "ok"
    assert probes["banner_none"]["value"] == "none"
    # L4-L5, L5-S1, L5-L6 (a lumbarized S1 disc), S2-L1 (no S2), L4 (not a disc), L4-L9, T1-T1, T13-L1, L6-S1
    assert probes["disc_labels"]["value"] == [True, True, True, False, False, False, True, False, True]
    # the same refusals as Anchors (ValueError) for an unshiftable disc, an unknown key, a missing when
    assert (
        probes["shift_l6_disc_refused"]["error"]
        and "counted range" in probes["shift_l6_disc_refused"]["error"]
    )
    assert probes["relabel_unknown"]["error"] and "to relabel" in probes["relabel_unknown"]["error"]
    assert probes["when_required"]["error"] and "when" in probes["when_required"]["error"]
    assert probes["export_needs_reason"]["error"] and "reason" in probes["export_needs_reason"]["error"]
    assert (
        probes["add_existing_refused"]["error"]
        and "already exists" in probes["add_existing_refused"]["error"]
    )
    assert probes["add_bad_centre"]["error"] and "three finite" in probes["add_bad_centre"]["error"]
    # an inferior normal is flipped to superior-pointing
    assert probes["add_inferior_normal_flipped"]["value"][2] > 0
    # the height of an added disc is the median of the others' heights (Anchors.add_disc)
    h = probes["add_height_median"]["value"]
    assert h["height"] == pytest.approx(float(np.median(h["others"])))
    assert probes["confirm_sets_status"]["value"] == "confirmed"


def test_python_refuses_the_same_requests(truth_anchors):
    a = Anchors.from_json(truth_anchors.to_json())
    with pytest.raises(ValueError):
        a.shift_labels(20)
    with pytest.raises(ValueError):
        a.relabel({"L5-S1": "L4-L5"})  # L4-L5 already exists
    assert a.disc_labels() == truth_anchors.disc_labels()  # nothing changed


def test_clean_export_and_python_default_policy_agree():
    policy = _policy()
    assert policy["confHuman"] == pytest.approx(DEFAULT.review_confidence_human)
    assert policy["confLow"] == pytest.approx(DEFAULT.viewer_conf_low)
    assert policy["confHigh"] == pytest.approx(DEFAULT.viewer_conf_high)


# ------------------------------------------------------------------ payload and page
def test_payload_embeds_anchors_and_policy(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    payload = build_payload(phantom_study, anchors, title="t")
    assert payload["anchors"] == json.loads(json.dumps(anchors.to_dict(), default=float))
    assert payload["policy"]["who"] == "viewer user"
    assert set(payload["policy"]) == {"who", "confHigh", "confLow", "confHuman", "confAi"}


def test_payload_without_labels_has_no_anchors(phantom_study):
    payload = build_payload(phantom_study, None, title="t")
    assert payload["anchors"] is None
    assert payload["discs"] == []


def test_payload_survives_numpy_scalars(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    anchors.numbering["count"] = np.int64(7)  # not JSON by default; the payload must still build
    payload = build_payload(phantom_study, anchors, title="t")
    assert payload["anchors"]["numbering"]["count"] == 7


def test_page_inlines_label_logic_and_offline(phantom, phantom_study, tmp_path):
    html = build_viewer(phantom_study, tmp_path / "v", anchors=Anchors.from_truth(phantom.truth)).read_text(
        "utf-8"
    )
    assert "SSLabels" in html and "exportAnchors" in html
    assert "<script src" not in html and "<link " not in html
    assert "https://" not in html and "http://" not in html
    assert "fetch(" not in html and "import(" not in html
    assert "window.SPINESCOPE_DATA" in html
    # every form control has an id and a name (the page's own, not generated by the browser)
    static_markup = html.split("<script>", 1)[
        0
    ]  # controls made by the page script are checked in the browser
    for m in re.finditer(r"<(select|input|textarea)\b([^>]*)>", static_markup):
        attrs = m.group(2)
        if 'type="hidden"' in attrs:
            continue
        assert re.search(r'\bid="', attrs), attrs


def test_page_and_export_contain_no_phi(phantom, phantom_study, phantom_dir, tmp_path):
    anchors = Anchors.from_truth(phantom.truth)
    path = build_viewer(phantom_study, tmp_path / "v", anchors=anchors)
    html = path.read_text(encoding="utf-8")
    spec = {
        "anchors": json.loads(json.dumps(anchors.to_dict(), default=float)),
        "policy": _policy(),
        "reason": REASON,
        "when": WHEN,
        "steps": [{"op": "shift", "n": 1}],
        "export": {"why": REASON, "when": WHEN},
    }
    export = _run_node(spec, tmp_path)["export"]
    for text in (html, export):
        for tag, value in FAKE_PHI.items():
            if len(value) >= 4:
                assert value not in text, f"{tag} leaked"
        assert not re.search(r"\d+\.\d+\.\d+\.\d+\.\d+", text), "DICOM UID-like string"
        assert str(phantom_dir["root"]) not in text
        assert "/tmp/" not in text and "/Users/" not in text
        assert "Krisp" not in text and "krisp" not in text
    assert "PHANTOM" not in export


def test_payload_json_is_stable_for_the_page(phantom, phantom_study):
    p1 = build_payload(phantom_study, Anchors.from_truth(phantom.truth), title="t")
    p2 = build_payload(phantom_study, Anchors.from_truth(phantom.truth), title="t")
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)
